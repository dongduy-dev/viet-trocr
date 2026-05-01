"""
05_verify_lmdb.py
=================
Reads generated LMDB datasets, decodes a few random labels, 
and reconstructs the image bytes back into PNG files for visual verification.
"""

import io
import lmdb
import random
from pathlib import Path
from PIL import Image

def verify_lmdb(lmdb_dir: str, output_dir: str, num_samples: int = 5):
    lmdb_path = Path(lmdb_dir)
    out_path = Path(output_dir)
    
    if not lmdb_path.exists():
        print(f"[SKIP] LMDB directory not found: {lmdb_path}")
        return

    # Create output directory for reconstructed images
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening LMDB: {lmdb_path}")
    # Open environment in read-only mode to prevent accidental corruption
    env = lmdb.open(
        str(lmdb_path), 
        max_readers=1, 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False
    )
    
    with env.begin(write=False) as txn:
        # 1. Read total number of samples
        num_samples_bytes = txn.get(b"num-samples")
        if not num_samples_bytes:
            print("[ERROR] 'num-samples' key not found. Export might have failed.")
            return
        
        total_samples = int(num_samples_bytes.decode("utf-8"))
        print(f"Total samples in LMDB: {total_samples}")
        
        # 2. Randomly pick N indices to test
        samples_to_check = min(num_samples, total_samples)
        if samples_to_check == 0:
            print("[WARN] LMDB is empty.")
            return
            
        indices = random.sample(range(1, total_samples + 1), samples_to_check)
        
        # 3. Reconstruct and verify
        for idx in sorted(indices):
            img_key = f"image-{idx:08d}".encode("utf-8")
            lbl_key = f"label-{idx:08d}".encode("utf-8")
            
            img_bytes = txn.get(img_key)
            lbl_bytes = txn.get(lbl_key)
            
            if not img_bytes or not lbl_bytes:
                print(f"  [WARN] Missing data for index {idx:08d}")
                continue
            
            label = lbl_bytes.decode("utf-8")
            
            try:
                # Convert bytes back to PIL Image
                img = Image.open(io.BytesIO(img_bytes))
                
                # Construct a clean filename indicating the source LMDB and index
                safe_label = "".join([c if c.isalnum() else "_" for c in label[:10]])
                save_name = f"{lmdb_path.parent.name}_{lmdb_path.name}_{idx:08d}_{safe_label}.png"
                
                img.save(out_path / save_name)
                print(f"  [OK] Index: {idx:08d} | Size: {img.size} | Label: '{label}'")
            except Exception as e:
                print(f"  [ERROR] Failed to reconstruct image at index {idx:08d}: {e}")
                
    env.close()
    print(f"  -> Samples saved to {out_path}\n")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    LMDB_ROOT = PROJECT_ROOT / "lmdb"
    OUTPUT_SAMPLES_DIR = PROJECT_ROOT / "lmdb_verification_samples"
    
    print(f"05_verify_lmdb.py")
    print(f"Project root: {PROJECT_ROOT}")
    print("-" * 55)
    
    # Target the primary datasets used for fine-tuning the model
    targets = [
        LMDB_ROOT / "line_printed" / "train",
        LMDB_ROOT / "line_handwritten" / "train",
        LMDB_ROOT / "word_printed" / "val",     # Check a validation set too
    ]
    
    for target in targets:
        verify_lmdb(str(target), str(OUTPUT_SAMPLES_DIR), num_samples=3)

    print("-" * 55)
    print(f"Verification complete. Check '{OUTPUT_SAMPLES_DIR.name}' folder for the extracted PNGs.")