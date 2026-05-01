"""
TARGETED DIAGNOSTIC v2: Position 1 with decoder_start=0
Checks if model learned [0,0]→char1 or if epochs 8-10 damaged it.
Also checks encoder→decoder cross-attention during generation.
Run in FRESH Colab runtime.
"""

import torch, torch.nn.functional as F, yaml, sys
sys.path.insert(0, "/content/trocr_viet")
from main import setup_model
from data.dataset import LMDBDataset

with open("/content/trocr_viet/config.yaml") as f:
    cfg = yaml.safe_load(f)
device = torch.device("cuda")
processor, model = setup_model(cfg, device)

# Load checkpoint
ckpt = torch.load("/content/drive/MyDrive/OCR/checkpoints/stage1_epoch_010.pt",
                   map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Verify config
print(f"decoder_start_token_id = {model.config.decoder_start_token_id}")
print(f"generation_config.use_cache = {model.generation_config.use_cache}")
print(f"decoder.config.use_cache = {model.decoder.config.use_cache}")

# Load sample
ds = LMDBDataset("/content/lmdb/line_printed/val",
                 target_h=128, target_w=1536, default_data_type="printed")
sample = ds[0]
ref = sample["label"]
pv = processor(images=sample["image"], return_tensors="pt").pixel_values.to(device)
print(f"\nReference: {ref}")
print(f"pixel_values shape: {pv.shape}")

# ═══ TEST A: Teacher-forced with decoder_start=0 ═══
print("\n" + "="*60)
print("TEST A: Teacher-forced forward pass (decoder_start=0)")
print("="*60)

text_enc = processor.tokenizer(ref, padding="max_length", max_length=128,
                                truncation=True, return_tensors="pt")
labels = text_enc.input_ids.to(device)
labels_masked = labels.clone()
labels_masked[labels_masked == processor.tokenizer.pad_token_id] = -100

print(f"Labels (first 10): {labels[0, :10].tolist()}")
print(f"Label tokens: {processor.tokenizer.convert_ids_to_tokens(labels[0, :10].tolist())}")

with torch.no_grad():
    outputs = model(pixel_values=pv, labels=labels_masked)
logits = outputs.logits

print(f"Forward loss: {outputs.loss.item():.4f}")
print()

for pos in range(min(8, logits.shape[1])):
    probs = F.softmax(logits[0, pos], dim=-1)
    top5_vals, top5_ids = probs.topk(5)
    target_id = labels[0, pos].item()
    target_prob = probs[target_id].item()
    target_tok = processor.tokenizer.convert_ids_to_tokens([target_id])[0]
    top5_toks = processor.tokenizer.convert_ids_to_tokens(top5_ids.tolist())
    print(f"  Pos {pos}: target='{target_tok}'(id={target_id}) p={target_prob:.6f}")
    print(f"         top5: {list(zip(top5_toks, top5_ids.tolist(), [f'{v:.4f}' for v in top5_vals.tolist()]))}")

# ═══ TEST B: Manual decoder call with explicit encoder_hidden_states ═══
print("\n" + "="*60)
print("TEST B: Manual decoder with explicit encoder_hidden_states")
print("="*60)

with torch.no_grad():
    enc_out = model.encoder(pixel_values=pv)
    enc_hs = enc_out.last_hidden_state
    print(f"Encoder output shape: {enc_hs.shape}")
    print(f"Encoder output mean/std: {enc_hs.mean().item():.4f} / {enc_hs.std().item():.4f}")

    # Feed first 3 tokens [0, 0, char1] to decoder manually
    first_tokens = labels[0, :3].unsqueeze(0)  # [bos, char1, char2]
    # Prepend decoder_start
    decoder_input = torch.cat([
        torch.tensor([[model.config.decoder_start_token_id]], device=device),
        first_tokens[:, :-1]
    ], dim=1)
    print(f"Decoder input ids: {decoder_input[0].tolist()}")

    dec_out = model.decoder(
        input_ids=decoder_input,
        encoder_hidden_states=enc_hs,
        use_cache=False,
    )
    dec_logits = dec_out.logits
    for pos in range(dec_logits.shape[1]):
        probs = F.softmax(dec_logits[0, pos], dim=-1)
        top5_vals, top5_ids = probs.topk(5)
        top5_toks = processor.tokenizer.convert_ids_to_tokens(top5_ids.tolist())
        print(f"  Pos {pos}: top5: {list(zip(top5_toks, top5_ids.tolist(), [f'{v:.4f}' for v in top5_vals.tolist()]))}")

# ═══ TEST C: Check cross-attention weights ═══
print("\n" + "="*60)
print("TEST C: Cross-attention check — is decoder looking at encoder?")
print("="*60)

with torch.no_grad():
    dec_out2 = model.decoder(
        input_ids=decoder_input,
        encoder_hidden_states=enc_hs,
        use_cache=False,
        output_attentions=True,
    )
    # Cross attention shape: (batch, heads, seq_len, encoder_seq_len)
    cross_attn = dec_out2.cross_attentions
    if cross_attn:
        last_layer = cross_attn[-1]  # Last decoder layer
        print(f"  Cross-attn shape: {last_layer.shape}")
        # Average over heads for position 1
        pos1_attn = last_layer[0, :, 1, :].mean(dim=0)  # avg over heads
        print(f"  Pos 1 cross-attn to encoder: "
              f"max={pos1_attn.max():.4f}, mean={pos1_attn.mean():.6f}, "
              f"sum={pos1_attn.sum():.4f}")
        print(f"  Top 5 encoder positions attended: {pos1_attn.topk(5).indices.tolist()}")
    else:
        print("  No cross-attention outputs available!")

# ═══ TEST D: Compare generation with ZERO encoder vs REAL encoder ═══
print("\n" + "="*60)
print("TEST D: Does the encoder matter? (zero encoder vs real)")
print("="*60)

with torch.no_grad():
    # Real encoder
    dec_real = model.decoder(
        input_ids=torch.tensor([[0, 0]], device=device),
        encoder_hidden_states=enc_hs,
        use_cache=False,
    )
    probs_real = F.softmax(dec_real.logits[0, 1], dim=-1)
    top3_real = probs_real.topk(3)

    # Zero encoder
    zero_hs = torch.zeros_like(enc_hs)
    dec_zero = model.decoder(
        input_ids=torch.tensor([[0, 0]], device=device),
        encoder_hidden_states=zero_hs,
        use_cache=False,
    )
    probs_zero = F.softmax(dec_zero.logits[0, 1], dim=-1)
    top3_zero = probs_zero.topk(3)

    real_toks = processor.tokenizer.convert_ids_to_tokens(top3_real.indices.tolist())
    zero_toks = processor.tokenizer.convert_ids_to_tokens(top3_zero.indices.tolist())

    print(f"  Real encoder pos1 top3: {list(zip(real_toks, [f'{v:.4f}' for v in top3_real.values.tolist()]))}")
    print(f"  Zero encoder pos1 top3: {list(zip(zero_toks, [f'{v:.4f}' for v in top3_zero.values.tolist()]))}")
    print(f"  Are they identical? {torch.allclose(probs_real, probs_zero, atol=1e-3)}")
