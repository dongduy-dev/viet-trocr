HOW TO SET UP AND RUN
Step 1 - Install dependencies (one time only)
pip install pandas matplotlib
Step 2 - Place your files in the same folder
my_folder/
    visualization_local.py
    your_training.log
Step 3 - Run it
python visualization_local.py --log your_training.log
This opens all 4 charts (CER, WER, Loss, Runtime) as interactive windows.
OPTIONAL FLAGS
To save charts as PNG files instead of pop-up windows:
python visualization_local.py --log your_training.log --save
This creates: chart_cer.png, chart_wer.png, chart_loss.png, chart_runtime.png
A CSV of parsed metrics is also always saved as parsed_training_metrics.csv.