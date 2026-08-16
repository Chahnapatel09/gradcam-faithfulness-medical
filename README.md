# Grad-CAM Faithfulness in Medical Imaging

Graduate mini-project for CSCI 5501 (Deep Learning Applications), Dalhousie University.

## Research Question
Does Grad-CAM (and its variants Grad-CAM++, HiResCAM) actually reflect what a CNN
uses to make a prediction, or does it produce heatmaps that look plausible without
being faithful to the model's real reasoning?

## Approach
A fine-tuned ResNet-18 binary classifier (PneumoniaMNIST, MedMNIST) is explained
with three CAM variants. Faithfulness is tested via a deletion protocol: mask the
top-X% most salient pixels (per CAM variant) and measure the resulting drop in
accuracy/confidence, compared against random-pixel masking as a control. Faithful
explanations should show a much larger drop under targeted masking than random
masking. Results are further split by correct vs. incorrect predictions and by
model confidence.

## Project Structure
```
src/
  data.py              # PneumoniaMNIST loading
  model.py             # ResNet-18 setup, fine-tuning logic
  train.py             # training loop with checkpointing
  gradcam_utils.py      # wrapper around pytorch-grad-cam (Grad-CAM, Grad-CAM++, HiResCAM)
  faithfulness_eval.py  # masking protocol, deletion curves, AUC
  analysis.py           # correct/incorrect + confidence-stratified analysis
  plots.py               # figures for the report
notebooks/
  main.ipynb            # Colab entry point
checkpoints/             # trained model weights (saved to Drive, not git)
results/                 # CSVs and plots (not git)
report/                  # 4-page IEEE report
```

## Setup
Local editing in VS Code; execution on Google Colab (T4 GPU). Code is synced via
GitHub (`git pull` in Colab); checkpoints/results persist via Colab's Drive mount.

```bash
pip install -r requirements.txt
```
