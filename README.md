# CPSC8430DL_HW
# Adarsha Neupane

## Overview
This project implements a **Sequence-to-Sequence (Seq2Seq) model with attention** for machine translation.  
Key enhancements include:
- **Scheduled Sampling** during training for improved robustness.  
- **Beam Search decoding** during inference for better translation quality.  

The final model outperformed the baseline BLEU score of **0.60** (provided in the homework instructions), achieving an **average BLEU score of 0.6364** using the checkpoint `ep199.pt`.

## Files
- **`submit.sh`** – Shell script for training the model. Handles hyperparameters, checkpoint saving, and execution.  
- **`hw2_seq2seq.sh`** – Shell script for running inference on the test set.  
- **`model_seq2seq.py`** – Core Python implementation of the Seq2Seq model with attention, scheduled sampling, and beam search.

## Training
To train the model:
```bash
bash submit.sh