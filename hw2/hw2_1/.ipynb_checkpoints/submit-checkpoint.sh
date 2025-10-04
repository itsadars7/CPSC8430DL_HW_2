#!/bin/bash
#SBATCH --job-name=hw2_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_p100|gpu_v100_16gb|gpu_v100_32gb'
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

########## USER SETTINGS ##########
# Path to the data root containing: training_data/, testing_data/, training_label.json, testing_label.json
DATA_DIR="../../MLDS_hw2_1_data/"

# Where to save checkpoints
SAVE_DIR="seq2seq_model"

# Training hyperparameters
EPOCHS=200
BATCH=32
LR=1e-3
HIDDEN=256
EMB=256
MAX_LEN=30
TF_START=1.0
TF_END=0.6
###################################

cd $SLURM_SUBMIT_DIR
mkdir -p logs
# cd Lab/Soil_Moisture/Sensitivity/LSTM
module purge
module load anaconda3/2023.09-0

echo "==> Starting training..."
python model_seq2seq.py train \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --device cuda \
  --input_dim 4096 \
  --hidden "${HIDDEN}" \
  --emb "${EMB}" \
  --max_len "${MAX_LEN}" \
  --tf_start "${TF_START}" \
  --tf_end "${TF_END}"
