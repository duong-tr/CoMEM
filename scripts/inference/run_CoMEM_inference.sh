#!/bin/bash

# Set the model name and script path according to your needs
# You can choose the following models:
# - qwen2
# - qwen2.5

# You can choose the following benchmarks:
# - OK-VQA: CoMEM-inference/OK-VQA/run_okvqa_finetunekv_clip.py
# - A-OKVQA: CoMEM-inference/AOK-VQA/run_aokvqa_finetunekv_clip.py
# - Infoseek: CoMEM-inference/infoseek/run_infoseek_finetunekv_clip.py
# - ViQUAE: CoMEM-inference/Viquae/run_viquae_finetunekv_clip.py
# - MRAG-Bench: CoMEM-inference/MRAG_Bench/run_mrag_finetunekv_clip.py
# - OVEN: CoMEM-inference/OVEN/run_oven_finetunekv_clip.py
# - ReasonVQA: CoMEM-inference/ReasonVQA/run_reasonvqa_finetunekv_clip.py

HF_REPO_ID="WenyiWU0111/continuous-memory-qwen2.5-800"
CHECKPOINT_PATH="checkpoints/continuous-memory-qwen2.5-800"

MODEL_NAME='qwen2.5'
SCRIPT_PATH="CoMEM-inference/ReasonVQA/run_reasonvqa_finetunekv_clip.py"
OUTPUT_DIR="CoMEM-inference/ReasonVQA/output" # Change this to your desired output directory
SIMILAR_NUM=10 # Number of relevant image-text pairs to retrieve
DS_ROOT_DIR="E:/Code/PhD/EKVQA" # Change this to your dataset root directory

# auto download checkpoint if not exists
checkpoint_is_complete() {
    local checkpoint_dir="$1"

    # A valid Transformers checkpoint should contain config.json
    [[ -f "${checkpoint_dir}/config.json" ]] || return 1

    # Accept either safetensors or PyTorch weight files.
    compgen -G "${checkpoint_dir}/*.safetensors" > /dev/null ||
    compgen -G "${checkpoint_dir}/pytorch_model*.bin" > /dev/null
}


download_checkpoint() {
    echo "Downloading checkpoint: ${HF_REPO_ID}"
    echo "Destination: ${CHECKPOINT_PATH}"

    mkdir -p "$CHECKPOINT_PATH"

    hf download "$HF_REPO_ID" --local-dir "$CHECKPOINT_PATH"

    if ! checkpoint_is_complete "$CHECKPOINT_PATH"; then
        echo "Error: checkpoint download appears incomplete." >&2
        exit 1
    fi

    echo "Checkpoint downloaded successfully."
}

if checkpoint_is_complete "$CHECKPOINT_PATH"; then
    echo "Checkpoint already exists at: ${CHECKPOINT_PATH}"
else
    echo "Checkpoint not found or incomplete."
    download_checkpoint
fi


chmod +x $SCRIPT_PATH
echo "Running finetune CoMEM inference..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --similar_num $SIMILAR_NUM --checkpoint_path $CHECKPOINT_PATH --ds_root_dir $DS_ROOT_DIR

echo "All runs completed!"

# # Note: For OVEN, you need to set the split to 'val_entity' and 'val_query' for the entity and query splits respectively.
# SPLIT='val_entity' 
# SCRIPT_PATH="CoMEM-inference/OVEN/run_oven_finetune_clip.py"
# OUTPUT_DIR="CoMEM-inference/OVEN/output" 
# chmod +x $SCRIPT_PATH
# echo "Running baseline inference..."
# CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH --model_name MODEL_NAME --output_dir $OUTPUT_DIR --split $SPLIT --similar_num $SIMILAR_NUM --checkpoint_path $CHECKPOINT_PATH