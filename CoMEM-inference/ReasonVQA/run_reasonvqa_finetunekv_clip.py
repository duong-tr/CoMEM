"""zeroshot Infoseek inference script."""
import os
import sys
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from src_vlm.training.qwenVL_inference2 import Qwen2VLForConditionalGeneration_new
import json
import torch
from PIL import Image
import argparse
from tqdm import tqdm
import time
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import generate_response_knowledge

from io import BytesIO
from streaming import StreamingDataset
import base64 
from tqdm import tqdm
from transformers import AutoProcessor
from pathlib import Path


def load_mds(mds_dir):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        records.append(sample)
    return records

def load_and_process_image(item):
    # Load and preprocess the image
    path = item["image_path"]
    raw_image = Image.open(path).convert("RGB")     
    if raw_image.size[0] > 512 or raw_image.size[1] > 512:
        raw_image = raw_image.resize((512, 512), Image.LANCZOS)       
    return raw_image, item["question"]

def process_images_in_batches(batch_data, question_ids, batch_size, prompt, args):
    ########## Get output saving path ###########
    file_path = os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.similar_num
                    ))
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            output = [json.loads(line) for line in f]
    else:
        output = []
    batch_data = batch_data[len(output):]
    question_ids = question_ids[len(output):]
    # setup device to use
    max_memory = { 
        0: "23GiB",
        1: "23GiB"
    }
    print("Load pretrained model...")
    if 'qwen2.5' in args.model_name:
        checkpoint_path = args.checkpoint_path
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=0,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    elif 'qwen2' in args.model_name:
        checkpoint_path = args.checkpoint_path
        print('load qwen2 model')
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
        model = Qwen2VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=0,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    
    print("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]
        question_ids_subset = question_ids[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_ids, answers = [], []

        # Load and preprocess the images
        start_time = time.time()
        for tmp_id, item in zip(question_ids_subset, batch_subset):
            tmp_img, tmp_q = load_and_process_image(item)
            batch_ids.append(tmp_id)
            tmp_q = prompt.format(tmp_q)
            ####### Find Similar Images #######
            def process_similar_infos(item, similar_num):
                similar_infos = item["retrieval_info"][:similar_num]
                similar_infos_dict = {}
                for idx, info in enumerate(similar_infos):
                    key = idx
                    image_data = base64.b64decode(info['image'])
                    fact_img = Image.open(BytesIO(image_data)).convert("RGB")   
                    if fact_img.size[0] > 512 or fact_img.size[1] > 512:
                        fact_img = fact_img.resize((512, 512), Image.LANCZOS)
                    fact_text = info["passage_content"] or ""
                    similar_infos_dict[key] = {"image": fact_img, "desc": fact_text}
                return similar_infos_dict
            similar_infos = process_similar_infos(item, args.similar_num)
            texts = [item['desc'] for item in similar_infos.values()]
            images = [item['image'] for item in similar_infos.values()]
            ans = generate_response_knowledge(args.model_name, processor, model, tmp_img, tmp_q, texts, images)
            print(ans)
            answers.append(ans)

        print(f"Time for batch {idx}: {time.time() - start_time}")
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
        # save output into jsonl
        with open(file_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
       
    return output
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2.5_clip", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="CoMEM", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/ReasonVQA/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    parser.add_argument("--ds_root_dir", type=str, default="", help="dataset root directory")
    args = parser.parse_args()

    data_path = "CoMEM-inference/ReasonVQA/reasonvqa_filtered.jsonl"

    # Read the input JSONL file
    # print('Read the input JSONL file')
    # batch_data = load_mds(data_path)
    # with open(data_path, 'r') as f:
    #     lang_batch_data = [json.loads(line) for line in f]
    #     lang_batch_data = {lang_item['question_id']: lang_item for lang_item in lang_batch_data}

    import pandas as pd
    batch_data = pd.read_json(data_path, lines=True).to_dict(orient='records')
    # rename column 'question_id' to 'data_id'
    for item in batch_data:
        item['data_id'] = item['question_id']
        del item['question_id']
        item['image_path'] = str(Path(args.ds_root_dir) / item['image_path']).replace("\\", "/")
    lang_batch_data = {lang_item['data_id']: lang_item for lang_item in batch_data}

    # double check data exists:
    not_exist = []
    clean_batch_data = []
    clean_question_ids = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        qid = item['data_id']
        path = item['image_path']
        # check path exists
        if not os.path.exists(path):
            print(f"Image path does not exist: {path}")
            not_exist.append(qid)
        else:
            clean_batch_data.append(item)
            clean_question_ids.append(qid)

    print(len(not_exist))
    print("Number of valid samples: ", len(clean_batch_data))
            
    # Desired batch size
    batch_size = args.batch_size

    PROMPT = """Question: {} 
    For this question, please reference to the given information and perform step-by-step reasoning, to obtain the final answer. 
    Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}"""
    
    # Run the batch processing function
    output = process_images_in_batches(clean_batch_data, clean_question_ids, batch_size, prompt=PROMPT, args=args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.similar_num
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")