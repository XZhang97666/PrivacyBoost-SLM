#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from fnmatch import fnmatch
import datasets
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy
import transformers
from accelerate.utils import set_seed
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from tensorboardX import SummaryWriter
from data import CSQAForT5, StrategyQAForT5, Com2SenseQAForT5, CreakQAForT5, OpenbookQAForT5, MedQAForT5,BioForT5,HocForT5, T5collate_fn
from transformers.utils.versions import require_version
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support



logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

try:
    nltk.data.find("tokenizers/punkt")
except :
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        action="store_true"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    return args

def copy_file(dst, src=os.getcwd()):
    pattern = "*.py"
    copy_dirs = [src,src+"/model"]
    pair_file_list = []
    for path, subdirs, files in os.walk(src):
        for name in files:
            if fnmatch(name, pattern):
                source_file = os.path.join(path, name)
                target_file = os.path.join(path, name).replace(src,dst)
                pair_file_list.append((source_file,target_file))
    for source_file,target_file in pair_file_list:
        if(os.path.dirname(source_file) in copy_dirs):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy(source_file, target_file)

def most_common_answer(cand_list):
    from collections import Counter
    data = Counter(cand_list)
    return data.most_common(1)[0][0]


def eval_checkpoint(model, tokenizer, eval_dataloader, gen_kwargs, args):

    samples_seen = 0
    correct_seen = 0
    ub_correct_seen = 0 # upper bound correctness

    answer_correct_seen = 0
    reasoning_path_correct_seen = 0
    answer_ub_correct_seen = 0
    reasoning_path_ub_correct_seen = 0
    consistency_seen = 0

    results = {}
    meta_list = []


    progress_bar = tqdm(range(len(eval_dataloader)), disable=False)
    # import pdb;pdb.set_trace()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch["src_ids"] = batch["src_ids"].to(args.device)
            batch["src_mask"] = batch["src_mask"].to(args.device)
            generated_tokens = model.generate(
                batch["src_ids"],
                attention_mask=batch["src_mask"],
                **gen_kwargs,
            )

            batch_size = len(batch["src_ids"])
            samples_seen += batch_size
            
            if batch["src_ids"].dim()==3:
                batch["src_ids"]=batch["src_ids"][:, 0, :]
            encoded_tokens = tokenizer.batch_decode(batch["src_ids"], skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if(args.dataset_name=="gsm8k"):
                batch["tgt_ids"][batch["tgt_ids"]<0] = 0
            decoded_labels = tokenizer.batch_decode(batch["tgt_ids"], skip_special_tokens=True)
            if(args.eval_mode=="standard"):
                for i in range(batch_size):
                    cand_list = []
                    meta_info = {}
                    meta_info["question"] = encoded_tokens[i]
                    reasoning_paths = []
                    predict_strings = []
                    for j in range(args.num_return_sequences):
                        reasoning_path = decoded_preds[i*args.num_return_sequences+j]
                        reasoning_paths.append(reasoning_path)
                        if(args.dataset_name=="csqa"):

                            if("Explanation:" in reasoning_path):
                                predict_string = reasoning_path.split("Explanation:")[0].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)","(e)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break

                            elif("answer is" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[-1].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)","(e)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break
                            else:
                                cand = reasoning_path.strip()

                        elif(args.dataset_name=="strategyqa"):

                            if("Explanation:" in reasoning_path):
                                predict_string = reasoning_path.split("Explanation:")[0].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["yes","no"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break

                            elif("answer is" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[-1].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["yes","no"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break
                            else:
                                cand = reasoning_path.strip()

                        elif(args.dataset_name=="obqa"):

                            if("Explanation:" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[0].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break

                            elif("answer is" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[-1].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break
                            else:
                                cand = reasoning_path.strip()
                        elif(args.dataset_name=="medqa"):

                            if("Explanation:" in reasoning_path):
                                predict_string = reasoning_path.split("Explanation:")[0].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break

                            elif("answer is" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[-1].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break
                            else:
                                cand = reasoning_path.strip()

                        elif(args.dataset_name=="mmlu"):

                            if("Explanation:" in reasoning_path):
                                predict_string = reasoning_path.split("Explanation:")[0].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break

                            elif("answer is" in reasoning_path):
                                predict_string = reasoning_path.split("answer is")[-1].strip()
                                predict_strings.append(predict_string)
                                cand = "(none)"
                                for choice in ["(a)","(b)","(c)","(d)"]:
                                    if choice in predict_string:
                                        cand = choice
                                        break
                            else:
                                cand = reasoning_path.strip()
                        

                        cand_list.append(cand)
                    meta_info["reasoning_paths"] = reasoning_paths
                    meta_info["predict_strings"] = predict_strings
                    meta_info["cand_answer_list"] = cand_list
                    meta_info["label"] = decoded_labels[i]
                    meta_info["answer"] = decoded_labels[i].split("####")[-1].strip()
                    cand_answer = most_common_answer(cand_list)
                    label_answer = meta_info["answer"]
                    meta_info["cand_answer"] = cand_answer
                    if(cand_answer==label_answer):
                        correct_seen += 1
                        meta_info["correct"] = True
                    else:
                        meta_info["correct"] = False

                    if(label_answer in cand_list):
                        ub_correct_seen += 1
                        meta_info["ub_correct_seen"] = True
                    else:
                        meta_info["ub_correct_seen"] = False
                    meta_list.append(meta_info)
            progress_bar.update(1)

    if(args.eval_mode=="standard"):

        results["acc"] = 100*correct_seen/samples_seen
        results["ub_acc"] = 100*ub_correct_seen/samples_seen

    results["ub_acc"] = 100*ub_correct_seen/samples_seen
    results["prediction"] = meta_list

    return results

