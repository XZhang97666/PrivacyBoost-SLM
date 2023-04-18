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
from accelerate import Accelerator
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
    T5ForConditionalGeneration
)
from modeling_t5 import FiDT5
from tensorboardX import SummaryWriter
from SLMReason.data import CSQAForT5, MedQAForFiD , MedQAForT5, OBQAForFiD, T5collate_fn,FiDCollator,MMLUForFiD,CSQAForFiD
from transformers.utils.versions import require_version
from run_eval_mem import eval_checkpoint, eval_hoc_checkpoint
import time
from modeling_t5 import FiDT5
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
        "--train_split",
        type=str,
        default=None,
        help="The split of train dataset.",
    )

    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="The split of eval dataset.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--train_max_target_length",
        type=int,
        default=1024,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--eval_max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--save_epoch_interval",
        default=1,
        type=int,
        help="how many epochs to do eval one time")
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--shot", type=int, default=-1, help="Use all training data by default"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If passed, will go to debug mode.",
    )
    #eval
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        action="store_true"
    )
    parser.add_argument("--masked_token_prob", type=float, default=0.0, help="Mask token probaility for training context")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Activate training and validation",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Activate test",
    )
    args = parser.parse_args()

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

def main():
    model=optimizer= train_dataloader=eval_dataloader=test_dataloader =lr_scheduler = None
    args = parse_args()
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        if args.train:
            os.makedirs(args.output_dir, exist_ok=True)
            args_dict = deepcopy(args.__dict__)
            args_dict["lr_scheduler_type"] = str(args.lr_scheduler_type)
            json.dump(args_dict, open(os.path.join(args.output_dir, 'train_args.json'), 'w'), sort_keys=True, indent=2)
            copy_file(args.output_dir+"/code")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tb_writer = SummaryWriter(args.output_dir)


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    

    if args.model_name_or_path:
        if("fid" in args.train_split):
            t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            model = FiDT5(t5.config)
            model.load_t5(t5.state_dict())
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    if(args.dataset_name=="gsm8k"):
        num_added_toks = tokenizer.add_tokens(["<<"])
        print("We have added", num_added_toks, "tokens")

    # model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if(args.dataset_name=="csqa"and 'fid' in args.train_split):
        train_dataset = CSQAForFiD(args.dataset_name, args.train_split, args.shot)
        eval_dataset  = CSQAForFiD (args.dataset_name,args.eval_split, shot=-1) # elif(args.dataset_name=="strategyqa"):
 
    elif (args.dataset_name=="obqa"and 'fid' in args.train_split):
        if args.train:
            train_dataset = OBQAForFiD(args.dataset_name, args.train_split, args.shot)
            eval_dataset  = OBQAForFiD (args.dataset_name,args.eval_split, shot=-1)
        if args.test:
            test_split= args.eval_split.replace("validation","test")
            test_dataset = OBQAForFiD(args.dataset_name,test_split,shot=-1)

    if args.train:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        for index in random.sample(range(len(eval_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")
    if args.test:
        for index in random.sample(range(len(test_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")


    train_datacollate_fn= FiDCollator(args.train_split,tokenizer,args.max_source_length , args.train_max_target_length,args.masked_token_prob )
    eval_datacollate_fn= FiDCollator(args.eval_split,tokenizer,args.max_source_length , args.eval_max_target_length )
    if args.train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=train_datacollate_fn, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_datacollate_fn, batch_size=args.per_device_eval_batch_size, shuffle=False)
    if args.test:
        test_dataloader = DataLoader(test_dataset, collate_fn=eval_datacollate_fn, batch_size=args.per_device_eval_batch_size, shuffle=False)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if(args.model_name_or_path=="t5-3b"):
        from transformers.optimization import Adafactor
        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.train:
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader,lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader,lr_scheduler
        )

    gen_kwargs = {
            "max_length": args.eval_max_target_length,
            "num_return_sequences": args.num_return_sequences,
            "temperature":args.temperature,
            "top_k":args.top_k,
            "do_sample":args.do_sample
        }
    # Train!
    args.device=model.device
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.train:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.

    
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        logging_loss = 0
        best_metric=-1
        t0 = time.time()
        for epoch in range(args.num_train_epochs):
            # outputs1=None
            # outputs2=None
            model.train()
            
            for step, batch in enumerate(train_dataloader):

                if(args.debug):
                    import pdb; pdb.set_trace()

                if 'fid' in args.train_split:
                    loss=model(input_ids=batch["src_ids"], attention_mask=batch["src_mask"], labels = batch["tgt_ids"])[0]
                else:
                    outputs = model(input_ids=batch["src_ids"], attention_mask=batch["src_mask"], labels = batch["tgt_ids"])
                    loss = outputs.loss
                # import pdb; pdb.set_trace()
                loss = loss / args.gradient_accumulation_steps
                logging_loss += loss.item()
                accelerator.backward(loss)

                
                if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if(accelerator.is_main_process):
                        tb_writer.add_scalar("loss", logging_loss, completed_steps)

                        logging_loss = 0
                if completed_steps >= args.max_train_steps:
                    break
            
            model.eval()

            if epoch==0:
                new_time=time.time()
                epoch_time= new_time- t0
                print('{} seconds'.format(epoch_time))
      
            results = eval_checkpoint(model, tokenizer, eval_dataloader, gen_kwargs, args)
            if(args.eval_mode=="standard"):
                metric = results["acc"]
                results["epoch"] = epoch
                if(metric > best_metric):
                    best_metric = metric
                    best_dev = results
                    best_dev_epoch = epoch
                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            os.makedirs(args.output_dir+"/best_model", exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir+"/best_model", save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(args.output_dir+"/best_model")
                            config.save_pretrained(args.output_dir+"/best_model")

            print(best_metric)
            if epoch-best_dev_epoch>10:
                print('early_stop')
                break
        
        meta_string = "split-"+args.eval_split+"_num_return_sequences-"+str(args.num_return_sequences)+"_do_sample-"+str(args.do_sample)+"_temperature-"+str(args.temperature) + "_top_k-"+str(args.top_k)
        best_dev['epoch_time']=epoch_time
        with open(os.path.join(args.output_dir, meta_string+"_best_dev_results.json"), "w") as f:
            json.dump(best_dev, f, indent=4)


  
    if args.test:
        output_file_path=''
        import glob
        file_paths = glob.glob(args.output_dir+"*")
        for file_path in file_paths:
            try:
                with open(os.path.join(file_path, meta_string+"_best_results.json"), "r") as f:
                    best_results=json.load(f)
            except:
                continue
            if len(best_results)!=0:
                output_file_path=file_path
                break
        # if len(best_results)!=0:
            #args.output_dir
        if output_file_path=='':
            output_file_path=args.output_dir
        state_dict = torch.load(output_file_path+"/best_model/pytorch_model.bin", map_location=torch.device(args.device))
        

        model.load_state_dict(state_dict)
        # import pdb;pdb.set_trace()
        model.eval()

        test_results = eval_checkpoint(model, tokenizer, test_dataloader, gen_kwargs, args)

        meta_string = "split-"+test_split+"_num_return_sequences-"+str(args.num_return_sequences)+"_do_sample-"+str(args.do_sample)+"_temperature-"+str(args.temperature) + "_top_k-"+str(args.top_k)
        with open(os.path.join(args.output_dir, meta_string+"_test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    main()
