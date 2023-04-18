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
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.


import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from fnmatch import fnmatch
import numpy as np
import torch
from torch.utils.data import DataLoader
import evaluate
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
    AutoModelForMultipleChoice,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from tensorboardX import SummaryWriter
from SLMReason.data import MedQAForBert,MMLUForBert, MedMACQAForBert, HeadQAForBert, DataCollatorForMultipleChoice
from transformers.utils.versions import require_version
import time
from modeling_bert import BertForMTL
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)




def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
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
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
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
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--mt_weight", type=float, default=None, help="Weight for multitask training")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--masked_token_prob", type=float, default=0.0, help="Mask token probaility for training context")
    
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
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    
    parser.add_argument(
        "--shot", type=int, default=-1, help="Use all training data by default"
    )

    parser.add_argument(
        "--eval_shot", type=int, default=-1, help="Use for eval debug"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
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
        model = BertForMTL.from_pretrained(args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,args=args)
                
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)
    if(args.dataset_name=="medqa"):
        train_dataset = MedQAForBert( args.dataset_name,args.train_split, args.shot)
        eval_dataset  = MedQAForBert(args.dataset_name, args.eval_split, shot=-1)
        if args.test:
            test_split= args.eval_split.replace("validation","test")
            test_dataset  = MedQAForBert(args.dataset_name, test_split, shot=-1)
    elif (args.dataset_name=='mmlu'):
        if args.test:
            test_split= args.eval_split.replace("validation","test")
            test_dataset  = MMLUForBert(args.dataset_name, test_split, shot=-1)

    elif (args.dataset_name=='medmcqa'):
        if args.train:
            train_dataset = MedMACQAForBert( args.dataset_name,args.train_split, args.shot)
            eval_dataset  = MedMACQAForBert(args.dataset_name, args.eval_split, shot=args.eval_shot)

    elif (args.dataset_name=='headqa'):
        if args.train:
            train_dataset = HeadQAForBert( args.dataset_name,args.train_split, args.shot)
            eval_dataset  = HeadQAForBert(args.dataset_name, args.eval_split, shot=args.eval_shot)
        
        if args.test:
            test_split= args.eval_split.replace("validation","test")
            test_dataset  = HeadQAForBert(args.dataset_name, test_split, shot=-1)

    if args.train:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    if args.test:
        for index in random.sample(range(len(test_dataset)), 1):
            logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    
    train_data_collator = DataCollatorForMultipleChoice(config, tokenizer,args.max_seq_length,args.masked_token_prob,args.train_split)
    eval_data_collator = DataCollatorForMultipleChoice(config, tokenizer,args.max_seq_length,args.masked_token_prob,args.eval_split)

    if args.train:
        train_dataloader = DataLoader(train_dataset,collate_fn=train_data_collator, batch_size=args.per_device_train_batch_size,shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_data_collator, batch_size=args.per_device_eval_batch_size, shuffle=False)
    if args.test:
        test_dataloader = DataLoader(test_dataset, collate_fn=eval_data_collator, batch_size=args.per_device_eval_batch_size, shuffle=False)

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

    

    # Scheduler and math around the number of training steps.
    if args.train :
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
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
    
        args.device=model.device
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader,lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,  eval_dataloader, test_dataloader,lr_scheduler
    )
    args.device=model.device

    completed_steps = 0
    logging_loss = 0
    best_metric=-1
    t0 = time.time()
    metric = evaluate.load("accuracy")
    meta_string = "split-"+args.eval_split
    best_results={}

    if args.train:
        for epoch in range(args.num_train_epochs):
            eval_predicted_result=[]
            results={}
            model.train()
            
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch,mt_weight=args.mt_weight)
                loss = outputs.loss

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
            if epoch==0:
                new_time=time.time()
                epoch_time= new_time- t0
                print('{} seconds'.format(epoch_time))

            model.eval()
            progress_bar_eval = tqdm(range(len(eval_dataloader)), disable=False)
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.mc_logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["mc_label"]))
                eval_predicted_result.extend(predictions.tolist())
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
                progress_bar_eval.update(1)
            eval_metric = metric.compute()
            accelerator.print(f"epoch {epoch}: {eval_metric}")
            results["epoch"] = epoch
            eval_acc=eval_metric['accuracy']
            results['dev_acc']=eval_acc
            results['dev_predict']=eval_predicted_result
            if(eval_acc> best_metric):
                best_metric = eval_acc
                best_results= results
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
            if epoch-best_dev_epoch>5:
                print('early_stop')
                break
    

    
    if args.test:
        test_predicted_result=[]
        output_file_path=''
        if len(best_results)==0:
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
        if output_file_path=='':
            output_file_path=args.output_dir
        state_dict = torch.load(output_file_path+"/best_model/pytorch_model.bin", map_location=torch.device(args.device))
        
        model.load_state_dict(state_dict)
        model.eval()
        progress_bar_test = tqdm(range(len(test_dataloader)), disable=False)
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.mc_logits.argmax(dim=-1)
            test_predicted_result.extend(predictions.tolist())
            predictions, references = accelerator.gather_for_metrics((predictions, batch["mc_label"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            progress_bar_test.update(1)

        test_metric = metric.compute()

        accelerator.print(f"test acc: {test_metric}")
        best_results['test_acc']=test_metric['accuracy']
        best_results['test_predict']= test_predicted_result



    if args.train:
        best_results['epoch_time']=epoch_time
    print(args.output_dir)
    with open(os.path.join(args.output_dir, meta_string+"_best_results.json"), "w") as f:
        json.dump(best_results, f, indent=4)
        



if __name__ == "__main__":
    main()
