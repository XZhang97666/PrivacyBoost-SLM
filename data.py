"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import json
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
from datasets import load_dataset
import numpy as np
import random
import glob
from itertools import chain
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizerBase,

)
from transformers.utils import PaddingStrategy
from typing import Optional, Union
import itertools

class DatasetForT5(Dataset):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split):

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.data)



class DatasetForBert(Dataset):

    def __init__(self,task_name,split):
        self.task_name = task_name
        self.split = split

    def __len__(self):
        return len(self.data)


class MedQAForBert(DatasetForBert):

    def __init__(self, task_name, split, shot):
        super().__init__(task_name,split)
        self.shot = shot
        if("train" in self.split):
            self.data = self.load_training_data()
            
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        
        elif("test" in self.split):
            self.data = self.load_test_data()


    def load_validation_data(self):

        if "qac_25" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-50-0.0.json") as f:
                keycontext_dataset = json.load(f)        
        elif "qac_75" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word'in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-random-0.0.json") as f:
                keycontext_dataset = json.load(f) 
        elif 'qac_random_span' in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f) 
        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg_no_pos-validation-0-1272-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-validation-0-1272-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/medqa/context_v6/org_contexteA-validation-0-1272-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            

        dataset = load_dataset('./data/MedQA/BertMC/',split="validation")

        self._num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        self.ending_names = [f"ending{i}" for i in range(self._num_choices )]
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ending in self.ending_names:
                            op=data_i[ending]
                            try:
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                                k+=1
                            except:
                                import pdb;
                                pdb.set_trace()
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i=="none":
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            ending=self.ending_names[choice_idx]
                            op=data_i[ending]
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]

                            
            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1


            data.append(data_i)

        return data

    def load_test_data(self):

        if "qac_25" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-50-0.0.json") as f:
                keycontext_dataset = json.load(f)

        elif "qac_75" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word' in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-random-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_span' in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg_no_pos-test-0-1273-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-test-0-1273-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/medqa/context_v6/org_contexteA-test-0-1273-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        dataset = load_dataset('./data/MedQA/BertMC/',split="test")

        self._num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        self.ending_names = [f"ending{i}" for i in range(self._num_choices )]
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ending in self.ending_names:
                            op=data_i[ending]
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i=="none":
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            ending=self.ending_names[choice_idx]
                            op=data_i[ending]
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]


            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)
        
        return data

    
    def load_training_data(self):

        if "qac_25" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-10178-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-10178-50-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_75" in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-10178-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word'in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-1000-random-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_span' in self.split:
            with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-1000-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f) 

        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg_no_pos-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/medqa/context_v6/org_contexteA-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)



        dataset = load_dataset('./data/MedQA/BertMC/',split="train")

        self._num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        self.ending_names = [f"ending{i}" for i in range(self._num_choices )]

        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]
        if 'wA' in self.split:
            self.context_names = ["sent2"]*self._num_choices 
            


        data=[]
        if 'qac' in self.split:
            indices=range(len(keycontext_dataset))
        else:
            indices=range(len(dataset))
        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    try:
                        data_i['overall']=keycontext_dataset[idx]['overall']
                    except:
                        import pdb;pdb.set_trace()

                    k=0
                    if "all" in self.split:
                        k=0
                        for ending in self.ending_names:
                            op=data_i[ending]
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i=="none":
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            ending=self.ending_names[choice_idx]
                            op=data_i[ending]
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]
            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)

        return data

    


    def __getitem__(self, idx):

        example = self.data[idx]

        second_sentence_ans = [example[end] for end in self.ending_names] 
        label = example['label']

        second_sentences= [example[sent2] for sent2 in self.context_names]
        example['overall'] = example['overall'].replace('\n', ' ')


        return {"sent1":  example["sent1"],
        "answers":  second_sentence_ans,
        "overall":  example['overall'],
        'contexts': second_sentences,
        "mc_label" :  label }



    def __len__(self):
        return len(self.data)




class HeadQAForBert(DatasetForBert):

    def __init__(self, task_name, split, shot):
        super().__init__(task_name,split)
        self.shot = shot
        if("train" in self.split):
            self.data = self.load_training_data()
            
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        
        elif("test" in self.split):
            self.data = self.load_test_data()


    def load_validation_data(self):
        if "eA" in self.split:
            cname='validation-0-1366'
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/headqa/context_v1/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)


        dataset = load_dataset('head_qa',"en",split="validation")

        self._num_choices = len(dataset[0]['answers'])
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ans_dict in data_i['answers']:
                            op=ans_dict['atext']
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i not in ["(a)", "(b)","(c)","(d)"]:
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            op=data_i['answers'][choice_idx]['atext']

                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]
            else:
                data_i['overall']=''
                k=0
                for ans_dict in data_i['answers']:
                    op=ans_dict['atext']
                    data_i[self.context_names[k]]=''
                    k+=1


            data.append(data_i)

        return data

    def load_test_data(self):
        if "eA" in self.split:
            cname='test-0-2742'
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/headqa/context_v1/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)


        dataset = load_dataset('head_qa',"en",split="test")

        self._num_choices = len(dataset[0]['answers'])
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ans_dict in data_i['answers']:
                            op=ans_dict['atext']
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i not in ["(a)", "(b)","(c)","(d)"]:
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            op=data_i['answers'][choice_idx]['atext']

                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                                
            else:
                data_i['overall']=''
                k=0
                for ans_dict in data_i['answers']:
                    op=ans_dict['atext']
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)

        return data

    
    def load_training_data(self):
     
        if "eA" in self.split:
            cname='train-0-2657'
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/headqa/context_v1/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/headqa/context_v1/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)


        dataset = load_dataset('head_qa',"en",split="train")

        self._num_choices = len(dataset[0]['answers'])
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

            

        data=[]
        if 'qac' in self.split:
            indices=range(len(keycontext_dataset))
        else:
            indices=range(len(dataset))
        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ans_dict in data_i['answers']:
                            op=ans_dict['atext']
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if  choice_i not in ["(a)", "(b)","(c)","(d)"]:
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            op=data_i['answers'][choice_idx]['atext']

                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]

            else:
                data_i['overall']=''
                k=0
                for ans_dict in data_i['answers']:
                    op=ans_dict['atext']
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)

        return data

    


    def __getitem__(self, idx):


        example = self.data[idx]

        second_sentence_ans = [ ans_dict['atext'] for ans_dict  in example['answers']] 
        label = example['ra']-1

        second_sentences= [example[sent2] for sent2 in self.context_names]
        example['overall'] = example['overall'].replace('\n', ' ')


        return {"sent1":  example["qtext"],
        "answers":  second_sentence_ans,
        "overall":  example['overall'],
        'contexts': second_sentences,
        "mc_label" :  label }



    def __len__(self):
        return len(self.data)


class MedMACQAForBert(DatasetForBert):
    def __init__(self, task_name, split, shot):
        super().__init__(task_name,split)

        self.shot = shot
        if("train" in self.split):
            self.data = self.load_training_data()
            
        elif("validation" in self.split):
            self.data = self.load_validation_data()

    def load_validation_data(self):


        if "eA" in self.split:
            cname='validation-0-4183'
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/medmcqa/context_v1/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/medmcqa/context_v1/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/medmcqa/context_v1/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        dataset = load_dataset("medmcqa",split="validation")



       
        self.ending_names = [elm for elm in dataset.features.keys() if elm.startswith('op')]
        self._num_choices = len(self.ending_names)
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ending in self.ending_names:
                            op=data_i[ending]
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1

                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[idx]['choice']
                        if choice_i=="none":
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[idx]['choice'][1])-ord('a')
                            ending=self.ending_names[choice_idx]
                            op=data_i[ending]
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=keycontext_dataset[idx][op]

            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1


            data.append(data_i)

        return data

    def load_training_data(self):

        cname='random-train-0-10000'
        if "woneg" in self.split and 'wopos' in self.split:
            with open("./data/GPT3/medmcqa/context_v1/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)

        elif "woneg" in self.split:
            with open("./data/GPT3/medmcqa/context_v1/contexteA_no_neg-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)
        else:
            with open("./data/GPT3/medmcqa/context_v1/org_contexteA-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)



        dataset = load_dataset("medmcqa",split="train")

        with open("./data/medmc/indexes_list_file.json","r") as f:
                idx_list=json.load(f)


        self.ending_names = [elm for elm in dataset.features.keys() if elm.startswith('op')]
        self._num_choices = len(self.ending_names)
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]
        if 'wA' in self.split:
            self.context_names = ["sent2"]*self._num_choices 
        data=[]
        if 'qac' in self.split:
            indices=idx_list[:len(keycontext_dataset)]
        else:
            indices=idx_list
        if(self.shot != -1):
            indices = indices[:self.shot]
        for i in range(len(indices)):
            idx=indices[i]
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[i]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    try:
                        data_i['overall']=keycontext_dataset[i]['overall']
                    except:
                        import pdb;pdb.set_trace()

                    if "all" in self.split:
                        for k, ending in enumerate(self.ending_names):
                            op=data_i[ending]
                            data_i[self.context_names[k]]=keycontext_dataset[i][op]
                    elif "gpt3" in self.split:
                        choice_i=keycontext_dataset[i]['choice']
                        if choice_i=="none":
                            for k in range(len(self.context_names)):
                                data_i[self.context_names[k]]=""
                        else:
                            choice_idx=ord(keycontext_dataset[i]['choice'][1])-ord('a')
                            ending=self.ending_names[choice_idx]
                            op=data_i[ending]
                            for k in range(len(self.context_names)):
                                try:
                                    data_i[self.context_names[k]]=keycontext_dataset[i][op]
                                except:
                                    import pdb;pdb.set_trace()
            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)
        return data



    def __getitem__(self, idx):

        example = self.data[idx]

        

        second_sentence_ans = [example[end] for end in self.ending_names] 
        label = example['cop']

        second_sentences= [example[sent2] for sent2 in self.context_names]
        example['overall'] = example['overall'].replace('\n', ' ')


        return {"sent1":  example['question'],
        "answers":  second_sentence_ans,
        "overall":  example['overall'],
        'contexts': second_sentences,
        "mc_label" :  label }



    def __len__(self):
        return len(self.data)

class MMLUForBert(DatasetForBert):

    def __init__(self, task_name, split, shot):
        super().__init__(task_name,split)
        self.shot = shot
        if("test" in self.split):
            self.data = self.load_test_data()



    def load_test_data(self):

        if "eA" in self.split:
            with open("./data/GPT3/mmlu/org_contexteA-test-0-272-0.0.json") as f:
                keycontext_dataset = json.load(f)

        dataset = load_dataset('./data/mmlu/professional_medicine/',split="test")

        self._num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        self.ending_names = [f"ending{i}" for i in range(self._num_choices )]
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'wA' in self.split:
                    kc=keycontext_dataset[idx]['context']
                    data_i['sent2']=kc
                elif 'eA' in self.split:
                    data_i['overall']=keycontext_dataset[idx]['overall']
                    if "all" in self.split:
                        k=0
                        for ending in self.ending_names:
                            op=data_i[ending]
                            data_i[self.context_names[k]]=keycontext_dataset[idx][op]
                            k+=1
                        

            else:
                data_i['overall']=''
                k=0
                for ending in self.ending_names:
                    op=data_i[ending]
                    data_i[self.context_names[k]]=''
                    k+=1

            data.append(data_i)
        
        return data

    


    def __getitem__(self, idx):
        example = self.data[idx]
        second_sentence_ans = [example[end] for end in self.ending_names] 
        label = example['label']

        second_sentences= [example[sent2] for sent2 in self.context_names]
        example['overall'] = example['overall'].replace('\n', ' ')


        return {"sent1":  example["sent1"],
        "answers":  second_sentence_ans,
        "overall":  example['overall'],
        'contexts': second_sentences,
        "mc_label" :  label }



    def __len__(self):
        return len(self.data)



class CSQAForT5(DatasetForT5):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split, shot, mask_incorrect_reasoning_path):
        super().__init__(max_src_len , max_tgt_len , task_name, tokenizer, split)
        self.shot = shot
        self.mask_incorrect_reasoning_path = mask_incorrect_reasoning_path
        if("train" in self.split):
            self.data = self.load_training_data()
        elif("debug" in self.split):
            self.data = self.debug_load_validation_data()
        elif("validation" in self.split):
            self.data = self.load_validation_data()

    def __getitem__(self, idx):

        if("multitask" in self.split):

            if(self.mask_incorrect_reasoning_path):
                tgt_ids2 = torch.tensor(self.tgt_text2s["input_ids"][idx],dtype=torch.long)*int(self.data[idx]["correct"])
                # import pdb; pdb.set_trace()
            else:
                tgt_ids2 = torch.tensor(targets2["input_ids"][idx],dtype=torch.long)
            return {"src_ids1": torch.tensor(self.src_text1s["input_ids"][idx],dtype=torch.long),
                    "src_ids2": torch.tensor(self.src_text2s["input_ids"][idx],dtype=torch.long),
                    "src_mask1":torch.tensor(self.src_text1s["attention_mask"][idx],dtype=torch.long),
                    "src_mask2":torch.tensor(self.src_text2s["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids1": torch.tensor(self.tgt_text1s["input_ids"][idx],dtype=torch.long),
                    "tgt_ids2": tgt_ids2}
        else:
            return {"src_ids": torch.tensor(self.src_texts["input_ids"][idx],dtype=torch.long),
                    "src_mask":torch.tensor(self.src_texts["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids": torch.tensor(self.tgt_texts["input_ids"][idx],dtype=torch.long)}

    
   

    def load_validation_data(self):
        data = []
        dataset = load_dataset('commonsense_qa',split="validation")

        if 'qae' in self.split:
            with open("./data/GPT3/csqa/explain-validation-0-1221-0.0.json") as f:
                keycontext_dataset  = json.load(f)
        else:
            with open("./data/GPT3/csqa/contextwA-validation-0-1221-0.0.json") as f:
                keycontext_dataset = json.load(f)
        for i in range(len(dataset)):
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['choices']["label"])):
                answer_choices += "("+dataset[i]['choices']["label"][j].lower()+") "+dataset[i]['choices']["text"][j]+"\n"
            question = "Q: "+dataset[i]['question'] +"\n" 
            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()

            if 'qakc' in self.split:
                src_text+=answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
            elif "qkca" in  self.split:
                src_text+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
            elif "qak" in  self.split:
                src_text+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nA:"
            elif ("qac" in  self.split) :
                src_text+= answer_choices+"Context: "+keycontext_dataset[i]['context']+"\nA:"
            elif 'qae' in self.split:
                src_text+= answer_choices+"Context: "+keycontext_dataset[i]['explanation']+"\nA:"
            else:
                src_text +=answer_choices+"A:"

            data_i["src_text"]= src_text
            data_i["tgt_text"] = "("+dataset[i]['answerKey'].lower()+")"
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_training_data(self):
        data = []
        indices = []
        with open("./data/GPT3/csqa/cot-train-0-9741-0.0.json") as f:
            cot_dataset = json.load(f)
        with open("./data/GPT3/csqa/explain-train-0-9741-0.0.json") as f:
            explain_dataset = json.load(f)
        with open("./data/GPT3/csqa/cotexplain-train-0-9741-0.0.json") as f:
            cotexplain_dataset = json.load(f)
        
        if 'withA' in self.split:
            with open("./data/GPT3/csqa/contextwA-train-0-9741-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'woA' in  self.split:
            with open("./data/GPT3/csqa/keywords-context-train-0-400-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'eachA' in  self.split:
            with open("./data/GPT3/csqa/keywords-eachA-train-0-400-0.0.json") as f:
                keycontext_dataset = json.load(f)

        for idx in range(len(cot_dataset)): #len(cot_dataset)
            if("correct" in self.split):
                if(cot_dataset[idx]["correct"]):
                    indices.append(idx)
            elif("all" in self.split):
                indices.append(idx)

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        
        dataset = load_dataset('commonsense_qa',split="train")
        for i in indices:
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['choices']["label"])):
                answer_choices += "("+dataset[i]['choices']["label"][j].lower()+") "+dataset[i]['choices']["text"][j]+"\n"
            question = "Q: "+dataset[i]['question'] +"\n"
            if("multitask" in self.split):
                src_text1 = "qta " + question
                src_text2 = "qtr " + question
                tgt_text1 = "("+dataset[i]['answerKey'].lower()+")"
                # import pdb; pdb.set_trace()
                if("EXPLANATION" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["explanation"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["explanation"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["explanation"]
                        data_i["correct"] = True
                elif("COT" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["cot"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["cot"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["cot"]
                        data_i["correct"] = True
                elif("PE" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["pe"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["pe"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["pe"]
                        data_i["correct"] = True

                if 'qakc' in self.split:
                    src_text1+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
                    src_text2+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
                elif "qkca" in  self.split:
                    src_text1+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
                    src_text2+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
                
                elif "qak" in  self.split:
                    src_text1+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nA:"
                    src_text2+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nA:"
                
                elif "qac" in  self.split:
                    src_text1+= answer_choices+"Context: "+keycontext_dataset[i]['context']+"\nA:"
                    src_text2+= answer_choices+"Context: "+keycontext_dataset[i]['context']+"\nA:"
                else:
                    src_text1 +=answer_choices+"A:"
                    src_text2 +=answer_choices+"A:"
                data_i["src_text1"] = src_text1
                data_i["src_text2"] = src_text2
                data_i["tgt_text1"] = tgt_text1
                data_i["tgt_text2"] = tgt_text2
                data.append(data_i)
            # elif("pe" in self.split):
            #     import pdb; pdb.set_trace()
            # elif("ep" in self.split):
            #     import pdb; pdb.set_trace()
            else:
                src_text = "qta " + question
                tgt_text = "("+dataset[i]['answerKey'].lower()+")"
                
                if 'qakc' in self.split:
                    src_text+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
                elif "qkca" in  self.split:
                    src_text+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
                elif "qak" in self.split:
                    src_text+= answer_choices+"Keyword: "+ keycontext_dataset[i]['keywords']+"\nA:"  
                elif "qac" in  self.split:
                    src_text+= answer_choices+"Context: "+keycontext_dataset[i]['context']+"\nA:"
                elif 'qae' in self.split:
                    src_text+= answer_choices+"Context: "+explain_dataset[i]['explanation']+"\nA:"
                else:
                    src_text +=answer_choices+"A:"

                data_i["src_text"] = src_text
                data_i["tgt_text"] = tgt_text
                data.append(data_i)

        if("multitask" in self.split):
            src_text1s = [example["src_text1"] for example in data]
            src_text2s = [example["src_text2"] for example in data]
            tgt_text1s = [example["tgt_text1"] for example in data]
            tgt_text2s = [example["tgt_text2"] for example in data]
            self.src_text1s = self.tokenizer(src_text1s, max_length=self.max_src_len)
            self.src_text2s = self.tokenizer(src_text2s, max_length=self.max_src_len)
            self.tgt_text1s = self.tokenizer(tgt_text1s, max_length=self.max_tgt_len)
            self.tgt_text2s = self.tokenizer(tgt_text2s, max_length=self.max_tgt_len)
        else:
            src_texts = [example["src_text"] for example in data]
            tgt_texts = [example["tgt_text"] for example in data]
            self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
            self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

class CSQAForFiD(Dataset):

    def __init__(self, task_name,split, shot):
        super().__init__()
        self.shot = shot
        self.split=split
        self.task_name=task_name
        if("train" in self.split):
            self.data = self.load_training_data()
            # import pdb;pdb.set_trace()
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            'src_text' : example['src_text'],
            'contexts' : example['contexts'],
            'tgt_text' : example['tgt_text']
        }

    def __len__(self):
        return len(self.data)


    def load_validation_data(self):

        dataset = load_dataset('commonsense_qa',split="validation")

        if "eA" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/csqa/context_v4/contexteA_no_neg_no_pos-validation-0-1221-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/csqa/context_v4/contexteA_no_neg-validation-0-1221-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            # elif "wopos" in self.split:
            #     with open("./data/GPT3/medqa/context_v6/contexteA_no_pos-validation-0-1272-0.0.json") as f:
            #         keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/csqa/context_v4/org_contexteA-validation-0-1221-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        
        indices=list(range(len(dataset)))
        
        data=[]
        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['choices']["label"])):
                answer_choices += "("+dataset[i]['choices']["label"][j].lower()+") "+dataset[i]['choices']["text"][j]+"\n"
                choices_text.append(dataset[i]['choices']["text"][j])
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            if 'qac' not in self.split:
                data_i['src_text']=src_text+'\nA:'
            else:
                data_i['src_text']=src_text
            data_i["tgt_text"]="("+dataset[i]['answerKey'].lower()+")"
            data_i['contexts']=contexts

            data.append(data_i)

        return data


    def load_training_data(self):

        dataset = load_dataset('commonsense_qa',split="train")

        if "eA" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/csqa/context_v4/contexteA_no_neg_no_pos-train-0-9741-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/csqa/context_v4/contexteA_no_neg-train-0-9741-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            # elif "wopos" in self.split:
            #     with open("./data/GPT3/medqa/context_v6/contexteA_no_pos-validation-0-1272-0.0.json") as f:
            #         keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/csqa/context_v4/org_contexteA-train-0-9741-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        
        indices=list(range(len(dataset)))

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        data=[]
        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['choices']["label"])):
                answer_choices += "("+dataset[i]['choices']["label"][j].lower()+") "+dataset[i]['choices']["text"][j]+"\n"
                choices_text.append(dataset[i]['choices']["text"][j])
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            if 'qac' not in self.split:
                data_i['src_text']=src_text+'A:'
            else:
                data_i['src_text']=src_text
            data_i["tgt_text"]="("+dataset[i]['answerKey'].lower()+")"
            data_i['contexts']=contexts

            data.append(data_i)

        return data

class OBQAForFiD(Dataset):

    def __init__(self, task_name,split, shot):
        super().__init__()
        self.shot = shot
        self.split=split
        self.task_name=task_name
        if("train" in self.split):
            self.data = self.load_training_data()
            # import pdb;pdb.set_trace()
        elif("validation" in self.split):
            self.data = self.load_validation_data()

        elif ('test' in self.split):
            self.data=self.load_test_data()
        

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            'src_text' : example['src_text'],
            'contexts' : example['contexts'],
            'tgt_text' : example['tgt_text']
        }

    def __len__(self):
        return len(self.data)


    def load_validation_data(self):



        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))

        
        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg_no_pos-validation-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg-validation-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/obqa/context_v1/org_contexteA-validation-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        
        indices=list(range(len(dataset)))

        
        data=[]
        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+" "
                choices_text.append(dataset[i]['question']['choices'][j]["text"])
            
            question = "Q: "+dataset[i]['question']["stem"]+'\n'
            src_text = "qta " + question + answer_choices

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            if 'qac' not in self.split:
                data_i['src_text']=src_text+'\nA:'
            else:
                data_i['src_text']=src_text
            data_i["tgt_text"]="("+dataset[i]['answerKey'].lower()+")"
            data_i['contexts']=contexts

            data.append(data_i)

        return data


    def load_training_data(self):
        
        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))


        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg_no_pos-train-0-4957-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg-train-0-4957-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/obqa/context_v1/org_contexteA-train-0-4957-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        
        indices=list(range(len(dataset)))

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        data=[]
        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+" "
                choices_text.append(dataset[i]['question']['choices'][j]["text"])
            
            question = "Q: "+dataset[i]['question']["stem"]+'\n'
            src_text = "qta " + question + answer_choices

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            if 'qac' not in self.split:
                data_i['src_text']=src_text+'\nA:'
            else:
                data_i['src_text']=src_text
            data_i["tgt_text"]="("+dataset[i]['answerKey'].lower()+")"
            data_i['contexts']=contexts

            data.append(data_i)

        return data

    def load_test_data(self):
        
        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))


        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg_no_pos-test-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/GPT3/obqa/context_v1/contexteA_no_neg-test-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/GPT3/obqa/context_v1/org_contexteA-test-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

        
        indices=list(range(len(dataset)))

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)
        data=[]
        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+" "
                choices_text.append(dataset[i]['question']['choices'][j]["text"])
            
            question = "Q: "+dataset[i]['question']["stem"]+'\n'
            src_text = "qta " + question + answer_choices

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            if 'qac' not in self.split:
                data_i['src_text']=src_text+'\nA:'
            else:
                data_i['src_text']=src_text
            data_i["tgt_text"]="("+dataset[i]['answerKey'].lower()+")"
            data_i['contexts']=contexts

            data.append(data_i)

        return data
class MMLUForFiD(Dataset):

    def __init__(self, task_name,split, shot):
        super().__init__()
        self.shot = shot
        self.split=split
        self.task_name=task_name
        if("train" in self.split):
            self.data = self.load_training_data()
            # import pdb;pdb.set_trace()
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        
        elif("test" in self.split):
            self.data = self.load_test_data()
        

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            'src_text' : example['src_text'],
            'contexts' : example['contexts'],
            'tgt_text' : example['tgt_text']
        }

    def __len__(self):
        return len(self.data)


    def load_validation_data(self):
        
        data = []
        indices = []
               
        with open("./data/GPT3/mmlu/contexteA_no_neg-validation-0-1272-0.0.json") as f:
            keycontext_dataset = json.load(f)
        dataset = load_dataset('./data/mmlu/professional_medicine/',split="validation")
        
        num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        ending_names = [f"ending{i}" for i in range(num_choices )]
        endings_dict = {ending_names[i]: chr(ord('a') + i) for i in range(num_choices)}
        

        indices=list(range(len(dataset)))


        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices: "
            for ending in ending_names:
                answer_choices += "("+endings_dict[ending]+") "+ dataset[i][ending] +" "
                choices_text.append(dataset[i][ending])
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            choice=endings_dict[ending_names[dataset[i]['label']]]
            tgt_text = "("+choice.lower()+")"
            

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            else:
                src_text+='\nA:'

            data_i['src_text']=src_text
            data_i["tgt_text"]=tgt_text
            data_i['contexts']=contexts

            data.append(data_i)
        
        return data

    def load_test_data(self):
        
        data = []
        
        if "eA" in self.split:
            with open("./data/GPT3/mmlu/contexteA_no_neg-test-0-272-0.0.json") as f:
                keycontext_dataset = json.load(f)


        dataset = load_dataset('./data/mmlu/professional_medicine/',split="test")
        
        num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        ending_names = [f"ending{i}" for i in range(num_choices )]
        endings_dict = {ending_names[i]: chr(ord('a') + i) for i in range(num_choices)}
        

        indices=list(range(len(dataset)))


        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices: "
            for ending in ending_names:
                answer_choices += "("+endings_dict[ending]+") "+ dataset[i][ending] +" "
                choices_text.append(dataset[i][ending])
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            choice=endings_dict[ending_names[dataset[i]['label']]]
            tgt_text = "("+choice.lower()+")"
            

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            else:
                src_text+='\nA:'

                


            data_i['src_text']=src_text
            data_i["tgt_text"]=tgt_text
            data_i['contexts']=contexts

        return data

    def load_training_data(self):

        return None



class StrategyQAForT5(DatasetForT5):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split, shot, mask_incorrect_reasoning_path):
        super().__init__(max_src_len , max_tgt_len , task_name, tokenizer, split)
        self.shot = shot
        self.mask_incorrect_reasoning_path = mask_incorrect_reasoning_path
        if("train" in self.split):
            self.data = self.load_training_data()
        elif("validation" in self.split):
            self.data = self.load_validation_data()

    def __getitem__(self, idx):

        if("multitask" in self.split):

            if(self.mask_incorrect_reasoning_path):
                tgt_ids2 = torch.tensor(self.tgt_text2s["input_ids"][idx],dtype=torch.long)*int(self.data[idx]["correct"])
            else:
                tgt_ids2 = torch.tensor(targets2["input_ids"][idx],dtype=torch.long)
            return {"src_ids1": torch.tensor(self.src_text1s["input_ids"][idx],dtype=torch.long),
                    "src_ids2": torch.tensor(self.src_text2s["input_ids"][idx],dtype=torch.long),
                    "src_mask1":torch.tensor(self.src_text1s["attention_mask"][idx],dtype=torch.long),
                    "src_mask2":torch.tensor(self.src_text2s["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids1": torch.tensor(self.tgt_text1s["input_ids"][idx],dtype=torch.long),
                    "tgt_ids2": tgt_ids2}
        else:
            return {"src_ids": torch.tensor(self.src_texts["input_ids"][idx],dtype=torch.long),
                    "src_mask":torch.tensor(self.src_texts["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids": torch.tensor(self.tgt_texts["input_ids"][idx],dtype=torch.long)}


    def load_validation_data(self):
        data = []
        with open("./data/strategyqa/dev.json") as f:
            dataset = json.load(f)
        for i in range(len(dataset)):
            data_i = {}
            question = "Q: "+dataset[i]['question'] +"\n" + "A:"
            if(dataset[i]["answer"]):
                answer = "yes"
            else:
                answer = "no"

            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()
            tgt_text = answer
            data_i["src_text"] = src_text
            data_i["tgt_text"] = tgt_text
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_training_data(self):

        data = []
        indices = []
        with open("./data/GPT3/strategyqa/cot-train-0-2061-0.0.json") as f:
            cot_dataset = json.load(f)
        with open("./data/GPT3/strategyqa/explain-train-0-2061-0.0.json") as f:
            explain_dataset = json.load(f)
        with open("./data/GPT3/strategyqa/cotexplain-train-0-2061-0.0.json") as f:
            cotexplain_dataset = json.load(f)

        for idx in range(len(cot_dataset)):
            if("correct" in self.split):
                if(cot_dataset[idx]["correct"]):
                    indices.append(idx)
            elif("all" in self.split):
                indices.append(idx)

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)

        with open("./data/strategyqa/train.json") as f:
            dataset = json.load(f)
        for i in indices:
            data_i = {}
            question = "Q: "+dataset[i]['question'] +"\n" + "A:"
            if(dataset[i]["answer"]):
                answer = "yes"
            else:
                answer = "no"
            if("multitask" in self.split):
                src_text1 = "qta " + question
                src_text2 = "qtr " + question
                tgt_text1 = answer
                # import pdb; pdb.set_trace()
                if("EXPLANATION" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["explanation"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["explanation"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["explanation"]
                        data_i["correct"] = True
                elif("COT" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["cot"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["cot"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["cot"]
                        data_i["correct"] = True
                elif("PE" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["pe"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["pe"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["pe"]
                        data_i["correct"] = True

                data_i["src_text1"] = src_text1
                data_i["src_text2"] = src_text2
                data_i["tgt_text1"] = tgt_text1
                data_i["tgt_text2"] = tgt_text2
                data.append(data_i)
            else:
                src_text = "qta " + question
                tgt_text = answer
                data_i["src_text"] = src_text
                data_i["tgt_text"] = tgt_text
                data.append(data_i)

        if("multitask" in self.split):
            src_text1s = [example["src_text1"] for example in data]
            src_text2s = [example["src_text2"] for example in data]
            tgt_text1s = [example["tgt_text1"] for example in data]
            tgt_text2s = [example["tgt_text2"] for example in data]
            self.src_text1s = self.tokenizer(src_text1s, max_length=self.max_src_len)
            self.src_text2s = self.tokenizer(src_text2s, max_length=self.max_src_len)
            self.tgt_text1s = self.tokenizer(tgt_text1s, max_length=self.max_tgt_len)
            self.tgt_text2s = self.tokenizer(tgt_text2s, max_length=self.max_tgt_len)
        else:
            src_texts = [example["src_text"] for example in data]
            tgt_texts = [example["tgt_text"] for example in data]
            self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
            self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

class CreakQAForT5(DatasetForT5):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split, shot, mask_incorrect_reasoning_path):
        super().__init__(max_src_len , max_tgt_len , task_name, tokenizer, split)
        self.shot = shot
        self.mask_incorrect_reasoning_path = mask_incorrect_reasoning_path
        if("train" in self.split):
            self.data = self.load_training_data()
        elif("validation" in self.split):
            self.data = self.load_validation_data()

    def __getitem__(self, idx):

        if("multitask" in self.split):

            if(self.mask_incorrect_reasoning_path):
                tgt_ids2 = torch.tensor(self.tgt_text2s["input_ids"][idx],dtype=torch.long)*int(self.data[idx]["correct"])
            else:
                tgt_ids2 = torch.tensor(targets2["input_ids"][idx],dtype=torch.long)
            return {"src_ids1": torch.tensor(self.src_text1s["input_ids"][idx],dtype=torch.long),
                    "src_ids2": torch.tensor(self.src_text2s["input_ids"][idx],dtype=torch.long),
                    "src_mask1":torch.tensor(self.src_text1s["attention_mask"][idx],dtype=torch.long),
                    "src_mask2":torch.tensor(self.src_text2s["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids1": torch.tensor(self.tgt_text1s["input_ids"][idx],dtype=torch.long),
                    "tgt_ids2": tgt_ids2}
        else:
            return {"src_ids": torch.tensor(self.src_texts["input_ids"][idx],dtype=torch.long),
                    "src_mask":torch.tensor(self.src_texts["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids": torch.tensor(self.tgt_texts["input_ids"][idx],dtype=torch.long)}


    def load_validation_data(self):
        data = []
        with open("./data/creak/dev.json") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        for i in range(len(dataset)):
            data_i = {}
            question = "claim: " + dataset[i]['sentence']
            if(dataset[i]["label"].lower()=="true"):
                answer = "true"
            elif(dataset[i]["label"].lower()=="false"):
                answer = "false"
            else:
                import pdb;
                pdb.set_trace()

            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()

            tgt_text = answer
            data_i["src_text"] = src_text
            data_i["tgt_text"] = tgt_text
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_training_data(self):

        data = []
        indices = []
        with open("./data/GPT3/creak/cot-train-0-1000-0.0.json") as f:
            cot_dataset = json.load(f)
            assert len(cot_dataset) == 1000
        with open("./data/GPT3/creak/cot-train-0-1000-0.0.json") as f:
            explain_dataset = json.load(f)
        with open("./data/GPT3/creak/cot-train-0-1000-0.0.json") as f:
            cotexplain_dataset = json.load(f)

        for idx in range(len(cot_dataset)):
            if("correct" in self.split):
                if(cot_dataset[idx]["correct"]):
                    indices.append(idx)
            elif("all" in self.split):
                indices.append(idx)

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)

        with open("./data/creak/train.json") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        for i in indices:
            data_i = {}
            question = "claim: " + dataset[i]['sentence']
            if(dataset[i]["label"].lower()=="true"):
                answer = "true"
            elif(dataset[i]["label"].lower()=="false"):
                answer = "false"
            else:
                import pdb;
                pdb.set_trace()
            if("multitask" in self.split):
                src_text1 = "qta " + question
                src_text2 = "qtr " + question
                tgt_text1 = answer
                # import pdb; pdb.set_trace()
                if("EXPLANATION" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["explanation"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["explanation"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["explanation"]
                        data_i["correct"] = True
                elif("COT" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["cot"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["cot"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["cot"]
                        data_i["correct"] = True
                elif("PE" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["pe"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["pe"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["pe"]
                        data_i["correct"] = True

                data_i["src_text1"] = src_text1
                data_i["src_text2"] = src_text2
                data_i["tgt_text1"] = tgt_text1
                data_i["tgt_text2"] = tgt_text2
                data.append(data_i)
            else:
                src_text = "qta " + question
                tgt_text = answer
                data_i["src_text"] = src_text
                data_i["tgt_text"] = tgt_text
                data.append(data_i)

        if("multitask" in self.split):
            src_text1s = [example["src_text1"] for example in data]
            src_text2s = [example["src_text2"] for example in data]
            tgt_text1s = [example["tgt_text1"] for example in data]
            tgt_text2s = [example["tgt_text2"] for example in data]
            self.src_text1s = self.tokenizer(src_text1s, max_length=self.max_src_len)
            self.src_text2s = self.tokenizer(src_text2s, max_length=self.max_src_len)
            self.tgt_text1s = self.tokenizer(tgt_text1s, max_length=self.max_tgt_len)
            self.tgt_text2s = self.tokenizer(tgt_text2s, max_length=self.max_tgt_len)
        else:
            src_texts = [example["src_text"] for example in data]
            tgt_texts = [example["tgt_text"] for example in data]
            self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
            self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

class OpenbookQAForT5(DatasetForT5):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split, shot, mask_incorrect_reasoning_path):
        super().__init__(max_src_len , max_tgt_len , task_name, tokenizer, split)
        self.shot = shot
        self.mask_incorrect_reasoning_path = mask_incorrect_reasoning_path
        if("train" in self.split):
            self.data = self.load_training_data()
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        elif("test" in self.split):
            self.data = self.load_test_data()

    def __getitem__(self, idx):

        if("multitask" in self.split):

            if(self.mask_incorrect_reasoning_path):
                tgt_ids2 = torch.tensor(self.tgt_text2s["input_ids"][idx],dtype=torch.long)*int(self.data[idx]["correct"])
            else:
                tgt_ids2 = torch.tensor(targets2["input_ids"][idx],dtype=torch.long)
            return {"src_ids1": torch.tensor(self.src_text1s["input_ids"][idx],dtype=torch.long),
                    "src_ids2": torch.tensor(self.src_text2s["input_ids"][idx],dtype=torch.long),
                    "src_mask1":torch.tensor(self.src_text1s["attention_mask"][idx],dtype=torch.long),
                    "src_mask2":torch.tensor(self.src_text2s["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids1": torch.tensor(self.tgt_text1s["input_ids"][idx],dtype=torch.long),
                    "tgt_ids2": tgt_ids2}
        else:
            return {"src_ids": torch.tensor(self.src_texts["input_ids"][idx],dtype=torch.long),
                    "src_mask":torch.tensor(self.src_texts["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids": torch.tensor(self.tgt_texts["input_ids"][idx],dtype=torch.long)}


    def load_validation_data(self):
        data = []
        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        for i in range(len(dataset)):
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+"\n"
            question = "Q: "+dataset[i]['question']["stem"] +"\n" + answer_choices+"A:"
            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()

            data_i["src_text"] = src_text
            data_i["tgt_text"] = "("+dataset[i]['answerKey'].lower()+")"
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_test_data(self):
        data = []
        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        for i in range(len(dataset)):
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+"\n"
            question = "Q: "+dataset[i]['question']["stem"] +"\n" + answer_choices+"A:"
            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()

            data_i["src_text"] = src_text
            data_i["tgt_text"] = "("+dataset[i]['answerKey'].lower()+")"
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_training_data(self):

        data = []
        indices = []
        with open("./data/GPT3/obqa/cot-train-0-4957-0.0.json") as f:
            cot_dataset = json.load(f)
        with open("./data/GPT3/obqa/explain-train-0-4957-0.0.json") as f:
            explain_dataset = json.load(f)
        with open("./data/GPT3/obqa/cotexplain-train-0-4957-0.0.json") as f:
            cotexplain_dataset = json.load(f)

        for idx in range(len(cot_dataset)):
            if("correct" in self.split):
                if(cot_dataset[idx]["correct"]):
                    indices.append(idx)
            elif("all" in self.split):
                indices.append(idx)

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)

        with open("./data/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        for i in indices:
            data_i = {}
            answer_choices = "Answer Choices:\n"
            for j in range(len(dataset[i]['question']['choices'])):
                answer_choices += "("+dataset[i]['question']['choices'][j]["label"].lower()+") "+dataset[i]['question']['choices'][j]["text"]+"\n"
            question = "Q: "+dataset[i]['question']["stem"] +"\n" + answer_choices+"A:"
            answer = "("+dataset[i]['answerKey'].lower()+")"
            if("multitask" in self.split):
                src_text1 = "qta " + question
                src_text2 = "qtr " + question
                tgt_text1 = answer
                # import pdb; pdb.set_trace()
                if("EXPLANATION" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["explanation"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["explanation"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["explanation"]
                        data_i["correct"] = True
                elif("COT" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["cot"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["cot"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["cot"]
                        data_i["correct"] = True
                elif("PE" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["pe"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["pe"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["pe"]
                        data_i["correct"] = True

                data_i["src_text1"] = src_text1
                data_i["src_text2"] = src_text2
                data_i["tgt_text1"] = tgt_text1
                data_i["tgt_text2"] = tgt_text2
                data.append(data_i)
            else:
                src_text = "qta " + question
                tgt_text = answer
                data_i["src_text"] = src_text
                data_i["tgt_text"] = tgt_text
                data.append(data_i)

        if("multitask" in self.split):
            src_text1s = [example["src_text1"] for example in data]
            src_text2s = [example["src_text2"] for example in data]
            tgt_text1s = [example["tgt_text1"] for example in data]
            tgt_text2s = [example["tgt_text2"] for example in data]
            self.src_text1s = self.tokenizer(src_text1s, max_length=self.max_src_len)
            self.src_text2s = self.tokenizer(src_text2s, max_length=self.max_src_len)
            self.tgt_text1s = self.tokenizer(tgt_text1s, max_length=self.max_tgt_len)
            self.tgt_text2s = self.tokenizer(tgt_text2s, max_length=self.max_tgt_len)
        else:
            src_texts = [example["src_text"] for example in data]
            tgt_texts = [example["tgt_text"] for example in data]
            self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
            self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

class Com2SenseQAForT5(DatasetForT5):

    def __init__(self, max_src_len , max_tgt_len , task_name, tokenizer, split, shot, mask_incorrect_reasoning_path):
        super().__init__(max_src_len , max_tgt_len , task_name, tokenizer, split)
        self.shot = shot
        self.mask_incorrect_reasoning_path = mask_incorrect_reasoning_path
        if("train" in self.split):
            self.data = self.load_training_data()
        elif("validation" in self.split):
            self.data = self.load_validation_data()

    def __getitem__(self, idx):

        if("multitask" in self.split):

            if(self.mask_incorrect_reasoning_path):
                tgt_ids2 = torch.tensor(self.tgt_text2s["input_ids"][idx],dtype=torch.long)*int(self.data[idx]["correct"])
            else:
                tgt_ids2 = torch.tensor(targets2["input_ids"][idx],dtype=torch.long)
            return {"src_ids1": torch.tensor(self.src_text1s["input_ids"][idx],dtype=torch.long),
                    "src_ids2": torch.tensor(self.src_text2s["input_ids"][idx],dtype=torch.long),
                    "src_mask1":torch.tensor(self.src_text1s["attention_mask"][idx],dtype=torch.long),
                    "src_mask2":torch.tensor(self.src_text2s["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids1": torch.tensor(self.tgt_text1s["input_ids"][idx],dtype=torch.long),
                    "tgt_ids2": tgt_ids2}
        else:
            return {"src_ids": torch.tensor(self.src_texts["input_ids"][idx],dtype=torch.long),
                    "src_mask":torch.tensor(self.src_texts["attention_mask"][idx],dtype=torch.long),
                    "tgt_ids": torch.tensor(self.tgt_texts["input_ids"][idx],dtype=torch.long)}

    def load_data(self,split):
        dataset = []
        with open("./data/com2sense/"+split+".json") as f:
            data = json.load(f)
        with open("./data/com2sense/pair_id_"+split+".json") as f:
            pair_id = json.load(f)
        id2data = {}
        for i in range(len(data)):
            id2data[data[i]["id"]] = data[i]
        key_set = set()
        for key, value in id2data.items():
            if(key in key_set):
                continue
            pair_key = pair_id[key]
            key_set.add(key)
            key_set.add(pair_key)
            dataset.append([value, id2data[pair_key]])
            dataset.append([id2data[pair_key], value])
        return dataset

    def load_validation_data(self):
        data = []
        dataset = self.load_data("dev")
        for i in range(len(dataset)):
            data_i = {}
            question = "Q: Which of the following sentence is logically correct?\nAnswer Choices:\n(a) " + dataset[i][0]["sent"] +"\n(b) "+ dataset[i][1]["sent"] +"\nA:"
            if(dataset[i][0]["label"].lower()=="true"):
                answer = "(a)"
            elif(dataset[i][1]["label"].lower()=="true"):
                answer = "(b)"
            else:
                import pdb; pdb.set_trace()

            if("qtr" in self.split):
                src_text = "qtr " + question
            elif("qta" in self.split):
                src_text = "qta " + question
            else:
                import pdb; pdb.set_trace()

            tgt_text = answer
            data_i["src_text"] = src_text
            data_i["tgt_text"] = tgt_text
            data.append(data_i)
        src_texts = [example["src_text"] for example in data]
        tgt_texts = [example["tgt_text"] for example in data]
        self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
        self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data

    def load_training_data(self):


        indices = []
        with open("./data/GPT3/com2sense_pair/pair-cot-train-0-1608-0.0.json") as f:
            cot_dataset = json.load(f)
        with open("./data/GPT3/com2sense_pair/pair-cot-train-0-1608-0.0.json") as f:
            explain_dataset = json.load(f)
        with open("./data/GPT3/com2sense_pair/pair-cot-train-0-1608-0.0.json") as f:
            cotexplain_dataset = json.load(f)

        for idx in range(len(cot_dataset)):
            if("correct" in self.split):
                if(cot_dataset[idx]["correct"]):
                    indices.append(idx)
            elif("all" in self.split):
                indices.append(idx)

        if(self.shot != -1):
            indices = random.sample(indices, self.shot)

        dataset = self.load_data("train")
        data = []
        for i in indices:
            data_i = {}
            question = "Q: Which of the following sentence is logically correct?\nAnswer Choices:\n(a) " + dataset[i][0]["sent"] +"\n(b) "+ dataset[i][1]["sent"] +"\nA:"
            if(dataset[i][0]["label"].lower()=="true"):
                answer = "(a)"
            elif(dataset[i][1]["label"].lower()=="true"):
                answer = "(b)"
            else:
                import pdb; pdb.set_trace()
            if("multitask" in self.split):
                src_text1 = "qta " + question
                src_text2 = "qtr " + question
                tgt_text1 = answer
                # import pdb; pdb.set_trace()
                if("EXPLANATION" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["explanation"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["explanation"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["explanation"]
                        data_i["correct"] = True
                elif("COT" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["cot"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["cot"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["cot"]
                        data_i["correct"] = True
                elif("PE" in self.split):
                    if("cotexplain" in self.split):
                        tgt_text2 = cotexplain_dataset[i]["pe"]
                        data_i["correct"] = True
                    elif("cot" in self.split):
                        tgt_text2 = cot_dataset[i]["pe"]
                        data_i["correct"] = cot_dataset[i]["correct"]
                    elif("explain" in self.split):
                        tgt_text2 = explain_dataset[i]["pe"]
                        data_i["correct"] = True

                data_i["src_text1"] = src_text1
                data_i["src_text2"] = src_text2
                data_i["tgt_text1"] = tgt_text1
                data_i["tgt_text2"] = tgt_text2
                data.append(data_i)
            # elif("pe" in self.split):
            #     import pdb; pdb.set_trace()
            # elif("ep" in self.split):
            #     import pdb; pdb.set_trace()
            else:
                src_text = "qta " + question
                tgt_text = answer
                data_i["src_text"] = src_text
                data_i["tgt_text"] = tgt_text
                data.append(data_i)

        if("multitask" in self.split):
            src_text1s = [example["src_text1"] for example in data]
            src_text2s = [example["src_text2"] for example in data]
            tgt_text1s = [example["tgt_text1"] for example in data]
            tgt_text2s = [example["tgt_text2"] for example in data]
            self.src_text1s = self.tokenizer(src_text1s, max_length=self.max_src_len)
            self.src_text2s = self.tokenizer(src_text2s, max_length=self.max_src_len)
            self.tgt_text1s = self.tokenizer(tgt_text1s, max_length=self.max_tgt_len)
            self.tgt_text2s = self.tokenizer(tgt_text2s, max_length=self.max_tgt_len)
        else:
            src_texts = [example["src_text"] for example in data]
            tgt_texts = [example["tgt_text"] for example in data]
            self.src_texts = self.tokenizer(src_texts, max_length=self.max_src_len)
            self.tgt_texts = self.tokenizer(tgt_texts, max_length=self.max_tgt_len)
        return data


def Bertcollate_fn(batch):


    output_batch={}
    mc_label_name = "mc_label" if "mc_label" in batch[0].keys() else "mc_labels"
    mc_labels = [int(feature.pop(mc_label_name)) for feature in batch]
    batch_size = len(batch)
    num_choices = len(batch[0]["input_ids"])
    flattened_batch = [
        [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in batch
    ]

    flattened_batch = sum(flattened_batch, [])

    if('mlm_labels' in flattened_batch[0]):
        mlm_labels = [torch.tensor(example['mlm_labels'],dtype=torch.long) for example in flattened_batch]
        mlm_labels = rnn_utils.pad_sequence(mlm_labels, batch_first=True,padding_value=-100)
        output_batch['mlm_labels']=mlm_labels


    input_ids = [torch.tensor(example['input_ids'] ,dtype=torch.long) for example in flattened_batch]
    token_type_ids = [torch.tensor(example['token_type_ids'],dtype=torch.long) for example in flattened_batch]
    attention_mask = [torch.tensor(example['attention_mask'],dtype=torch.long) for example in flattened_batch]

    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = rnn_utils.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask= rnn_utils.pad_sequence(attention_mask, batch_first=True,padding_value=0)

    output_batch['input_ids']=input_ids
    output_batch['token_type_ids']=token_type_ids
    output_batch['attention_mask']=attention_mask


    output_batch = {k: v.view(batch_size, num_choices, -1) for k, v in output_batch.items()}

    output_batch['mc_label']=torch.tensor(mc_labels,dtype=torch.long)


    return output_batch






def T5collate_fn(batch):

    if('src_ids' in batch[0]):

        src_ids = [example['src_ids'] for example in batch]
        src_mask = [example['src_mask'] for example in batch]
        tgt_ids = [example['tgt_ids'] for example in batch]

        src_ids = rnn_utils.pad_sequence(src_ids, batch_first=True, padding_value=0)
        src_mask = rnn_utils.pad_sequence(src_mask, batch_first=True, padding_value=0)
        tgt_ids = rnn_utils.pad_sequence(tgt_ids, batch_first=True,padding_value=0)
        # tgt_ids = rnn_utils.pad_sequence(tgt_ids, batch_first=True,padding_value=-100)
        # tgt_ids[tgt_ids[:, :] == 0] = -100

        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "tgt_ids":tgt_ids}

    elif('src_ids1' in batch[0]):

        src_ids1 = [example['src_ids1'] for example in batch]
        src_ids2 = [example['src_ids2'] for example in batch]
        src_mask1 = [example['src_mask1'] for example in batch]
        src_mask2 = [example['src_mask2'] for example in batch]
        tgt_ids1 = [example['tgt_ids1'] for example in batch]
        tgt_ids2 = [example['tgt_ids2'] for example in batch]
        src_ids1 = rnn_utils.pad_sequence(src_ids1, batch_first=True, padding_value=0)
        src_ids2 = rnn_utils.pad_sequence(src_ids2, batch_first=True, padding_value=0)
        src_mask1 = rnn_utils.pad_sequence(src_mask1, batch_first=True, padding_value=0)
        src_mask2 = rnn_utils.pad_sequence(src_mask2, batch_first=True, padding_value=0)
        tgt_ids1 = rnn_utils.pad_sequence(tgt_ids1, batch_first=True,padding_value=-100)
        tgt_ids2 = rnn_utils.pad_sequence(tgt_ids2, batch_first=True,padding_value=-100)
        tgt_ids1[tgt_ids1[:, :] == 0] = -100
        tgt_ids2[tgt_ids2[:, :] == 0] = -100

        return {"src_ids1": src_ids1,
                "src_ids2":src_ids2,
                "src_mask1":src_mask1,
                "src_mask2":src_mask2,
                "tgt_ids1":tgt_ids1,
                "tgt_ids2":tgt_ids2
                }

class MedQAForFiD(Dataset):

    def __init__(self, task_name,split, shot):
        super().__init__()
        self.shot = shot
        self.split=split
        self.task_name=task_name
        if("train" in self.split):
            self.data = self.load_training_data()
            # import pdb;pdb.set_trace()
        elif("validation" in self.split):
            self.data = self.load_validation_data()
        
        elif("test" in self.split):
            self.data = self.load_test_data()
        

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            'src_text' : example['src_text'],
            'contexts' : example['contexts'],
            'tgt_text' : example['tgt_text']
        }

    def __len__(self):
        return len(self.data)


    def load_validation_data(self):
        
        data = []
        indices = []

        if "eA" in self.split:
            with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-validation-0-1272-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "wA" in self.split:
            with open("./data/GPT3/medqa/context_v2/contextwA-validation-0-1272-0.0.json") as f:
                keycontext_dataset = json.load(f)
        else:
            with open("./data/GPT3/medqa/context_v1/context-validation-0-1272-0.0.json") as f:
                keycontext_dataset = json.load(f)


        with open("./data/MedQA/questions/US/4_options/phrases_no_exclude_dev.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))

        indices=list(range(len(keycontext_dataset)))


        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices: "
            for choice,text in dataset[i]['options'].items():
                answer_choices += "("+choice.lower()+") "+ text +" "
                choices_text.append(text)
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            choice=dataset[i]['answer_idx']
            tgt_text = "("+choice.lower()+")"
            

            if 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            data_i['src_text']=src_text
            data_i["tgt_text"]=tgt_text
            data_i['contexts']=contexts

            data.append(data_i)

        return data

    def load_test_data(self):
        data = []
        
        if "eA" in self.split:
            with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-test-0-1273-0.0.json") as f:
                keycontext_dataset = json.load(f)


        with open("./data/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))
        

        indices=list(range(len(keycontext_dataset)))

        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices: "
            for choice,text in dataset[i]['options'].items():
                answer_choices += "("+choice.lower()+") "+ text +" "
                choices_text.append(text)
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            choice=dataset[i]['answer_idx']
            tgt_text = "("+choice.lower()+")"
            
            if 'qakc' in self.split:
                src_text+= "\nKeyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
            elif "qkca" in  self.split:
                src_text+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
            elif "qak" in self.split:
                src_text+= "\nKeyword: "+ keycontext_dataset[i]['keywords']+"\nA:"  
            elif 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            data_i['src_text']=src_text
            data_i["tgt_text"]=tgt_text
            data_i['contexts']=contexts

            data.append(data_i)

        return data

    def load_training_data(self):
        
        data = []
        indices = []

        if 'eA' in self.split :
            with open("./data/GPT3/medqa/context_v6/contexteA_no_neg-train-0-4000-0.0.json") as f:
                keycontext_dataset = json.load(f)


        with open("./data/MedQA/questions/US/4_options/phrases_no_exclude_train.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))


        

        indices=list(range(len(keycontext_dataset)))


        
        if(self.shot != -1):
            indices = random.sample(indices, self.shot)


        for i in indices:
            choices_text=[]
            contexts=[]
            data_i = {}
            answer_choices = "Answer Choices: "
            for choice,text in dataset[i]['options'].items():
                answer_choices += "("+choice.lower()+") "+ text +" "
                choices_text.append(text)
            question = "Q: "+dataset[i]['question'] +"\n" 
            src_text = "qta " + question + answer_choices

            choice=dataset[i]['answer_idx']
            tgt_text = "("+choice.lower()+")"
            
            if 'qakc' in self.split:
                src_text+= "\nKeyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\nA:"
            elif "qkca" in  self.split:
                src_text+= "Keyword: "+ keycontext_dataset[i]['keywords']+"\nContext: "+keycontext_dataset[i]['context']+"\n"+answer_choices+"A:"
            elif "qak" in self.split:
                src_text+= "\nKeyword: "+ keycontext_dataset[i]['keywords']+"\nA:" 
            elif 'qac' in self.split:
                if 'eA' in self.split:
                    for j,choice in enumerate(choices_text):
                        try:
                            contexts.append(keycontext_dataset[i]['overall']+" "+keycontext_dataset[i][choice]+'\nA:')
                        except:
                            contexts.append(keycontext_dataset[i]['overall']+'\nA:')

            data_i['src_text']=src_text
            data_i["tgt_text"]=tgt_text
            data_i['contexts']=contexts

            data.append(data_i)

        return data

class DataCollatorForMultipleChoice(object):
    
    def __init__(self, config, tokenizer,max_seq_len,mlm_probability,split):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability=mlm_probability
        self.split=split
        self.config=config
        
    def __call__(self, batch):

        output_batch={}
        sep = self.tokenizer.sep_token_id
        cls = self.tokenizer.cls_token_id
        assert(batch[0]['mc_label'] != None)
        mc_labels = [ item['mc_label'] for item in batch]
        batch_size=len(batch)
        num_choices=len(batch[0]['contexts'])
        first_sentences=[[item['sent1']] * num_choices  for item in batch ]
        first_sentences = sum(first_sentences, [])
        if 'overall' in self.split: 
            second_sentences=[f"{item['answers'][i]} {item['overall']}" for item in batch for i in range(num_choices) ]
        elif 'special' in self.split:
            second_sentences=[f"{item['answers'][i]} {item['contexts'][i]}" for item in batch for i in range(num_choices) ]
        else:
            second_sentences=[f"{item['answers'][i]} {item['overall']} {item['contexts'][i]} " for item in batch for i in range(num_choices) ]

        if self.config.model_type == "gpt2":
            first_sentences  = [s + self.tokenizer.sep_token for s in first_sentences ]
            second_sentences = [s + self.tokenizer.sep_token for s in second_sentences]

        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = [torch.tensor(input_id,dtype=torch.long) for input_id in tokenized_examples['input_ids']]
        atten_masks = [torch.tensor(atten_mask,dtype=torch.long) for atten_mask in tokenized_examples['attention_mask']]
        if self.config.model_type=="bert":
            token_types = [torch.tensor(token_type,dtype=torch.long) for token_type in tokenized_examples['token_type_ids']]

        input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
        atten_masks= rnn_utils.pad_sequence(atten_masks, batch_first=True,padding_value=0)
        if self.config.model_type=="bert":
            token_types = rnn_utils.pad_sequence(token_types, batch_first=True, padding_value=0)

        
        output_batch['input_ids']=input_ids
        output_batch['attention_mask']=atten_masks
        if self.config.model_type=="bert":
            output_batch['token_type_ids']=token_types 

        output_batch = {k: v.view(batch_size, num_choices, -1) for k, v in output_batch.items()}
        output_batch['mc_label']=torch.tensor(mc_labels,dtype=torch.long)
        return output_batch


class FiDCollator(object):
    def __init__(self, split,tokenizer ,max_src_len,max_tgt_len,noise_density=0):
        self.tokenizer = tokenizer
        self.split=split
        self.max_tgt_len = max_tgt_len
        self.max_src_len = max_src_len
        self.mlm_probability = noise_density

    def __call__(self, batch):
        assert(batch[0]['tgt_text'] != None)

        batch_size=len(batch)
        target = [ex['tgt_text'] for ex in batch]
        target = self.tokenizer(target,max_length=self.max_tgt_len,truncation=True)
        target_ids = [torch.tensor(input_ids,dtype=torch.long) for input_ids in target["input_ids"]]  
        target_ids = rnn_utils.pad_sequence(target_ids, batch_first=True,padding_value=-100)
        target_ids[target_ids[:, :] == 0] = -100

        if 'qac' in  self.split:
            num_choice=len(batch[0]['contexts'])
            def append_question(example):
                if example['contexts'] is None:
                    return [example['src_text']]
                return [example['src_text'] + "\nContext: " + t for t in example['contexts']]
            text_passages = [append_question(example) for example in batch]
            flat_text_passages=list(itertools.chain(*text_passages))
            srcs = self.tokenizer(flat_text_passages,max_length=self.max_src_len,truncation=True)
            src_input_ids=[torch.tensor(input_ids,dtype=torch.long) for input_ids in srcs['input_ids']]
            src_masks=[torch.tensor(attention_mask,dtype=torch.long) for attention_mask in srcs['attention_mask']]

            src_input_ids = rnn_utils.pad_sequence(src_input_ids, batch_first=True, padding_value=0)
            src_masks = rnn_utils.pad_sequence(src_masks, batch_first=True, padding_value=0)

            return {"src_ids": src_input_ids.view(batch_size,num_choice,-1),
                    "src_mask":src_masks.view(batch_size,num_choice,-1),
                    "tgt_ids":target_ids
                    }
        else:
            num_choice=1
            src_texts= [example['src_text'] for example in batch]
            srcs = self.tokenizer(src_texts,max_length=self.max_src_len,truncation=True)
            src_input_ids=[torch.tensor(input_ids,dtype=torch.long) for input_ids in srcs['input_ids']]
            src_masks=[torch.tensor(attention_mask,dtype=torch.long) for attention_mask in srcs['attention_mask']]

            src_input_ids = rnn_utils.pad_sequence(src_input_ids, batch_first=True, padding_value=0)
            src_masks = rnn_utils.pad_sequence(src_masks, batch_first=True, padding_value=0)

            return {"src_ids": src_input_ids.view(batch_size,num_choice,-1),
                    "src_mask":src_masks.view(batch_size,num_choice,-1),
                    "tgt_ids":target_ids
                    }




