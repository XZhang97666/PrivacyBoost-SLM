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
            with open("./data/MedQA/context/org_contexteA-validation-0-1272-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/MedQA/context/org_contexteA-validation-0-1272-50-0.0.json") as f:
                keycontext_dataset = json.load(f)        
        elif "qac_75" in self.split:
            with open("./data/MedQA/context/org_contexteA-validation-0-1272-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word'in self.split:
            with open("./data/MedQA/context/org_contexteA-validation-0-1272-random-0.0.json") as f:
                keycontext_dataset = json.load(f) 
        elif 'qac_random_span' in self.split:
            with open("./data/MedQA/context/org_contexteA-validation-0-1272-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f) 
        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/MedQA/context/contexteA_no_neg_no_pos-validation-0-1272-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/MedQA/context/contexteA_no_neg-validation-0-1272-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/MedQA/context/org_contexteA-validation-0-1272-0.0.json") as f:
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
                if 'eA' in self.split:
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
            with open("./data/MedQA/context/org_contexteA-test-0-1273-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/MedQA/context/org_contexteA-test-0-1273-50-0.0.json") as f:
                keycontext_dataset = json.load(f)

        elif "qac_75" in self.split:
            with open("./data/MedQA/context/org_contexteA-test-0-1273-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word' in self.split:
            with open("./data/MedQA/context/org_contexteA-test-0-1273-random-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_span' in self.split:
            with open("./data/MedQA/context/org_contexteA-test-0-1273-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/MedQA/context/contexteA_no_neg_no_pos-test-0-1273-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/MedQA/context/contexteA_no_neg-test-0-1273-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/MedQA/context/org_contexteA-test-0-1273-0.0.json") as f:
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
                if 'eA' in self.split:
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
            with open("./data/MedQA/context/org_contexteA-train-0-10178-25-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_50" in self.split:
            with open("./data/MedQA/context/org_contexteA-train-0-10178-50-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif "qac_75" in self.split:
            with open("./data/MedQA/context/org_contexteA-train-0-10178-75-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_word'in self.split:
            with open("./data/MedQA/context/org_contexteA-train-0-1000-random-0.0.json") as f:
                keycontext_dataset = json.load(f)
        elif 'qac_random_span' in self.split:
            with open("./data/MedQA/context/org_contexteA-train-0-1000-randomspan-0.0.json") as f:
                keycontext_dataset = json.load(f) 

        elif "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/MedQA/context/contexteA_no_neg_no_pos-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/MedQA/context/contexteA_no_neg-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/MedQA/context/org_contexteA-train-0-10178-0.0.json") as f:
                    keycontext_dataset = json.load(f)



        dataset = load_dataset('./data/MedQA/BertMC/',split="train")

        self._num_choices = len([elm for elm in dataset.features.keys() if elm.startswith('ending')])
        self.ending_names = [f"ending{i}" for i in range(self._num_choices )]

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
                if 'eA' in self.split:
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
                with open("./data/headqa/context/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/headqa/context/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/headqa/context/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)


        dataset = load_dataset('head_qa',"en",split="validation")

        self._num_choices = len(dataset[0]['answers'])
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'eA' in self.split:
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
                with open("./data/headqa/context/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/headqa/context/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/headqa/context/org_contexteA-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)


        dataset = load_dataset('head_qa',"en",split="test")

        self._num_choices = len(dataset[0]['answers'])
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

        data=[]
        indices=range(len(dataset))
        
        for idx in indices:
            data_i=dataset[idx]
            if "qac" in self.split:
                if 'eA' in self.split:
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
                with open("./data/headqa/context/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/headqa/context/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/headqa/context/org_contexteA-"+cname+"-0.0.json") as f:
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
                if 'eA' in self.split:
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
                with open("./data/medmc/context/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/medmc/context/contexteA_no_neg-"+cname+"-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/medmc/context/org_contexteA-"+cname+"-0.0.json") as f:
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
                if 'eA' in self.split:
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
            with open("./data/medmc/context/contexteA_no_neg_no_pos-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)

        elif "woneg" in self.split:
            with open("./data/medmc/context/contexteA_no_neg-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)
        else:
            with open("./data/medmc/context/org_contexteA-"+cname+"-0.0.json") as f:
                keycontext_dataset = json.load(f)



        dataset = load_dataset("medmcqa",split="train")

        with open("./data/medmc/indexes_list_file.json","r") as f:
                idx_list=json.load(f)


        self.ending_names = [elm for elm in dataset.features.keys() if elm.startswith('op')]
        self._num_choices = len(self.ending_names)
        self.context_names = [f"sent2_{i}" for i in range(self._num_choices )]

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
                if 'eA' in self.split:
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
            with open("./data/mmlu/context/org_contexteA-test-0-272-0.0.json") as f:
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
                if 'eA' in self.split:
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
                with open("./data/csqa/context/contexteA_no_neg_no_pos-validation-0-1221-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/csqa/context/contexteA_no_neg-validation-0-1221-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/csqa/context/org_contexteA-validation-0-1221-0.0.json") as f:
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
                with open("./data/csqa/context/contexteA_no_neg_no_pos-train-0-9741-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/csqa/context/contexteA_no_neg-train-0-9741-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/csqa/context/org_contexteA-train-0-9741-0.0.json") as f:
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



        with open("./data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))

        
        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/obqa/context/contexteA_no_neg_no_pos-validation-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/obqa/context/contexteA_no_neg-validation-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/obqa/context/org_contexteA-validation-0-500-0.0.json") as f:
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
        
        with open("./data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))


        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/obqa/context/contexteA_no_neg_no_pos-train-0-4957-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/obqa/context/contexteA_no_neg-train-0-4957-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/obqa/context/org_contexteA-train-0-4957-0.0.json") as f:
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
        
        with open("./data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl") as f:
            dataset = []
            for line in f.readlines():
                dataset.append(json.loads(line))


        if "qac" in self.split:
            if "woneg" in self.split and 'wopos' in self.split:
                with open("./data/obqa/context/contexteA_no_neg_no_pos-test-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)

            elif "woneg" in self.split:
                with open("./data/obqa/context/contexteA_no_neg-test-0-500-0.0.json") as f:
                    keycontext_dataset = json.load(f)
            else:
                with open("./data/obqa/context/org_contexteA-test-0-500-0.0.json") as f:
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




