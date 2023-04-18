# PrivacyBoost-SLM

## 1. Set up environment and data

### Environment
Run the following commands to create a conda environment:
```bash
conda create -n PBSLM python=3.7
source activate PBSLM
pip install -r requirements.txt
```

### Dataset
You can download the generated context and processed dataset (if not directly available from Hugging Face) on which we evaluated our method from [here]. Simply download this zip file and extract its contents.
This includes:
#### Biomedical domain: [MedQA-USMLE](https://github.com/jind11/MedQA), [HEADQA](https://github.com/aghie/head-qa), [MedMCQA](https://medmcqa.github.io/), and [MMLU-professional medicine](https://github.com/hendrycks/test).
#### General domain: [CommonSenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) and [OpenbookQA](https://allenai.org/data/open-book-qa).

## 2. Experiment results

###  Full-training 

#### BioLinkBert
To evaluate the performance of the Fine-Tuning with Context (FTC) using BioLinkBert-Base as the backbone on MedQA, HEADQA and MedMCQA, run
```
sh ./scripts/run_training_eval_bert.sh 
```

To evaluate the performance using BioLinkBert-Large, replace the `model_name_or_path` argument with "michiyasunaga/BioLinkBERT-large" in the previous commands.

#### BioMedLM
To directly evaluate the performance of the FTC using BioMedLM as the backbone on various datasets, we upload the BioMedLM checkpoints [here]. Simply download this zip file and unzip its contents. Then follow the instructions below to adpot the fine-tuned model.
```
sh ./scripts/run_training_eval_gpt.sh 
```

#### T5
To further evaluate the performance on general domain e.g. CommonsenseQA and OpenbookQA dataset, run

```
sh ./scripts/run_training_eval_t5.sh 
```


###  Few-shot setting

Change the `shot= {100 200 500}` to run experiments in few-shot setting. For example, to evluate few-shot performance on MedQA dataset with BioLinkBert-Base as backbone, run
```
sh ./scripts/run_training_eval_bert_fewshot.sh 
```

###  Out-of-Domain (OOD) Performance
To evaluate the Out-of-Domain (OOD) performance of FTC using BioLinkBERT-Base as the backbone, without additional training, you need to modify the script to use the best performance model from the source domain and directly apply it to the target domain. You can do this by changing the `dataset_name`  to the target domain name (e.g.,  `{medqa headqa medmcqa mmlu}` ) and `input_dataset_name` to the source domain name (e.g., `{medqa headqa medmcqa}`). For example, run `medqa` -> `mmlu`:

```
run_training_eval_bert_OOD.sh
```







