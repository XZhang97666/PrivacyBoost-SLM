# PrivacyBoost-SLM

## Set up environment and data

### Environment
Run the following commands to create a conda environment:
```bash
conda create -n PBSLM python=3.7
source activate PBSLM
pip install -r requirements.txt
```

### Dataset

You can download the generated context and processed dataset (if not directly available from Hugging Face) on which we evaluated our method from [here](https://drive.google.com/file/d/1JPtOkqpku540NL1wQLtWJkuGWIs2HfA0/view?usp=sharing). Simply download this zip file and extract its contents.
This includes:
#### Biomedical domain: [MedQA-USMLE](https://github.com/jind11/MedQA), [HEADQA](https://github.com/aghie/head-qa), [MedMCQA](https://medmcqa.github.io/), and [MMLU-professional medicine](https://github.com/hendrycks/test).
#### General domain: [CommonSenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) and [OpenbookQA](https://allenai.org/data/open-book-qa).


### Resulting file structure

The resulting file structure should look like this:

```plain
.
├── README.md
├── data/
    ├── MedQA/  
        ├── apicall/             (prompts,keywords for GPT 3.5 API context generation)
        ├── context/             (genearted context for SLM training)
        ├── ...
    ├── headqa/
    ├── medmc/
    ├── csqa/
    └── obqa/
```


To generate context on your own, you can use the `api_call.sh`. To do this, follow the steps below:
- Set `dataset_name` to the desired dataset, for example: `{medqa headqa medmcqa}`.
- Set `split` to the desired dataset split, `{train validation|dev test}`.
- Specify the `start_idx` and `end_idx` to define the starting and ending index of the dataset you want to iterate through.
- Call the OpenAI API with the configured settings.

## Experiment results

###  Full-training 

#### BioLinkBert
To evaluate the performance of the Fine-Tuning with Context (FTC) using BioLinkBert-Base as the backbone on MedQA, HEADQA and MedMCQA, run
```
sh run_training_eval_bert.sh 
```

To evaluate the performance using BioLinkBert-Large, replace the `model_name_or_path` argument with "michiyasunaga/BioLinkBERT-large" in the previous commands.

#### BioMedLM
To directly evaluate the performance of the FTC using BioMedLM as the backbone on various datasets, we upload the BioMedLM checkpoints [here](https://drive.google.com/file/d/1gB-V6D_3xaRaYDkUrdUhJE8j6RQjNHou/view?usp=sharing). Simply download this zip file and unzip its contents. Then follow the instructions below to adpot the fine-tuned model.
```
sh run_training_eval_gpt.sh 
```

#### T5
To further evaluate the performance on general domain e.g. CommonsenseQA and OpenbookQA dataset, run

```
sh run_training_eval_t5.sh 
```


###  Few-shot setting

Change the `shot= {100 200 500}` to run experiments in few-shot setting. For example, to evluate few-shot performance on MedQA dataset with BioLinkBert-Base as backbone, run
```
sh run_training_eval_bert_fewshot.sh 
```

###  Out-of-Domain (OOD) Performance
To evaluate the Out-of-Domain (OOD) performance of FTC using BioLinkBERT-Base as the backbone, without additional training, you need to modify the script to use the best performance model from the source domain and directly apply it to the target domain. You can do this by changing the `dataset_name`  to the target domain name (e.g.,  `{medqa headqa medmcqa mmlu}` ) and `input_dataset_name` to the source domain name (e.g., `{medqa headqa medmcqa}`). For example, run `medqa` -> `mmlu`:

```
run_training_eval_bert_OOD.sh
```



## Citation

If you found this repository useful, please consider cite our paper:

```bibtex
@misc{zhang2023enhancing,
      title={Enhancing Small Medical Learners with Privacy-preserving Contextual Prompting}, 
      author={Xinlu Zhang and Shiyang Li and Xianjun Yang and Chenxin Tian and Yao Qin and Linda Ruth Petzold},
      year={2023},
      eprint={2305.12723},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## Acknowledgements

We would like to express our gratitude to Zhiyu Chen from Meta Reality Labs, Ming Yi, and Hong Wang from the Computer Science Department at UCSB, as well as the anonymous reviewers, for their invaluable feedback. Additionally, we extend our thanks to Rachael A Callcut and Anamaria J Roble for their insightful discussions and guidance on medical prompt designs. Furthermore, we gratefully acknowledge the generous financial support provided by the National Institutes for Health (NIH) grant NIH 7R01HL149670.

