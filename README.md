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



