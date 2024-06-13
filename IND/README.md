# WhoIsWho-IND-KDD-2024

## Team information
- Team name: DOCOMOLABS
- Rank: 7th
- Score: 0.80487
- Parameters: 10,213,351,506
- Total GPU Memory: 24GB

## Prerequisites
- AMI
-- Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.2.0 (Amazon Linux 2) 20240604
-AWS EC2 
-- g5.2xlarge（GPU 24GB）

## Getting Started

### Installation
For``IND``,
```
pip install -r Code/requirements.txt
```
### Execute
Please execute in the following order. 
You can start from any number within the same number.

- Code/0_xx => Code/1_xx => Code/2_xx

# Detail
## raw
Provided data

## Code
### Code/0_xxxx
Clean raw data and create word2vec model

### Code/1_xxxx
Generate a feature vector for the paper

### Code/2_xxxx
Make inferences using features and output results

### test_feature/xxxx.parquet
Feature data

## test_data/xxxx
Data such as word2vec

# Folder
├── README.md
├── raw
│   ├── ind_valid_author.json
│   ├── ind_valid_author_submit.json
│   ├── pid_to_info_all.json
│   └── train_author.json
├── Code
│   ├── 0_preprocessing.ipynb
│   ├── 0_preprocessing_light.ipynb
│   ├── 1_get_basic_emb.py
│   ├── 1_get_bert_emb_de.py
│   ├── 1_get_bert_emb_e5.py
│   ├── 1_get_bert_emb_glm4.py
│   ├── 1_get_bert_emb_minilm.py
│   ├── 1_get_bert_emb_oag.py
│   ├── 1_get_bert_emb_sci.py
│   ├── 1_get_graph_emb.py
│   ├── 1_get_jaccard_emb.py
│   ├── 1_get_tfidf_emb.py
│   ├── 1_get_w2v_emb.py
│   └── 2_predict.ipynb
├── test_data
│   ├── cleaned_pid_to_info_all_v6.parquet
│   ├── cleaned_pid_to_info_all_v6_light.parquet
│   ├── ind_test_author_filter_public.parquet
│   ├── ind_valid_author.parquet
│   ├── train_author.parquet
│   ├── w2v_concat_cbow_128dim_min2_window10_neg5_epoch30_v6.bin
│   ├── w2v_concat_cbow_128dim_min2_window10_neg5_epoch30_v6.bin.syn1neg.npy
│   ├── w2v_concat_cbow_128dim_min2_window10_neg5_epoch30_v6.bin.wv.vectors.npy
│   └── w2v_org_cbow_128dim_window5_min5_neg5_epoch30_v6.bin
└── test_feature
    ├── test_basic.parquet
    ├── test_deberta.parquet
    ├── test_glm.parquet
    ├── test_graph.parquet
    ├── test_jaccard.parquet
    ├── test_minilm.parquet
    ├── test_multilingual_e5_large.parquet
    ├── test_oag_bert.parquet
    ├── test_scibert_nli.parquet
    ├── test_tfidf.parquet
    ├── test_w2v.parquet
    ├── train_basic.parquet
    ├── train_deberta.parquet
    ├── train_glm.parquet
    ├── train_graph.parquet
    ├── train_jaccard.parquet
    ├── train_minilm.parquet
    ├── train_multilingual_e5_large.parquet
    ├── train_oag_bert.parquet
    ├── train_scibert_nli.parquet
    ├── train_tfidf.parquet
    └── train_w2v.parquet

