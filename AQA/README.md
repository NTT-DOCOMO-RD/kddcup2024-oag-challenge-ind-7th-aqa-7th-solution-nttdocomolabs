# OAG-AQA Task 7th Solution

## Solution Overview
* 2stage prediction
* Stage1.Retrieve
  - Add LLM-generated answer of question to query(HyDE)
  - Vector search by contrastive learning(SimCSE)
* Stage2. Reranking
  - Binary classification by encoder model(deberta V3 large model) & ensemble
<img width="1024" alt="スクリーンショット 2024-06-13 18 09 31" src="https://github.com/NTT-DOCOMO-RD/kddcup2024-oag-challenge-ind-7th-aqa-7th-solution-nttdocomolabs/assets/111550364/2d35567f-9a37-48e5-a58a-b32d5d0e8c05">

## Prerequisites
* Linux 
* Python 3.10.12
* Tesla T4(CPU memory 50GB, GPU memory 16GB)

## Installation
``` 
pip install -r requirements.txt
```

## Dataset

* phase1 [https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA.zip](https://www.dropbox.com/scl/fi/2ckwl9fcpbik88z1cekot/AQA.zip?rlkey=o7ttmrvpdbvbu3rcr6t33jrx7&dl=1)

* phase2 [https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA-test-public.zip](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/AQA-test-public.zip)


## Directory structure
```
│── README.md
│── requirements.txt 
│── data/ 
│      │── raw/ # Competition dataset
│      │  　 │── pid_to_title_abs_new.json
│      │  　 │── pid_to_title_abs_update_filter.json
│      │  　 │── qa_train.txt
│      │  　 │── qa_test_wo_ans_new.txt
│      │── llm_genrated_output/ # LLM-generated output
│      │  　 │──train_llm_answer_question_body.csv
│      │  　 │──test_llm_answer_question_body.csv
│      │  　 │──train_llm_summarize_question_body.csv
│      │  　 │──test_llm_summarize_question_body.csv
│      │── candidates  # Supervised candidate dataset
│      │  　 │──train.csv
│      │  　 │──test.csv
│      │── submit_file # pred & ensemble file
│      │  　 │──pred_rerank_neg_sample1.csv
│      │  　 │──pred_rerank_neg_sample2.csv
│      │  　 │──pred_rerank_neg_sample3.csv
│      │  　 │──ensemble_preds.txt
│── model/ 
│      │── stage1/ # embedding model checkpoint
│      │── stage2/ # encoder model checkpoint
│      │  　 │──rerank_neg_sample_1 
│      │  　 │──rerank_neg_sample_2 
│      │  　 │──rerank_neg_sample_3
│── code/ 
│      │── llm_generate/
│      │  　 │── train_llm_answer_question_body.py
│      │  　 │── test_llm_answer_question_body.py
│      │  　 │── train_llm_summarize_question_body.py
│      │  　 │── test_llm_summarize_question_body.py
│      │── create_candidate/ # stage1.candidate
│      │  　 │── train_create_candidate_dataset.py
│      │  　 │── test_create_candidate_dataset.py
│      │── reranker/ # stage2.reranker
│      │  　 │── train_reranker_neg_sample1.py # train
│      │  　 │── train_reranker_neg_sample2.py # train
│      │  　 │── train_reranker_neg_sample3.py # train
│      │  　 │── inferencet_reranker_neg_sample1.py # inference
│      │  　 │── inference_reranker_neg_sample2.py # inference
│      │  　 │── inference_reranker_neg_sample3.py # inference
│      │── ensemble/ 
│      │  　 │── ensemble.py
```

## LLM generate
* Add LLM-generated answer of question to query(HyDE)
  
```
python code/llm_generate/train_llm_answer_question_body.py
```
``` 
python code/llm_generate/test_llm_answer_question_body.py
```
```
python code/llm_generate/train_llm_summarize_question_body.py 
```
```
python code/llm_generate/test_llm_summarize_question_body.py 
```

## Stage1. Create candidate
* Perform contrastive learning(SimCSE: Simple Contrastive Learning of Sentence Embeddings) by train query vs passage
* Vector search by the trained model above and perform knn
* Preprocess text(Add prefix before text: <question>, <llm answer> etc., head-tail word :Use words from the first half and second half of the text.)
* In train, a lot of data are used in the passage to increase the number of positive examples (train, papers, etc.). In test, however, papers that were added in phase 2 are only used for inference
  
``` 
git clone https://github.com/rapidsai/rapidsai-csp-utils.git
```

``` 
python rapidsai-csp-utils/colab/pip-install.py
```

``` 
python code/create_candidate/train_create_candidate_dataset.py
```

``` 
python code/create_candidate/test_create_candidate_dataset.py
```

## Stage2. Reranking
* Obtain top 100 candidates by gte small model, and then perform binary classification to the obtained candidates using deberta V3　large model
* To reduce negative candidate examples, perform negative down sampling
* NOTE: Without installing numpy 1.23.5, we had dependency errors. For this reason, re-installing numpy is encouraged here

``` 
pip install numpy==1.23.5
```

``` 
python code/reranker/train_rerank_neg_sample_1.py
```

``` 
python code/reranker/train_rerank_neg_sample_2.py
```

``` 
python code/reranker/train_rerank_neg_sample_3.py
```

``` 
python code/reranker/inference_rerank_neg_sample_1.py
```

``` 
python code/reranker/inference_rerank_neg_sample_2.py
```

```
python code/reranker/inference_rerank_neg_sample_3.py
```

## Ensemble
* Ensemble 3 reranker preds(0.0~1.0) by simply averaging

``` 
python code/ensemble/ensemble.py
```

## Score

<img width="893" alt="スクリーンショット 2024-06-14 10 00 58" src="https://github.com/NTT-DOCOMO-RD/kddcup2024-oag-challenge-ind-7th-aqa-7th-solution-nttdocomolabs/assets/111550364/1ce788ab-a400-4a99-97ed-a2d0b6fb1a77">



