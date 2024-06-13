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

` pip install -r requirements.txt `

## Dataset

* phase1 [https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA.zip](https://www.dropbox.com/scl/fi/2ckwl9fcpbik88z1cekot/AQA.zip?rlkey=o7ttmrvpdbvbu3rcr6t33jrx7&dl=1)

* phase2 [https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA-test-public.zip](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/AQA/AQA-test-public.zip)

## LLM generate
* Add LLM-generated answer of question to query(HyDE)
* Mthod overview
  - Input "question + body" information into LLM
  - Delete body links
  - Use only the first and second halves of the combined question + body text.
  
` python code/llm_generate/train_llm_answer_question_body.py `

` python code/llm_generate/test_llm_answer_question_body.py `

` python code/llm_generate/train_llm_summarize_question_body.py `

` python code/llm_generate/test_llm_summarize_question_body.py `

## Create candidate
* Perform contrastive learning(SimCSE: Simple Contrastive Learning of Sentence Embeddings) by train query vs passage
* Vector search by the trained model above and perform knn
* Preprocess text(Add prefix before text: <question>, <llm answer> etc., head-tail word :Use words from the first half and second half of the text.)
* In train, a lot of data are used in the passage to increase the number of positive examples (train, papers, etc.). In test, however, papers that were added in phase 2 are only used for inference
  
` git clone https://github.com/rapidsai/rapidsai-csp-utils.git `

` python rapidsai-csp-utils/colab/pip-install.py `

` python code/create_candidate/train_create_candidate_dataset.py `

` python code/create_candidate/test_create_candidate_dataset.py `

## Reranking
* Obtain top 100 candidates by gte small model, and then perform binary classification to the obtained candidates using deberta V3　large model
* To reduce negative candidate examples, perform negative down sampling
* NOTE: Without installing numpy 1.23.5, we had dependency errors. For this reason, re-installing numpy is encouraged here

` pip install numpy==1.23.5 `

` python code/reranker/train_rerank_neg_sample_1.py `

` python code/reranker/train_rerank_neg_sample_2.py `

` python code/reranker/train_rerank_neg_sample_3.py `

` python code/reranker/inference_rerank_neg_sample_1.py `

` python code/reranker/inference_rerank_neg_sample_2.py `

` python code/reranker/inference_rerank_neg_sample_3.py `

## Ensemble
* Ensemble 3 reranker preds(0.0~1.0) by simply averaging

` python code/ensemble/ensemble.py` 

## Score

<img width="654" alt="スクリーンショット 2024-06-13 18 10 23" src="https://github.com/NTT-DOCOMO-RD/kddcup2024-oag-challenge-ind-7th-aqa-7th-solution-nttdocomolabs/assets/111550364/0fa89646-0ade-4a69-b3d4-eb4df946469f">




