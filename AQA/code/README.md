# OAG-AQA Task 7place Solution

## Prerequisites
* Linux 
* Python 3.10.12
* Tesla T4(CPU memory 50GB, GPU memory 16GB)

## Installation
pip install -r requirements.txt

## Dataset
phase1 https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA.zip

phase2 https://www.biendata.xyz/competition/aqa_kdd_2024/data/AQA-test-public.zip

## Stage0.LLM generate
* Generate LLM and add LLM-generated output to query
  
` python code/llm_generate/train_llm_answer_question_body.py `

` python code/llm_generate/test_llm_answer_question_body.py `

` python code/llm_generate/train_llm_summarize_question_body.py `

` python code/llm_generate/test_llm_summarize_question_body.py `

## Stage1.create candidate
* Contrastive learning(SimCSE: Simple Contrastive Learning of Sentence Embeddings) by train query vs passage
* Vector search by using Contrastive learning model weight and knn
* Preprocessing text(Add prefix before text: <question>, <llm answer> etc., head-tail word :Use words from the first half and second half of the text.)
* In train, a lot of data are used in the passage to increase the number of positive examples (train, papers, etc.).  In test, Use only papers data added in Phase 2
` git clone https://github.com/rapidsai/rapidsai-csp-utils.git `

` python rapidsai-csp-utils/colab/pip-install.py `

` python code/create_candidate/train_create_candidate_dataset.py `

` python code/create_candidate/test_create_candidate_dataset.py `

## Stage2.Reranking
* Binary classification candidates top 100 by deberta V3 large model
* Negative down sampling due to many negative candidate examples
` pip install numpy==1.23.5 `

` python code/reranker/train_rerank_neg_sample_1.py `

` python code/reranker/train_rerank_neg_sample_2.py `

` python code/reranker/train_rerank_neg_sample_3.py `

` python code/reranker/inference_rerank_neg_sample_1.py `

` python code/reranker/inference_rerank_neg_sample_2.py `
` python code/reranker/inference_rerank_neg_sample_3.py `

## Ensemble
* Ensemble 3 reranker preds(0.0~1.0) by simple average
` python code/ensemble/ensemble.py

