# WhoIsWho-IND-KDD-2024

## Team information
- Team name: DOCOMOLABS
- Rank: 7th
- Score: 0.80487
- Parameters: 10,213,351,506
- Total GPU Memory: 24GB

## Prerequisites
* AMI 
    * Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.2.0 (Amazon Linux 2) 20240604 
* AWS EC2 
    * g5.2xlarge（GPU 24GB） 

## Summary
![alt text](<summary.png>)

## Getting Started

### Installation
For``IND``,
```
pip install -r Code/requirements.txt
```
### Execute
Please execute in the following order. 
You can start from any number within the same number.

* Code/0_xx => Code/1_xx => Code/2_xx

### Notes on Execution

If you want to start with feature generation, unzip raw.zip, 
extract it to the same hierarchy as "Code", 
and execute in order from Code/0_xxx.

If you want to skip feature generation (Code/0_xxxxx, Code/1_xxxxx), 
unzip test_data.zip & test_feature.zip, 
extract them to the same hierarchy as "Code", and then execute Code/2_XXX

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