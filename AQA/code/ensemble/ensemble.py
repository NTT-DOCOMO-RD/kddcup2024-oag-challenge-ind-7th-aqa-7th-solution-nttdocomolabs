import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import gc
import json
from os.path import join
import warnings
warnings.filterwarnings('ignore')

# Config
class Config:
  ensemble = 'ensemble_preds'
  pred_1 = 'pred_rerank_neg_sample_1'
  pred_2 = 'pred_rerank_neg_sample_2'
  pred_3 = 'pred_rerank_neg_sample_3'
  seed = 42

# PATH
INPUT_DIR = 'data/submit_file/'
OUTPUT_DIR = 'data/submit_file/'

# Utils
def str2list(x):
  x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
  l = [i for i in x.split() if i]
  return l


def main():

    # Load
    test_pred1 = pd.read_csv(INPUT_DIR + f"{Config.pred_1}.csv")
    test_pred2 = pd.read_csv(INPUT_DIR + f"{Config.pred_2}.csv")
    test_pred3 = pd.read_csv(INPUT_DIR + f"{Config.pred_3}.csv")

    # preprocess
    test_pred1 = test_pred1.rename(columns={'preds':'pred1'})
    test_pred2 = test_pred2.rename(columns={'preds':'pred2'})
    test_pred3 = test_pred3.rename(columns={'preds':'pred3'})
    test = test_pred1.merge(test_pred2[['q_id','pids','pred2']], on=['q_id','pids'], how='left')
    test = test.merge(test_pred3[['q_id','pids','pred3']], on=['q_id','pids'], how='left')
    test['pred1'] = test['pred1'].fillna(0)
    test['pred2'] = test['pred2'].fillna(0)
    test['pred3'] = test['pred3'].fillna(0)

    # ensemble
    test['preds'] = (test['pred1'] + test['pred2'] + test['pred3'] ) / 3

    # create submit file
    eval_df = pd.DataFrame()
    for q_id in test['q_id'].unique():
        q_id_df = test[test['q_id']==q_id].reset_index(drop=True)
        q_id_df = q_id_df.sort_values('preds', ascending=False).reset_index(drop=True)
        pred_pids_lst = [pred for pred in q_id_df['pids']]
        q_id_df['pred_pids_lst'] = str(list(pred_pids_lst))
        q_id_df['pred_pids_lst'] = q_id_df['pred_pids_lst'].apply(str2list)
        eval_df = pd.concat([eval_df, q_id_df])
        eval_df = eval_df[['q_id','pred_pids_lst']].drop_duplicates(subset='q_id').reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    eval_df['pred_pids_lst'] = eval_df['pred_pids_lst'].apply(lambda x:x[:20])
    eval_df['pred_pids_lst'] = eval_df['pred_pids_lst'].apply(lambda x:(','.join(map(str, x))))

    # save
    eval_df['pred_pids_lst'].to_csv(OUTPUT_DIR + f'{Config.ensemble}.txt', sep='\t', escapechar='\b',index=False, header=False,)

if __name__ == "__main__":
    main()