#Import
import os
import random
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


#CFG
class CFG:
  exp = 'rerank_neg_sample_1'

  inf_epoch = 2
  seed = 42
  batch_size = 64
  num_workers = 4
  gradient_checkpointing = True
  scheduler = 'cosine'
  batch_scheduler = True
  fc_dropout = 0.0
  num_cycles = 0.5
  num_warmup_steps = 0
  min_lr = 1e-6
  eps = 1e-6
  betas = (0.9, 0.999)
  model = "microsoft/deberta-v3-large"
  encoder_lr = 2e-5
  decoder_lr = 2e-5
  min_lr = 1e-6
  weight_decay = 0.01
  gradient_accumulation_steps =1
  max_grad_norm=1000
  target = 'target'
  max_len = 512
  apex = True
  print_freq = 100

  # use train model
  train_exp = exp
  train_dir = f'model/stage2/{train_exp}/'

  # query, answer token length
  question_body_length = 256
  title_abs_length = 256
  debug = False

# PATH
TEST_DATASET_DIR ='data/candidates/'
OUTPUT_DIR = f'data/submit_file/'

# Utils
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

def str2list(x):
  x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
  l = [i for i in x.split() if i]
  return l


# TestDataset
class TestDataset(Dataset):
  def __init__(self, cfg, df):
    self.cfg = cfg
    self.text = df['text'].values
    # self.target = df[cfg.target].values

  def __len__(self):
    return len(self.text)

  def __getitem__(self, idx):
    text = self.text[idx]
    inputs = self.cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens = True,
        max_length=self.cfg.max_len,
        pad_to_max_length=True,
        truncation = True
        )
    for k, v in inputs.items():
      inputs[k] = torch.tensor(v, dtype=torch.long)

    # target = torch.tensor(self.target[idx], dtype=torch.float)
    return inputs #, target

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

# Model
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.

        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

# test_fn
def test_fn(test_loader, model, device):
  model.eval()
  preds = []
  for step, (inputs) in enumerate(tqdm(test_loader)):
    for k, v in inputs.items():
      inputs = collate(inputs)
      inputs[k] = v.to(device)
    with torch.no_grad():
      y_preds = model(inputs)
    preds.append(y_preds.sigmoid().to('cpu').numpy())

  predictions = np.concatenate(preds)
  return predictions



def main():

    # Load
    test = pd.read_csv(TEST_DATASET_DIR + "test_candidates.csv")
    test = test[['q_id','pids','question_llm_answer_summarize','preds_text','target']]

    # preprocess
    test['question_llm_answer_summarize'] = test['question_llm_answer_summarize'].apply(lambda x:' '.join(x.split(' ')[:CFG.question_body_length]))
    test['preds_text'] =  test['preds_text'].apply(lambda x:' '.join(x.split(' ')[:CFG.title_abs_length]))
    test['text'] = test['question_llm_answer_summarize'] + '[SEP]' + test['preds_text']

    # tokenizer
    tokenizer=AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer=tokenizer

    # Dataset
    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=CFG.num_workers)

    # model
    CFG.config_path=CFG.train_dir+'config.pth'
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    CFG.path = CFG.train_dir
    state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_epoch{CFG.inf_epoch}.pth",
                        map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    model.to(device)

    # inference
    predictions = test_fn(test_loader, model, device)
    eval_df = pd.DataFrame()
    test['preds'] = predictions
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

    # Save
    test.to_csv(OUTPUT_DIR + f'pred_{CFG.exp}.csv',index=False)


if __name__ == "__main__":
    main()