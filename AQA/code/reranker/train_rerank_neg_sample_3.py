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
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


#CFG
class CFG:
  exp = 'rerank_neg_sample_3'

  epochs = 3
  seed = 42
  batch_size = 32
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
  fc_dropout=0.2

  # query, answer token length
  question_body_length = 256
  title_abs_length = 256
  # negative sampling
  neg_sample_ratio = 0.05
  debug = False #True

# PATH
TRAIN_DATASET_DIR = 'data/candidates/'
OUTPUT_DIR = f'model/stage2/{CFG.exp}/'

# Utils
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

# Negative down sampling
def neg_down_sample(df):

  print('Negative down sampling')
  pos_row = df[df['target']==1].reset_index(drop=True)
  neg_row = df[df['target']==0].reset_index(drop=True)

  print(' Before neg sampling')
  print(' - pos record:',len(pos_row))
  print(' - neg record:',len(neg_row))

  neg_row = neg_row.sample(frac=CFG.neg_sample_ratio, random_state=CFG.seed).reset_index(drop=True)

  print(' After neg sampling')
  print(' - pos record:',len(pos_row))
  print(' - neg record:',len(neg_row))

  df = pd.concat([pos_row, neg_row])
  df = df.sort_values('q_id',ascending=True).reset_index(drop=True)

  return df


#TrainDataset
class TrainDataset(Dataset):
  def __init__(self, cfg, df):
    self.cfg = cfg
    self.text = df['text'].values
    self.target = df[cfg.target].values

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

    target = torch.tensor(self.target[idx], dtype=torch.float)
    return inputs, target

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

        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
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
        output = self.fc(self.fc_dropout(feature))
        return output
    
#Helper function
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

#train_fn
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
  model.train()
  scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
  losses = AverageMeter()
  start = end = time.time()
  global_steps = 0

  for step, (inputs, targets) in enumerate(train_loader):

    for k, v in inputs.items():
      inputs = collate(inputs)
      inputs[k] = v.to(device)
    targets = targets.to(device)
    batch_size = targets.size(0)

    with torch.cuda.amp.autocast():
      y_preds = model(inputs)

      loss = criterion(y_preds.squeeze(1), targets)
    if CFG.gradient_accumulation_steps > 1:
        loss = loss / CFG.gradient_accumulation_steps
    losses.update(loss.item(), batch_size)
    scaler.scale(loss).backward()

    grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), CFG.max_grad_norm)
    if (step + 1) % CFG.gradient_accumulation_steps==0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
      global_steps += 1
      if CFG.batch_scheduler:
        scheduler.step()

    end = time.time()
    if step % CFG.print_freq==0 or step==(len(train_loader)-1):
      print(f'epoch: [{epoch+1}][{step}/{len(train_loader)}] '
      f'Elapsed{timeSince(start, float(step+1)/len(train_loader)):s} '
      f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
      f'Grad: {grad_norm:.4f} '
      f'LR: {scheduler.get_lr()[0]:.8f} '
      )

  return losses.avg, model


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
  optimizer_parameters = [
      {'params':[p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
      'lr':encoder_lr, 'weight_decay': weight_decay},
      {'params':[p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
      'lr':encoder_lr, 'weight_decay': 0.0},
      {'params':[p for n, p in model.named_parameters() if 'model' not in n],
      'lr':decoder_lr, 'weight_decay': 0.0},
  ]
  return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
  if cfg.scheduler == 'linear':
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=cfg.num_warmup_steps,
                                                num_training_steps=num_train_steps)
  elif cfg.scheduler == 'cosine':
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=cfg.num_warmup_steps,
                                                num_training_steps=num_train_steps,
                                                num_cycles=CFG.num_cycles)
  return scheduler


def main():

    # Load
    # Supervides dataset
    train = pd.read_csv(TRAIN_DATASET_DIR + "train.csv")
    
    # preprocess
    train['question_llm_answer_summarize'] = train['question_llm_answer_summarize'].apply(lambda x:' '.join(x.split(' ')[:CFG.question_body_length]))
    train['preds_text'] =  train['preds_text'].apply(lambda x:' '.join(x.split(' ')[:CFG.title_abs_length]))
    train['text'] = train['question_llm_answer_summarize'] + '[SEP]' + train['preds_text']

    #tokenizer
    tokenizer=AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer=tokenizer

    # Sampling
    # ------ delet only negative sample ------- #
    print('Init')
    print(' train q_id:, ', train['q_id'].nunique())
    print(' train row:, ', len(train))
    pos_train = train[train[CFG.target] == 1].reset_index(drop=True)
    neg_train = train[train[CFG.target] == 0].reset_index(drop=True)
    pos_train_q_id = pos_train['q_id'].unique()
    neg_train_q_id = neg_train['q_id'].unique()
    not_only_neg_train = neg_train[neg_train['q_id'].isin(pos_train_q_id)].reset_index(drop=True)
    print('Drop only negative')
    train = pd.concat([pos_train, not_only_neg_train]).reset_index(drop=True)
    print(' train q_id after drop only neg:, ', train['q_id'].nunique())
    print(' train row after drop only neg:, ', len(train))
    train = neg_down_sample(train)
    print('Final')
    print(' train row: ', len(train))
    print(' train pos row: ', len(train[train[CFG.target] == 1]))
    print(' train q_id: ', train['q_id'].nunique())
    del pos_train, neg_train, not_only_neg_train, pos_train_q_id, neg_train_q_id
    gc.collect()
    # ------------------------------------------ #

    # Dataset
    train_dataset = TrainDataset(CFG, train)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=CFG.num_workers)

    # model
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)

    # train
    optimizer_parameters = get_optimizer_params(model, encoder_lr=CFG.encoder_lr, decoder_lr=CFG.decoder_lr,weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    num_train_steps = int(len(train)/CFG.batch_size*CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(CFG.epochs):
        start_time = time.time()

        avg_loss, model = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        elapesed =time.time() - start_time

        print(f'Epoch: {epoch+1} - avg_train_loss:{avg_loss:.4f}')
        torch.save({'model':model.state_dict(),}, OUTPUT_DIR+f"{CFG.model.replace('/','-')}_epoch{epoch+1}.pth")

    torch.cuda.empty_cache()
    gc.collect()



if __name__ == "__main__":
    main()


