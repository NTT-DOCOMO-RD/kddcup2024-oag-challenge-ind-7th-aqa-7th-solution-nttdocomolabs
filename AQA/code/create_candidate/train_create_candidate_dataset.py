# =========================================================================================
# Libraries
# =========================================================================================
import os
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import cupy as cp
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
# %env TOKENIZERS_PARALLELISM=false
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config
class Config:

  model_name = 'Supabase/gte-small'
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  batch_size = 32
  seed = 42
  fold_num = 4
  gradient_checkpointing = True
  scheduler = 'cosine'
  batch_scheduler = True
  dropout = 0.0
  num_cycles = 0.5
  num_warmup_steps = 0
  min_lr = 1e-6
  eps = 1e-6
  betas = (0.9, 0.999)
  encoder_lr = 2e-5
  decoder_lr = 2e-5
  min_lr = 1e-6
  weight_decay = 0.01
  gradient_accumulation_steps =1
  max_grad_norm=1000
  apex = True
  print_freq = 100
  num_workers = 4

  max_length = 256
  question_body_length = 126
  title_abs_length = 126

  epochs = 1

  top_n = 1000
  pp_top_n = 100

  debug = False

# PATH
INPUT_DIR = 'data/raw/'
OUTPUT_MODEL_DIR = 'model/stage1/train_emb_model/'
OUTPUT_DIR = 'data/candidates/'
LLM_ANSWER_DIR = 'data/llm_generated_output/'
LLM_SUMMARIZE_DIR = 'data/llm_generated_output/'


# Utils
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(Config.seed)

def str2list(x):
  x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
  l = [i for i in x.split() if i]
  return l

def load_data(data):
  if data == 'train':
    with open(INPUT_DIR + "qa_train.txt", 'r', encoding='utf-8') as file:
        df = pd.DataFrame()
        for idx, line in enumerate(file):
          data = json.loads(line)
          df.loc[idx,'question'] = data["question"]
          df.loc[idx,'body'] = data["body"]
          df.loc[idx,'pids'] = str(data["pids"])
    df['pids'] = df['pids'].apply(lambda x:str2list(x))

  else: # valid or test
    with open(INPUT_DIR + f"qa_{data}_wo_ans.txt", 'r', encoding='utf-8') as file:
        df = pd.DataFrame()
        for idx, line in enumerate(file):
          data = json.loads(line)
          df.loc[idx,'question'] = data["question"]
          df.loc[idx,'body'] = data["body"]

  return df


def shorten_text(text, max_words):
    """
    Retrieve and reconstruct the first and last words to bring the text below the specified word count.

    Args: text (str)
        text (str): Original text
        max_words (int): Maximum number of words

    Returns: text
        str: Shortened text
    """
    words = text.split()

    if len(words) <= max_words:
        return text

    get_words = max_words // 2

    start_words = words[:get_words]
    end_words = words[-get_words:]

    start_words = ' '.join(start_words)
    end_words = ' '.join(end_words)

    shortened_text = start_words + '...' + end_words

    return shortened_text

def remove_links(text):
    """
    Remove URLs and links from text
    """
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'\[[^\]]+\]\([^\)]+\)', '', text)
    text = re.sub(r'\[[^\]]+\]\s*\([^\)]*\)', '', text)

    text = re.sub(r'<a[^>]*>', '', text)
    text = re.sub(r'</a>', '', text)

    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'\([^\)]+\)', '', text)

    return text

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors = None,
        add_special_tokens = True,
        return_token_type_ids=True,
        max_length = cfg.max_length,
        truncation = True,
        padding = 'max_length',

    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_paper_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs

class uns_query_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['question_llm_answer_summarize'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs

# =========================================================================================
# Mean pooling class
# =========================================================================================
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


# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):

    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name)
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, input):
        outputs = self.feature(input)
        return outputs


# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []

    for step, inputs in enumerate(loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds


# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(paper, query, cfg, fold):
    # Create products_train dataset
    paper_dataset = uns_paper_dataset(paper, cfg)
    # Create sessions_train dataset
    query_dataset = uns_query_dataset(query, cfg)
    # Create products_train and sessions_train dataloaders
    paper_loader = DataLoader(
        paper_dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers,
        pin_memory = True,
        drop_last = False
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers,
        pin_memory = True,
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg, pretrained=True)
    model.to(device)


    # pretrained model
    state = torch.load(OUTPUT_MODEL_DIR + f'fold_{fold}_' + f"{Config.model_name.replace('/', '-')}_epoch{cfg.epochs}.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])


    # Get embeddings
    print(' ')
    print('Getting embeddings...')
    # Predict products_train
    paper_vec = get_embeddings(paper_loader, model, device)
    query_vec = get_embeddings(query_loader, model, device)
    # Transfer predictions to gpu
    paper_vec_gpu = cp.array(paper_vec)
    query_vec_gpu = cp.array(query_vec)
    # Release memory
    torch.cuda.empty_cache()
    del paper_dataset, query_dataset, paper_loader, query_loader, paper_vec, query_vec
    gc.collect()

    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = cfg.top_n, metric = 'cosine')
    neighbors_model.fit(paper_vec_gpu)
    indices = neighbors_model.kneighbors(query_vec_gpu, return_distance = False)
    predictions = []
    preds_text = []
    for k in range(len(indices)):
        pred = indices[k]
        p = [','.join([paper.loc[int(ind), 'pids'] for ind in pred])]
        t = ['<pred_separator>'.join([paper.loc[int(ind), 'text'] for ind in pred])]
        predictions.extend(p[:cfg.top_n])
        preds_text.extend(t[:cfg.top_n])
    query['pids_prediction'] = list(predictions)
    query['preds_text'] = list(preds_text)
    query = query.sort_values('q_id').reset_index(drop=True)
    # Release memory
    del neighbors_model, predictions, indices
    gc.collect()
    return query


# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(paper, query, cfg):
    # Create lists for training
    query_ids = []
    paper_ids = []
    question_body = []
    title_abs = []
    preds_texts = []
    targets = []
    # Iterate over each topic
    for k in tqdm(range(len(query))):
        row = query.iloc[k]
        query_id = row['q_id']
        query_question_body = row['question_llm_answer_summarize']
        pids_prediction = row['pids_prediction']
        preds_text = row['preds_text']
        ground_truth = row['pids_list']
        for pids_pred, preds_t in zip(pids_prediction, preds_text):
            paper_title_abs = paper.loc[paper[paper['pids']==pids_pred].index[0]]['title_abstract']
            query_ids.append(query_id)
            paper_ids.append(pids_pred)
            question_body.append(query_question_body)
            preds_texts.append(preds_t)
            title_abs.append(paper_title_abs)
            # If pred is in ground truth, 1 else 0
            if pids_pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {'q_id': query_ids,
         'pids': paper_ids,
         'question_llm_answer_summarize': question_body,
         'title_abstract': title_abs,
         'preds_text': preds_texts,
         'target': targets}
    )
    # Release memory
    del query_ids, paper_ids, question_body, title_abs, targets
    gc.collect()
    return train

def remove_special_chars(text):
    # regular expression pattern to replace special characters with blanks
    pattern = r'[^\w\s]'
    cleaned_text = re.sub(pattern, ' ', str(text))
    cleaned_text = cleaned_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').replace('\\', ' ').replace('\newline', ' ').replace('\'', ' ').replace('\"', ' ').replace('\a', ' ').replace('\b', ' ').replace('\f', ' ').replace('\v', ' ')
    return cleaned_text

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

#Dataset
class Dataset(Dataset):
  def __init__(self, cfg, df):
    self.cfg = cfg
    self.inputs_question_body = df['question_llm_answer_summarize'].values
    self.inputs_title_abs = df['title_abstract'].values

  def __len__(self):
    return len(self.inputs_question_body)

  def __getitem__(self, idx):

    question_body = self.inputs_question_body[idx]
    inputs_question_body = self.cfg.tokenizer.encode_plus(
        question_body,
        return_tensors=None,
        add_special_tokens = True,
        max_length=self.cfg.max_length,
        pad_to_max_length=True,
        truncation = True,
        return_token_type_ids=True,
        )

    title_abs = self.inputs_title_abs[idx]
    inputs_title_abs = self.cfg.tokenizer.encode_plus(
        title_abs,
        return_tensors=None,
        add_special_tokens = True,
        max_length=self.cfg.max_length,
        pad_to_max_length=True,
        truncation = True,
        return_token_type_ids=True,
        )

    for k, v in inputs_question_body.items():
      inputs_question_body[k] = torch.tensor(v, dtype=torch.long)

    for k, v in inputs_title_abs.items():
      inputs_title_abs[k] = torch.tensor(v, dtype=torch.long)

    return inputs_question_body, inputs_title_abs
  
# Model
class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.model = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        
#Simcse loss
def simcse_unsup_loss(feature_last_item, feature_next_item) -> 'tensor':
    y_true = torch.arange(0, feature_last_item.size(0), device=device)
    sim = F.cosine_similarity(feature_last_item.unsqueeze(1), feature_next_item.unsqueeze(0), dim=2)
    sim = sim / 0.05
    loss = F.cross_entropy(sim, y_true)
    loss = torch.mean(loss)
    return loss

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
                                                num_cycles=Config.num_cycles)
  return scheduler



def main():

    ## Load
    # paper
    papers = pd.read_json(INPUT_DIR + "pid_to_title_abs_update_filter.json")
    papers = papers.T.reset_index().rename(columns={'index':'pids'})
    papers['title'] = papers['title'].fillna('')
    papers['abstract'] = papers['abstract'].fillna('')
    for col in papers.select_dtypes(include=['object']).columns:
        papers[col] = papers[col].apply(remove_special_chars)
    # train
    train = load_data('train')
    train['q_id'] = np.arange(len(train))
    # llm answer question
    train_llm_answer= pd.read_csv(LLM_ANSWER_DIR + "train_llm_answer_question_body.csv")
    # llm summarize question
    train_llm_summarize = pd.read_csv(LLM_SUMMARIZE_DIR + "train_llm_summarize_question_body.csv")

    ## CV
    gkf = GroupKFold(n_splits=Config.fold_num)
    for fold, ( _, val_) in enumerate(gkf.split(X=train, groups=train.q_id)):
        train.loc[val_ , "kfold"] = int(fold)
    train["kfold"] = train["kfold"].astype(int)

    ## Preprocess
    # paper
    papers['title'] = papers['title'].fillna('')
    papers['abstract'] = papers['abstract'].fillna('')
    # add prefix
    papers['title'] = '<title>'  + ' ' + papers['title']
    papers['abstract'] = '<abstract>'  + ' ' + papers['abstract']
    papers['title_abstract'] = papers['title'] + ' ' +  papers['abstract']

    # llm_answer
    train_llm_answer = train_llm_answer.dropna(subset=['question','body','create_llm_answer_question']).reset_index(drop=True)
    train_llm_answer = train_llm_answer.drop_duplicates(subset=['question']).reset_index(drop=True)

    # llm_summarize
    train_llm_summarize = train_llm_summarize.dropna(subset=['question','body','create_llm_summarize']).reset_index(drop=True)
    train_llm_summarize = train_llm_summarize.drop_duplicates(subset=['question']).reset_index(drop=True)

    # train
    train = train.merge(train_llm_answer[['question','create_llm_answer_question']], how='left', on=['question'])
    train = train.merge(train_llm_summarize[['question','create_llm_summarize']], how='left', on=['question'])
    train['create_llm_answer_question'] = train['create_llm_answer_question'].fillna('')
    train['create_llm_summarize'] = train['create_llm_summarize'].fillna('')
    train['create_llm_answer_question'] = train['create_llm_answer_question'].apply(lambda x:shorten_text(x, Config.max_length))
    train['create_llm_summarize'] = train['create_llm_summarize'].apply(lambda x:shorten_text(x, Config.max_length))
    train = train.explode('pids').reset_index(drop=True)
    train = train.merge(papers, how='left', on='pids')
    # add prefix
    train['question'] = '<question>'  + ' ' + train['question']
    train['body'] = '<body>'  + ' ' + train['body']
    train['create_llm_answer_question'] = '<llm answer>'  + ' ' + train['create_llm_answer_question']
    train['create_llm_summarize'] = '<llm summarize>'  + ' ' + train['create_llm_summarize']
    # contrative leranring query
    train['question_llm_answer_summarize'] = train['question'] + ' ' + train['create_llm_answer_question'] + ' ' + train['create_llm_summarize']
    # contrative leranring passage
    train['title_abstract'] =  train['title_abstract'].apply(lambda x:shorten_text(x, Config.title_abs_length))

    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
    Config.tokenizer = tokenizer
   
    oof_df = pd.DataFrame()

    for fold in range(Config.fold_num):
        print(f' ======= fold : {fold} =======')
        train_fold = train[train['kfold']!=fold].reset_index(drop=True)
        valid_fold = train[train['kfold']==fold].reset_index(drop=True)

        # ----- Contrastive learning ----- #

        train_dataset = Dataset(Config, train_fold)
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True, num_workers=Config.num_workers)

        Config.pooling = 'pooler'
        model = SimcseModel(Config.model_name, Config.pooling).to(device)

        losses = AverageMeter()
        scaler = torch.cuda.amp.GradScaler(enabled=Config.apex)
        start = end = time.time()
        global_steps = 0

        optimizer_parameters = get_optimizer_params(model, encoder_lr=Config.encoder_lr, decoder_lr=Config.decoder_lr,weight_decay=Config.weight_decay)
        optimizer = AdamW(optimizer_parameters, lr=Config.encoder_lr, eps=Config.eps, betas=Config.betas)

        num_train_steps = int(len(train)/Config.batch_size*Config.epochs)
        scheduler = get_scheduler(Config, optimizer, num_train_steps)


        for epoch in tqdm(range(Config.epochs)):
            print(f'epoch:{epoch}')

            for step, (question_body, title_abs) in enumerate(train_loader):

                batch_size = Config.batch_size #question_body.size(0)
                with torch.cuda.amp.autocast(enabled=Config.apex):
                    question_body = question_body.to(device)
                    title_abs = title_abs.to(device)
                    question_body_pid = model(question_body['input_ids'], question_body['attention_mask'], question_body['token_type_ids'])
                    title_abs_pid = model(title_abs['input_ids'], title_abs['attention_mask'], title_abs['token_type_ids'])
                    loss = simcse_unsup_loss(question_body_pid, title_abs_pid)

                if Config.gradient_accumulation_steps > 1:
                    loss = loss / Config.gradient_accumulation_steps
                losses.update(loss.item(), batch_size)
                scaler.scale(loss).backward()

                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), Config.max_grad_norm)
                if (step + 1) % Config.gradient_accumulation_steps==0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_steps += 1
                    if Config.batch_scheduler:
                        scheduler.step()


                end = time.time()
                if step % Config.print_freq==0 or step==(len(train_loader)-1):
                    print(f'epoch: [{epoch+1}][{step}/{len(train_loader)}] '
                    f'Elapsed{timeSince(start, float(step+1)/len(train_loader)):s} '
                    f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                    #f'Loss: {loss:.4f} '
                    f'Grad: {grad_norm:.4f} '
                    f'LR: {scheduler.get_lr()[0]:.8f} '
                    )

            torch.save({'model':model.state_dict()},OUTPUT_MODEL_DIR + f'fold_{fold}_' +f"{Config.model_name.replace('/','-')}_epoch{epoch+1}.pth")

            del question_body, title_abs, question_body_pid, title_abs_pid
            torch.cuda.empty_cache()
            gc.collect()

        # ------------------------------ #


        # --------- Retrieve ---------- #

        # question
        tmp_question = train_fold.copy()
        tmp_question = tmp_question.rename(columns={'question': 'text'})
        tmp_question = tmp_question[['text', 'pids']]
        # question + body
        tmp_question_body = train_fold.copy()
        tmp_question_body['body'] = tmp_question_body['body'].apply(remove_links)
        tmp_question_body['question_body'] = tmp_question_body['question']+ ' ' + tmp_question_body['body']
        tmp_question_body['question_body'] =  tmp_question_body['question_body'].apply(lambda x:shorten_text(x, Config.question_body_length))
        tmp_question_body = tmp_question_body.rename(columns={'question_body': 'text'})
        tmp_question_body = tmp_question_body[['text', 'pids']]
        # question + title + abstract
        tmp_question_body_title_abstract = train_fold.copy()
        tmp_question_body_title_abstract['question_title_abstract'] = tmp_question_body_title_abstract['question'] + ' ' + tmp_question_body_title_abstract['title_abstract']
        tmp_question_body_title_abstract = tmp_question_body_title_abstract.rename(columns={'question_title_abstract': 'text'})
        tmp_question_body_title_abstract = tmp_question_body_title_abstract[['text', 'pids']]
        # question + llm_answer
        tmp_train_question_llm_answer = train_fold.copy()
        tmp_train_question_llm_answer['question_llm_answer'] = tmp_train_question_llm_answer['question'] + ' ' + tmp_train_question_llm_answer['create_llm_answer_question']
        tmp_train_question_llm_answer = tmp_train_question_llm_answer[['question_llm_answer', 'pids']]
        tmp_train_question_llm_answer = tmp_train_question_llm_answer.rename(columns={'question_llm_answer': 'text'})
        tmp_train_question_llm_answer = tmp_train_question_llm_answer[['text', 'pids']]
        # question + llm_summarize
        tmp_train_question_llm_summarize = train_fold.copy()
        tmp_train_question_llm_summarize['question_llm_summarize'] = tmp_train_question_llm_summarize['question'] + ' ' + tmp_train_question_llm_summarize['create_llm_summarize']
        tmp_train_question_llm_summarize = tmp_train_question_llm_summarize[['question_llm_summarize', 'pids']]
        tmp_train_question_llm_summarize = tmp_train_question_llm_summarize.rename(columns={'question_llm_summarize': 'text'})
        tmp_train_question_llm_summarize = tmp_train_question_llm_summarize[['text', 'pids']]
        # llm answer
        tmp_train_llm_answer = train_fold.copy()
        tmp_train_llm_answer['create_llm_answer_question'] = tmp_train_llm_answer['create_llm_answer_question']
        tmp_train_llm_answer = tmp_train_llm_answer.rename(columns={'create_llm_answer_question': 'text'})
        tmp_train_llm_answer = tmp_train_llm_answer[['text', 'pids']]
        # llm summarize
        tmp_train_llm_summarize = train_fold.copy()
        tmp_train_llm_summarize['create_llm_summarize'] = tmp_train_llm_summarize['create_llm_summarize']
        tmp_train_llm_summarize = tmp_train_llm_summarize.rename(columns={'create_llm_summarize': 'text'})
        tmp_train_llm_summarize = tmp_train_llm_summarize[['text', 'pids']]
        # question + llm_answer + llm_summarize
        tmp_train_llm_answer_summarize = train_fold.copy()
        tmp_train_llm_answer_summarize = tmp_train_llm_answer_summarize[['question_llm_answer_summarize', 'pids']]
        tmp_train_llm_answer_summarize = tmp_train_llm_answer_summarize.rename(columns={'question_llm_answer_summarize': 'text'})
        tmp_train_llm_answer_summarize = tmp_train_llm_answer_summarize[['text', 'pids']]
        # title + abstract
        tmp_title_abstract = papers[['title_abstract', 'pids']]
        tmp_title_abstract = tmp_title_abstract.rename(columns={'title_abstract': 'text'})
        tmp_title_abstract = tmp_title_abstract[['text', 'pids']]

        # passage
        passage = pd.concat([tmp_question[['text', 'pids']], tmp_question_body[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_question_body_title_abstract[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_train_question_llm_answer[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_train_question_llm_summarize[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_train_llm_answer[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_train_llm_summarize[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_train_llm_answer_summarize[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = pd.concat([passage[['text', 'pids']], tmp_title_abstract[['text', 'pids']]], axis=0).reset_index(drop=True)
        passage = passage.dropna().reset_index(drop=True)

        del train_fold, tmp_question, tmp_question_body, tmp_question_body_title_abstract, tmp_train_question_llm_answer, tmp_train_question_llm_summarize, tmp_train_llm_answer, tmp_train_llm_summarize, tmp_train_llm_answer_summarize, tmp_title_abstract
        gc.collect()

        # Run nearest neighbors
        valid_fold = get_neighbors(passage, valid_fold, Config, fold)
        valid_fold['pids_prediction'] = valid_fold['pids_prediction'].apply(lambda x: list(x.split(',')))
        valid_fold['preds_text'] = valid_fold['preds_text'].apply(lambda x: list(x.split('<pred_separator>')))

        # postprocess
        valid_fold_pp = valid_fold.explode(['pids_prediction','preds_text']).reset_index(drop=True)
        valid_fold_pp = valid_fold_pp.drop_duplicates(subset=['q_id', 'pids_prediction']).reset_index(drop=True)
        valid_fold_pids_prediction = valid_fold_pp.groupby('q_id')['pids_prediction'].apply(lambda x: [pred for pred in x][:Config.pp_top_n]).reset_index()
        valid_fold_preds_text = valid_fold_pp.groupby('q_id')['preds_text'].apply(lambda x: [pred for pred in x][:Config.pp_top_n]).reset_index()
        valid_fold = valid_fold.drop(['pids_prediction', 'preds_text'],axis=1)
        valid_fold_pp = valid_fold_pids_prediction.merge(valid_fold_preds_text, how='left', on='q_id')
        valid_fold_pp = valid_fold_pp.merge(valid_fold, how='left', on='q_id')

        oof_df = pd.concat([oof_df, valid_fold_pp]).reset_index(drop=True)

        del valid_fold, passage, valid_fold_pp, valid_fold_pids_prediction, valid_fold_preds_text, model
        gc.collect()

        # ------------------------------ #


    ## Create train dataset

    # create list of groung trues pids 
    oof_df_pids_lst = oof_df.groupby('q_id')['pids'].apply(lambda x:list(x)).reset_index().rename(columns={'pids':'pids_list'})
    oof_df_pp = oof_df.merge(oof_df_pids_lst, how='left', on='q_id')

    # creat Supervised dataset
    test_oof_df = build_training_set(papers, oof_df_pp, Config)

    # drop duplicate
    test_oof_df = test_oof_df.drop_duplicates(subset=['q_id', 'pids']).reset_index(drop=True)

    # Save
    test_oof_df.to_csv(OUTPUT_DIR + 'train_candidates.csv', index = False)


if __name__ == "__main__":
    main()