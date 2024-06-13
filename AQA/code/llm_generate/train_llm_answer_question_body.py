import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import gc
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
from pytz import timezone
import warnings
warnings.filterwarnings('ignore')

# Config
class Config:
    llm_model_name = "mlabonne/Marcoro14-7B-slerp" #LLM model
    max_length = 256 
    debug = False 

# PATH
INPUT_DIR = 'data/raw/' 
OUTPUT_DIR = 'data/llm_generated_output/' 

# utils
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


def remove_links(text):

    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'\[[^\]]+\]\([^\)]+\)', '', text)
    text = re.sub(r'\[[^\]]+\]\s*\([^\)]*\)', '', text)

    text = re.sub(r'<a[^>]*>', '', text)
    text = re.sub(r'</a>', '', text)

    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'\([^\)]+\)', '', text)

    return text

def remove_newlines(df):

    return df.apply(lambda x: str(x).replace("\n", ""))



def shorten_text(text, max_words):

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


def generate_prompt(question):


  prompt = """
  ### Instruction
  You are a researcher with expertise in cutting-edge technology.
  You are able to answer highly technical questions about technology in a straightforward and clear manner.
  Please answer technical questions posted on the Internet according to the following constraints.
  Also include any technical keywords that are relevant to this question.

  ### Constraints
  Please limit your summary to 150 words or less.
  Do not include URLs or links in your summary.

  ### Technical Question
  {question}

  ### Output
  """

  # Fill in prompt
  prompt = prompt.format(question=question,)

  return prompt



def main():
    
    # load
    qa_train = load_data('train')
    
    # remove_links
    qa_train['body'] = qa_train['body'].apply(remove_links)

    # remove_newlines
    qa_train['body'] = remove_newlines(qa_train['body'])

    # get head tail 
    text_length = Config.max_length
    qa_train['question_body'] = qa_train['question'] + '[SEP]' + qa_train['body']
    qa_train['question_body'] = qa_train['question_body'].apply(lambda x:shorten_text(x, text_length))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
      Config.llm_model_name,
      use_fast=True,
      padding = True,
      max_length = Config.max_length,
      truncation = True
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
      Config.llm_model_name,
      load_in_8bit=True,
      torch_dtype=torch.float16,
      device_map="auto",
    )

    # generate 
    create_llm_answer_question = qa_train.copy()
    for idx in range(len(create_llm_answer_question)):

        question = create_llm_answer_question['question_body'][idx]
        prompt = generate_prompt(question)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        attention_mask = torch.ones_like(input_ids)

        output = model.generate(input_ids,
                                max_new_tokens=160,
                                attention_mask=attention_mask,
                                num_return_sequences=1,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.85,
                                pad_token_id=tokenizer.eos_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                )

        # decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        create_llm_answer_question.loc[idx,'create_llm_answer_question'] = generated_text[len(prompt):]

        del question, prompt, input_ids, attention_mask, output, generated_text
        torch.cuda.empty_cache()
        gc.collect()

    # save
    create_llm_answer_question.to_csv(OUTPUT_DIR + 'train_llm_answer_question_body.csv', index=False)



if __name__ == "__main__":
    main()