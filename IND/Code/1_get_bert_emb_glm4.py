#!/usr/bin/python

import os
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs("../test_feature/", exist_ok=True)
warnings.simplefilter("ignore")


def embed(docs, model, tokenizer):
    # tokenize
    vec, query_list = [], []
    batch = 1
    for doc in docs:
        text = ""
        if len(doc) == 0:
            doc = ""
        text = f"{doc}"
        query_list.append(text)
    for start in range(0, len(query_list), batch):
        tokens = tokenizer(
            query_list[start : start + batch],
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
            last_hidden_state = out.hidden_states[-1][0]
            last_hidden = last_hidden_state.masked_fill(
                ~tokens["attention_mask"][..., None].bool(), 0.0
            )
            doc_embeds = (
                last_hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
            )
            doc_embeds = torch.where(
                torch.isinf(doc_embeds) | torch.isnan(doc_embeds),
                torch.tensor(0.0, device=doc_embeds.device),
                doc_embeds,
            )
            doc_embeds = F.normalize(doc_embeds, p=2, dim=1)
        vec.extend(doc_embeds.cpu().float().numpy().tolist())
    return vec


def concat_text(row):
    text = ""
    for col in ["title", "abstract", "keywords", "venue"]:
        if len(row[col]) > 0:
            text = text + " " + row[col].rstrip(".")
    row["all"] = text.strip()
    return row


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc(df_id, emb, prefix, target):
    df_list, type_list = [], []
    cols = [f"col_{i}" for i in range(len(emb[0]))]
    df_emb = pd.DataFrame(emb, columns=cols)
    df_emb = pd.concat([df_id, df_emb], axis=1)
    df_emb = df_emb.set_index(["author_id", "paper_id", "label"])
    df_list.append(df_emb)
    type_list.append("vec")

    author_vec = df_emb.mean().values

    df_dot = df_emb.dot(df_emb.T)
    df_norm = pd.DataFrame(np.linalg.norm(df_emb.values, axis=1), index=df_emb.index)
    df_norm = df_norm.dot(df_norm.T)
    df_cos = df_dot / df_norm
    for i in range(len(df_cos)):
        df_cos.iloc[i, i] = 0
    df_list.append(df_cos)
    type_list.append("cos")

    col_list = []
    df_feature = pd.DataFrame()
    for df_per, n in zip(df_list, type_list):
        df_tmp = pd.concat(
            [
                df_per.mean(axis=1),
                df_per.median(axis=1),
                df_per.max(axis=1),
                df_per.min(axis=1),
                df_per.std(axis=1),
                df_per.var(axis=1),
                df_per.quantile(0.25, axis=1),
                df_per.quantile(0.75, axis=1),
            ],
            axis=1,
        )
        cols = [
            f"{prefix}_{n}_mean",
            f"{prefix}_{n}_median",
            f"{prefix}_{n}_max",
            f"{prefix}_{n}_min",
            f"{prefix}_{n}_std",
            f"{prefix}_{n}_var",
            f"{prefix}_{n}_q25",
            f"{prefix}_{n}_q75",
        ]
        df_tmp.columns = cols
        df_tmp[f"{prefix}_{n}_range"] = (
            df_tmp[f"{prefix}_{n}_max"] - df_tmp[f"{prefix}_{n}_min"]
        )
        df_tmp[f"{prefix}_{n}_iqr"] = (
            df_tmp[f"{prefix}_{n}_q75"] - df_tmp[f"{prefix}_{n}_q25"]
        )
        df_tmp[f"{prefix}_{n}_skew"] = df_tmp.apply(skew, axis=1)
        df_tmp[f"{prefix}_{n}_kurtosis"] = df_tmp.apply(kurtosis, axis=1)
        col_list.extend(cols)
        df_feature = pd.concat([df_feature, df_tmp], axis=1)
    return df_feature


def glm(row, proc, df_master, model, tokenizer):
    author_id = row["id"]
    if proc == "train":
        paper_ids = row["normal_data"].tolist() + row["outliers"].tolist()
        labels = [1] * len(row["normal_data"]) + [0] * len(row["outliers"])
    elif proc == "test":
        paper_ids = row["papers"].tolist()
        labels = [0] * len(paper_ids)
    df_ids = pd.DataFrame(
        {"author_id": author_id, "paper_id": paper_ids, "label": labels}
    )
    df = df_ids.merge(df_master, left_on="paper_id", right_on="id", how="left")
    df = df.apply(concat_text, axis=1)

    target_col = ["all"]
    id_col_list = ["author_id", "paper_id", "label"]
    org_col = [col for col in df.columns.tolist() if col not in id_col_list]
    feature_list = []

    f_col = f"glm4_all"
    vec = embed(df["all"].tolist(), model, tokenizer)
    feature_list.append(calc(df_ids, vec, f_col, "all"))

    df = df.set_index(["author_id", "paper_id", "label"])
    for df_f in feature_list:
        df = df.join(df_f, how="left")
    df = df.drop(org_col, axis=1)
    df = df.reset_index()
    print(df.head(2))
    return df


def main():
    df_master = pd.read_parquet("../test_data/cleaned_pid_to_info_all_v6_light.parquet")
    df_train_master = pd.read_parquet("../test_data/train_author.parquet")
    df_test_master = pd.read_parquet(
        "../test_data/ind_test_author_filter_public.parquet"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_id = "THUDM/glm-4-9b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True).half().to(device)
    )
    model.eval()
    print(sum(p.numel() for p in model.parameters()))
    dfs_train = []

    for index, row in df_train_master.iterrows():
        print(index)
        dfs_train.append(glm(row, "train", df_master, model, tokenizer))
    df_train = pd.concat(dfs_train)
    print(df_train.shape)
    print(df_train.head())
    df_train.to_parquet("../test_feature/train_glm.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        print(index)
        dfs_test.append(glm(row, "test", df_master, model, tokenizer))
    df_test = pd.concat(dfs_test)
    print(df_test.shape)
    print(df_test.head())
    df_test.to_parquet("../test_feature/test_glm.parquet", index=False)


if __name__ == "__main__":
    main()
