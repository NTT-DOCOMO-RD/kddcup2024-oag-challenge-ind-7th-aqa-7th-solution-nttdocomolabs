#!/usr/bin/python

import os
import torch
import warnings
import scipy.stats
import numpy as np
import pandas as pd
from cogdl.oag import oagbert
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

os.makedirs("../test_feature/", exist_ok=True)
warnings.simplefilter("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def concat_text(row):
    text = ""
    for col in ["title", "abstract", "keywords", "venue"]:
        if len(row[col]) > 0:
            text = text + row[col].rstrip(".") + "."
    if len(text) == 0:
        text = "No information."
    row["all"] = text
    return row


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc_km(emb):
    k = KMeans(n_clusters=2)
    k.fit(emb)
    dis = k.transform(emb).min(axis=1)
    return dis


def calc_gm(emb):
    gm = GaussianMixture(n_components=2, covariance_type="full").fit(emb)
    pred = gm.score_samples(emb)
    return pred


def calc_dbs(emb):
    db = DBSCAN(
        eps=0.3,
        min_samples=4,
        metric="cosine",
    )
    db.fit(emb)
    label = db.labels_
    labels = []
    for p in label:
        if p == -1:
            labels.append(0)
        else:
            labels.append(1)
    return labels


def calc_ocs(emb):
    clf = OneClassSVM(nu=0.2, gamma="auto", kernel="rbf")
    clf.fit(emb)
    pred = clf.decision_function(emb)
    label = clf.predict(emb)
    return label, pred


def calc_lof(emb):
    n = int(len(emb) * 0.2)
    if n == 0:
        n = int(n / 2)
        if n == 0:
            n = 1
    clf = LocalOutlierFactor(contamination="auto", n_neighbors=n)
    label = clf.fit_predict(emb)
    emb = clf.negative_outlier_factor_
    pred = scipy.stats.zscore(emb)
    return label, pred


def calc_ilf(emb):
    clf = IsolationForest(contamination="auto", n_estimators=100, random_state=1)
    clf.fit(emb)
    pred = clf.decision_function(emb)
    label = clf.predict(emb)
    return label, pred


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
        df_feature = pd.concat([df_feature, df_tmp], axis=1)
    author_cos_list = []
    for i, r in df_emb.iterrows():
        author_cos_list.append(cos_sim(author_vec, r.values))
    df_feature[f"{prefix}_cos_author"] = author_cos_list
    if target in ["all", "title", "abstract"]:
        label_dbs = calc_dbs(df_emb.values)
        pred_km = calc_km(df_emb.values)
        pred_gm = calc_gm(df_emb.values)
        label_ocs, pred_ocs = calc_ocs(df_emb.values)
        label_lof, pred_lof = calc_lof(df_emb.values)
        label_ilf, pred_ilf = calc_ilf(df_emb.values)
        df_feature[f"{prefix}_label_dbs"] = label_dbs
        df_feature[f"{prefix}_label_lof"] = label_lof
        df_feature[f"{prefix}_label_ilf"] = label_ilf
        df_feature[f"{prefix}_label_ocs"] = label_ocs

        df_pred = pd.DataFrame()
        df_pred[f"{prefix}_pred_lof"] = pred_lof
        df_pred[f"{prefix}_pred_ilf"] = pred_ilf
        df_pred[f"{prefix}_pred_ocs"] = pred_ocs
        df_pred[f"{prefix}_pred_km"] = pred_km
        df_pred[f"{prefix}_pred_gm"] = pred_gm
        df_feature.reset_index(inplace=True)
        df_feature = pd.concat([df_feature, df_pred], axis=1)
        df_feature = df_feature.set_index(["author_id", "paper_id", "label"])
    return df_feature


def bert(row, proc, df_master, model, tokenizer):
    author_id = row["id"]
    author_name = row["name"]
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

    id_col_list = ["author_id", "paper_id", "label"]
    org_col = [col for col in df.columns.tolist() if col not in id_col_list]
    vec = []
    for i, r in df.iterrows():
        names, orgs = [], []
        for a in r["authors"]:
            name, org = a[0].split(","), a[1].split(",")
            if author_name not in name:
                for n in name:
                    if n != "NULL":
                        names.append(n)
                for o in org:
                    if o != "NULL":
                        orgs.append(o)
        (
            input_ids,
            input_masks,
            token_type_ids,
            masked_lm_labels,
            position_ids,
            position_ids_second,
            masked_positions,
            num_spans,
        ) = model.build_inputs(
            title=r["title"],
            abstract=r["abstract"],
            venue=r["venue"],
            concepts=r["keywords"].split(","),
            authors=names,
            affiliations=orgs,
        )
        try:
            _, emb = model.bert.forward(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
                token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device),
                attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device),
                position_ids_second=torch.LongTensor(position_ids_second)
                .unsqueeze(0)
                .to(device),
            )
            emb = F.normalize(emb, p=2, dim=1)
            emb = emb.cpu().detach().numpy().tolist()[0]
        except Exception as e:
            emb = np.zeros((768,), dtype="float32")
            emb = emb.tolist()
        vec.append(emb)

    f_col = f"oag_all"
    df_emb = calc(df_ids, vec, f_col, "all")
    df_emb = df_emb.reset_index()
    return df_emb


def main():
    df_master = pd.read_parquet("../test_data/cleaned_pid_to_info_all_v6_light.parquet")
    df_train_master = pd.read_parquet("../test_data/train_author.parquet")
    df_test_master = pd.read_parquet(
        "../test_data/ind_test_author_filter_public.parquet"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    tokenizer, model = oagbert("oagbert-v2-sim")
    model.to(device)
    model.eval()
    print(sum(p.numel() for p in model.parameters()))

    dfs_train = []
    for index, row in df_train_master.iterrows():
        print(index)
        dfs_train.append(bert(row, "train", df_master, model, tokenizer))
    df_train = pd.concat(dfs_train)
    print(df_train.shape)
    print(df_train.head())
    df_train.to_parquet("../test_feature/train_oag_bert.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        print(index)
        dfs_test.append(bert(row, "test", df_master, model, tokenizer))
    df_test = pd.concat(dfs_test)
    print(df_test.shape)
    print(df_test.head())

    df_test.to_parquet("../test_feature/test_oag_bert.parquet", index=False)


if __name__ == "__main__":
    main()
