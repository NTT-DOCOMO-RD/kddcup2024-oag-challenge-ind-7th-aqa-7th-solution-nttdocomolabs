#!/usr/bin/python
import os
import warnings
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from sklearn.covariance import MinCovDet
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

os.makedirs("../test_feature/", exist_ok=True)
warnings.simplefilter("ignore")


def get_emb(words, model):
    vec = np.zeros((128,), dtype="float32")
    num = 0
    for word in words:
        try:
            vec = np.add(vec, model.wv[word])
            num += 1
        except:
            pass
    if num > 0:
        vec = np.divide(vec, num)
    return vec


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
    clf = OneClassSVM(nu=0.1, gamma="auto", kernel="rbf")
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


def calc_mah(emb):
    mcd = MinCovDet()
    t = mcd.fit(emb)
    print(t)
    pred = mcd.mahalanobis(emb)
    return pred


def concat_text(row):
    temp = []
    temp.extend(row["title"])
    temp.extend(row["abstract"])
    row["title_abstract"] = temp
    return row


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc(df):
    df = df.set_index(["author_id", "paper_id", "label"])
    num_col_list = [
        "t_num",
        "a_num",
        "k_num",
        "v_num",
        "n_num",
        "o_num",
        "year",
        "num_authors",
    ]
    base_mean = df[num_col_list].mean()
    for col in num_col_list:
        df[f"{col}_diff"] = df[col] - base_mean[col]
    return df


def trans(row, author_name):
    lang_list = ["en", "Nothing"]
    if row["t_lang"] not in lang_list:
        row["t_lang"] = "other"
    if row["a_lang"] not in lang_list:
        row["a_lang"] = "other"
    if row["k_lang"] not in lang_list:
        row["k_lang"] = "other"
    if row["v_lang"] not in lang_list:
        row["v_lang"] = "other"
    row["t_num"] = len(row["title"])
    row["a_num"] = len(row["abstract"])
    row["k_num"] = len(row["keywords"])
    row["v_num"] = len(row["venue"])
    names, orgs = [], []
    for a in row["authors"]:
        name, org = a[0].split(","), a[1].split(",")
        if author_name not in name:
            for n in name:
                if n != "NULL":
                    names.append(n)
            for o in org:
                if o != "NULL":
                    orgs.append(o)
    row["n_num"] = len(names)
    row["o_num"] = len(orgs)
    if (
        len(row["title"]) == 0
        and len(row["abstract"]) == 0
        and len(row["abstract"]) == 0
        and len(row["venue"]) == 0
    ):
        row["no_data"] = 1
    else:
        row["no_data"] = 0
    return row


def basic(row, proc, df_master):
    author_id = row["id"]
    if proc == "train":
        paper_ids = row["normal_data"].tolist() + row["outliers"].tolist()
        labels = [1] * len(row["normal_data"]) + [0] * len(row["outliers"])
    elif proc == "test":
        paper_ids = row["papers"].tolist()
        labels = [0] * len(paper_ids)
    df = pd.DataFrame({"author_id": author_id, "paper_id": paper_ids, "label": labels})
    id_col_list = ["author_id", "paper_id", "label"]
    df = df.merge(df_master, left_on="paper_id", right_on="id", how="left")
    df = df.drop("id", axis=1)
    drop_col = [
        col
        for col in df.columns.tolist()
        if col not in id_col_list + ["year", "num_author"]
    ]

    df["year"] = df["year"].fillna(df["year"].mean())

    df = df.apply(trans, author_name=row["name"], axis=1)
    df = calc(df)

    df = df[[col for col in df.columns.tolist() if col not in drop_col]]

    prefix = "basic"
    scaler = StandardScaler()
    emb = scaler.fit_transform(df)
    label_dbs = calc_dbs(emb)
    pred_km = calc_km(emb)
    pred_gm = calc_gm(emb)
    label_ocs, pred_ocs = calc_ocs(emb)
    label_lof, pred_lof = calc_lof(emb)
    label_ilf, pred_ilf = calc_ilf(emb)
    df[f"{prefix}_label_dbs"] = label_dbs
    df[f"{prefix}_label_lof"] = label_lof
    df[f"{prefix}_label_ilf"] = label_ilf
    df[f"{prefix}_label_ocs"] = label_ocs
    df[f"{prefix}_pred_lof"] = pred_lof
    df[f"{prefix}_pred_ilf"] = pred_ilf
    df[f"{prefix}_pred_ocs"] = pred_ocs
    df[f"{prefix}_pred_km"] = pred_km
    df[f"{prefix}_pred_gm"] = pred_gm
    print(df.head(2))
    return df


def main():
    df_master = pd.read_parquet("../test_data/cleaned_pid_to_info_all_v6.parquet")
    df_train_master = pd.read_parquet("../test_data/train_author.parquet")
    df_test_master = pd.read_parquet(
        "../test_data/ind_test_author_filter_public.parquet"
    )

    dfs_train = []
    for index, row in df_train_master.iterrows():
        dfs_train.append(basic(row, "train", df_master))
        print(index)
    df_train = pd.concat(dfs_train)
    df_train = df_train.reset_index()
    print(df_train.shape)
    print(df_train.head())
    df_train.to_parquet("../test_data/train_basic.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        dfs_test.append(basic(row, "test", df_master))
        print(index)
    df_test = pd.concat(dfs_test)
    df_test = df_test.reset_index()
    print(df_test.shape)
    print(df_test.head())
    df_test.to_parquet("../test_feature/test_basic.parquet", index=False)


if __name__ == "__main__":
    main()
