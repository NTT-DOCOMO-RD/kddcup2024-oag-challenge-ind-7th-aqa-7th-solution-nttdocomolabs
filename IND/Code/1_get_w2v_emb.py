#!/usr/bin/python
import os
import gensim
import warnings
import itertools
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope


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


def calc_eel(emb):
    try:
        eel = EllipticEnvelope(contamination=0.1)
        eel.fit(emb)
        label = eel.predict(emb)
        pred = eel.decision_function(emb)
    except Exception as e:
        print(emb)
        eel = EllipticEnvelope(contamination=0.1, support_fraction=0.8)
        eel.fit(emb)
        label = eel.predict(emb)
        pred = eel.decision_function(emb)
    return label, pred


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
    clf = LocalOutlierFactor(contamination=0.1, n_neighbors=n)
    label = clf.fit_predict(emb)
    emb = clf.negative_outlier_factor_
    pred = scipy.stats.zscore(emb)
    return label, pred


def calc_ilf(emb):
    clf = IsolationForest(contamination=0.1, n_estimators=100, random_state=1)
    clf.fit(emb)
    pred = clf.decision_function(emb)
    label = clf.predict(emb)
    return label, pred


def concat_text(row):
    temp = []
    temp.extend(row["title"])
    temp.extend(row["abstract"])
    row["title_abstract"] = temp
    return row


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calc(df_id, emb, prefix, author_vec_v2, target):
    df_list, type_list = [], []
    cols = ["paper_id"] + [f"col_{i}" for i in range(128)]
    df_emb = pd.DataFrame(emb, columns=cols)
    df_emb = df_id.merge(df_emb, on="paper_id", how="left")
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

    # if flag:
    #     df_mah =df_emb.parallel_apply(calc_mahalanobis, emb=df_emb.values, axis=1)
    #     df_list.append(df_mah)
    #     type_list.append('mah')

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
        ]
        df_tmp.columns = cols
        col_list.extend(cols)
        df_feature = pd.concat([df_feature, df_tmp], axis=1)
    author_cos_list, author_cos_list_v2 = [], []
    for i, r in df_emb.iterrows():
        author_cos_list.append(cos_sim(author_vec, r.values))
        author_cos_list_v2.append(cos_sim(author_vec_v2, r.values))
    df_feature[f"{prefix}_cos_author"] = author_cos_list
    df_feature[f"{prefix}_cos_author_v2"] = author_cos_list_v2
    if target in ["all", "title", "abstract"]:
        label_dbs = calc_dbs(df_emb.values)
        pred_km = calc_km(df_emb.values)
        pred_gm = calc_gm(df_emb.values)
        # label_eel,pred_eel = calc_eel(df_emb.values)
        label_ocs, pred_ocs = calc_ocs(df_emb.values)
        label_lof, pred_lof = calc_lof(df_emb.values)
        label_ilf, pred_ilf = calc_ilf(df_emb.values)
        df_feature[f"{prefix}_label_dbs"] = label_dbs
        df_feature[f"{prefix}_label_lof"] = label_lof
        df_feature[f"{prefix}_label_ilf"] = label_ilf
        df_feature[f"{prefix}_label_ocs"] = label_ocs
        # df_feature[f'{prefix}_label_eel'] = label_eel

        df_pred = pd.DataFrame()
        df_pred[f"{prefix}_pred_lof"] = pred_lof
        df_pred[f"{prefix}_pred_ilf"] = pred_ilf
        df_pred[f"{prefix}_pred_ocs"] = pred_ocs
        # df_pred[f'{prefix}_pred_eel'] = pred_eel
        df_pred[f"{prefix}_pred_km"] = pred_km
        df_pred[f"{prefix}_pred_gm"] = pred_gm
        ss = StandardScaler()
        ss.fit(df_pred)
        df_pred = pd.DataFrame(ss.transform(df_pred), columns=df_pred.columns.tolist())
        df_feature.reset_index(inplace=True)
        df_feature = pd.concat([df_feature, df_pred], axis=1)
        df_feature = df_feature.set_index(["author_id", "paper_id", "label"])
    return df_feature


def w2v(row, proc, df_master, model, o_model):
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
    target_col = ["title", "abstract", "keywords", "venue", "orgs", "title_abstract"]
    id_col_list = ["author_id", "paper_id", "label"]
    org_col = [col for col in df.columns.tolist() if col not in id_col_list]
    all_emb, all_words = [], []
    for index, r in df.iterrows():
        words = []
        words.extend(r["title"])
        words.extend(r["abstract"])
        words.extend(r["keywords"])
        if r["venue"] is not None:
            words.extend(r["venue"])
        if r["year"] > 0:
            words.extend([str(int(r["year"]))])
        all_words.extend(words)
        all_emb.append(
            list(itertools.chain.from_iterable([[r["id"]], get_emb(words, model)]))
        )

    all_vec = get_emb(all_words, model)

    feature_list = []
    feature_list.append(calc(df_ids, all_emb, "w2v_all", all_vec, "all"))

    for col in target_col:
        per_words = []
        f_col = f"w2v_{col}"
        emb_list = []
        if col == "orgs":
            for index, r in df.iterrows():
                p_id = r["id"]
                words = []
                for a in r["authors"]:
                    name, org = a[0].split(","), a[1].split(",")
                    if author_name not in name:
                        for o in org:
                            if o != "NULL":
                                words.append(o)
                per_words.extend(words)
                emb_list.append(
                    list(
                        itertools.chain.from_iterable([[p_id], get_emb(words, o_model)])
                    )
                )
            all_vec = get_emb(per_words, o_model)
        else:
            for index, r in df.iterrows():
                p_id = r["id"]
                per_words.extend(r[col])
                emb_list.append(
                    list(
                        itertools.chain.from_iterable([[p_id], get_emb(r[col], model)])
                    )
                )
            all_vec = get_emb(per_words, model)
        feature_list.append(calc(df_ids, emb_list, f_col, all_vec, col))

    df = df.set_index(["author_id", "paper_id", "label"])
    for df_f in feature_list:
        df = df.join(df_f, how="left")
    df = df.drop(org_col, axis=1)
    df = df.reset_index()
    print(df.head())
    return df


def main():
    df_master = pd.read_parquet("../test_data/cleaned_pid_to_info_all_v6.parquet")
    df_train_master = pd.read_parquet("../test_data/train_author.parquet")
    df_test_master = pd.read_parquet("../test_data/ind_valid_author.parquet")

    model = gensim.models.Word2Vec.load(
        "../test_data/w2v_concat_cbow_128dim_min2_window10_neg5_epoch30_v6.bin"
    )
    o_model = gensim.models.Word2Vec.load(
        "../test_data/w2v_org_cbow_128dim_window5_min5_neg5_epoch30_v6.bin"
    )

    dfs_train = []
    for index, row in df_train_master.iterrows():
        dfs_train.append(w2v(row, "train", df_master, model, o_model))
        print(index)
    df_train = pd.concat(dfs_train)
    print(df_train.shape)
    print(df_train.head())
    df_train.to_parquet("../feature/train_w2v_latest.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        dfs_test.append(w2v(row, "test", df_master, model, o_model))
        print(index)
    df_test = pd.concat(dfs_test)
    print(df_test.shape)
    print(df_test.head())
    df_test.to_parquet("../feature/test_w2v_latest.parquet", index=False)


if __name__ == "__main__":
    main()
