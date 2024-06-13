#!/usr/bin/python
import os
import warnings
import numpy as np
import itertools
import pandas as pd
import scipy.stats
from nltk.stem import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from scipy.stats import skew, kurtosis
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

os.makedirs("../test_feature/", exist_ok=True)
warnings.simplefilter("ignore")


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_vec(x, stopwords):
    N = 128
    check = [t for t in x if t != ""]
    if len(check) != 0:
        vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2))
        x = vectorizer.fit_transform(x)
        n_samples, n_features = x.shape
        x = x.toarray()
        if N > n_features:
            N = int(n_features / 2)
        try:
            svd = TruncatedSVD(n_components=N)
            x = svd.fit_transform(x)
            return x
        except Exception as e:
            print(x)
            return np.zeros((len(x), N))

    else:
        return np.zeros((len(x), N))


def calc_mah(emb):
    try:
        mcd = MinCovDet()
        mcd.fit(emb)
        pred = mcd.mahalanobis(emb)
    except Exception as e:
        pred = np.zeros((len(emb),), dtype="float32")
    return pred


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


def calc(df_id, emb, prefix, target):
    df_list, type_list = [], []
    cols = ["paper_id"] + [f"col_{i}" for i in range(len(emb[0]) - 1)]
    df_emb = pd.DataFrame(emb, columns=cols)
    df_emb = df_id.merge(df_emb, on="paper_id", how="left")
    df_emb = df_emb.set_index(["author_id", "paper_id", "label"])
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
    author_cos_list = []
    for i, r in df_emb.iterrows():
        author_cos_list.append(cos_sim(author_vec, r.values))
    df_feature[f"{prefix}_cos_author"] = author_cos_list

    if target in ["all", "title", "abstract"]:
        # pred_mah = calc_mah(df_emb.values)
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
        # df_pred[f'{prefix}_pred_mah'] = pred_mah
        df_pred[f"{prefix}_pred_lof"] = pred_lof
        df_pred[f"{prefix}_pred_ilf"] = pred_ilf
        df_pred[f"{prefix}_pred_ocs"] = pred_ocs
        # df_pred[f'{prefix}_pred_eel'] = pred_eel
        df_pred[f"{prefix}_pred_km"] = pred_km
        df_pred[f"{prefix}_pred_gm"] = pred_gm
        # ss = StandardScaler()
        # ss.fit(df_pred)
        # df_pred= pd.DataFrame(ss.transform(df_pred), columns=df_pred.columns.tolist())
        df_feature.reset_index(inplace=True)
        df_feature = pd.concat([df_feature, df_pred], axis=1)
        df_feature = df_feature.set_index(["author_id", "paper_id", "label"])

    return df_feature


def tf_idf(row, proc, df_master, stopwords):
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
    target_col = ["title", "abstract", "keywords", "venue", "orgs"]
    id_col_list = ["author_id", "paper_id", "label"]
    org_col = [col for col in df.columns.tolist() if col not in id_col_list]
    all_list = []

    for index, r in df.iterrows():
        temp_list = (
            r["title"].tolist()
            + r["abstract"].tolist()
            + r["keywords"].tolist()
            + r["venue"].tolist()
        )
        all_list.append(" ".join(temp_list))

    all_emb = get_vec(all_list, stopwords)
    all_emb = [
        list(itertools.chain.from_iterable([[i], emb]))
        for i, emb in zip(df["id"].tolist(), all_emb)
    ]
    feature_list = []
    feature_list.append(calc(df_ids, all_emb, "tfidf_all", "all"))

    for col in target_col:
        embs = []
        w_list = df[col].tolist()
        if col == "orgs":
            w_list = []
            for index, r in df.iterrows():
                words = []
                for a in r["authors"]:
                    name, org = a[0].split(","), a[1].split(",")
                    if author_name not in name:
                        for o in org:
                            if o != "NULL":
                                words.append(o)
                w_list.append(words)
        w_list = [" ".join(w) for w in w_list]
        w_list = get_vec(w_list, stopwords)
        embs = [
            list(itertools.chain.from_iterable([[i], emb]))
            for i, emb in zip(df["id"].tolist(), w_list)
        ]
        feature_list.append(calc(df_ids, embs, f"tfidf_{col}", col))

    df = df.set_index(["author_id", "paper_id", "label"])
    for df_f in feature_list:
        df = df.join(df_f, how="left")
    df = df.drop(org_col, axis=1)
    df = df.reset_index()
    # feature_col = [col for col in df.columns.tolist() if col not in id_col_list]
    # # ss = StandardScaler()
    # # ss.fit(df[feature_col])
    # # df[feature_col] = pd.DataFrame(ss.transform(df[feature_col]), columns=feature_col)
    print(df.head(2))
    return df


def get_stopword():
    stopwords = [
        "at",
        "based",
        "in",
        "of",
        "for",
        "on",
        "and",
        "to",
        "an",
        "using",
        "with",
        "the",
        "by",
        "we",
        "be",
        "is",
        "are",
        "can",
        "university",
        "univ",
        "china",
        "department",
        "dept",
        "laboratory",
        "lab",
        "school",
        "al",
        "et",
        "institute",
        "inst",
        "college",
        "chinese",
        "beijing",
        "journal",
        "science",
        "international",
        "a",
        "was",
        "were",
        "that",
        "2",
        "key",
        "1",
        "technology",
        "0",
        "sciences",
        "as",
        "from",
        "r",
        "3",
        "academy",
        "this",
        "nanjing",
        "shanghai",
        "state",
        "s",
        "research",
        "p",
        "results",
        "peoples",
        "4",
        "which",
        "5",
        "high",
        "materials",
        "study",
        "control",
        "method",
        "group",
        "c",
        "between",
        "or",
        "it",
        "than",
        "analysis",
        "system",
        "sci",
        "two",
        "6",
        "has",
        "h",
        "after",
        "different",
        "n",
        "national",
        "japan",
        "have",
        "cell",
        "time",
        "zhejiang",
        "used",
        "data",
        "these",
    ]
    ps = PorterStemmer()
    result = []
    for s in stopwords:
        result.extend([ps.stem(s)])
    return list(set(stopwords + result))


def main():
    df_master = pd.read_parquet("../test_data/cleaned_pid_to_info_all_v6.parquet")
    df_train_master = pd.read_parquet("../test_data/train_author.parquet")
    df_test_master = pd.read_parquet(
        "../test_data/ind_test_author_filter_public.parquet"
    )

    stopwords = get_stopword()
    dfs_train = []
    for index, row in df_train_master.iterrows():
        dfs_train.append(tf_idf(row, "train", df_master, stopwords))
        print(index)
    df_train = pd.concat(dfs_train)
    print(df_train.shape)
    print(df_train.head())
    df_train.to_parquet("../test_feature/train_tfidf.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        dfs_test.append(tf_idf(row, "test", df_master, stopwords))
        print(index)
    df_test = pd.concat(dfs_test)
    print(df_test.shape)
    print(df_test.head())
    df_test.to_parquet("../test_feature/test_tfidf.parquet", index=False)


if __name__ == "__main__":
    main()
