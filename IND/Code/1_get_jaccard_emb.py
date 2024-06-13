#!/usr/bin/python
import os
import random
import warnings
import itertools
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
from scipy.stats import skew, kurtosis
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

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
    try:
        mcd = MinCovDet()
        mcd.fit(emb)
        pred = mcd.mahalanobis(emb)
    except Exception as e:
        pred = np.zeros((len(emb),), dtype="float32")
    return pred


class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, names, orgs, conf):
        temp = set()
        for i, toks in names.iterrows():
            p, a = toks[0], toks[1]
            if p not in self.paper_author:
                self.paper_author[p] = []
            self.paper_author[p].append(a)
            if a not in self.author_paper:
                self.author_paper[a] = []
            self.author_paper[a].append(p)
        temp.clear()

        for i, toks in orgs.iterrows():
            p, a = toks[0], toks[1]
            if p not in self.paper_org:
                self.paper_org[p] = []
            self.paper_org[p].append(a)
            if a not in self.org_paper:
                self.org_paper[a] = []
            self.org_paper[a].append(p)
        temp.clear()

        for i, toks in conf.iterrows():
            p, a = toks[0], toks[1]
            if p not in self.paper_conf:
                self.paper_conf[p] = []
            self.paper_conf[p].append(a)
            if a not in self.conf_paper:
                self.conf_paper[a] = []
            self.conf_paper[a].append(p)
        temp.clear()

        # print("#papers ", len(self.paper_conf))
        # print("#authors", len(self.author_paper))
        # print("#org_words", len(self.org_paper))
        # print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, numwalks, walklength):
        path_list = []
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks
                paper = paper0
                outline = []
                i = 0
                while i < walklength:
                    i = i + 1
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]
                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline.append(paper)
                            # outline += " " + paper

                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)
                        word = words[wordid]
                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            # outline += " " + paper
                            outline.append(paper)
                path_list.append(outline)

        if list(itertools.chain.from_iterable((path_list))) == []:
            path_list = []
            for i in self.paper_author.keys():
                path_list.append([i])
        return path_list


def gen_edge(df, t_name):
    name_list, org_list, conf_list = [], [], []
    for index, row in df.iterrows():
        for a in row["authors"]:
            name = a[0].split(",")
            org = a[1].split(",")
            temp_name_list, temp_org_list = [], []
            for n in name:
                if n != t_name:
                    if n != "NULL":
                        temp_name_list.append(n)
                else:
                    for o in org:
                        if o != "NULL":
                            temp_org_list.append(o)
                    for o in list(set(temp_org_list)):
                        org_list.append([row["paper_id"], o])
            for n in list(set(temp_name_list)):
                name_list.append([row["paper_id"], n])

        row["venue"] = list(set(row["venue"]))
        if len(row["venue"]) == 0:
            conf_list.append([row["paper_id"], "null"])
        else:
            for v in list(set(row["venue"])):
                conf_list.append([row["paper_id"], v])
    df_name = pd.DataFrame(name_list, columns=["paper_id", "key"])
    df_name = df_name.drop_duplicates(["paper_id", "key"])
    df_org = pd.DataFrame(org_list, columns=["paper_id", "key"])
    df_org = df_org.drop_duplicates(["paper_id", "key"])
    df_conf = pd.DataFrame(conf_list, columns=["paper_id", "key"])
    df_conf = df_conf.drop_duplicates(["paper_id", "key"])
    return df_name, df_org, df_conf


def get_ids(row, proc, master):
    author_id = row["id"]
    merge_keys = ["author_id", "paper_id", "label"]
    if proc == "train":
        paper_ids = row["normal_data"].tolist() + row["outliers"].tolist()
        labels = [1] * len(row["normal_data"]) + [0] * len(row["outliers"])
    elif proc == "test":
        paper_ids = row["papers"].tolist()
        labels = [0] * len(paper_ids)
    dfs = pd.DataFrame({"author_id": author_id, "paper_id": paper_ids, "label": labels})
    df = dfs.merge(master, left_on="paper_id", right_on="id", how="left")
    return df, df[merge_keys], merge_keys


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


def calc(df_id, emb, prefix):
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
    author_cos_list = []
    for i, r in df_emb.iterrows():
        author_cos_list.append(cos_sim(author_vec, r.values))
    df_feature[f"{prefix}_cos_author"] = author_cos_list

    pred_mah = calc_mah(df_emb.values)
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
    df_pred[f"{prefix}_pred_mah"] = pred_mah
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


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def jaccard_similarity(list1, list2):
    if not list1 or not list2:
        return 0
    intersection = len(set(list1) & set(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union if union != 0 else 0


def calc_ja(lst):
    lst = list(lst)
    co_score, jaccard_score = [], []
    for i, l_1 in enumerate(lst):
        c, s = 0, 0
        for j, l_2 in enumerate(lst):
            if i != j:
                s += jaccard_similarity(l_1, l_2)
                if len(list(set(l_1) & set(l_2))) > 0:
                    c += 1
        co_score.append(c / len(lst))
        jaccard_score.append(s / len(lst))
    return co_score, jaccard_score


def basic(row, proc, df_master):
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
    dfs = df_ids.merge(df_master, left_on="paper_id", right_on="id", how="left")

    name_list, org_list = [], []
    for index, r in dfs.iterrows():
        tmp_name_list, tmp_org_list = [], []
        for a in r["authors"]:
            name, org = a[0].split(","), a[1].split(",")
            if author_name not in name:
                for n in name:
                    if n != "NULL":
                        tmp_name_list.append(n)
                for o in org:
                    if o != "NULL":
                        tmp_org_list.append(o)
        name_list.append(tmp_name_list)
        org_list.append(tmp_org_list)

    venue_list = [o.tolist() for o in dfs["venue"].tolist()]
    keywords_list = [o.tolist() for o in dfs["keywords"].tolist()]

    co_name_score, jaccard_name_score = calc_ja(name_list)
    co_org_score, jaccard_org_score = calc_ja(org_list)
    co_venue_score, jaccard_venue_score = calc_ja(venue_list)
    co_keywords_score, jaccard_keywords_score = calc_ja(keywords_list)
    dfs["co_name_score"] = co_name_score
    dfs["jaccard_name_score"] = jaccard_name_score
    dfs["co_org_score"] = co_org_score
    dfs["jaccard_org_score"] = jaccard_org_score
    dfs["co_venue_score"] = co_venue_score
    dfs["jaccard_venue_score"] = jaccard_venue_score
    dfs["co_keywords_score"] = co_keywords_score
    dfs["jaccard_keywords_score"] = jaccard_keywords_score
    feature_col = [
        "co_name_score",
        "co_org_score",
        "co_venue_score",
        "co_keywords_score",
        "jaccard_name_score",
        "jaccard_org_score",
        "jaccard_venue_score",
        "jaccard_keywords_score",
    ]
    ss = StandardScaler()
    ss.fit(dfs[feature_col])
    df_feature = calc(df_ids, ss.transform(dfs[feature_col]), "jaccrd")

    dfs = dfs[["author_id", "paper_id", "label"] + feature_col]
    dfs = dfs.set_index(["author_id", "paper_id", "label"])
    dfs = df_feature.join(dfs, how="left")
    dfs = dfs.reset_index()
    print(dfs.head())
    return dfs


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
    print(df_train.shape)
    print(df_train.head())

    df_train.to_parquet("../test_feature/train_jaccard.parquet", index=False)

    dfs_test = []
    for index, row in df_test_master.iterrows():
        dfs_test.append(basic(row, "test", df_master))
        print(index)
    df_test = pd.concat(dfs_test)
    print(df_test.shape)
    print(df_test.head())
    df_test.to_parquet("../test_feature/test_jaccard.parquet", index=False)


if __name__ == "__main__":
    main()
