import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def get_embs_labels(emb_path, labels_path):
    embs = np.load(emb_path)
    with open(labels_path) as file:
        labels = [line.rstrip() for line in file]
    return embs, labels

def norm_sim_matrix(cos_sim):
    g_means = np.mean(cos_sim, axis=1)
    g_var = np.std(cos_sim, axis=1, ddof=1)
    S = np.zeros_like(cos_sim)
    for i in range(len(cos_sim)):
        S[i] = (cos_sim[i] - g_means[i]) / g_var[i]
    return S

def make_sim_matrix(embs):
    sim_matrix = cosine_similarity(embs)
    sim_matrix = norm_sim_matrix(sim_matrix)
    sim_matrix = np.maximum(sim_matrix, sim_matrix.T)
    return sim_matrix

def get_clan_mapping(path="data/Pfam-A.clans.tsv"):
    clans = pd.read_csv(path, sep='\t', header=None, names=['acc', 'clan', 'clan_name', 'prot_name', 'prot_desc'])
    clans['clan'].fillna('NOCLAN', inplace=True)
    clan_mapping = dict(zip(clans['acc'], clans['clan']))
    return clan_mapping

def make_result(sim_matrix, labels, th=-5, only_clans=True, num_pairs_return=100000):
    result = []
    clan_mapping = get_clan_mapping(35)

    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            fam1, fam2 = labels[i], labels[j]
            score = sim_matrix[i, j]
            if score < th:
                continue

            if fam1 not in clan_mapping or fam2 not in clan_mapping: # if we don't know these families
                raise ValueError(f'Unknown family {fam1} or {fam2} -- check clans file')
            clan1, clan2 = clan_mapping[fam1], clan_mapping[fam2]

            if clan1 == 'NOCLAN' or clan2 == 'NOCLAN':
                flag = 2
            else:
                if clan1 == clan2:
                    flag = 1
                else:
                    flag = 0

            if only_clans and flag == 2:
                continue

            result.append((score, fam1, fam2, clan1, clan2, flag))

    print(f'Number of pairs above threshold {th}: {len(result)}, returning top {min(num_pairs_return, len(result))}',)
    result.sort(key=lambda x: -x[0])
    return result[:num_pairs_return]

def make_tp_fp(pairs, fp_th=1000, do_liberal_eval=False):  
    tps, fps = [], []
    true_cnt, false_cnt = 0, 0
    
    for cnt, elem in enumerate(pairs):
        flag = elem[-1]
        if flag == 0 or (do_liberal_eval and flag == 2):
            false_cnt += 1
            fps.append(false_cnt)
            tps.append(true_cnt)
        elif flag == 1:
            true_cnt += 1
            
        if false_cnt == fp_th:
            break
            
    return tps, fps


