# Adapted based on this repo: https://github.com/HKUNLP/icl-selective-annotation/tree/main
import os
import json
import argparse
import random
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data_file(path_to_file: str) -> dict:
    with open(path_to_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def save_data_file(path_to_file: str, data):
    with open(path_to_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def fast_votek(vote_stat: dict, select_num: int) -> list:
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)

    bar = tqdm(range(select_num), desc='Perform fast vote-k')
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue

            for one_support in candidates:
                if one_support not in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])

        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        if int(cur_selected_idx) not in selected_indices:
            selected_indices.append(int(cur_selected_idx))
            bar.update(1)

        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def calculate_nearest_neighbours(client_utts, k: int, path_to_knn_file: str, args) -> dict:
    if os.path.exists(path_to_knn_file):
        print("loading nn file...")
        vote_stat = read_data_file(path_to_knn_file)
    else:
        embeddings = calculate_sentence_embeddings(client_utts, args)
        bar = tqdm(range(len(embeddings)), desc=f'Calculate nearest neighbours')
        vote_stat = defaultdict(list)

        for i in range(len(embeddings)):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
            for idx in sorted_indices:
                if idx != i:
                    vote_stat[idx].append(i)
            bar.update(1)
        save_data_file(path_to_knn_file, vote_stat)
    return vote_stat


def calculate_sentence_embeddings(texts_to_encode: List[str], args):
    num = len(texts_to_encode)
    embeddings = []
    emb_model = SentenceTransformer(args.embedding_model)

    # bar = tqdm(range(0, num, 20), desc='Calculate Embeddings')
    for i in range(0, num, 20):
        embeddings += emb_model.encode(texts_to_encode[i:i + 20]).tolist()
        # bar.update(1)

    embeddings = torch.tensor(embeddings)
    if num > 1:
        mean_embeddings = torch.mean(embeddings, 0, True)
        embeddings = embeddings - mean_embeddings  # why need to minus the mean embedding (???)
    return embeddings


def get_client_utts(mi_data: dict, get_therapist_utt: bool, path_to_ids_file: str) -> Tuple[List[str], List[str]]:
    ids, utts = [], []
    if os.path.exists(path_to_ids_file):
        print('loading mapping file...')
        with open(path_to_ids_file, 'r', encoding='utf-8') as file:
            for line in file:
                if len(line.strip().split('\t')) == 2:
                    ids.append(line.strip().split('\t')[0])
                    utts.append(line.strip().split('\t')[1])
    else:
        bar = tqdm(range(len(mi_data.items())), desc='Process client utterances')
        for _id, info in mi_data.items():
            if info['utt_info']['interlocutor'] == 'client':
                ids.append(_id)
                if get_therapist_utt:
                    utts.append(f"Therapist: {info['utt_info']['prev_utt']} ; Client: {info['utt_info']['text']}")
                else:
                    utts.append(info['utt_info']['text'])
            bar.update(1)

        assert len(ids) == len(utts)
        with open(path_to_ids_file, 'w', encoding='utf-8') as file:
            for idx in range(len(ids)):
                file.write(f"{ids[idx]}\t{utts[idx]}\n")
    return ids, utts


def get_selected_candidates(selected_indices: list, ids: list, mi_utts: dict):
    selected_candidates = {}
    bar = tqdm(range(len(selected_indices)), desc='Get selected candidates')
    for idx in selected_indices:
        selected_candidates[ids[idx]] = mi_utts[ids[idx]]
        bar.update(1)
    return selected_candidates


def get_top_candidates():
    args.output_file = f"{args.annotation_size}_{args.selective_annotation_method}.json"
    if not os.path.exists(args.path_2_knn_dir):
        os.makedirs(args.path_2_knn_dir, exist_ok=True)

    mi_utts = read_data_file(args.path_2_input_file)
    client_utts_ids, client_utts = get_client_utts(mi_utts, args.get_therapist_utt_as_context,
                                                   path_to_ids_file=f"{args.path_2_knn_dir}/{args.mapping_file}")
    nearest_neighbours = calculate_nearest_neighbours(client_utts, args.k,
                                                      path_to_knn_file=f"{args.path_2_knn_dir}/{args.knn_file}")

    selected_indices = fast_votek(nearest_neighbours, args.annotation_size)
    selected_candidates = get_selected_candidates(selected_indices, client_utts_ids, mi_utts)

    save_data_file(path_to_file=f"{args.path_2_knn_dir}/{args.output_file}",
                   data=selected_candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=85, type=int)
    parser.add_argument('--path_2_knn_dir', default='../data/knn')
    parser.add_argument('--path_2_input_file', default='../data/AnnoMI-full-utt.json')
    parser.add_argument('--knn_file', type=str, default='nearest_neighbours.json')
    parser.add_argument('--mapping_file', type=str, default='mapping_knn.txt')
    parser.add_argument('--embedding_model', type=str,
                        default='sentence-transformers/paraphrase-mpnet-base-v2')
    parser.add_argument('--selective_annotation_method', default='fast-votek', type=str,
                        choices=['fast-votek', 'votek', 'diversity'])
    parser.add_argument('--prompt_retrieval_method', default='similar', type=str)
    parser.add_argument('--annotation_size', default=300, type=int)
    parser.add_argument('--k', default=150, type=int, help='Number of nearest neighbours.')
    parser.add_argument('--get_therapist_utt_as_context', action='store_true',
                        help='Whether we use the previous therapist utterance to calculate similarity score.')
    # parser.add_argument('--task', type=str, default='get_top', choices=['get_top'])
    args = parser.parse_args()
    set_seed(args.seed)

    get_top_candidates()
