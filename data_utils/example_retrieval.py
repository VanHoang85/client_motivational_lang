# Disclaimer: Adapted from https://github.com/HKUNLP/icl-selective-annotation/blob/main/two_steps.py

import random
import torch
import numpy as np


def example_retrieval(train_embs, example_emb, args):
    selected_indices = []
    if args.example_retrieval_method == 'random':
        while len(selected_indices) < args.num_examples:
            num = random.randint(0, len(train_embs))
            if num not in selected_indices:
                selected_indices.append(num)

    elif args.example_retrieval_method == 'similarity':
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        test_e_reshape = example_emb.reshape(1, -1)
        scores = cos(test_e_reshape, train_embs).numpy()
        sorted_indices = np.argsort(scores)

        num_indices = len(sorted_indices)
        for idx in range(num_indices - 1, -1, -1):
            # check if the two embs are the same
            # here an embedding consists of both therapist and client utts
            # if args.example_retrieval_method == 'similar' and scores[sorted_indices[idx]] == 1:
            #    continue

            if len(selected_indices) < args.num_examples:
                selected_indices.append(int(sorted_indices[idx]))
            else:
                break  # break the loop once we get enough examples
    return selected_indices
