# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Metrics used to evaluate our model

"""
import numpy as np
from collections import Counter
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score


def build_allocation_matrix(entity_labels, block_labels):
    num_entities = entity_labels.max() + 1
    num_blocks = block_labels.max() + 1
    allocation_matrix = torch.zeros((num_entities, num_blocks))

    for entity, predicted_block in zip(entity_labels, block_labels):
        allocation_matrix[entity, predicted_block] += 1

    return allocation_matrix


class Metrics:

    # Code taken from the work
    # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
    def cluster_acc(self, Y_pred, Y):
        Y_pred, Y = np.array(Y_pred), np.array(Y)
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[row[i], col[i]] for i in range(row.shape[0])]) * 1.0 / Y_pred.size

    def nmi(self, Y_pred, Y):
        Y_pred, Y = np.array(Y_pred), np.array(Y)
        assert Y_pred.size == Y.size
        return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')

    def accuracy_score(self, Y_pred, Y):
        allocation_matrix = build_allocation_matrix(Y_pred, Y)
        acc = (allocation_matrix / allocation_matrix.sum(axis=1, keepdims=True)).max(axis=1).mean()
        return acc

    def dispersal_score(self, Y_pred, Y):
        allocation_matrix = build_allocation_matrix(Y_pred, Y)
        num_blocks = allocation_matrix.shape[1]
        allocation_matrix = allocation_matrix / allocation_matrix.sum(axis=1, keepdims=True)
        total = allocation_matrix.sum()
        incidences = allocation_matrix.sum(axis=0)
        perfect_fit_rate = 1 / num_blocks
        errors = (incidences / total) - perfect_fit_rate
        max_error = (num_blocks - 1) * 2 * perfect_fit_rate
        return 1 - np.abs(errors).sum() / max_error

    def nmi(self, Y_pred, Y):
        Y_pred, Y = np.array(Y_pred), np.array(Y)
        assert Y_pred.size == Y.size
        return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
