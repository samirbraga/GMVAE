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


# Code taken from the work
# VaDE (Variational Deep Embedding:A Generative Approach to Clustering)

def _index_of(tensor, elem):
    return (tensor == elem).nonzero(as_tuple=True)[0]


def build_allocation_matrix(num_blocks, entity_labels, block_labels):
    indexed_entity_labels = torch.unique(entity_labels)
    num_entities = len(indexed_entity_labels)
    allocation_matrix = torch.zeros((num_entities, num_blocks))

    for entity, predicted_block in zip(entity_labels, block_labels):
        entity_index = _index_of(indexed_entity_labels, entity)
        allocation_matrix[entity_index, predicted_block] += 1

    return allocation_matrix


def blocks_by_entities(num_blocks, entity_labels, block_labels):
    unique_entity_labels = torch.unique(entity_labels)
    block_by_entity = {entity.item(): np.zeros(num_blocks) for entity in unique_entity_labels}
    for entity, predicted_block in zip(entity_labels, block_labels):
        block_by_entity[entity.item()][predicted_block.item()] += 1

    return {entity: np.argmax(allocs) for entity, allocs in block_by_entity.items()}


def cluster_acc(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum([w[row[i], col[i]] for i in range(row.shape[0])]) * 1.0 / Y_pred.size


def nmi(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')


def reachability_score(num_blocks, block_labels, entity_labels):
    allocation_matrix = build_allocation_matrix(num_blocks, entity_labels, block_labels)
    acc = torch.mean(torch.max(allocation_matrix / allocation_matrix.sum(axis=1, keepdims=True), dim=1).values)
    return acc


def blocking_accuracy(true_blocking, pred_blocking):
    total = 0
    rits = 0
    for entity, predicted_block in pred_blocking.items():
        total += 1
        if predicted_block == true_blocking[entity]:
            rits += 1
    return rits / total


def dispersal_score(num_blocks, block_labels, entity_labels):
    allocation_matrix = build_allocation_matrix(num_blocks, entity_labels, block_labels)
    num_blocks = allocation_matrix.shape[1]
    allocation_matrix = allocation_matrix / allocation_matrix.sum(axis=1, keepdims=True)
    total = allocation_matrix.sum()
    incidences = allocation_matrix.sum(axis=0)
    perfect_fit_rate = 1 / num_blocks
    errors = (incidences / total) - perfect_fit_rate
    max_error = (num_blocks - 1) * 2 * perfect_fit_rate
    return 1 - torch.abs(errors).sum() / max_error


def nmi(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
