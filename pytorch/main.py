"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Main file to execute the model on the MNIST dataset

"""
import matplotlib
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as utils_data
from model.GMVAEBlocking import *

matplotlib.use('agg')


class Args(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

default_args = {
    'seed': 1,

    ## GPU
    'cuda': 0,
    'gpuID': 0,

    ## Training
    'epochs': 100,
    'batch_size': 128,
    'batch_size_val': 200,
    'learning_rate': 1e-3,
    'decay_epoch': -1,
    'lr_decay': 0.5,

    ## Architecture
    'num_classes': 3,
    'gaussian_size': 32,
    'input_size': 784,

    ## Partition parameters
    'train_proportion': 1.0,

    ## Gumbel parameters
    'init_temp': 1.0,
    'decay_temp': 1,
    'hard_gumbel': 0,
    'min_temp': 0.5,
    'decay_temp_rate': 0.013862944,

    ## Loss function parameters
    'w_gauss': 1,
    'w_categ': 1,
    'w_rec': 1,
    'w_blocking': -40,
    'rec_type': 'bce',

    ## Others
    'verbose': 0
}


def get_args(**args):
    args_dics = dict(list(default_args.items()) + list(args.items()))
    return Args(args_dics)


## Random Seed
SEED = default_args['seed']
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if default_args['cuda']:
  torch.cuda.manual_seed(SEED)

BASE_PATH = '/home/samir/Documents/atividades/pml/trabalho-final'

#########################################################
## Read Data
#########################################################
# text_data = pd.read_parquet(BASE_PATH + '/small_sample_wiki_links-use_emb.parquet')
# text_data['full_context_use_emb'] = text_data['full_context_use_emb'].apply(lambda x: x[0]['values'])
# target_map = dict([(wiki_url, index) for index, wiki_url in enumerate(text_data['wiki_url'].unique())])
# text_data['target'] = text_data['wiki_url'].apply(lambda x: target_map[x])
# data_target = torch.tensor(text_data['target'].values.astype(int))
# train = torch.tensor(np.stack(text_data['full_context_use_emb'].values)).float()
# train_dataset = utils_data.dataset.TensorDataset(train, data_target)


#########################################################
## Data Partition
#########################################################
# def partition_dataset(n, proportion=0.8):
#   train_num = int(n * proportion)
#   indices = np.random.permutation(n)
#   train_indices, val_indices = indices[:train_num], indices[train_num:]
#   return train_indices, val_indices

# we use all train dataset without partitioning
# train_indices, val_indices = partition_dataset(len(train_dataset))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size'], shuffle=True)
# val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size_val'], sampler=SubsetRandomSampler(val_indices))

print("Loading mnist dataset...")
# Download or load downloaded MNIST dataset
train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())

def partition_dataset(n, proportion=0.8):
  train_num = int(n * proportion)
  indices = np.random.permutation(n)
  train_indices, val_indices = indices[:train_num], indices[train_num:]
  return train_indices, val_indices

if default_args['train_proportion'] == 1.0:
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size'], shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=default_args['batch_size_val'], shuffle=False)
  val_loader = test_loader
else:
  train_indices, val_indices = partition_dataset(len(train_dataset), default_args['train_proportion'])
  # Create data loaders for train, validation and test datasets
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size'], sampler=SubsetRandomSampler(train_indices))
  val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=default_args['batch_size_val'], sampler=SubsetRandomSampler(val_indices))
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=default_args['batch_size_val'], shuffle=False)


if __name__ == "__main__":
    retrain = False
    w_blocking = 50
    num_classes = 16
    args = get_args(num_classes=num_classes, w_blocking=w_blocking)
    gmvae = GMVAEBlocking(args)
    history_loss = gmvae.train(train_loader, val_loader)
    reachy, dispersal, accuracy, predicted_labels = gmvae.test(train_loader)
    latent_features, true_labels = gmvae.latent_features(train_loader, True)

    print(accuracy)
