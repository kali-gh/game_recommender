import logging
import pandas as pd
import csv
import time
import torch

from typing import List
from pathlib import Path
from gensim.utils import simple_preprocess
from gensim.models import LsiModel
from gensim import corpora
from gensim.matutils import corpus2csc

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    Defines a dataset for our processing
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_loader(df, x_cols, batch_size, shuffle=True):
    """
    Builds a loader to use for training / validation / test

    Args:
        df: dataframe to pull loader from
    Returns:
        the loader to use
    """

    X = torch.FloatTensor(df[x_cols].values)
    y = torch.FloatTensor(df.rating.values).reshape(-1, 1)

    dataset = CustomDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader