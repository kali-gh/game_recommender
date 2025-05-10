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

