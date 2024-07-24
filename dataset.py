import pandas as pd
import numpy as np
import torch
import scanpy as sc
import scipy.sparse as sp
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from graph_construction import calcADJ



class MyDatasetTrans(Dataset):
    """Operations with the datasets."""

    def __init__(self, normed_data, coor_df, image, transform=None):
        """
        Args:
            normed_data: Normalized data extracted from original AnnData object.
            coor_df: Spatial location extracted from original AnnData object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = normed_data.values.T
        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.exp = np.array_split(self.data, np.ceil(len(self.data) / 50))
        self.image_feature = np.array_split(self.image, np.ceil(len(self.image) / 50))
        self.adj = [calcADJ(coord=i, k=4, pruneTag='NA') for i in self.coord]
    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        exp = torch.tensor(self.exp[idx])
        coord = torch.tensor(self.coord[idx])
        image = torch.tensor(self.image_feature[idx])
        adj = self.adj[idx]
        sample = (exp, coord, image)

        return sample

class MyDatasetTrans2(Dataset):
    """Operations with the datasets."""

    def __init__(self, coor_df, image, transform=None):
        """
        Args:
            normed_data: Normalized data extracted from original AnnData object.
            coor_df: Spatial location extracted from original AnnData object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.image_feature = np.array_split(self.image, np.ceil(len(self.image) / 50))

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        coord = torch.tensor(self.coord[idx])
        image = torch.tensor(self.image_feature[idx])

        sample = (coord, image)

        return sample