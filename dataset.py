from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        row = self.df.iloc[item]

        image_id1 = row[1]
        image_id2 = row[2]
        label = row[3:8].values.astype("float32")
        #print(type(label))
        #print(label)
        sum_ = sum(label)
        label = np.array([s/sum_ for s in label])
        image_path1 = self.images_path+ str(image_id1.strip())+".jpg"
        image_path2 = self.images_path+ str(image_id2.strip())+".jpg"
        image1 = default_loader(image_path1)
        image2 = default_loader(image_path2)
        x1 = self.transform(image1)
        x2 = self.transform(image2)
        return x1, x2, label
