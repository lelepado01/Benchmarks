
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms

# path = "/lustre/orion/cli900/world-shared/users/kurihana/C6.1/L1/"
# norm_path = "/lustre/orion/cli900/proj-shared/common"

class ModisCloudPatchesDataset(Dataset):
    def __init__(self, split: str = "train", sensor: str = "aqua", method : str = 'normalize'):
        if sensor not in ["aqua", "terra"]:
            raise ValueError(f"Invalid sensor: {sensor}")
        if method not in ["standardize", "normalize"]:
            raise ValueError(f"Invalid method: {method}")

        self.sensor = sensor
        self.method = method

        patches_path = "/lustre/orion/cli900/proj-shared/train_cloud_npypatches_10M/"
        statsdir = "/lustre/orion/cli900/proj-shared/common"

        self.data = [patches_path + p for p in os.listdir(patches_path) if sensor in p]

        if split == "train":
            self.data = self.data[:int(len(self.data) * 0.8)]
        elif split == "val":
            self.data = self.data[int(len(self.data) * 0.8):int(len(self.data) * 0.9)]
        elif split == "test":
            self.data = self.data[int(len(self.data) * 0.8):]
        else: 
            raise ValueError(f"Invalid split: {split}")

        mean_terra_file = os.path.join(statsdir, "mod02_ocean_band28_29_31_gmean.npy")
        std_terra_file = os.path.join(statsdir, "mod02_ocean_band28_29_31_gstdv.npy")
        mean_aqua_file = os.path.join(statsdir, "myd02_ocean_band28_29_31_gmean.npy")
        std_aqua_file = os.path.join(statsdir, "myd02_ocean_band28_29_31_gstdv.npy")

        self.terra_mean = np.load(mean_terra_file)
        self.terra_std = np.load(std_terra_file)
        self.aqua_mean = np.load(mean_aqua_file)
        self.aqua_std = np.load(std_aqua_file)

    def _transform(self, filename, mean, std, method, nsigma=2): 
        """ function to apply different data processing
            filename (str): npy file 
            Options:
                standardize: mean 0, std 1 
                normalize: [0,1] range. | Value | <= 2 * signma is boundary 0 and 1 
        """
        try:
            data = torch.tensor(np.load(filename))
        except ValueError as e:
            print(f" Train/Test fileformat should be NPY : {e}", flush=True)
    
        if method == 'standardize':
            # channel last for 1d mean and std operation
            sdata = (data - mean) / std
            # fill na with 0 i.e. mean value
            nan_idx = np.where(np.isnan(sdata))
            sdata[nan_idx] = 0
            # swith back to channel first
            return sdata.permute(2, 0, 1)
        
        elif method == 'normalize':
            # channel last for 1d mean and std operation
            ulim = mean + nsigma * std
            sdata = data / ulim
            # fill na with 0 i.e. mean value
            nan_idx = np.where(np.isnan(sdata))
            sdata[nan_idx] = 0
            # fill 1 where pixel values over n-sigma
            upper_index = np.where(sdata > 1.000)
            sdata[upper_index] = 1.000
            # just in case remove negative values
            zero_index = np.where(sdata < 0.000)
            if len(zero_index[0]) > 0:
                sdata[zero_index] = 0.00

            # swith back to channel first
            return sdata.permute(2, 0, 1)


    def __getitem__(self, index):
        x = self.data[index]

        mean = self.terra_mean if self.sensor == 'terra'  else self.aqua_mean
        std  = self.terra_std  if self.sensor == 'terra' else self.aqua_std
        data = self._transform(x, mean, std, method=self.method).float()
        return data

    def __len__(self):
        return len(self.data)
