
from torch.utils.data import Dataset
import numpy as np
import torch
from datasets.modis_dataset import ModisCloudPatchesDataset


class MaskGenerator:
    # Reference: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        # if input size 128, mask_patch_size 32, model_patch_size 4, ratio 0.6
        # mask shape (32,32) that includes either 0 or 1
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class ModisMaskedPatchesDataset(Dataset): 
    def __init__(self, split: str = "train", mask_channel: int = 0, method='normalize'):
        self.mask_channel = mask_channel
        self.data = ModisCloudPatchesDataset(split=split, method=method)

        # self.mask_generator = MaskGenerator(input_size=128, mask_patch_size=32,model_patch_size=4, mask_ratio=0.6)

    def __getitem__(self, index):
        # C, H, W = x.shape
        x = self.data[index]
        
        # mask = self.mask_generator()
        # mask = mask[None].repeat(self.mask_channel, axis=0) 

        y = torch.clone(x)
        # set the mask channel to 0
        #x[self.mask_channel] = 0
        
        return x, y

    def __len__(self):
        return len(self.data)