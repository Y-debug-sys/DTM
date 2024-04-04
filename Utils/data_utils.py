import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Utils.random_utils import random_mask, select


def build_dataloader(
    data_root,
    dataset_name,
    flow_known_rate,
    link_known_rate,
    batch_size,
    random_seed,
    train_size=3000,
    test_size=672,
    window=12,
    mode='TME'
):
    if dataset_name == 'abilene':
        scale = 10**9
        tm_filename = 'abilene_tm.csv'
        rm_filename = 'abilene_rm.csv'
    elif dataset_name == 'geant':
        scale = 10**7
        tm_filename = 'geant_tm.csv'
        rm_filename = 'geant_rm.csv'

    tm_filepath = os.path.join(data_root, tm_filename)
    rm_filepath = os.path.join(data_root, rm_filename)

    if mode == 'TME':
        train_dataset = TMEDataset(tm_filepath, rm_filepath, train_size, test_size, period='train', scale=scale,
                                   flow_known_rate=flow_known_rate, link_known_rate=link_known_rate, seed=random_seed,
                                   window=window, exclude_zeros=False)
        test_dataset = TMEDataset(tm_filepath, rm_filepath, train_size, test_size, period='test', scale=scale,
                                  flow_known_rate=flow_known_rate, link_known_rate=link_known_rate, seed=random_seed,
                                  window=window, exclude_zeros=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    else:
        train_dataset = TMEDataset(tm_filepath, rm_filepath, train_size, test_size, period='train', scale=scale,
                                   flow_known_rate=flow_known_rate, link_known_rate=link_known_rate, seed=random_seed,
                                   window=window, exclude_zeros=False)
        if mode == 'TMC':
            test_dataset = TMCDataset(tm_filepath, train_size, period='test', scale=scale, known_rate=flow_known_rate, seed=random_seed)
        else:
            test_dataset = TMDataset(tm_filepath, train_size, period='test', scale=scale, known_rate=flow_known_rate, seed=random_seed)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    rm, rm_pinv = train_dataset.rm, train_dataset.rm_pinv
    return train_loader, test_loader, rm, rm_pinv


class TMEDataset(Dataset):
    def __init__(
        self, 
        tm_filepath, 
        rm_filepath, 
        train_size, 
        test_size,
        window=12,
        period='train',
        scale=10**9,
        flow_known_rate=0.,
        link_known_rate=1.,
        exclude_zeros=False,
        seed=2023
    ):
        super(TMEDataset, self).__init__()
        assert period in ['train', 'test_train', 'test'], ''
        self.period, self.window = period, window

        traffic, self.link, self.traffic_clean, self.rm =\
            self.read_data(tm_filepath, rm_filepath, train_size, test_size, scale, period)

        self.traffic, _, self.traffic_masks =\
            random_mask(traffic.cpu().numpy(), missing_ratio=1-flow_known_rate, seed=seed, exclude_zeros=exclude_zeros)
                
        self.rm_pinv = torch.linalg.pinv(self.rm)

        self.len, self.dim_1 = self.link.shape
        _, self.dim_2 = self.traffic.shape

        select_unob = select(self.dim_1, ratio=1-link_known_rate, seed=seed)
        link_masks = ~np.isnan(self.link.cpu().numpy())
        link_masks[:, select_unob] = False
        self.link_masks = torch.from_numpy(link_masks).float()

        if period == 'train':
            self.sample_num = max(self.len - self.window + 1, 0)
        else:
            assert self.len % self.window==0, ''
            self.sample_num = int(self.len / self.window) if int(self.len / self.window) > 0 else 0

        self.traffic, self.traffic_clean, self.link, self.link_masks, self.traffic_masks =\
              self.__getsamples(self.traffic, self.traffic_clean, self.link, self.link_masks, self.traffic_masks, period)

    def read_data(
        self,
        tm_filepath,
        rm_filepath, 
        train_size, 
        test_size, 
        scale, 
        period='train'
    ):
        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)

        traffic = df.values[:(train_size+test_size)] / scale
        quantile = np.percentile(df.values / scale, q=99)
        traffic[traffic > quantile] = quantile
        traffic = traffic / quantile
        traffic = torch.from_numpy(traffic).float()

        rm_df = pd.read_csv(rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)

        rm = torch.from_numpy(rm_df.values).float()
        link = traffic @ rm

        if period == 'train':
            traffic = traffic[:train_size, :]
            link = link[:train_size, :]
        else:
            traffic = traffic[-test_size:, :]
            link = link[-test_size:, :]

        return traffic, link, traffic.clone(), rm

    def __getitem__(self, ind):
        if self.period == 'train':
            x = self.traffic[ind, :, :]
            y = self.link[ind, :, :]
            m1 = self.traffic_masks[ind, :, :]
            m2 = self.link_masks[ind, :, :]
            return x, y, m1, m2
        x = self.traffic[ind, :, :]
        y = self.link[ind, :, :]
        m = torch.ones_like(y).float()
        return x, y, m

    def __len__(self):
        return self.sample_num
    
    def update(self, traffic):
        if self.period != 'train':
            raise NotImplementedError()
        self.traffic = traffic
    
    def __getsamples(self, data1, data2, data3, mask1, mask2, period):
        x1 = torch.zeros((self.sample_num, self.window, self.dim_2))
        x2 = torch.zeros((self.sample_num, self.window, self.dim_2))
        x3 = torch.zeros((self.sample_num, self.window, self.dim_1))
        m1 = torch.zeros((self.sample_num, self.window, self.dim_1))
        m2 = torch.zeros((self.sample_num, self.window, self.dim_2))

        if period == 'train':
            for i in range(self.sample_num):
                start = i
                end = i + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
                m1[i, :, :] = mask1[start:end, :]
                m2[i, :, :] = mask2[start:end, :]
        else:
            j = 0
            for i in range(0, self.sample_num):
                start = j
                end = j + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
                m1[i, :, :] = mask1[start:end, :]
                m2[i, :, :] = mask2[start:end, :]
                j = end
        
        return x1, x2, x3, m1, m2


class TMCDataset(Dataset):
    def __init__(
        self, 
        tm_filepath, 
        train_size, 
        window=12,
        period='train',
        scale=10**9,
        known_rate=0.,
        seed=2023
    ):
        super(TMCDataset, self).__init__()
        assert period in ['train', 'test_train', 'test'], ''
        self.period, self.window = period, window

        traffic, self.traffic_clean =\
            self.read_data(tm_filepath, train_size, scale)
        
        self.traffic, self.irregular_masks, self.traffic_masks =\
                      random_mask(traffic.cpu().numpy(), missing_ratio=1-known_rate, seed=seed)

        self.len, self.dim = self.traffic.shape

        if period == 'train':
            self.sample_num = max(self.len - self.window + 1, 0)
        else:
            assert self.len % self.window==0, ''
            self.sample_num = int(self.len / self.window) if int(self.len / self.window) > 0 else 0

        self.traffic, self.irregular_masks, self.traffic_masks =\
              self.__getsamples(self.traffic, self.irregular_masks, self.traffic_masks, period)

    def read_data(
        self,
        tm_filepath, 
        train_size, 
        scale
    ):
        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)

        traffic = df.values[:train_size] / scale
        quantile = np.percentile(df.values / scale, q=99)
        traffic[traffic > quantile] = quantile
        traffic = traffic / quantile
        traffic = torch.from_numpy(traffic).float()
        return traffic, traffic.clone()

    def __getitem__(self, ind):
        x = self.traffic[ind, :, :]
        m1 = self.irregular_masks[ind, :, :]
        m2 = self.traffic_masks[ind, :, :]
        return x, m1, m2

    def __len__(self):
        return self.sample_num
    
    def __getsamples(self, data1, data2, data3, period):
        x1 = torch.zeros((self.sample_num, self.window, self.dim))
        x2 = torch.zeros((self.sample_num, self.window, self.dim))
        x3 = torch.zeros((self.sample_num, self.window, self.dim))
        if period == 'train':
            for i in range(self.sample_num):
                start = i
                end = i + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
        else:
            j = 0
            for i in range(0, self.sample_num):
                start = j
                end = j + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
                j = end

        return x1, x2, x3


class TMDataset(Dataset):
    def __init__(
        self, 
        tm_filepath, 
        rm_filepath, 
        train_size, 
        test_size,
        window=12,
        period='train',
        scale=10**9,
        flow_known_rate=0.,
        link_known_rate=1.,
        exclude_zeros=False,
        seed=2023
    ):
        super(TMDataset, self).__init__()
        assert period in ['train', 'test_train', 'test'], ''
        self.period, self.window = period, window

        traffic, self.link, self.traffic_clean, self.rm =\
            self.read_data(tm_filepath, rm_filepath, train_size, test_size, scale, period)

        self.traffic, self.irregular_masks, self.traffic_masks =\
            random_mask(traffic.cpu().numpy(), missing_ratio=1-flow_known_rate, seed=seed, exclude_zeros=exclude_zeros)
                
        self.rm_pinv = torch.linalg.pinv(self.rm)

        self.len, self.dim_1 = self.link.shape
        _, self.dim_2 = self.traffic.shape

        select_unob = select(self.dim_1, ratio=1-link_known_rate, seed=seed)
        link_masks = ~np.isnan(self.link.cpu().numpy())
        link_masks[:, select_unob] = False
        self.link_masks = torch.from_numpy(link_masks).float()

        if period == 'train':
            self.sample_num = max(self.len - self.window + 1, 0)
        else:
            assert self.len % self.window==0, ''
            self.sample_num = int(self.len / self.window) if int(self.len / self.window) > 0 else 0

        self.traffic, self.irregular_masks, self.link, self.link_masks, self.traffic_masks =\
              self.__getsamples(self.traffic, self.irregular_masks, self.link, self.link_masks, self.traffic_masks, period)

    def read_data(
        self,
        tm_filepath,
        rm_filepath, 
        train_size, 
        test_size, 
        scale, 
        period='train'
    ):
        df = pd.read_csv(tm_filepath, header=None)
        df.drop(df.columns[-1], axis=1, inplace=True)

        traffic = df.values[:(train_size+test_size)] / scale
        quantile = np.percentile(df.values / scale, q=99)
        traffic[traffic > quantile] = quantile
        traffic = traffic / quantile
        traffic = torch.from_numpy(traffic).float()

        rm_df = pd.read_csv(rm_filepath, header=None)
        rm_df.drop(rm_df.columns[-1], axis=1, inplace=True)

        rm = torch.from_numpy(rm_df.values).float()
        link = traffic @ rm

        if period == 'train':
            traffic = traffic[:train_size, :]
            link = link[:train_size, :]
        else:
            traffic = traffic[-test_size:, :]
            link = link[-test_size:, :]

        return traffic, link, traffic.clone(), rm

    def __getitem__(self, ind):
        x = self.traffic[ind, :, :]
        y = self.link[ind, :, :]
        m = self.link_masks[ind, :, :]
        m1 = self.irregular_masks[ind, :, :]
        m2 = self.traffic_masks[ind, :, :]
        return x, y, m, m1, m2

    def __len__(self):
        return self.sample_num
    
    def __getsamples(self, data1, data2, data3, mask1, mask2, period):
        x1 = torch.zeros((self.sample_num, self.window, self.dim_2))
        x2 = torch.zeros((self.sample_num, self.window, self.dim_2))
        x3 = torch.zeros((self.sample_num, self.window, self.dim_1))
        m1 = torch.zeros((self.sample_num, self.window, self.dim_1))
        m2 = torch.zeros((self.sample_num, self.window, self.dim_2))
        if period == 'train':
            for i in range(self.sample_num):
                start = i
                end = i + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
                m1[i, :, :] = mask1[start:end, :]
                m2[i, :, :] = mask2[start:end, :]
        else:
            j = 0
            for i in range(0, self.sample_num):
                start = j
                end = j + self.window
                x1[i, :, :] = data1[start:end, :]
                x2[i, :, :] = data2[start:end, :]
                x3[i, :, :] = data3[start:end, :]
                m1[i, :, :] = mask1[start:end, :]
                m2[i, :, :] = mask2[start:end, :]
                j = end
        
        return x1, x2, x3, m1, m2
