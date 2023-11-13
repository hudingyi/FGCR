#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2023/06
# author:Dingyi Hu
# emai:hudingyi@buaa.edu.cn

import os
import pickle
import numpy as np

import torch
import torch.utils.data as data


class KernelWSILoader(torch.utils.data.Dataset):
    def __init__(self, list_path, max_node_number, max_kernel_num=25):
        with open(list_path, 'rb') as f:
            data = pickle.load(f)
        self. dl = data['list']
        self.list_dir = data['base_dir']
        self.maxno = max_node_number
        with open(self.get_wsi_data_path(0), 'rb') as f:
            wsi_data = pickle.load(f)
        self.nk = max_kernel_num
        self.feat_dim = wsi_data['feats'].shape[1]


        
    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        with open(self.get_wsi_data_path(idx), 'rb') as f:
            wsi_data = pickle.load(f)

        num_node = min(wsi_data['feats'].shape[0], self.maxno)
        features = wsi_data['feats'][:num_node]

        if np.sum(wsi_data['knumber'])>self.nk:
            kernel_level_percent = self.nk/np.sum(wsi_data['knumber'])*np.array(wsi_data['knumber'])
            kernel_lel_num = np.ceil(kernel_level_percent).astype(int)
            if kernel_lel_num[0] > np.sum(kernel_lel_num)-self.nk:
                kernel_lel_num[0] -= np.sum(kernel_lel_num)-self.nk
        else:
            kernel_lel_num = wsi_data['knumber']
        k_idx = [wsi_data['k_idx'][i][0:kernel_lel_num[i]] for i in range(len(kernel_lel_num))]
        rd_k = [wsi_data['rd'][kanchor_idx,:num_node] for kanchor_idx in k_idx]
        wsi_rd = []
        pos_radius = []
        for i in range(len(rd_k)):
            rd_2 = rd_k[i]*rd_k[i]
            wsi_rd.append(np.exp(-rd_2 / (2*wsi_data['npks'][i])))
            pos_radius.append(kernel_lel_num[i]*[wsi_data['npks'][i]])
        rd = np.concatenate(wsi_rd, axis=0)
        pos = [wsi_data['pos'][kanchor_idx] for kanchor_idx in k_idx]
        pos = np.concatenate(pos, axis=0)
        pos_radius = np.concatenate(pos_radius, axis=0)
        pos = np.hstack((pos,pos_radius.reshape(-1,1)))
    
        all_pos = wsi_data['pos'][:num_node]

        wsi_label = int(self.dl[idx][1])
        text_index = wsi_data['text']
        promote = wsi_data['prompt']
        data = self.pack_data(features, rd, num_node, text_index, promote, pos, all_pos, wsi_label)    

        return data, wsi_label, self.get_wsi_data_path(idx).split('/')[-1][:-4]



    def pack_data(self, feat, rd, num_node, text_feat, promote, pos, all_pos, wsi_label, text_length = 150):
        num_anchor = rd.shape[0]

        wsi_feat = np.zeros((self.maxno, self.feat_dim))
        wsi_rd = np.zeros((self.nk, self.maxno))
        text = np.zeros((text_length,1), int)
        pm = np.zeros((text_length,1), int)
        pos_out = np.zeros((self.nk, 3))
        all_pos_out = np.zeros((self.maxno, 2))

        text_node = len(text_feat)
        pm_node = len(promote)
        wsi_feat[:num_node] = np.squeeze(feat)
        wsi_rd[:num_anchor, :num_node] = rd
        text[:text_node] = np.squeeze(text_feat).reshape((text_node,1))
        pm[:pm_node] = np.squeeze(promote).reshape((pm_node,1))
        pos_out[:num_anchor] = pos
        all_pos_out[:num_node] = all_pos

        token_mask = np.zeros((self.maxno, 1), int)
        token_mask[:num_node] = 1
        kernel_mask = np.zeros((self.nk, 1), int)
        kernel_mask[:num_anchor] = 1
        text_mask = np.zeros((text_length,1), int)
        text_mask[:text_node] = 1
        pm_mask = np.zeros((text_length,1), int)
        pm_mask[:pm_node] = 1

        return wsi_feat, wsi_rd, text, pm, token_mask, kernel_mask, text_mask, pm_mask, pos_out, all_pos_out, wsi_label



    def get_wsi_data_path(self, idx):
        return os.path.join(self.list_dir, self.dl[idx][0])

    def get_feat_dim(self):
        return self.feat_dim
        
    def get_weights(self):
        labels = np.asarray([path[1] for path in self.dl])
        labels -= np.min(labels)
        tmp = np.bincount(labels)
        weights = 1 / np.asarray(tmp[labels], np.float)
        
        return weights

class DistributedWeightedSampler(data.DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):

        super(DistributedWeightedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


