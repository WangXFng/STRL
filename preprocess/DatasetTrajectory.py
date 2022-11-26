from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import transformer.Constants as Constants
import numpy as np
import os
# import networkx as nx

import random
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
import scipy

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

class Dataset(object):
    def __init__(self):
        # users_num = {
        #     'Dataset': 24941,
        #     'Gowalla': 18737,
        #     'Yelp': 30887,
        #     'Yelp2020': 28038
        # }
        # pois_num = {
        #     'Dataset': 28593,
        #     'Gowalla': 32510,
        #     'Yelp': 18995,
        #     'Yelp2020': 15745
        # }

        # self.user_num = users_num.get(Constants.DATASET)
        # self.poi_num = pois_num.get(Constants.DATASET)
        self.user_num = Constants.USER_NUMBER
        self.poi_num = Constants.TYPE_NUMBER
        self.place_coords = self.read_poi_coos()

        self.training_user = self.read_training_data()
        self.tuning_user = self.read_tuning_data()
        self.test_user = self.read_test_data()

        self.user_data, self.user_valid = self.getDataByTxt()
        # self.G = None
        # self.getDataByGraph()

    def getDataByTxt(self):

        user_data = []  # [] for i in range(self.user_num)
        user_valid = []  # [] for i in range(self.user_num)
        for i in range(self.user_num):
            index = np.array(self.training_user[i]) - 1
            user_data.append((self.training_user[i], self.tuning_user[i],
                                   self.place_coords[index, :][:, index].toarray(),
                                   # [i + 1 for j in range(len(self.training_user[i]))],
                                   [i + 1],
                                   ),
                                  )
            valid_input = self.training_user[i].copy()
            valid_input.extend(self.tuning_user[i])
            index2 = np.array(valid_input) - 1
            user_valid.append((valid_input, self.test_user[i],
                                    self.place_coords[index2, :][:, index2].toarray(),
                                    # [i + 1 for j in range(len(valid_input))],
                                    [i + 1],
                                    ), )

        # poi_data = []  # [] for i in range(self.poi_num)
        # for i in range(self.poi_num):
        #     poi_data.append((training_poi[i],  # tuning_poi[i],
        #                           [i + 1],
        #                           ), )
        return user_data, user_valid

    # def getDataByGraph(self):
    #     directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
    #     edge_file = directory_path + "{dataset}_edge_file.txt".format(dataset=Constants.DATASET)
    #     if not os.path.isfile(edge_file):
    #         print('Creating Graph ..')
    #         f = open(edge_file, "w")
    #         for (tra_traj, tes_traj) in zip(self.training_user, self.tuning_user):
    #             user_traj = tra_traj.copy()
    #             user_traj.extend(tes_traj)
    #             for i in range(len(user_traj) - 1):
    #                 lid, lid2 = user_traj[i], user_traj[i + 1]
    #                 node_from, node_to = lid, lid2
    #                 if node_from != node_to:
    #                     f.write("%s %s %s\n" % (node_from, node_to, 1))
    #
    #         f.close()
    #     self.G = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
    #
    # def getWalks(self):
    #     walks = []
    #     all_nodes = list(self.G.nodes)
    #     random.shuffle(all_nodes)
    #     for node in all_nodes:
    #         walk = [node]
    #         while len(walk) < 100:
    #             cur_walk = walk[-1]
    #             cur_neighbor = list(self.G.neighbors(cur_walk))
    #             if len(cur_neighbor) > 0:
    #                 walk.append(random.choice(cur_neighbor))
    #             else:
    #                 break
    #         walks.append(walk)
    #
    #     user_data = []  # [] for i in range(self.user_num)
    #     for i in walks:
    #         len_ = int(len(i)*0.875)
    #         i = np.array(i).astype(np.int)
    #         index = i[:len_] - 1
    #         user_data.append((i[:len_].tolist(), i[len_:].tolist(),
    #                                self.place_coords[index, :][:, index].toarray(),
    #                                [0],
    #                                ),
    #                               )
    #     return user_data
    #     # return walks

    def read_training_data(self):
        user_traj = [[] for i in range(self.user_num)]
        # poi_traj = [[] for i in range(self.poi_num)]
        directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
        train_file = '{dataset}_train.txt'.format(dataset=Constants.DATASET)
        all_train_data = open(directory_path + train_file, 'r').readlines()
        for eachline in all_train_data:
            uid, lid, time = eachline.strip().split()
            if Constants.DATASET == 'Gowalla':
                uid, lid = int(uid), int(lid)
            else:
                uid, lid = int(uid), int(lid)+1
            user_traj[uid].append(lid)
            # poi_traj[lid].append(uid+1)
        # return user_traj, poi_traj
        return user_traj

    def read_tuning_data(self):
        user_traj = [[] for i in range(self.user_num)]
        # poi_traj = [[] for i in range(self.poi_num)]
        directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
        tune_file = '{dataset}_tune.txt'.format(dataset=Constants.DATASET)
        all_tune_data = open(directory_path + tune_file, 'r').readlines()
        for eachline in all_tune_data:
            uid, lid, time = eachline.strip().split()
            if Constants.DATASET == 'Gowalla':
                uid, lid = int(uid), int(lid)
            else:
                uid, lid = int(uid), int(lid)+1
            user_traj[uid].append(lid)
            # poi_traj[lid].append(uid+1)
        # return user_traj, poi_traj
        return user_traj

    def read_test_data(self):
        user_traj = [[] for i in range(self.user_num)]
        # poi_traj = [[] for i in range(self.poi_num)]
        directory_path = './data/{dataset}/'.format(dataset=Constants.DATASET)
        tune_file = '{dataset}_test.txt'.format(dataset=Constants.DATASET)
        all_test_data = open(directory_path + tune_file, 'r').readlines()
        for eachline in all_test_data:
            uid, lid, time = eachline.strip().split()
            if Constants.DATASET == 'Gowalla':
                uid, lid = int(uid), int(lid)
            else:
                uid, lid = int(uid), int(lid)+1
            user_traj[uid].append(lid)
            # poi_traj[lid].append(uid+1)
        # return user_traj, poi_traj
        return user_traj

    def read_poi_coos(self):
        sparse_mx = scipy.sparse.load_npz('./data/{dataset}/place_correlation_gamma60.npz'.format(dataset=Constants.DATASET))
        # sparse_mx = sparse_mx.tocoo().astype(np.float32)
        # indices = torch.from_numpy(
        #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        # values = torch.from_numpy(sparse_mx.data)
        # shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.DoubleTensor(indices, values, shape)
        return sparse_mx  # .todense()
    #
    # def get_data(self):
    #     return self.data

    def paddingLong2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [Constants.PAD] * (max_len - len(inst))
            for inst in insts])
        # print(batch_seq)
        return torch.tensor(batch_seq, dtype=torch.long)

    def padding2D(self, insts):
        """ Pad the instance to the max seq length in batch. """
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst[:max_len] + [Constants.PAD] * (max_len - len(inst))
            for inst in insts])
        # print(batch_seq)
        return torch.tensor(batch_seq, dtype=torch.float32)

    def padding3D(self, insts):  # (16, L)
        """ Pad the instance to the max seq length in batch. """
        # print(insts)
        max_len = max(len(inst) for inst in insts)
        inner_dis = []
        for i, io in enumerate(insts):
            len_ = max_len - len(io)
            pad_width1 = ((0, len_), (0, len_))
            inner_dis.append(
                np.pad(io, pad_width=pad_width1, mode='constant', constant_values=0))  # [:max_len,:max_len]
        return torch.tensor(np.array(inner_dis), dtype=torch.float32)

    def user_fn(self, insts):
        """ Collate function, as required by PyTorch. """
        ds = insts
        # print(ds[0])
        (event_type, test_label, inner_dis, user_type) = list(zip(*ds))  # list(zip(*ds))
        event_type = self.paddingLong2D(event_type)
        # print(event_type)
        test_label = self.paddingLong2D(test_label)
        inner_dis = self.padding3D(inner_dis)
        user_ids = self.paddingLong2D(user_type)
        return event_type, test_label, inner_dis.clone().detach(), user_ids
        # return event_type, test_label, user_ids

    def get_user_dl(self, batch_size):
        d = self.user_data
        # d2 = self.getWalks()
        # d2 = random.sample(d2, int(len(d2)*0.30))
        # d2.extend(d)

        # d = self.getWalks()
        user_dl = torch.utils.data.DataLoader(
            d,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )
        return user_dl

    def get_user_valid_dl(self, batch_size):
        f = self.user_valid
        # fs = ((f[0], ), (f[1], ), (f[2], ), (f[3], ))
        # print(f[0:5])
        # print(1/0)
        user_valid_dl = torch.utils.data.DataLoader(
            f,
            num_workers=0,
            batch_size=batch_size,
            collate_fn=self.user_fn,
            shuffle=True
        )

        return user_valid_dl
