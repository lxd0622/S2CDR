import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
import utils
import random
from models import SSBasemodel
import pdb
from dataloader_tgt import read_graph, getUserPosItems
# from dataloader_src import read_graph, getUserPosItems

class Run(object):
    def __init__(self,
                 config
                 ):
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batch_size = self.batchsize_test

        self.max_K = "[10]"
        self.topks = eval(self.max_K)

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.train_path = self.input_root + '/train_users.csv'
        self.test_path = self.input_root + '/test_users_tgt.csv'
        # self.test_path = self.input_root + '/test_users_src.csv'

        self.read_graph = read_graph


    def get_data(self):
        print('========Reading data========')

        UserItemNet, test_data, test_rating, item_map_src_list, mask_matrix = self.read_graph(self.use_cuda, self.train_path, self.test_path, self.batchsize_train)
        adj_mat = UserItemNet.tolil()

        return adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, mask_matrix

    def get_model(self, adj_mat, mask_matrix):
        if self.base_model == 'SS':
            model = SSBasemodel(adj_mat, mask_matrix, self.device)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model
    

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items) 
        ndcg, hit_rate = [], [], [], []
        for k in self.topks:
            ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
            hit_rate.append(utils.HitRate_atK(groundTrue, r, k))
        return { 'ndcg':np.array(ndcg),
                'hit_rate': np.array(hit_rate)}

    
    def eval_mae(self, model, adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, batch_size):
        print('Evaluating:')
        # model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        testDict = test_data
        results = {'precision': np.zeros(len(self.topks)),
                   'recall': np.zeros(len(self.topks)),
                   'ndcg': np.zeros(len(self.topks)),
                   'hit_rate': np.zeros(len(self.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            users_list = []
            rating_list = []
            groundTrue_list = []
            start = time.time()
            total_time = 0
            for batch_users in utils.minibatch(users, test_batch_size=batch_size):
                allPos = getUserPosItems(batch_users, item_map_src_list)
                groundTrue = [testDict[u] for u in batch_users]  #128
                
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.device)

                rating = model.getUsersRating(batch_users)
                rating = torch.from_numpy(rating)
                rating = rating.to(self.device)
                
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1<<10)

                include_index = []
                include_items = []
                for range_i, groundTrue_items in enumerate(groundTrue):
                    all_items_set = set(range(rating.size(1))) 
                    interacted_items_set = set(groundTrue_items)  
                    negative_items = list(all_items_set - interacted_items_set)  

                    negative_samples = random.sample(negative_items, 999)

                    positive_item = random.choice(groundTrue_items)

                    groundTrue[range_i] = [positive_item]

                    include_index.extend([range_i] * 1000)
                    include_items.extend([positive_item] + negative_samples)
                    
                new_rating = torch.full_like(rating, -(1 << 10)) 
                new_rating[include_index, include_items] = rating[include_index, include_items]  
                score, rating_K = torch.topk(new_rating, k=10)
                new_rating = new_rating.cpu().numpy()
                del new_rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            X = zip(rating_list, groundTrue_list)
            pre_results = []
            for x in X:
                pre_results.append(self.test_one_batch(x))

            for result in pre_results:
                results['ndcg'] += result['ndcg']
                results['hit_rate'] += result['hit_rate']
            results['ndcg'] /= float(len(users))
            results['hit_rate'] /= float(len(users))
            return results


    def CDR(self, model, adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, batch_size):
        print('==========S2CDR==========')
        model.train()
        results = self.eval_mae(model, adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, batch_size)
        print(results)

    def main(self):
        device = self.device
        adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, mask_matrix = self.get_data()
        model = self.get_model(adj_mat, mask_matrix)
        batch_size = self.batch_size
        self.CDR(model, adj_mat, test_data, UserItemNet, test_rating, item_map_src_list, batch_size)
