import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pdb


def read_graph(use_cuda, train_path, test_path, batchsize):
        train_data = pd.read_csv(train_path, header=None)
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser, test_testUser, test_testItem = [], [], [], [], []
        n_user = 0
        m_item = 0
        seen_users = set()
        seen_items = set()
        uid_map = {}
        item_map = {}
        item_map_src = {}
        item_map_src_list = []
        item_map_tgt = {}
        item_map_tgt_list = []
        cols = ['uid', 'pos_seq_src_item', 'pos_seq_src_y', 'pos_seq_tgt_item', 'pos_seq_tgt_y']
        train_data.columns = cols
        for _, row in train_data.iterrows():
            uid = str(row['uid'])
            pos_seq_src_item = list(map(str, eval(row['pos_seq_src_item'])))
            pos_seq_tgt_item = list(map(str, eval(row['pos_seq_tgt_item'])))

            if uid not in uid_map:
                uid_map[uid] = len(uid_map)
            mapped_uid = uid_map[uid]
            
            mapped_src_items = []
            for item in pos_seq_src_item:
                if item not in item_map:
                    item_map[item] = len(item_map)
                    item_map_src[item] = len(item_map) - 1
                    item_map_src_list.append(item_map_src[item])
                mapped_src_items.append(item_map[item])

            mapped_tgt_items = []
            for item in pos_seq_tgt_item:
                if item not in item_map:
                    item_map[item] = len(item_map)
                    item_map_tgt[item] = len(item_map) - 1
                    item_map_tgt_list.append(item_map_tgt[item])
                mapped_tgt_items.append(item_map[item])

            pos_seq_src_item_str = " ".join(map(str, mapped_src_items))
            pos_seq_tgt_item_str = " ".join(map(str, mapped_tgt_items))
            new_row = f"{mapped_uid} {pos_seq_src_item_str} {pos_seq_tgt_item_str}".split(' ')
            items = [int(i) for i in new_row[1:] if i.strip()]
            mapped_uid = int(new_row[0])
            trainUniqueUsers.append(mapped_uid)
            trainUser.extend([mapped_uid] * len(items))
            trainItem.extend(items)
        trainUniqueUsers = np.array(trainUniqueUsers) 
        trainUser = np.array(trainUser)  
        trainItem = np.array(trainItem)  

        test_data = pd.read_csv(test_path, header=None)
        test_data.columns = cols
        test_rating = list()
        for _, row in test_data.iterrows():
            uid = str(row['uid'])
            pos_seq_src_item = list(map(str, eval(row['pos_seq_src_item'])))
            pos_seq_tgt_item = list(map(str, eval(row['pos_seq_tgt_item'])))
            pos_seq_tgt_y = eval(row['pos_seq_tgt_y'])
            pos_seq_tgt_y = [int(float(item)) for item in pos_seq_tgt_y]
    
            if uid not in uid_map:
                uid_map[uid] = len(uid_map)
            mapped_uid = uid_map[uid]

            mapped_src_items = []
            for item in pos_seq_src_item:
                if item not in item_map:
                    item_map[item] = len(item_map)
                    item_map_src[item] = len(item_map) - 1
                    item_map_src_list.append(item_map_src[item])
                mapped_src_items.append(item_map[item])
            
            mapped_tgt_items = []
            for item in pos_seq_tgt_item:
                if item not in item_map:
                    item_map[item] = len(item_map)
                    item_map_tgt[item] = len(item_map) - 1
                    item_map_tgt_list.append(item_map_tgt[item])
                mapped_tgt_items.append(item_map[item])
            
            pos_seq_src_item_str = " ".join(map(str, mapped_src_items))
            pos_seq_tgt_item_str = " ".join(map(str, mapped_tgt_items))
            new_row = f"{mapped_uid} {pos_seq_src_item_str}".split(' ')
            test_new_row = f"{mapped_uid} {pos_seq_tgt_item_str}".split(' ')
            items = [int(i) for i in new_row[1:] if i.strip()]
            test_items = [int(i) for i in test_new_row[1:] if i.strip()]
            mapped_uid = int(new_row[0])
            testUniqueUsers.append(mapped_uid)
            testUser.extend([mapped_uid] * len(items))
            testItem.extend(items)
            test_testUser.extend([mapped_uid] * len(test_items))
            test_testItem.extend(test_items)
            pos_seq_tgt_y = [pos_seq_tgt_y]
            test_rating.extend(pos_seq_tgt_y)
        testUser = np.array(testUser)  
        testItem = np.array(testItem)  
        item_map_src_list = np.array(item_map_src_list)
        item_map_tgt_list = np.array(item_map_tgt_list)
        test_testUser = np.array(test_testUser)  
        test_testItem = np.array(test_testItem)  
        all_users = np.concatenate([trainUser, testUser])  
        all_items = np.concatenate([trainItem, testItem])  
        Graph = None
        UserItemNet = csr_matrix((np.ones(len(all_users)), (all_users, all_items)), shape=(len(uid_map), len(item_map)))
        test_data = {}  
        for i, item in enumerate(test_testItem):
            user = test_testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        
        mask_matrix_src = UserItemNet.copy().tolil()
        mask_matrix_src[:, item_map_src_list] = 0  
        mask_matrix_src = mask_matrix_src.tocsr()
        mask_matrix_src = mask_matrix_src.tolil()

        mask_matrix_tgt = UserItemNet.copy().tolil()
        mask_matrix_tgt[:, item_map_tgt_list] = 0  
        mask_matrix_tgt = mask_matrix_tgt.tocsr()
        mask_matrix_tgt = mask_matrix_tgt.tolil()
        return UserItemNet, test_data, test_rating, item_map_src_list, mask_matrix_src

def getUserPosItems(users, item_map_src_list):
    posItems = []
    for user in users:
        posItems.append(item_map_src_list)

    return posItems