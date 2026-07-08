import pandas as pd
import numpy as np
import gzip
import json
import tqdm
import random
import os
import pdb
from ordered_set import OrderedSet

class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

class DataPreprocessingReady():
    def __init__(self,
                 root,
                 src_tgt_pairs,
                 task,
                 ratio):
        self.root = root
        self.src = src_tgt_pairs[task]['src']
        self.tgt = src_tgt_pairs[task]['tgt']
        self.ratio = ratio

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt

    def get_src(self, data, uid_set):
        pos_seq_dict_1 = {}
        pos_seq_dict_2 = {}
        for uid in tqdm.tqdm(uid_set):
            pos_1 = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_2 = data[(data.uid == uid) & (data.y > 3)].y.values.tolist()
            pos_seq_dict_1[uid] = pos_1
            pos_seq_dict_2[uid] = pos_2
        return pos_seq_dict_1, pos_seq_dict_2

    def get_tgt(self, data, uid_set):
        pos_seq_dict_1 = {}
        pos_seq_dict_2 = {}
        for uid in tqdm.tqdm(uid_set):
            pos_1 = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_2 = data[(data.uid == uid) & (data.y > 3)].y.values.tolist()
            pos_seq_dict_1[uid] = pos_1
            pos_seq_dict_2[uid] = pos_2
        return pos_seq_dict_1, pos_seq_dict_2

    def split(self, src, tgt):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = OrderedSet(src.uid.unique())
        tgt_users = OrderedSet(tgt.uid.unique())
        src_user_interactions = src.groupby('uid').size()
        tgt_user_interactions = tgt.groupby('uid').size()
        filtered_src_users = [uid for uid in src_users if src_user_interactions[uid] >= 5]
        filtered_tgt_users = [uid for uid in tgt_users if tgt_user_interactions[uid] >= 5]
        co_users = OrderedSet(filtered_src_users) & OrderedSet(filtered_tgt_users)
        test_users = OrderedSet(random.sample(co_users, round(self.ratio[1] * len(co_users))))
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
        pos_seq_dict_1 = self.get_src(src, co_users)[0]
        pos_seq_dict_2 = self.get_src(src, co_users)[1]
        pos_seq_dict_3 = self.get_tgt(tgt, co_users)[0]
        pos_seq_dict_4 = self.get_tgt(tgt, co_users)[1]
        train_users = co_users - test_users
        train_users = pd.DataFrame({'uid': train_users})
        test_users = pd.DataFrame({'uid': test_users})
        train_users['pos_seq_src_item'] = train_users['uid'].map(pos_seq_dict_1)
        train_users['pos_seq_src_y'] = train_users['uid'].map(pos_seq_dict_2)
        train_users['pos_seq_tgt_item'] = train_users['uid'].map(pos_seq_dict_3)
        train_users['pos_seq_tgt_y'] = train_users['uid'].map(pos_seq_dict_4)
        test_users['pos_seq_src_item'] = test_users['uid'].map(pos_seq_dict_1)
        test_users['pos_seq_src_y'] = test_users['uid'].map(pos_seq_dict_2)
        test_users['pos_seq_tgt_item'] = test_users['uid'].map(pos_seq_dict_3)
        test_users['pos_seq_tgt_y'] = test_users['uid'].map(pos_seq_dict_4)
        return train_src, train_tgt, train_users, test_users

    def save(self, train_src, train_tgt, train_users, test_users):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_users.to_csv(output_root +  '/train_users.csv', sep=',', header=None, index=False)
        test_users.to_csv(output_root + '/test_users.csv', sep=',', header=None, index=False)

    def main(self):
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt = self.mapper(src, tgt)
        train_src, train_tgt, train_users, test_users = self.split(src, tgt)
        self.save(train_src, train_tgt, train_users, test_users)
