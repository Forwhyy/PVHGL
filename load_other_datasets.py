#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains functions for loading the following datasets:
        co-authorship: (dblp, cora)
        walmart-trips (From cornell)
        Amazon-reviews
        U.S. House committee
"""

import torch
import os
import pickle
# import ipdb
import sys

import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('../')

def load_LE_dataset(path=None, dataset="ModelNet40", train_percent = 0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))


    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'#[V;E]
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')


    projected_features = torch.FloatTensor(np.array(features.todense()))

    
    # From adjacency matrix to edge_list
    edge_index = edges.T 
#     ipdb.set_trace()
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    # ipdb.set_trace()
    
    # build torch data class
    data = Data(
#             x = projected_features, 
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),#[V|E;E|V]
            y = labels[:num_nodes])

    
    # ipdb.set_trace()
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            



#     ipdb.set_trace()
    
#     # generate train, test, val mask.
    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    
    data.train_percent = train_percent
    data.num_hyperedges = num_he
    
    return data

def load_citation_dataset(path='../hyperGCN/data/', dataset = 'cora', train_percent = 0.025):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            

    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)
    
    return data

def load_yelp_dataset(path='../data/raw_data/yelp_raw_datasets/', dataset = 'yelp', 
        name_dictionary_size = 1000,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding. 
        - city, in one-hot coding. 
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            

    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()
    
    return data

def load_cornell_dataset(path='../data/raw_data/', dataset = 'mimic_iii',
        feature_noise = 0.1,
        feature_dim = 100,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    import dill
    from torch.nn.init import xavier_normal_

    print(f'Loading hypergraph dataset : {dataset}')

    labels_matrix_path = os.path.join(path,dataset,"labels_matrix.pkl")
    new_records_path = os.path.join(path,dataset,"new_records_subjects.pkl")
    voc_path = os.path.join(path,dataset,"continue_all_codes_voc.pkl")

    # 1. 加载标签矩阵、记录数据和代码词典
    with open(labels_matrix_path, 'rb') as f:
        labels_matrix = dill.load(f) 
    with open(new_records_path, 'rb') as f:
        new_records = dill.load(f)  
    with open(voc_path, 'rb') as f:
        vocs = dill.load(f)  

    diag_voc = vocs['diag_voc']  # 诊断代码词典
    med_voc = vocs['med_voc']  # 药物代码词典
    pro_voc = vocs['pro_voc']  # 手术代码词典

    # 计算节点总数（诊断 + 药物 + 手术）
    num_diag_codes = len(diag_voc.word2idx)
    num_med_codes = len(med_voc.word2idx)
    num_pro_codes = len(pro_voc.word2idx)
    num_nodes = num_diag_codes + num_med_codes + num_pro_codes

    labels_tensor = torch.tensor(labels_matrix, dtype=torch.float32)

    num_patients = labels_matrix.shape[0]
    print(f"加载的标签矩阵维度: {labels_matrix.shape} (患者数, 诊断代码数)")

    # 2. 构建特征矩阵
    # 初始化特征矩阵
    # 使用 Xavier 初始化
    features_tensor = torch.empty((num_nodes, feature_dim), dtype=torch.float32)
    xavier_normal_(features_tensor)
    print(f"生成的特征矩阵维度: {features_tensor.shape} (节点数, 特征维度)")

    # 3. 构建超图边
    edge_index = []  # 用于存储患者和代码之间的边，形如（患者住院节点，代码节点）
    he_id = num_nodes  # 超边的 ID，从医疗代码节点编号之后开始

    # 遍历每位患者的记录
    for patient_id, visits in enumerate(new_records):
        for visit in visits:
            # 解构每次住院记录：诊断代码、药物代码和手术代码
            diag_codes, med_codes, pro_codes = visit
            codes = diag_codes + med_codes + pro_codes
            # 构建双向边：超边 -> 代码，节点 -> 超边
            for code in codes:
                edge_index.append([he_id, code])  # 超边到代码节点
                edge_index.append([code, he_id])  # 代码节点到超边
            he_id += 1  # 更新超边 ID

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"生成的边索引维度: {edge_index_tensor.shape} (2, 边数)")

    # 4. 创建 PyTorch Geometric 数据对象
    data = Data(x=features_tensor, edge_index=edge_index_tensor, y=labels_tensor)
    n_x = num_nodes
    num_class = labels_matrix.shape[-1]
    data.n_x = n_x
    # 添加元数据


    data.train_percent = train_percent  # 训练数据占比
    data.num_hyperedges = he_id-num_nodes
    print(f"超边数量：{data.num_hyperedges}")
    print("数据加载完成，生成超图数据结构。")

    return data

if __name__ == '__main__':
    data = load_cornell_dataset(dataset = 'mimic_iii', feature_noise = 0.1)


