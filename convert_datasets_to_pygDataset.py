
from torch_geometric.data import InMemoryDataset

from load_other_datasets import * 


def save_data_to_pickle(data, p2root = '../data/', file_name = None):
    '''
    if file name not specified, use time stamp.
    '''

    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, args,root = '../data/pyg_data/hypergraph_dataset_updated/', name = None,
                 p2raw = None,
                 train_percent = 0.7,
                 feature_noise = None,
                 transform=None, pre_transform=None):
        '''
        root: 新构建的数据集的路径
        name: 数据集的名字
        p2raw:存储原始数据集的路径 p2raw = 'data/raw_data/'
        train_percent:
        feature_noise: (1.0)
        transformer:
        pre_transformer:
        '''
        
        existing_dataset = ['mimic_iii', 'mimic_iv','mimic_iii_sorted',"mimic_iv_version9_sorted","mimic_iv_version9_sorted_10000"]
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name # 传入的数据集的名字
        
        self.feature_noise = feature_noise

        self._train_percent = train_percent  # 划分数据集的比例？

        #os.path.isdir() 函数的调用，用于检查某个路径是否是一个 目录。
        # 如果 p2raw 是一个有效的目录路径，这个函数会返回 True；如果路径不存在或不是一个目录，则返回 False。
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        # 如果root不是一个目录，就创建一个root的目录出来
        if not osp.isdir(root):# root是存储处理好的数据集的地方
            os.makedirs(root)
            
        self.root = root # data/pyg_data/hypergraph_dataset_updated
        self.feature_dim=args.feature_dim
        self.myraw_dir = osp.join(root, self.name, 'raw') # 处理后的原始路径
        # 处理好的数据集的目录：root + 数据集名称 + processed
        self.myprocessed_dir = osp.join(root, self.name, 'processed')# 完全处理好的数据集的路径？

        # 通过 super() 调用了父类的初始化方法
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"self.data.train_percent 的类型: {type(self.data.train_percent)}")

        self.train_percent = self.data.train_percent



    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            # file_names:sinate_committes_noise_1.0
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features


    def download(self):
        for name in self.raw_file_names:
            # self.myraw_dir:'./data/pyg_data/hypergraph_dataset_updated\\senate-committees-100\\raw'
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print("p2f:",p2f)# 处理好的原始的
                print("p2raw:",self.p2raw)# 原来的原始的
                print("name:",self.name)# 数据集名字

                if self.feature_noise is None:
                    raise ValueError(f'for cornell datasets, feature noise cannot be {self.feature_noise}')

                tmp_name = self.name
                tmp_data = load_cornell_dataset(path = self.p2raw,
                    dataset = tmp_name,#我设置的是为 self.name
                    feature_dim = self.feature_dim,# 100,自己设置的
                    feature_noise = self.feature_noise,
                    train_percent = self._train_percent)
                    
                _ = save_data_to_pickle(tmp_data, 
                                          p2root = self.myraw_dir,
                                          file_name = self.raw_file_names[0])
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        '''
        self.myraw_dir:'./data/pyg_data/hypergraph_dataset_updated\\senate-committees-100\\raw'
        '''
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        # collate 是 InMemoryDataset 提供的静态方法，用于将多个 Data 对象合并为一个 Data 对象，
        # 并生成分块索引 slices
        # 如果只有一个图数据，collate 会直接处理并返回一个元组 (data, slices)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':

    p2root = '../data/pyg_data/hypergraph_dataset_updated/'
    p2raw = '../data/raw_data/'


    for f in ['mimic_iii', ]:# 'house-committees', 'amazon-reviews']:
        for feature_noise in [0.1, 1]:
            dd = dataset_Hypergraph(root = p2root, 
                    name = f,
                    feature_noise = feature_noise,
                    p2raw = p2raw)

            assert dd.data.num_nodes in dd.data.edge_index[0]
            print(dd, dd.data)

