import argparse
import random
from torch_geometric.utils import remove_self_loops
from utils import add_self_loops
from tqdm import tqdm
from logger import Logger
from dataset import load_dataset
from eval import evaluate, eval_recall_at_k, eval_w_f1
import warnings
warnings.filterwarnings('ignore')
from pvhgl import *

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):

    print(args)
    fix_seed(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # 加载数据集
    # 包含graph和 lable
    dataset = load_dataset(args)
    dataset.label = dataset.label.to(device)

    # 划分数据集
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]

    print(f"训练集长度：{len(split_idx_lst[0]['train'])}")
    print(f"验证集长度：{len(split_idx_lst[0]['valid'])}")
    print(f"测试集长度：{len(split_idx_lst[0]['test'])}")


    n = dataset.graph['num_binodes']  # number of tokens(节点加超边的总数) tensor(12304)
    num_nodes = dataset.graph['num_nodes']  # number of nodes:3604
    e = dataset.graph['H'].shape[1]  # number of hyperedges:8661
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])  # number of class :1956
    d = dataset.graph['node_feat'].shape[1]  # 特征维度

    print(f"dataset {args.dataset} | num token {n} | num node {num_nodes} | num edge {e}| num node feats {d} | num classes {c}")

    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.graph['edge_index_bipart'] = dataset.graph['edge_index_bipart'].to(device)
    dataset.graph['H'] = dataset.graph['H'].to(device)

    model = PVHGL(n, num_nodes, d, args.hidden_channels, c, e, num_layers=args.num_layers, dropout=args.dropout,
                  num_heads=args.num_heads, use_bn=args.use_bn,
                  use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                  return_att=args.return_att).to(device)


    
    criterion = torch.nn.BCELoss()

    # 评估指标
    if args.metric == 'recall':
        eval_func = eval_recall_at_k
    elif args.metric == 'w_f1':
        eval_func = eval_w_f1


    logger = Logger(args.runs, args)
    model.train()
    print('MODEL:', model)

    adjs = []
    adj_bipart, _ = remove_self_loops(dataset.graph['edge_index_bipart'])
    adj_bipart, _ = add_self_loops(adj_bipart, num_nodes=dataset.graph['num_binodes'])
    adjs.append(adj_bipart)
    dataset.graph['adjs'] = adjs  # len2

    for run in tqdm(range(args.runs)):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        fix_seed(args.seed)
        args.seed += 1
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        patience = 10  # 设定早停的耐心值
        counter = 0  # 计数器，记录连续未改进的次数
        best_visit_weights = None

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            if args.return_att:
                out, visit_weights_per_patient,hyperedge_embeddings = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'],args.tau)
                loss = criterion(out[train_idx], dataset.label[train_idx])
            else:
                out = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'],args.tau)
                loss = criterion(out[train_idx], dataset.label[train_idx])

            loss.backward()
            optimizer.step()

            if epoch % args.eval_step == 0:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
                logger.add_result(run, result[:-1])

                if result[1] > best_val:
                    best_val = result[1]
                    counter = 0  # 验证集结果改进时重置计数器
                    if args.save_model:
                        torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}-run{run}.pkl')
                    if args.return_att:

                        best_visit_weights = [
                            w.clone().detach().cpu()
                            for w in visit_weights_per_patient
                        ]

                else:
                    counter += 1  # 未改进，计数器加1

                # 打印准确率
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')

            # 检查早停条件
            if counter >= patience:
                print(f"Early stopping at epoch {epoch:02d}. Best validation accuracy: {100 * best_val:.2f}%")
                break  # 结束当前运行的训练循环

        logger.print_statistics(run)
        # # —— 训练结束后，保存“最优那一次”的 visit_weights ——
        # if args.return_att and best_visit_weights is not None:
        #     os.makedirs(args.model_dir, exist_ok=True)
        #     save_path = os.path.join(
        #         args.model_dir,
        #         f'{args.dataset}-{run}-best_visit_weights.pkl'
        #     )
        #     with open(save_path, 'wb') as f:
        #         dill.dump(best_visit_weights, f)

    results = logger.print_statistics()


if __name__ == "__main__":
    args = argparse.Namespace(
        dataset="mimic_iii_sorted",
        rand_split=True,
        metric="recall",
        topk = 10,
        method="PVHGL",
        lr=4e-3,
        weight_decay=2e-1,
        num_layers= 3,
        hidden_channels = 256,
        num_heads=1,
        tau=0.2,
        use_bn=True,
        use_residual=True,
        use_gumbel= False,
        use_act=True,
        use_jk=True,
        runs= 10,
        epochs= 200,
        dropout= 0.5,
        device=0,
        data_dir="./data/pyg_data/hypergraph_dataset_updated",
        encode="CODEVISIT",
        hefeat="mean",
        feature_noise=1.0,
        seed=2025,
        cpu=False,
        train_prop=0.7,
        valid_prop=0.1,
        eval_step=1,
        save_model=False,
        model_dir="./saved_models/",
        records_length=r"/root/autodl-tmp/project/data/raw_data/mimic_iii_sorted/record_lengths.pkl",
        cooccurrence=r"/root/autodl-tmp/project/data/raw_data/mimic_iii_sorted/dia_coo_matrix.csv",
        feature_dim = 128,
        alpha = 0.2,
        return_att = False
    )
    main(args)

