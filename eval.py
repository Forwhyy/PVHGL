import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def eval_recall_at_k(y_true, y_pred, k=10):
    """
    计算多标签分类任务的 R@k (Recall at k)。

    Args:
        y_true (torch.Tensor): 真实标签，形状为 (num_samples, num_classes)，值为 0 或 1。
        y_pred (torch.Tensor): 模型预测的 logits，形状为 (num_samples, num_classes)。
        k (int): 选择概率前 k 大的标签。

    Returns:
        float: 平均 R@k 值。
    """
    
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
   
    recall_list = []
    for true_labels, pred_logits in zip(y_true, y_pred):
        # 获取前 k 个预测标签的索引
        top_k_indices = np.argsort(pred_logits)[-k:] 
        true_positive = np.sum(true_labels[top_k_indices])# 预测对的个数
        possible_positive = np.sum(true_labels)# 总共有多少个类别
        recall = true_positive / possible_positive if possible_positive > 0 else 0.0
        recall_list.append(recall)
    # 返回平均召回率
    return np.mean(recall_list)


def eval_w_f1(y_true, y_pred):

    from sklearn.metrics import f1_score
    import numpy as np

    # Convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    # Sort predictions by descending order of scores
    y_pred_sorted = np.argsort(y_pred, axis=-1)[:, ::-1]  # Sort indices by descending order

    # Create an empty result matrix with the same shape as y_true
    result = np.zeros_like(y_true)

    # Populate the result matrix with top predictions based on the number of true labels
    for i in range(len(result)):
        true_number = np.sum(y_true[i] == 1)  # Number of true labels for this sample
        result[i][y_pred_sorted[i][:true_number]] = 1  # Select top true_number predictions

    # Compute weighted F1 score
    return f1_score(y_true=y_true, y_pred=result, average='weighted', zero_division=0)

def eval_f1(y_true, y_pred, threshold=0.5, average='micro'):
    y_true = y_true.detach().cpu().numpy()
    y_pred = (y_pred.sigmoid().detach().cpu().numpy() > threshold).astype(int)  # 转换为二进制分类结果
    # 计算多标签 F1 分数
    f1 = f1_score(y_true, y_pred, average=average)  # 使用 sklearn 的 F1 计算函数
    return f1


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()
    # Forward pass
    if args.return_att:
         out, _ ,_= model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'], args.tau)
    else:
        out = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'], args.tau)

    # Calculate validation loss (if needed, currently set to 0)
    valid_loss = 0

    if args.metric == "recall":
        # Calculate multi-label recall for train, validation, and test sets
        train_eva = eval_recall_at_k(dataset.label[split_idx['train']], out[split_idx['train']], args.topk)
        valid_eva = eval_recall_at_k(dataset.label[split_idx['valid']], out[split_idx['valid']], args.topk)
        test_eva = eval_recall_at_k(dataset.label[split_idx['test']], out[split_idx['test']], args.topk)
    else:
        # Calculate W-F1 for train, validation, and test sets
        train_eva = eval_w_f1(dataset.label[split_idx['train']], out[split_idx['train']])
        valid_eva = eval_w_f1(dataset.label[split_idx['valid']], out[split_idx['valid']])
        test_eva = eval_w_f1(dataset.label[split_idx['test']], out[split_idx['test']])

    return train_eva, valid_eva, test_eva, valid_loss, out



