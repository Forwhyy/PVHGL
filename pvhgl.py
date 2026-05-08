import math
import os
import dill
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
BIG_CONSTANT = 1e8
import pandas as pd

def one_step_message_passing(args, query, key, value, tau=0.25, return_att=True):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query = query.squeeze(0)  # (N,H,M)
    key = key.squeeze(0)  # (N,H,M)
    value = value.squeeze(0)
    # 将节点数 (N) 和特征维度 (M) 的位置调整
    query = query.permute(1, 0, 2)  # (H, N, M)
    key = key.permute(1, 0, 2)  # (H, N, M)
    value = value.permute(1, 0, 2)
    key_transpose = key.transpose(-1, -2)  # (H, N, M) -> (H, M, N)
    # 计算注意力分数
    attention_scores = torch.matmul(query, key_transpose)
    # 对注意力分数进行缩放 (Scale)
    M = query.size(-1)  # 特征维度
    attention_scores = attention_scores / math.sqrt(M)  # 缩放

    with open(args.cooccurrence, "rb") as f:
        cooccurrence_matrix = pd.read_csv(f)
    cooccurrence_matrix = cooccurrence_matrix.to_numpy().astype(np.float64)
    # 归一化（针对 NumPy 数组）
    row_sums = cooccurrence_matrix.sum(axis=1, keepdims=True)  
    row_sums += 1e-8  
    cooccurrence_matrix = cooccurrence_matrix / row_sums  # 每行和为1
    cooccurrence_matrix = torch.tensor(cooccurrence_matrix, dtype=torch.float32)
    cooccurrence_matrix = cooccurrence_matrix.unsqueeze(0)
    # 提取 Attention Scores 中节点与节点的部分
    num_nodes = cooccurrence_matrix.size(1)  # 节点数
    node_attention_scores = attention_scores[:, :num_nodes, :num_nodes]  
    visit_attention_scores = torch.softmax(attention_scores[:, num_nodes:, num_nodes:],dim=-1) 
    # 归一化
    node_attention_scores = torch.softmax(node_attention_scores, dim=-1)  # 归一化到 [0, 1]
    # 将共现矩阵与注意力分数相加
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")  
    cooccurrence_matrix = cooccurrence_matrix.to(device)
    weighted_node_attention = node_attention_scores + args.alpha * cooccurrence_matrix 
    # 提取其他部分
    other_parts = attention_scores[:, num_nodes:, :]  
    # 拼接新张量
    new_attention_scores = torch.cat([
        torch.cat([weighted_node_attention, attention_scores[:, :num_nodes, num_nodes:]], dim=-1),
        other_parts
    ], dim=-2)

    # 对全局 Attention Scores 做归一化 (Softmax)
    attention_weights = torch.softmax(new_attention_scores, dim=-1)

    z_output = torch.matmul(attention_weights, value)  # (1,12337,256)
    z_output = z_output.permute(1, 0, 2)  # (12337,1,256)
    z_output = z_output.unsqueeze(0)  # (1,12337,1,256)

    if return_att:
        return z_output, torch.softmax(weighted_node_attention, dim=-1)
    else:
        return z_output

class Hypergraph_Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, projection_matrix_type='a',
                 return_att=True):

        super(Hypergraph_Transformer, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)# output层
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.projection_matrix_type = projection_matrix_type
        self.return_att = return_att

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wk.weight)  # Xavier 初始化
        nn.init.zeros_(self.Wk.bias)  # 偏置初始化为 0
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.zeros_(self.Wq.bias)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.zeros_(self.Wv.bias)
        nn.init.xavier_uniform_(self.Wo.weight)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, args, z, adjs, tau):
        # 获取 batch和节点数
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)

        if self.return_att:
            z_next, att = one_step_message_passing(args, query, key, value, tau, self.return_att)
        else:
            z_next = one_step_message_passing(args, query, key, value, tau, self.return_att)

        # 聚合多个头的结果
        z_next = self.Wo(z_next.flatten(-2, -1))

        if self.return_att: # 返回权重
            return z_next, att
        else:
            return z_next


class EncodeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.sparse.mm(input, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output

class DualDynamicEncoder(nn.Module):
    def __init__(self, num_nodes, num_edges, hidden_channels):
        super().__init__()
        # ---- 超图的节点结构编码 ----
        self.node_encoder = EncodeLinear(num_edges, hidden_channels)
        self.node_gate = EncodeLinear(num_edges, 1)

        # ---- 超边编码通路 ----
        self.edge_encoder = EncodeLinear(num_nodes, hidden_channels)
        self.edge_gate = EncodeLinear(num_nodes, 1)


    def forward(self, H,zx,zy):
        # 节点特征编码 --------------------------------------------------
        node_feat = self.node_encoder(H)  # (N, D)
        node_gate = torch.sigmoid(self.node_gate(H))  # (N, 1)
        # node_gate = torch.softmax(self.node_gate(H),dim=0)  # (N, 1)
        node_feat = node_gate * node_feat + (1 - node_gate) * zx  # (N, D)
        # 超边特征编码 --------------------------------------------------
        HT = H.transpose(0,1)  # (E, N)
        edge_feat = self.edge_encoder(HT)  # (E, D)
        edge_gate = torch.sigmoid(self.edge_gate(HT))  # (E, 1)
        # edge_gate = torch.softmax(self.edge_gate(HT),dim=-1)  # (E, 1)
        edge_feat = edge_gate * edge_feat + (1 - edge_gate) * zy  # (E, D)
        return node_feat,  edge_feat

class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))  # 权重矩阵
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))  # 偏置

    def forward(self, A_hat, X):
        # 图卷积计算（稀疏矩阵乘法）
        return torch.sparse.mm(A_hat, torch.mm(X, self.weight)) + self.bias

class VisitAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, visits):
        # visits: (num_visits, hidden_dim)
        attn_scores = torch.matmul(visits, self.query) / torch.sqrt(torch.tensor(visits.size(-1)))
        attn_weights = torch.softmax(attn_scores, dim=0)
        patient_embed = torch.sum(visits * attn_weights.unsqueeze(-1), dim=0)
        return patient_embed,attn_weights

class PVHGL(nn.Module):
    def __init__(self,num_tokens, num_nodes, in_channels, hidden_channels, out_channels, num_hes, num_layers=2, num_heads=4, dropout=0.0,
                 use_bn=True,
                 use_residual=True, use_act=False, use_jk=False,  return_att=True):
        super(PVHGL, self).__init__()
        self.convs = nn.ModuleList()
        # self.gcnlayer = GraphConvolutionLayer(hidden_channels, hidden_channels)
        self.fcs = nn.ModuleList()
        self.classfier=nn.Softmax(dim=-1)
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                Hypergraph_Transformer(hidden_channels, hidden_channels, num_heads=num_heads,
                                       return_att=return_att))
            self.bns.append(nn.LayerNorm(hidden_channels))

        if use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels))
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.n = num_tokens
        self.dual_encoder = DualDynamicEncoder(
            num_nodes=num_nodes,
            num_edges=num_hes,
            hidden_channels=hidden_channels
        )
        self.return_att = return_att

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'weight') and bn.weight is not None:
                nn.init.ones_(bn.weight) 
            if hasattr(bn, 'bias') and bn.bias is not None:
                nn.init.zeros_(bn.bias)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

        # dual_encoder 的 reset
        self.dual_encoder.node_encoder.reset_parameters()
        self.dual_encoder.node_gate.reset_parameters()
        self.dual_encoder.edge_encoder.reset_parameters()
        self.dual_encoder.edge_gate.reset_parameters()

    def getadj(self, H):
        # 从 H 中提取非零元素的索引
        m, n = H.shape
        H_indices = H.coalesce()._indices()  # shape: [2, nnz]
        # H_indices[0] 是节点索引，H_indices[1] 是对应的超边索引
        # 构造扩展图边的索引：
        # 对于每个非零元素 H[i, j] = 1，我们添加两条边：
        # 1. 从节点 i 到超边 (m + j)
        # 2. 从超边 (m + j) 到节点 i
        edge_from = torch.stack([H_indices[0], H_indices[1] + m], dim=0)
        edge_to = torch.stack([H_indices[1] + m, H_indices[0]], dim=0)
        # 将两个方向的边合并
        edge_indices = torch.cat([edge_from, edge_to], dim=1)

        # 如果需要，可以为每条边设置权重，通常这里都是 1.0
        edge_values = torch.ones(edge_indices.shape[1], dtype=torch.float).to(H.device)

        # 构造扩展图的邻接矩阵 A'
        A_prime = torch.sparse_coo_tensor(edge_indices, edge_values, (m + n, m + n))
        return A_prime

    def nomadj(self, args, A_prime):
        # 计算度矩阵 D
        D = torch.sparse.sum(A_prime, dim=1).to_dense()  # 每个节点的度数（度矩阵 D）
        num = A_prime.shape[0]
        # 计算 D^(-1/2) 对角矩阵
        # 使用稀疏矩阵计算归一化的邻接矩阵 A'：A_hat = D^(-1/2) * A_prime * D^(-1/2)
        # 这里要避免转成密集矩阵，因此直接使用稀疏矩阵相乘
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")  # 确认设备
        # 转换度矩阵 D_inv_sqrt 为稀疏对角矩阵形式（保持计算的稀疏性）
        D_inv_sqrt_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(0, num), torch.arange(0, num)]),
            values=D.pow(-0.5),
            size=(num, num),
            device=device
        )
        # 使用稀疏矩阵乘法进行归一化 A_hat = D_inv_sqrt * A_prime * D_inv_sqrt
        A_hat = torch.sparse.mm(D_inv_sqrt_sparse, A_prime)
        A_hat = torch.sparse.mm(A_hat, D_inv_sqrt_sparse)
        return A_hat

    def forward(self, args, x, adjs, H, tau=1.0):

        layer_ = []
        att_map=[]
        z = self.fcs[0](x)
        num_hyperedges = H.shape[1]
        zx = z[num_hyperedges:] 
        zy = z.squeeze(0)[-num_hyperedges:] 
        node_pe, edge_pe = self.dual_encoder(H,zx,zy)  

        # 判断位置编码的组合情况，并拼接相应的嵌入
        if 'CODE' in args.encode and 'VISIT' not in args.encode:
            z = torch.cat((node_pe, zy), dim=0).unsqueeze(0) 
        elif 'VISIT' in args.encode and 'CODE' not in args.encode:
            z = torch.cat((zx, edge_pe), dim=0).unsqueeze(0) 
        elif 'CODE' in args.encode and 'VISIT' in args.encode:
            z = torch.cat((node_pe, edge_pe), dim=0).unsqueeze(0)  
        else:
            z = z.unsqueeze(0)  # 不使用位置编码，直接使用原始嵌入

        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)
        for i, conv in enumerate(self.convs):
            if self.return_att:
                z, att = conv(args, z, adjs, tau)
                att_map.append(att.detach().cpu())
            else:
                z = conv(args, z, adjs, tau)
            if self.use_residual:# 如果使用残差
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)# 把每一层卷积后的全部特征矩阵都保存在里面

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)

        num_hyperedges = H.shape[1]
        code_embeddings = z.squeeze(0)[num_hyperedges:]  
        hyperedge_embeddings = z.squeeze(0)[-num_hyperedges:] 
        hidden_dim = hyperedge_embeddings.shape[-1]
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") 

        attn_layer = VisitAttention(hidden_dim).to(device)
        # 根据 record_lengths 计算每个患者的嵌入
        patient_embeddings = []
        visit_weights_per_patient = []
        start_idx = 0

        record_length_filepath=args.records_length
        with open(record_length_filepath,"rb")as file :
            record_lengths=dill.load(file)

        for num_visits in record_lengths:
            end_idx = start_idx + num_visits
            visits = hyperedge_embeddings[start_idx:end_idx]
            patient_embed, visit_weights = attn_layer(visits)
            patient_embeddings.append(patient_embed)
            visit_weights_per_patient.append(visit_weights)
            start_idx = end_idx
        patient_embeddings = torch.stack(patient_embeddings, dim=0)  # (num_patients, hidden_dim)

        # 使用患者嵌入进行诊断分类,得到logit
        x_out = self.fcs[-1](patient_embeddings)  # (num_patients, out_channels)
        x_out = self.classfier(x_out)

        if self.return_att:
            return x_out, visit_weights_per_patient,hyperedge_embeddings
        else:
            return x_out
