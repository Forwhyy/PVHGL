import torch
from typing import Optional
import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module


def ExtractV2E(edge_index,num_nodes,num_hyperedges):
    '''
    edge_index 原始数据包含节点到超边和超边到节点的双向连接。
    V2E 提取的结果是节点到超边的连接，忽略了超边到节点的反向连接部分。
    '''
    # Assume edge_index = [V|E;E|V]
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    if not ((num_nodes+num_hyperedges-1) == edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    V2E = edge_index[:, :cidx].type(torch.LongTensor)
    return V2E

def ConstructH(edge_index_0,num_nodes):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    参数：edge_index_0节点到边的对应，第一行是节点，第二行是对应的边
    """
#     ipdb.set_trace()
    edge_index = torch.zeros_like(edge_index_0,dtype=edge_index_0.dtype)
    edge_index[0]=edge_index_0[0]-edge_index_0[0].min()
    edge_index[1]=edge_index_0[1]-edge_index_0[1].min()
    v=torch.ones(edge_index.shape[1])
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_hyperedges = edge_index[1].max()+1
    H=torch.sparse.FloatTensor(edge_index, v, torch.Size([num_nodes, num_hyperedges]))# (282，315)
    # H 的形状为 [num_nodes, num_hyperedges]，其中：
    # 行表示节点编号
    # 列表示超边编号
    # 值表示节点和超边之间的连接关系（权重)
    return H

def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    
    N = num_nodes

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # if edge_index.min() > 0:
    #     loop_index = loop_index + edge_index.min()

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


