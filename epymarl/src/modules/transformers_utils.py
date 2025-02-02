import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from scipy import sparse as sp

"""
    General Multi-Head Attention Layer without DGL dependency
"""

def laplacian_positional_encoding(nx_graph, pos_enc_dim):

    A = nx.to_scipy_sparse_array(nx_graph).astype(float)
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(np.power(degrees, -0.5))
    
    L = sp.eye(nx_graph.number_of_nodes()) - D_inv_sqrt * A * D_inv_sqrt
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)

    EigVec = EigVec[:, EigVal.argsort()]
    
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    
    return lap_pos_enc


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, neigh_attention=True):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.neigh_attention = neigh_attention
        
        # Linear layers for Q, K, V
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
    
    def forward(self, adj_matrix, h):
        # adj_matrix: [batch_size, num_nodes, num_nodes]
        # h: [batch_size, num_nodes, in_dim]
        batch_size, num_nodes, _ = h.size()

        # Linear projections for Q, K, V
        Q_h = self.Q(h)  # [batch_size, num_nodes, num_heads * out_dim]
        K_h = self.K(h)  # [batch_size, num_nodes, num_heads * out_dim]
        V_h = self.V(h)  # [batch_size, num_nodes, num_heads * out_dim]

        # Reshaping into [batch_size, num_nodes, num_heads, out_dim]
        Q_h = Q_h.view(batch_size, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.view(batch_size, num_nodes, self.num_heads, self.out_dim)
        V_h = V_h.view(batch_size, num_nodes, self.num_heads, self.out_dim)

        # Compute attention scores
        scores = torch.einsum('bnhd,bmhd->bhnm', Q_h, K_h)  # [batch_size, num_heads, num_nodes, num_nodes]
        scores = scores / torch.sqrt(torch.tensor(self.out_dim, dtype=torch.float32))

        # Apply adjacency matrix mask
        adj_matrix_expanded = adj_matrix.unsqueeze(0).expand(self.num_heads, -1, -1)  # [batch_size, num_heads, num_nodes, num_nodes]
        if self.neigh_attention:
            scores = scores.masked_fill(adj_matrix_expanded == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, num_nodes, num_nodes]

        # Weighted sum of values
        out = torch.einsum('bhnm,bmhd->bnhd', attn_weights, V_h)  # [batch_size, num_heads, num_nodes, out_dim]

        # Concatenate all the heads' outputs
        out = out.reshape(batch_size, num_nodes, self.num_heads * self.out_dim)  # [batch_size, num_nodes, num_heads * out_dim]
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, layer_norm=False, batch_norm=True, residual=True, use_bias=False,
                 neigh_attention=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias, neigh_attention)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, adj_matrix, h):
        h_in1 = h  # for first residual connection
        
        # Multi-head attention output
        attn_out = self.attention(adj_matrix, h)
        
        h = self.O(attn_out)
        
        if self.residual:
            h = h_in1 + h  # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h  # for second residual connection
        
        # Feed-forward network (FFN)
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

