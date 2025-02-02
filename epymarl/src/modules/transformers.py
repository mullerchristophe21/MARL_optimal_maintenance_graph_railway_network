# Code without switch of device, does not work with cuda!

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
import numpy as np
import networkx as nx
from scipy import sparse as sp

from modules.transformers_utils import *


from data_graph_module import Graph


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')

        # Load the graph
        graph = Graph.load_graph_json(self.args.env_args["graph_file_json"])
        self.n_nodes = graph.adj_matrix.shape[0]
        nx_graph = nx.from_numpy_array(graph.adj_matrix.values)

        lat = graph.bins_df["lat"].values
        lon = graph.bins_df["lon"].values
        node_degrees = np.array(list(dict(nx.degree(nx_graph)).values()))
        closeness_centrality = np.array(list(nx.closeness_centrality(nx_graph).values()))
        between_centrality = np.array(list(nx.betweenness_centrality(nx_graph).values()))
        eigenvector_centrality = np.array(list(nx.eigenvector_centrality(nx_graph, max_iter=1000).values()))
        node_eccentricity = np.array(list(nx.eccentricity(nx_graph).values()))
        graph_center = np.mean(np.array(graph.bins_df[['lat', 'lon']]), axis=0)
        dist_to_center = np.linalg.norm(np.array(graph.bins_df[['lat', 'lon']]) - graph_center, axis=1)
        node_degrees_neigh = np.array([np.mean([node_degrees[neigh] for neigh in nx_graph.neighbors(node)]) + node_degrees[node] for node in nx_graph.nodes])
        node_degrees_2_neigh = np.array([np.mean([node_degrees_neigh[neigh] for neigh in nx_graph.neighbors(node)]) + node_degrees_neigh[node] for node in nx_graph.nodes])

        self.lat_std = (lat - lat.mean()) / lat.std()
        self.lon_std = (lon - lon.mean()) / lon.std()
        self.node_degrees_std = node_degrees - node_degrees.mean()
        self.closeness_centrality_std = (closeness_centrality - closeness_centrality.mean()) / closeness_centrality.std()
        self.between_centrality_std = (between_centrality - between_centrality.mean()) / between_centrality.std()
        self.eigenvector_centrality_std = (eigenvector_centrality - eigenvector_centrality.mean()) / eigenvector_centrality.std()
        self.node_eccentricity_std = (node_eccentricity - node_eccentricity.mean()) / node_eccentricity.std()
        self.dist_to_center_std = (dist_to_center - dist_to_center.mean()) / dist_to_center.std()
        self.node_degrees_neigh_std = (node_degrees_neigh - node_degrees_neigh.mean()) / node_degrees_neigh.std()
        self.node_degrees_2_neigh_std = (node_degrees_2_neigh - node_degrees_2_neigh.mean()) / node_degrees_2_neigh.std()

        self.fixed_node_features = np.stack([self.lat_std, 
                                             self.lon_std, 
                                             self.node_degrees_std, 
                                             self.closeness_centrality_std, 
                                             self.between_centrality_std, 
                                             self.eigenvector_centrality_std, 
                                             self.node_eccentricity_std, 
                                             self.dist_to_center_std, 
                                             self.node_degrees_neigh_std, 
                                             self.node_degrees_2_neigh_std], axis=1)


        if args.use_6_graph_input:
            self.fixed_node_features = self.fixed_node_features[:, [0, 1, 2, 3, 5, 8]]

        self.fixed_node_features = torch.tensor(self.fixed_node_features, dtype=torch.float32).to(self.device)

        self.EigVec = laplacian_positional_encoding(nx_graph, args.pos_enc_dim).to(self.device)
        self.Adj_Matrix = torch.from_numpy(graph.adj_matrix.values).float().to(self.device)

        # Layers
        if args.use_graph_input:
            self.input_encoder_dim = args.env_args["n_obs"] + self.fixed_node_features.shape[1]
        else:
            self.input_encoder_dim = args.env_args["n_obs"]

        
        if args.transformer_encoder_concat:
            self.input_encoder_dim += args.pos_enc_dim
        else:
            self.encoder_EigVec = nn.Linear(args.pos_enc_dim, args.encoder_dim_transformer)
        
        
        self.encoder_input = nn.Linear(self.input_encoder_dim, args.encoder_dim_transformer)

        self.layers = nn.ModuleList([GraphTransformerLayer(args.encoder_dim_transformer, args.encoder_dim_transformer, args.num_heads,
                                                           args.layer_norm, args.batch_norm, args.residual, args.use_bias, args.neigh_attention) for _ in range(args.n_layers_transformers)])

    def forward(self, x):


        if x.dim() == 2:
            # Actor input: [batch_size * n_nodes, n_obs]
            batch_size = x.size(0) // self.n_nodes
            x = x.view(batch_size, self.n_nodes, -1)

            if self.args.use_graph_input:
                x = torch.cat([x, self.fixed_node_features.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
            
            if self.args.transformer_encoder_concat:
                sign_flip = torch.rand(self.EigVec.size(1)).to(self.device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                EigVec_flipped = self.EigVec * sign_flip.unsqueeze(0)
                x = torch.cat([x, EigVec_flipped.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
                h = self.encoder_input(x)
            else:
                sign_flip = torch.rand(self.EigVec.size(1)).to(self.device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                EigVec_flipped = self.EigVec * sign_flip.unsqueeze(0)
                h_lap_pos_enc = self.encoder_EigVec(EigVec_flipped)
                h = self.encoder_input(x)            
                h = h + h_lap_pos_enc
            
            for conv in self.layers:
                h = conv(self.Adj_Matrix, h)

            x = h.view(-1, self.args.encoder_dim_transformer)

        elif x.dim() == 4:
            # Critic input: [batch_size, n_timesteps, n_nodes, n_obs]
            batch_size, n_timesteps, n_nodes, n_obs = x.size(0), x.size(1), x.size(2), x.size(3)
            x = x.view(-1, self.n_nodes, n_obs)  # Flatten batch_size and n_timesteps

            if self.args.use_graph_input:
                x = torch.cat([x, self.fixed_node_features.unsqueeze(0).expand(x.size(0), -1, -1)], dim=-1)
            
            if self.args.transformer_encoder_concat:
                sign_flip = torch.rand(self.EigVec.size(1)).to(self.device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                EigVec_flipped = self.EigVec * sign_flip.unsqueeze(0)
                x = torch.cat([x, EigVec_flipped.unsqueeze(0).expand(x.size(0), -1, -1)], dim=-1)
                h = self.encoder_input(x)
            else:
                sign_flip = torch.rand(self.EigVec.size(1)).to(self.device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                EigVec_flipped = self.EigVec * sign_flip.unsqueeze(0)
                h_lap_pos_enc = self.encoder_EigVec(EigVec_flipped)
                h = self.encoder_input(x)            
                h = h + h_lap_pos_enc
            
            for conv in self.layers:
                h = conv(self.Adj_Matrix, h)

            x = h.view(batch_size, n_timesteps, self.n_nodes, -1)

        return x
    