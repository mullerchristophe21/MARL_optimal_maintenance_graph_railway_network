# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from data_graph_module import Graph


class CTCEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CTCEAgent, self).__init__()
        self.args = args
        
        self.batch_size = args.batch_size_run
        graph = Graph.load_graph_json(self.args.env_args["graph_file_json"])
        self.n_nodes = graph.adj_matrix.shape[0]

        self.fc1 = nn.Linear(self.n_nodes * input_shape, self.n_nodes * args.n_actions)
        self.fc2 = nn.Linear(self.n_nodes * args.n_actions,  self.n_nodes * args.n_actions)
        self.fc3 = nn.Linear(self.n_nodes * args.n_actions,  self.n_nodes * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):    # INPUTS.SHAPE = [n_envs x n_nodes, n_obs]

        x = inputs.view(-1, self.n_nodes,  inputs.shape[1])    # x.shape = [n_envs, n_nodes, n_obs]

        x = x.view(x.shape[0], -1)      # x.shape = [n_envs, n_nodes * n_obs]

        # fully connected layers
        x = F.relu(self.fc1(x))    # x.shape = [n_envs, n_nodes * n_actions]
        x = F.relu(self.fc2(x))    # x.shape = [n_envs, n_nodes * n_actions]
        x = self.fc3(x)            # x.shape = [n_envs, n_nodes * n_actions]

        x = x.view(-1, self.args.n_actions)   # x.shape = [n_envs * n_nodes, n_actions]
        
        return x, hidden_state

