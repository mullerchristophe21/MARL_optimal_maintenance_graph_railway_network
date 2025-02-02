
import torch.nn as nn
import torch.nn.functional as F
from modules.GNN import SharedGNN
import torch as th


class Actor_GNN(nn.Module):
    def __init__(self, input_shape, args):
        super(Actor_GNN, self).__init__()
        self.args = args
        if self.args.use_shared_gnn:
            self.gnn = self.args.shared_gnn
        else:
            self.gnn = SharedGNN(self.args)

        input_shape = args.hidden_dim_gnn + input_shape

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None): # INPUTS.SHAPE = [n_env x n_nodes, n_obs]
        x = self.gnn(inputs)    # x.shape = [n_env x n_nodes, hidden_dim_gnn]
        concatenated_input = th.cat([x, inputs], dim=-1)    # concatenated_input.shape = [n_env x n_nodes, hidden_dim_gnn + n_obs]
        x = F.relu(self.fc1(concatenated_input))    # x.shape = [n_env x n_nodes, hidden_dim]
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))   # h.shape = [n_env x n_nodes, hidden_dim]
        q = self.fc2(h)
        return q, h     # q.shape = [n_env x n_nodes, n_actions]

