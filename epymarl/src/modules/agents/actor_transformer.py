
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import Transformer
import torch as th


class Actor_Transformer(nn.Module):
    def __init__(self, input_shape, args):
        super(Actor_Transformer, self).__init__()
        self.args = args

        if args.use_shared_transformer:
            self.transformer = args.shared_transformer
        else:
            self.transformer = Transformer(args)

        self.fc1 = nn.Linear(args.encoder_dim_transformer + input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None): # INPUTS.SHAPE = [n_env x n_nodes, n_obs]

        x = self.transformer(inputs)    # x.shape = [n_env x n_nodes, encoder_dim_transformer]
        concatenated_input = th.cat([x, inputs], dim=-1)    # concatenated_input.shape = [n_env x n_nodes, encoder_dim_transformer + n_obs]

        x = F.relu(self.fc1(concatenated_input))    # x.shape = [n_env x n_nodes, hidden_dim]
        if self.args.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.hidden_dim)
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))   # h.shape = [n_env x n_nodes, hidden_dim]
        q = self.fc2(h)
        return q, h     # q.shape = [n_env x n_nodes, n_actions]

