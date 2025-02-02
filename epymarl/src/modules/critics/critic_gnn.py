import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.GNN import SharedGNN


class Critic_GNN(nn.Module):
    def __init__(self, scheme, args):
        super(Critic_GNN, self).__init__()

        self.args = args
        self.scheme = scheme
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        if self.args.use_shared_gnn:
            self.gnn = self.args.shared_gnn
        else:
            self.gnn = SharedGNN(self.args)

        input_shape = args.hidden_dim_gnn + scheme["obs"]["vshape"]
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs_gnn(batch, t=t)      # INPUT SHAPE = [n_env, n_steps (+1?), n_nodes, n_obs]
        x = self.gnn(inputs)                                        # x.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim_gnn]
        # in_act, bs, max_t = self._build_inputs_act(batch, t=t) 
        concatenated_input = th.cat([x, inputs], dim=-1)            # concatenated_input.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim_gnn + n_obs]
        x = F.relu(self.fc1(concatenated_input))                    # x.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim]
        x = F.relu(self.fc2(x))                                     # x.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim]
        q = self.fc3(x) 
        return q                                                    # q SHAPE = [n_env, n_steps (+1?), n_nodes, 1]

    def _build_inputs_gnn(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        inputs.append(batch["obs"][:, ts])

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t
    
    def _build_inputs_act(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        inputs.append(batch["actions_onehot"][:, ts])
        inputs.append(batch["obs"][:, ts])

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        return input_shape
