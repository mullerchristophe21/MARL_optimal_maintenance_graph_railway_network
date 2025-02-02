import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import Transformer


class Critic_Transformer(nn.Module):
    def __init__(self, scheme, args):
        super(Critic_Transformer, self).__init__()

        self.args = args
        self.scheme = scheme
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        if args.use_shared_transformer:
            self.transformer = args.shared_transformer
        else:
            self.transformer = Transformer(args)

        self.output_type = "v"

        input_shape = args.encoder_dim_transformer + scheme["obs"]["vshape"]


        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)      # INPUT SHAPE = [n_env, n_steps (+1?), n_nodes, n_obs]
        x = self.transformer(inputs)
        concatenated_input = th.cat([x, inputs], dim=-1)            # concatenated_input.shape = [n_env, n_steps (+1?), n_nodes, input_shape + n_obs]
        x = F.relu(self.fc1(concatenated_input))                    # x.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim]
        x = F.relu(self.fc2(x))                                     # x.shape = [n_env, n_steps (+1?), n_nodes, hidden_dim]
        q = self.fc3(x) 
        return q                                                    # q SHAPE = [n_env, n_steps (+1?), n_nodes, 1]

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        inputs.append(batch["obs"][:, ts])

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        return input_shape
