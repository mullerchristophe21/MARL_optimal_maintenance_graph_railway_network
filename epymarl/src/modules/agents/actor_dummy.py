
import torch.nn as nn
import torch.nn.functional as F


class Actor_Dummy(nn.Module):
    def __init__(self, input_shape, args):
        super(Actor_Dummy, self).__init__()
        self.args = args

        self.dummy_policy_action = self.args.dummy_policy_action
        # if dummy policy action is a dictrionay, then use as threshold:
        self.threshold_strat = False
        if type(self.dummy_policy_action) == dict:
            self.threshold_strat = True
            self.thresholds = self.dummy_policy_action["thesholds"]
            self.dummy_actions = self.dummy_policy_action["actions"]

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        if self.threshold_strat:

            q[:,:] = -10000
            q[:,0] = -0.00001

            nodes_threshold_1 = inputs[:,0] > self.thresholds[0]
            nodes_threshold_2 = inputs[:,1] > self.thresholds[1]

            if len(nodes_threshold_1) > 0:
                q[nodes_threshold_1,:] = -10000
                q[nodes_threshold_1, self.dummy_actions[0]] = -0.00001
            
            if len(nodes_threshold_2) > 0:
                q[nodes_threshold_2,:] = -10000
                q[nodes_threshold_2, self.dummy_actions[1]] = -0.00001

        else:

            q = q * 0
            q[:,:] = -10000
            q[:, self.dummy_policy_action] = -0.00001

        return q, h

