# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F


# class COMACritic(nn.Module):
#     def __init__(self, scheme, args):
#         super(COMACritic, self).__init__()

#         self.args = args
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents

#         input_shape = self._get_input_shape(scheme)
#         self.output_type = "q"

#         # Set up network layers
#         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
#         self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
#         self.fc3 = nn.Linear(args.hidden_dim, self.n_actions)

#     def forward(self, batch, t=None):
#         inputs = self._build_inputs(batch, t=t)
#         x = F.relu(self.fc1(inputs))
#         x = F.relu(self.fc2(x))
#         q = self.fc3(x)
#         return q

#     def _build_inputs(self, batch, t=None):
#         bs = batch.batch_size
#         max_t = batch.max_seq_length if t is None else 1
#         ts = slice(None) if t is None else slice(t, t+1)
#         inputs = []
#         # state
#         inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

#         # observation
#         if self.args.obs_individual_obs:
#             inputs.append(batch["obs"][:, ts])

#         # actions (masked out by agent)
#         actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
#         agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
#         agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
#         inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

#         # last actions
#         if self.args.obs_last_action:
#             if t == 0:
#                 inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
#             elif isinstance(t, int):
#                 inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
#             else:
#                 last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
#                 last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
#                 inputs.append(last_actions)

#         if self.args.obs_agent_id:
#             inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

#         inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
#         return inputs

#     def _get_input_shape(self, scheme):
#         # state
#         input_shape = scheme["state"]["vshape"]
#         # observation
#         if self.args.obs_individual_obs:
#             input_shape += scheme["obs"]["vshape"]
#         # actions
#         input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
#         # last action
#         if self.args.obs_last_action:
#             input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
#         # agent id
#         if self.args.obs_agent_id:
#             input_shape += self.n_agents
#         return input_shape


import torch as th
import torch.nn as nn
import torch.nn.functional as F

class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_actions)

        # Print layer sizes
        print(f"Initialized COMACritic with layers:")
        print(f"fc1: {self.fc1.weight.shape}")
        print(f"fc2: {self.fc2.weight.shape}")
        print(f"fc3: {self.fc3.weight.shape}")

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        print(f"Input size: {inputs.size()}")  # Print size of inputs

        x = F.relu(self.fc1(inputs))
        print(f"After fc1: {x.size()}")  # Print size after fc1

        x = F.relu(self.fc2(x))
        print(f"After fc2: {x.size()}")  # Print size after fc2

        q = self.fc3(x)
        print(f"Output Q-values size: {q.size()}")  # Print size of output Q-values

        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        state = batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        inputs.append(state)
        print(f"State size: {state.size()}")  # Print size of state

        # observation
        if self.args.obs_individual_obs:
            obs = batch["obs"][:, ts]
            inputs.append(obs)
            print(f"Observation size: {obs.size()}")  # Print size of observation

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
        inputs.append(masked_actions)
        print(f"Masked actions size: {masked_actions.size()}")  # Print size of masked actions

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                last_action = th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            elif isinstance(t, int):
                last_action = batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_action)
            print(f"Last actions size: {last_action.size()}")  # Print size of last actions

        # agent id
        if self.args.obs_agent_id:
            agent_id = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            inputs.append(agent_id)
            print(f"Agent ID size: {agent_id.size()}")  # Print size of agent ID

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        print(f"Concatenated inputs size: {inputs.size()}")  # Print size of concatenated inputs

        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
