REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .actor_dummy import Actor_Dummy
from .actor_gnn import Actor_GNN
from .actor_transformer import Actor_Transformer
from .ctce_agent import CTCEAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["actor_gnn"] = Actor_GNN
REGISTRY["rnn_dummy"] = Actor_Dummy
REGISTRY["actor_transformer"] = Actor_Transformer
REGISTRY["ctce"] = CTCEAgent
