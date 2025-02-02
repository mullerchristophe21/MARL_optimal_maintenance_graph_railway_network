import os
import pickle
# import jax
import sys
import re
import networkx as nx
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from data_graph_utils import *
from gp_on_graph_wrapper import GraphMaternKernel

# from jax.lib import xla_bridge
from datetime import datetime

print(f"Running on PyMC v{pm.__version__}")
# print(xla_bridge.get_backend().platform)
# print(jax.devices())


#####################################################################################################
print("\nLoading Data...")
####

sample_posterior = sys.argv[1]
tune = sys.argv[2]
chains = sys.argv[3]
name_model = sys.argv[4]
graph_file_path = sys.argv[5]
n_cores = sys.argv[6]
note_save = sys.argv[7]
prc_test = int(sys.argv[8])/100
target_accept = int(sys.argv[9])/100
var_d1 = sys.argv[10]
var_d2 = sys.argv[11]

sample_posterior = int(sample_posterior)
tune = int(tune)
chains = int(chains)
n_cores = int(n_cores)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%m%d_%H%M")
print("\tStarting time: ", current_datetime)

graph_txt_bool = re.search(r"/(.*?)\.p", graph_file_path)
if graph_txt_bool:
    graph_txt = graph_txt_bool.group(1)
else:
    graph_txt = "GRAPH"

output_file = f"./storage_inference/{formatted_datetime}___{name_model}___{sample_posterior}s_{tune}t_{chains}c___{graph_txt}__{var_d1}{var_d2}__{note_save}.nc"

print("\tsample post: ", sample_posterior)
print("\ttune: ", tune)
print("\tchains: ", chains)
print("\tname_model: ", name_model)
print("\tgraph_file: ", graph_file_path)
print("\tn_cores: ", n_cores)
print("\tprc_test: ", prc_test)
print("\ttarget_accept: ", target_accept)
print("\toutput_file: ", output_file)

if os.path.exists(graph_file_path):
    with open(graph_file_path, "rb") as file:
        graph = pickle.load(file)
else:
    raise ValueError("Graph file does not exist")

# Extracting the indicator time series for the selected nodes
indicator_d1 = graph.indicator_ts_dict[var_d1]
indicator_d2 = graph.indicator_ts_dict[var_d2]


used_ids = indicator_d1.index[(indicator_d1.isna().sum(axis=1) == 0) &
                                                (indicator_d2.isna().sum(axis=1) == 0)].values

if "lat" not in graph.bins_df.columns:
    graph.bins_df["lat"], graph.bins_df["lon"] = lv95_to_lat_long(graph.bins_df["mean_x"], graph.bins_df["mean_y"])

# subgraph laplacian:
G = nx.from_pandas_adjacency(graph.adj_matrix)
laplacian = nx.laplacian_matrix(G).todense()
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
eigenpairs = (eigenvalues, eigenvectors)

indicator_d1 = graph.indicator_ts_dict["lonle_d1"]
indicator_d1_used = indicator_d1.iloc[used_ids, :]
indicator_d1_d_used = indicator_d1_used.diff(axis=1)
print("\tIndicator d1 shape: ", indicator_d1_used.shape)

indicator_d2 = graph.indicator_ts_dict["lonle_d2"]
indicator_d2_used = indicator_d2.iloc[used_ids, :]
indicator_d2_d_used = indicator_d2_used.diff(axis=1)
print("\tIndicator d2 shape: ", indicator_d2_used.shape)

# action ts:
action_ts = graph.actions_ts_adjusted
action_ts_used = action_ts.iloc[used_ids, :]
print("\tAction ts shape: ", action_ts_used.shape)

PROP_OF_TEST_NODES = prc_test
RND_SEED = 42
np.random.seed(RND_SEED)

n_test_nodes = int(PROP_OF_TEST_NODES * len(used_ids))
test_ids = np.random.choice(used_ids, n_test_nodes, replace=False)
train_ids = np.setdiff1d(used_ids, test_ids)

indicator_d1_train = indicator_d1_used.loc[train_ids, :]
indicator_d1_test = indicator_d1_used.loc[test_ids, :]
indicator_d2_train = indicator_d2_used.loc[train_ids, :]
indicator_d2_test = indicator_d2_used.loc[test_ids, :]
action_ts_train = action_ts_used.loc[train_ids, :]
action_ts_test = action_ts_used.loc[test_ids, :]

indicator_d1_d_train = indicator_d1_d_used.loc[train_ids, :]
indicator_d1_d_test = indicator_d1_d_used.loc[test_ids, :]
indicator_d2_d_train = indicator_d2_d_used.loc[train_ids, :]
indicator_d2_d_test = indicator_d2_d_used.loc[test_ids, :]

print("\tProb of test: ", PROP_OF_TEST_NODES)
print("\tRandom Seed: ", RND_SEED)
print("\tTrain nodes:", len(train_ids))
print("\tTest nodes:", len(test_ids))


#####################################################################################################
print("\nBuilding the model...")
####

n_timesteps = indicator_d1_train.shape[1]

with pm.Model() as m:

    ### Data input
    actions_ts = pm.MutableData("actions", action_ts_train.values.tolist())
    stacked_actions_ts = pt.concatenate([actions_ts, actions_ts], axis=0)
    ind_d1_t = []
    ind_d2_t = []
    ind_stacked = []
    for t in range(0, n_timesteps):
        ind_d1_t.append(pm.MutableData(f"indicator_d1_{t}", indicator_d1_train.values[:,t].tolist()))
        ind_d2_t.append(pm.MutableData(f"indicator_d2_{t}", indicator_d2_train.values[:,t].tolist()))
        ind_stacked.append(pm.MutableData(f"indicator_stacked_{t}", np.concatenate([indicator_d1_train.values[:,t], indicator_d2_train.values[:,t]], axis=0).tolist()))

    ### Initial process
    # Prior
    mu_init_d1 = pm.Laplace("mu_init_d1", mu=0.5, b=1)
    mu_init_d2 = pm.Laplace("mu_init_d2", mu=0.5, b=1)

    nu_g_init_d1 = 0.5
    kappa_g_init_d1 = pm.HalfNormal('kappa_g_init_d1', sigma=4)
    sigma_g_init_d1 = pm.HalfNormal('sigma_g_init_d1', sigma=2)

    # nu_g_init_d2 = pm.HalfNormal('nu_g_init_d2', sigma=3)
    nu_g_init_d2 = 0.5
    kappa_g_init_d2 = pm.HalfNormal('kappa_g_init_d2', sigma=4)
    sigma_g_init_d2 = pm.HalfNormal('sigma_g_init_d2', sigma=2)

    dummy_cov = pm.gp.cov.Matern52(1, ls=999)
    dummy_input = pt.stack([train_ids, pt.ones(train_ids.shape[0])], axis=1)
    cov_init_d1 = GraphMaternKernel(eigenpairs=eigenpairs, vertex_dim=1, point_kernel=dummy_cov, nu=nu_g_init_d1, kappa=kappa_g_init_d1, sigma_f=sigma_g_init_d1)
    cov_init_d2 = GraphMaternKernel(eigenpairs=eigenpairs, vertex_dim=1, point_kernel=dummy_cov, nu=nu_g_init_d2, kappa=kappa_g_init_d2, sigma_f=sigma_g_init_d2)
    
    cov_mat_init_d1 = cov_init_d1(dummy_input)
    cov_mat_init_d2 = cov_init_d2(dummy_input)

    #Likelihood
    init_d1 = pm.MvNormal("init_d1", mu=mu_init_d1, cov=cov_mat_init_d1, observed=ind_d1_t[0])
    init_d2 = pm.MvNormal("init_d2", mu=mu_init_d2, cov=cov_mat_init_d2, observed=ind_d2_t[0])
    
    ### Rest of the process
    # Prior
    mu_d1 = pm.Laplace("mu_d1", mu=[0.1,0,-0.5,-1], b=1)
    mu_d2 = pm.Laplace("mu_d2", mu=[0.1,0,-0.5,-1], b=1)
    nu_d = pm.Laplace("nu_d1", mu=5, b=0.2)
    k_d1 = 2 * pm.Beta(f"k_d1", alpha=10, beta=(10,15,15), shape=(3,))
    k_d1 = pt.concatenate([[1], k_d1])
    k_d2 = 2 * pm.Beta(f"k_d2", alpha=10, beta=(10,15,15), shape=(3,))
    k_d2 = pt.concatenate([[1], k_d2])
    lambda_act_d1 = pm.Normal("lambda_act_d1", mu=[0,1,2], sigma=0.5, shape=(3,))
    lambda_act_d1 = pt.concatenate([[0], lambda_act_d1])
    lambda_act_d2 = pm.Normal("lambda_act_d2", mu=[0,1,2], sigma=0.5, shape=(3,))
    lambda_act_d2 = pt.concatenate([[0], lambda_act_d2])
    lambda_ind = pm.Normal("lambda_ind", mu=[0], sigma=0.5, shape=(1,))
    lambda_ind = pt.concatenate([[0], lambda_ind])

    ### Covariance Matrix
    nu_g = 0.1
    kappa_g = pm.TruncatedNormal('kappa_g', mu=0.5, sigma=0.25, lower=0.001)
    sigma_g = pm.HalfNormal('sigma_g', sigma=2)
    ls_act = pm.HalfNormal("ls_act", sigma=0.5)
    ls_ind = pm.HalfNormal("ls_ind", sigma=0.5)

    cov_act_ind = pm.gp.cov.Matern52(2, ls=[ls_act, ls_ind])
    cov = GraphMaternKernel(eigenpairs=eigenpairs, vertex_dim=2, point_kernel=cov_act_ind, nu=nu_g, kappa=kappa_g, sigma_f=sigma_g)
    
    ## Likelihood
    for t in range(1, n_timesteps):

        input_gp = pt.stack([
            train_ids.tolist() * 2,                                         # Nodes id (graph-structure) 
            stacked_actions_ts[:,t-1],                                      # Actions
            [0] * train_ids.shape[0] + [1] * train_ids.shape[0]], axis=1)   # Indicator type
    
        cov_matrix_t = cov(input_gp)
        diag_t = pt.eye(train_ids.shape[0] * 2) * pt.diag(cov_matrix_t)

        effect_action_t_d1 = lambda_act_d1[actions_ts[:,t-1]]
        effect_action_t_d2 = lambda_act_d2[actions_ts[:,t-1]]
        effect_action_t = pt.concatenate([effect_action_t_d1, effect_action_t_d2], axis=0)

        effect_ind_t = lambda_ind[[0] * train_ids.shape[0] + [1] * train_ids.shape[0]]

        diag_2_t = diag_t * effect_action_t + diag_t * effect_ind_t

        final_cov_matrix_t = cov_matrix_t + diag_2_t
        
        mu_prev_d1 = mu_d1[actions_ts[:,t-1]] + k_d1[actions_ts[:,t-1]] * ind_d1_t[t-1]
        mu_prev_d2 = mu_d2[actions_ts[:,t-1]] + k_d2[actions_ts[:,t-1]] * ind_d2_t[t-1]
        mu_prev_t = pm.Deterministic(f"mu_{t}", pt.concatenate([mu_prev_d1, mu_prev_d2], axis=0))

        obs_t = pm.MvStudentT(f"obs_{t}", mu=mu_prev_t, scale=final_cov_matrix_t, nu=nu_d, observed=ind_stacked[t])


#####################################################################################################
print("\nSampling the model...")
####

with m:
    # trace = pm.sample(sample_posterior, tune=tune, chains=chains, cores=n_cores, target_accept=target_accept, nuts_sampler="numpyro") ### Install and import JAX to run numpyro sampler
    trace = pm.sample(sample_posterior, tune=tune, chains=chains, cores=n_cores, target_accept=target_accept)
    az.to_netcdf(trace, output_file)

print("\nDone!")