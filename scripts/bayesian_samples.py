import os
import pickle
# import jax
import sys
import re
import networkx as nx
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import numpy as np


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

trace_name = sys.argv[1]
graph_file_path = sys.argv[2]
prc_test = int(sys.argv[3])/100
var_d1 = sys.argv[4]
var_d2 = sys.argv[5]
n = int(sys.argv[6])
fixed_input = bool(sys.argv[7])
use_mean = bool(sys.argv[8])

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%m%d_%H%M")
print("\tStarting time: ", current_datetime)

graph_txt_bool = re.search(r"/(.*?)\.p", graph_file_path)
if graph_txt_bool:
    graph_txt = graph_txt_bool.group(1)
else:
    graph_txt = "GRAPH"


print("\tgraph_file: ", graph_file_path)
print("\tprc_test: ", prc_test)

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

print("\tPrc of test: ", PROP_OF_TEST_NODES)
print("\tRandom Seed: ", RND_SEED)
print("\tTrain nodes:", len(train_ids))
print("\tTest nodes:", len(test_ids))

#### Load trace
trace_path = f"./storage_inference/{trace_name}"
trace_name = trace_name.split(".")[0]
trace = az.from_netcdf(trace_path)

if use_mean:
    file_save = f"./storage_inference/{trace_name}_samples_post_mean.pkl"
else:
    file_save = f"./storage_inference/{trace_name}_samples_post.pkl"

from tqdm.auto import tqdm
import scipy.stats as stats

nodes = test_ids

n_test = len(test_ids)
n_timesteps_sample = indicator_d1_test.shape[1]

t_extract = az.extract(trace, group="posterior", combined=True).transpose("sample", ...)
samples_post = np.zeros((n, n_test, n_timesteps_sample, 2))

for i in tqdm(range(n)):

    try:

        if use_mean:
            t_extract = trace.posterior.mean(dim=["chain", "draw"])
        else:
            # choose random sample
            sample_t = np.random.choice(t_extract.indexes["sample"].shape[0], 1)
            t_extract = t_extract.isel(sample=sample_t)

        # INITIAL PROCESS
        mu_init_d1 = t_extract["mu_init_d1"].values.ravel()
        mu_init_d2 = t_extract["mu_init_d2"].values.ravel()
        kappa_g_init_d1 = t_extract["kappa_g_init_d1"].values.ravel()
        sigma_g_init_d1 = t_extract["sigma_g_init_d1"].values.ravel()
        kappa_g_init_d2 = t_extract["kappa_g_init_d2"].values.ravel()
        sigma_g_init_d2 = t_extract["sigma_g_init_d2"].values.ravel()

        # REST OF THE PROCESS
        mu_d1 = t_extract["mu_d1"].values.ravel()
        mu_d2 = t_extract["mu_d2"].values.ravel()
        nu_d1 = t_extract["nu_d1"].values.ravel()
        k_d1 = 2 * t_extract["k_d1"].values.ravel()
        k_d1 = np.concatenate([[0], k_d1])
        k_d2 = 2 * t_extract["k_d2"].values.ravel()
        k_d2 = np.concatenate([[0], k_d2])
        lambda_act_d1 = t_extract["lambda_act_d1"].values.ravel()
        lambda_act_d1 = np.concatenate([[0], lambda_act_d1])
        lambda_act_d2 = t_extract["lambda_act_d2"].values.ravel()
        lambda_act_d2 = np.concatenate([[0], lambda_act_d2])
        lambda_ind = t_extract["lambda_ind"].values.ravel()
        lambda_ind = np.concatenate([[0], lambda_ind])

        # COVARIANCE MATRIX
        nu_g = 0.1
        kappa_g = t_extract["kappa_g"].values.ravel()
        sigma_g = t_extract["sigma_g"].values.ravel()
        ls_act = t_extract["ls_act"].values.ravel()
        ls_ind = t_extract["ls_ind"].values.ravel()
        ls = pt.concatenate([ls_act, ls_ind], axis=0)


        # SAMPLE INIT
        if fixed_input:
            samples_post[i,:,0,0] = indicator_d1_test.values[:,0]
            samples_post[i,:,0,1] = indicator_d2_test.values[:,0]
        else:
            raise("Not implemented")
        

        cov = pm.gp.cov.Matern52(2, ls=ls)
        cov_fun = GraphMaternKernel(eigenpairs=eigenpairs, vertex_dim=2, point_kernel=cov, nu=nu_g, kappa=kappa_g, sigma_f=sigma_g)        
        
        # SAMPLE REST
        for t in range(1, n_timesteps_sample):

            input_gp = pt.stack([
                nodes.tolist()*2, 
                action_ts_test.values[:,t-1].tolist()*2,
                [0]*len(nodes)+[1]*len(nodes)], axis=1)

            cov_matrix_t = cov_fun(input_gp)
            diag_t = pt.eye(len(nodes)*2) * pt.diag(cov_matrix_t)

            effect_action_t_d1 = lambda_act_d1[action_ts_test.values[:,t-1].tolist()]
            effect_action_t_d2 = lambda_act_d2[action_ts_test.values[:,t-1].tolist()]
            effect_action_t = pt.concatenate([effect_action_t_d1, effect_action_t_d2], axis=0)

            effect_ind_t = lambda_ind[[0]*len(nodes)+[1]*len(nodes)]

            diag_2_t = diag_t * effect_action_t + diag_t * effect_ind_t

            final_cov_matrix_t = (cov_matrix_t + diag_2_t).eval()

            mu_prev_d1 = mu_d1[action_ts_test.values[:,t-1].astype(int)] + k_d1[action_ts_test.values[:,t-1].astype(int)] * samples_post[i,:,t-1,0]
            mu_prev_d2 = mu_d2[action_ts_test.values[:,t-1].astype(int)] + k_d2[action_ts_test.values[:,t-1].astype(int)] * samples_post[i,:,t-1,1]

            mu_prev_t = np.concatenate([mu_prev_d1, mu_prev_d2], axis=0)

            samples_t = stats.multivariate_t.rvs(loc=mu_prev_t, shape=final_cov_matrix_t, df=nu_d1)

            samples_post[i,:,t,0] = samples_t[:n_test]
            samples_post[i,:,t,1] = samples_t[n_test:]


    except Exception as e:
        print(e)
        continue

    if i % 10 == 0 and i > 0:
        with open(file_save, "wb") as file:
            pickle.dump(samples_post, file)
    if i == n-1:
        with open(file_save, "wb") as file:
            pickle.dump(samples_post, file)