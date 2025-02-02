####
#
# This environment differs from MARL_env_fake_cov by its scaling of the covariance matrix.
#
# You can set the reduce_var_coef attribute to scale the covariance matrix as needed.
#
###

import os
import arviz as az
import networkx as nx
import sys
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import gym
import scipy.stats as stats
from sklearn.gaussian_process.kernels import Matern
import datetime
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

from data_graph_module import Graph
from gp_on_graph_wrapper import GraphMaternKernel_numpy_efficient

class MARL_env_fake_cov_small(gym.Env):

    def __init__(self, graph_file_json, trace_path, rewards_args, n_actions, n_obs, max_timesteps, node_length=150, trace_mean=True, seed=None):

        if seed:
            self.seed(seed)
    
        self.obs_size = n_obs
        self.n_actions = n_actions
        self.min_emission = -5
        self.max_emission = 7
        self.max_timesteps = max_timesteps
        self.node_length = node_length

        self.graph = Graph.load_graph_json(graph_file_json)
        self.graph_nx = nx.from_numpy_array(self.graph.adj_matrix.values)
        laplacian = nx.laplacian_matrix(self.graph_nx).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        self.eigenpairs = (eigenvalues, eigenvectors)


        self.trace = az.from_netcdf(trace_path)

        self.n_nodes_graph = len(self.graph.bins_df)

        self.trace_mean = trace_mean

        # Cost emissions
        self.thresholds = rewards_args["thresholds"]
        self.cost_threshold = rewards_args["cost_threshold"]
        self.cost_ellipse_n = rewards_args["n"]

        # Costs
        # Renewal
        self.fixed_cost_renewal = rewards_args["fixed_cost_renewal"]
        self.variable_cost_renewal = rewards_args["variable_cost_renewal"]
        self.rate_of_decay_renewal = rewards_args["rate_of_decay_renewal"]
        # Tamping
        self.fixed_cost_tamping = rewards_args["fixed_cost_tamping"]
        self.variable_cost_tamping = rewards_args["variable_cost_tamping"]
        self.rate_of_decay_tamping = rewards_args["rate_of_decay_tamping"]
        
        np.fill_diagonal(self.graph.adj_matrix.values,0)

        self.observation_space = gym.spaces.Box(
            low=self.min_emission,
            high=self.max_emission,
            shape=(self.n_nodes_graph,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_nodes_graph)    

    def reset(self, seed=None):

        # draw a random iteration from the trace
        if self.trace_mean:
            self.trace_iteration = self.trace.posterior.mean(dim=["chain", "draw"])
        else:
            trace = az.extract(self.trace, combined=True, group="posterior").transpose("sample", ...)
            id_trace_draw = np.random.randint(0, len(trace.sample))
            self.trace_iteration = trace.isel(sample=id_trace_draw)
        
        self.save_args_model()

        self.renewal_2_step_before = np.array([False] * self.n_nodes_graph)
        self.renewal_1_step_before = np.array([False] * self.n_nodes_graph)

        self.timestep = 0
        self.current_emissions = self.sample_initial_emissions()

        info = {}

        return self.current_emissions, info
    

    def save_args_model(self):

        list_vars = list(self.trace_iteration.data_vars)
        self.args_model = {var: self.trace_iteration[var].values.ravel() for var in list_vars}   

        self.args_model["k_d1"] = np.concatenate([[1], 2 * self.args_model["k_d1"]])
        self.args_model["k_d2"] = np.concatenate([[1], 2 * self.args_model["k_d2"]])

        self.args_model["lambda_act_d1"] = np.concatenate([[1], self.args_model["lambda_act_d1"]]) -1
        self.args_model["lambda_act_d2"] = np.concatenate([[1], self.args_model["lambda_act_d2"]]) -1 

        self.args_model["lambda_ind"] = np.concatenate([[1], self.args_model["lambda_ind"]]) -1

        cov_act_ind = Matern(length_scale=self.args_model["ls_act"] + self.args_model["ls_ind"], nu=5/2)

        self.cov = GraphMaternKernel_numpy_efficient(self.eigenpairs, vertex_dim=2, point_kernel=cov_act_ind,
                                                        nu=20, kappa=1.50, 
                                                        sigma_f=self.args_model["sigma_g"])

    def sample_initial_emissions(self):

        mu_init_d1 = self.args_model["mu_init_d1"]
        mu_init_d2 = self.args_model["mu_init_d2"]

        if "nu_g_init_d1" not in self.args_model:
            nu_g_init_d1 = 0.5
            nu_g_init_d2 = 0.5
        else:
            nu_g_init_d1 = self.args_model["nu_g_init_d1"]
            nu_g_init_d2 = self.args_model["nu_g_init_d2"]

        kappa_g_init_d1 = self.args_model["kappa_g_init_d1"]
        sigma_g_init_d1 = self.args_model["sigma_g_init_d1"]

        kappa_g_init_d2 = self.args_model["kappa_g_init_d2"]
        sigma_g_init_d2 = self.args_model["sigma_g_init_d2"]

        cov_init_d1 = GraphMaternKernel_numpy_efficient(self.eigenpairs, vertex_dim=0, point_kernel=None, 
                                                        nu=nu_g_init_d1, kappa=kappa_g_init_d1, sigma_f=sigma_g_init_d1)
        cov_init_d2 = GraphMaternKernel_numpy_efficient(self.eigenpairs, vertex_dim=0, point_kernel=None,
                                                        nu=nu_g_init_d2, kappa=kappa_g_init_d2, sigma_f=sigma_g_init_d2)
        
        sigma_d1 = cov_init_d1(np.arange(self.n_nodes_graph)[:,None])
        sigma_d2 = cov_init_d2(np.arange(self.n_nodes_graph)[:,None])

        mean_d1 = np.repeat(mu_init_d1, self.n_nodes_graph)
        mean_d2 = np.repeat(mu_init_d2, self.n_nodes_graph)

        emissions_1 = stats.multivariate_normal.rvs(mean=mean_d1, cov=sigma_d1)
        emissions_2 = stats.multivariate_normal.rvs(mean=mean_d2, cov=sigma_d2)

        return np.array([emissions_1, emissions_2]).T


    def step(self, actions):

        actions = np.array(actions)

        # step since renewal
        self.renewal_2_step_before = self.renewal_1_step_before
        self.renewal_1_step_before = (actions == 3)

        # reward
        rewards, dict_rewards = self.rewards_fun(self.current_emissions, actions)

        # emissions
        emissions = self.sample_emissions(self.current_emissions, actions)
        self.prev_emission = self.current_emissions
        self.current_emissions = emissions

        # done
        self.timestep += 1
        done = (self.timestep == self.max_timesteps)

        if done:
            done_dict = {i:True for i in range(self.n_nodes_graph)}
            # done_dict["__all__"] = True
        else:
            done_dict = {i:False for i in range(self.n_nodes_graph)}
            # done_dict["__all__"] = False

        info = {"dict_rewards":dict_rewards}
        return emissions, rewards, done_dict, info


    def rewards_fun(self, emissions, actions):

        cost_ems = self.cost_threshold * (self.are_points_inside_superellipse(emissions,
                                                                              a=self.thresholds[0], 
                                                                              b=self.thresholds[1], 
                                                                              n=self.cost_ellipse_n) == False)

        cost_a = self.compute_node_costs_actions(self.graph_nx, actions)
        dict_rewards = {"cost_emissions":cost_ems, "cost_actions": cost_a}

        total_cost = cost_ems + cost_a

        return -total_cost, dict_rewards
    

    def are_points_inside_superellipse(self, emissions, a=1, b=1.5, n=2.5):

        points = np.asarray(emissions)
        x = np.maximum(0, points[:, 0])  # Set negative x values to zero
        y = np.maximum(0, points[:, 1])  # Set negative y values to zero
        inside = (x / a)**n + (y / b)**n <= 1
        return inside

    def compute_price_renewal(self, components):
        price_renewal = np.zeros(self.n_nodes_graph)

        for component in components:
            n_nodes_component = len(component)
            cost_per_meter = self.variable_cost_renewal / (np.power(n_nodes_component, self.rate_of_decay_renewal)) + self.fixed_cost_renewal
            price_renewal[list(component)] += cost_per_meter * self.node_length
        
        return price_renewal

    def compute_node_costs_actions(self, G, actions):
        
        # Get the nodes for each action
        nodes_a = []
        for a in range(self.n_actions):
            nodes_a.append([node for node, action in enumerate(actions) if action == a])

        # Find connected components for each action
        subgraph_a = [G.subgraph(nodes) for nodes in nodes_a]
        components_a = [list(nx.connected_components(subgraph)) for subgraph in subgraph_a]
        
        price_tamping = self.compute_price_fun(components_a[2], var=self.variable_cost_tamping, fix=self.fixed_cost_tamping, nu=self.rate_of_decay_tamping)
        price_renewal = self.compute_price_fun(components_a[3], var=self.variable_cost_renewal, fix=self.fixed_cost_renewal, nu=self.rate_of_decay_renewal)

        # SPEED: round up to digits
        N_DIGITS = 4
        price_tamping = np.round(price_tamping, N_DIGITS)
        price_renewal = np.round(price_renewal, N_DIGITS)

        return price_tamping + price_renewal

    def compute_price_fun(self, components, var, fix, nu):
         
        costs = np.zeros(self.n_nodes_graph)

        for component in components:
            n_nodes_component = len(component)
            cost_per_meter = var / (np.power(n_nodes_component, nu)) + fix
            costs[list(component)] += cost_per_meter * self.node_length
        
        return costs

    def get_final_cov_matrix(self, actions):
        
        input_gp = np.stack([
            np.arange(self.n_nodes_graph).tolist() * 2, 
            actions.tolist() * 2,
            [0] * self.n_nodes_graph + [1] * self.n_nodes_graph
        ]).T

        cov_matrix_t = self.cov(input_gp)
        reduce_var_coef = 2.5
        cov_matrix_t = cov_matrix_t / reduce_var_coef

        diag_t = np.eye(self.n_nodes_graph * 2) * np.diag(cov_matrix_t)
        effect_action_t_d1 = self.args_model["lambda_act_d1"][actions]
        effect_action_t_d2 = self.args_model["lambda_act_d2"][actions]
        effect_action_t = np.concatenate([effect_action_t_d1, effect_action_t_d2], axis=0)

        effect_ind_t = self.args_model["lambda_ind"][input_gp[:,2]]

        diag_2_t = diag_t * effect_action_t + diag_t * effect_ind_t
        final_cov_matrix_t = (cov_matrix_t + diag_2_t)

        return final_cov_matrix_t
    

    def sample_emissions(self, prev_emissions, actions):

        final_cov_matrix_t = self.get_final_cov_matrix(actions)

        k_d1 = self.args_model["k_d1"]
        mu_d1 = self.args_model["mu_d1"]
        mean_d1 = prev_emissions[:, 0] * k_d1[actions] + mu_d1[actions]
                
        k_d2 = self.args_model["k_d2"]
        mu_d2 = self.args_model["mu_d2"]
        mean_d2 = prev_emissions[:, 1] * k_d2[actions] + mu_d2[actions]

        mean_t = np.concatenate([mean_d1, mean_d2], axis=0)

        # SPEED UP: Make sparse
        threshold = 0.00001
        final_cov_matrix_t[final_cov_matrix_t < threshold] = 0
        # Initialize a small value to add to the diagonal if needed
        epsilon = 1e-10
        max_retries = 10
        success = False

        for attempt in range(max_retries):
            try:
                # Try to generate the multivariate t-distribution samples
                obs_t = stats.multivariate_t.rvs(loc=mean_t, shape=final_cov_matrix_t, df=self.args_model["nu_d1"])
                emissions_1 = obs_t[:self.n_nodes_graph]
                emissions_2 = obs_t[self.n_nodes_graph:]
                success = True
                break

            except np.linalg.LinAlgError as e:
                print(f"An error occurred on attempt {attempt + 1}: {e}")
                # Add a small value to the diagonal to make the matrix more stable
                np.fill_diagonal(final_cov_matrix_t, np.diag(final_cov_matrix_t) + epsilon)
                epsilon *= 10  # Increase epsilon for the next iteration

        if not success:
            print("Failed to generate samples after multiple attempts")
            raise RuntimeError("Unable to generate multivariate samples due to numerical instability.")

        return np.array([emissions_1, emissions_2]).T

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)