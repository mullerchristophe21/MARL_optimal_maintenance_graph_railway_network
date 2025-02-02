import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import os


def lv95_to_lat_long(lv95_x, lv95_y):
    import pyproj
    return pyproj.Transformer.from_crs("epsg:2056", "epsg:4326").transform(lv95_y, lv95_x)

def returns_only_long_enough_gtg(all_positions_df, limit=150):
    all_positions = all_positions_df.drop_duplicates()
    temp = all_positions.groupby("gtg_neid").agg({"gtg_position":(lambda x: x.max() - x.min())})
    valid_gtg = temp[temp["gtg_position"] >= limit].index
    return valid_gtg.values

def create_bins_faster(all_positions, len_bin=150):
    
    all_gtgs = all_positions["gtg_neid"].unique()

    # Create groups for vectorized operations
    grouped = all_positions.groupby('gtg_neid')

    bins = []
    for gtg in tqdm(all_gtgs, desc="Creating bins"):
        df_gtg = grouped.get_group(gtg)

        # Vectorized min, max, and initial mean calculations
        min_pos, max_pos = df_gtg["gtg_position"].min(), df_gtg["gtg_position"].max()
        mean_x, mean_y = df_gtg[["LV95_x", "LV95_y"]].mean().values  # As a NumPy array

        if max_pos - min_pos < len_bin:
            bins.append({"gtg_neid": gtg, "gtg_position_s": min_pos, "gtg_position_e": max_pos, "mean_x": mean_x, "mean_y": mean_y})
            continue

        # Efficient bin generation with NumPy
        bin_starts = np.arange(min_pos, max_pos, len_bin)
        bin_starts = bin_starts[:-1]  # Consider last bin separately
        bin_ends = np.minimum(bin_starts + len_bin, max_pos)

        # Mask creation for filtering within each bin
        masks = [(df_gtg['gtg_position'] >= start) & (df_gtg['gtg_position'] < end) for start, end in zip(bin_starts, bin_ends)]

        # Vectorized mean calculations within each bin
        bin_means = [df_gtg.loc[mask, ["LV95_x", "LV95_y"]].mean().values for mask in masks]  # List of [mean_x, mean_y] arrays

        # Build bin dictionaries efficiently
        bins.extend([{"gtg_neid": gtg, "gtg_position_s": start, "gtg_position_e": end, "mean_x": means[0], "mean_y": means[1]}
                     for start, end, means in zip(bin_starts, bin_ends, bin_means)])
        
        # Handle the last partial bin if needed (same logic as before)
        last_end = bin_starts[-1] + len_bin
        if max_pos - last_end < (len_bin/2):
            mean_x = df_gtg[(df_gtg["gtg_position"] >= bins[-1]["gtg_position_s"]) & (df_gtg["gtg_position"] < max_pos)]["LV95_x"].mean()
            mean_y = df_gtg[(df_gtg["gtg_position"] >= bins[-1]["gtg_position_s"]) & (df_gtg["gtg_position"] < max_pos)]["LV95_y"].mean()
            bins[-1]["gtg_position_e"] = max_pos
            bins[-1]["mean_x"] = mean_x
            bins[-1]["mean_y"] = mean_y
        else:
            mean_x = df_gtg[(df_gtg["gtg_position"] >= last_end) & (df_gtg["gtg_position"] < max_pos)]["LV95_x"].mean()
            mean_y = df_gtg[(df_gtg["gtg_position"] >= last_end) & (df_gtg["gtg_position"] < max_pos)]["LV95_y"].mean()
            bins.append({"gtg_neid": gtg, "gtg_position_s": last_end, "gtg_position_e": max_pos, "mean_x": mean_x, "mean_y": mean_y})

    return pd.DataFrame(bins)

def create_bins_faster_new(all_positions, len_bin=150):
    all_gtgs = all_positions["gtg_neid"].unique()

    # Create groups for vectorized operations
    grouped = all_positions.groupby('gtg_neid')

    bins = []
    for gtg in tqdm(all_gtgs, desc="Creating bins"):
        df_gtg = grouped.get_group(gtg)

        # Vectorized min, max, and initial median calculations
        min_pos, max_pos = df_gtg["gtg_position"].min(), df_gtg["gtg_position"].max()

        if max_pos - min_pos < len_bin:
            median_pos = df_gtg["gtg_position"].median()
            median_row = df_gtg.loc[(df_gtg["gtg_position"] - median_pos).abs().idxmin()]
            bins.append({"gtg_neid": gtg, "gtg_position_s": min_pos, "gtg_position_e": max_pos, "mean_x": median_row["LV95_x"], "mean_y": median_row["LV95_y"]})
            continue

        # Efficient bin generation with NumPy
        bin_starts = np.arange(min_pos, max_pos, len_bin)
        bin_starts = bin_starts[:-1]  # Consider last bin separately
        bin_ends = np.minimum(bin_starts + len_bin, max_pos)

        for start, end in zip(bin_starts, bin_ends):
            mask = (df_gtg['gtg_position'] >= start) & (df_gtg['gtg_position'] < end)
            df_bin = df_gtg.loc[mask]
            if df_bin.empty:
                continue
            median_pos = df_bin["gtg_position"].median()
            median_row = df_bin.loc[(df_bin["gtg_position"] - median_pos).abs().idxmin()]
            bins.append({"gtg_neid": gtg, "gtg_position_s": start, "gtg_position_e": end, "mean_x": median_row["LV95_x"], "mean_y": median_row["LV95_y"]})
        
        # Handle the last partial bin if needed
        last_end = bin_starts[-1] + len_bin
        if max_pos - last_end < (len_bin / 2):
            mask = (df_gtg["gtg_position"] >= bins[-1]["gtg_position_s"]) & (df_gtg["gtg_position"] < max_pos)
            df_bin = df_gtg.loc[mask]
            if not df_bin.empty:
                median_pos = df_bin["gtg_position"].median()
                median_row = df_bin.loc[(df_bin["gtg_position"] - median_pos).abs().idxmin()]
                bins[-1]["gtg_position_e"] = max_pos
                bins[-1]["mean_x"] = median_row["LV95_x"]
                bins[-1]["mean_y"] = median_row["LV95_y"]
        else:
            mask = (df_gtg["gtg_position"] >= last_end) & (df_gtg["gtg_position"] < max_pos)
            df_bin = df_gtg.loc[mask]
            if not df_bin.empty:
                median_pos = df_bin["gtg_position"].median()
                median_row = df_bin.loc[(df_bin["gtg_position"] - median_pos).abs().idxmin()]
                bins.append({"gtg_neid": gtg, "gtg_position_s": last_end, "gtg_position_e": max_pos, "mean_x": median_row["LV95_x"], "mean_y": median_row["LV95_y"]})

    return pd.DataFrame(bins)



def create_bins_df(global_path, lines_list, limit_gtg_length=1, len_bin=150, new=False):

    # load all_positions:
    usecols = ["LV95_x","LV95_y","gtg_neid","gtg_position", "line", "line-km", "date_of_measurement"]

    dfs_pos = []
    for line in tqdm(lines_list, desc="Loading positions"):
        file = f"{line}.c000.csv"
        df = pd.read_csv(global_path + "/" + file, sep=";", usecols=usecols)
        dfs_pos.append(df)

    all_positions = pd.concat(dfs_pos)

    # gtg long enough:
    gtg_long_enough = returns_only_long_enough_gtg(all_positions, limit_gtg_length)

    # filter all_position on gtg long enough:
    all_positions = all_positions[all_positions["gtg_neid"].isin(gtg_long_enough)].copy()
    all_positions['count_dates'] = all_positions.groupby(usecols[:-1])['date_of_measurement'].transform('nunique')

    max_measurement_date = all_positions["date_of_measurement"].max()
    min_measurement_date = all_positions["date_of_measurement"].min()

    if new:
        bins_df = create_bins_faster_new(all_positions, len_bin)
    else:
        bins_df = create_bins_faster(all_positions, len_bin)
    bins_df = bins_df.rename_axis("node_id")


    return bins_df, (min_measurement_date, max_measurement_date)

def choose_dates(start_date, end_date, days_between_days=180):
    # Initialize a list to store the chosen dates
    chosen_dates = []

    # Convert start_date and end_date to datetime objects
    current_date = start_date
    end_datetime = end_date + timedelta(days=1)  # Add 1 day to include end_date

    # Increment the start_date by half of days_between_days
    # current_date += timedelta(days=days_between_days/2)

    # Iterate over the range of dates from start_date to end_date with step of days_between_days
    while current_date < end_datetime:
        chosen_dates.append(current_date)
        current_date += timedelta(days=days_between_days)

    return chosen_dates

def compute_statistics(df_filtered, list_dates, indicator):

    mean_indicator = []

    for i in range(len(list_dates)):
        from_date = list_dates[i]
        if i == len(list_dates) - 1:
            to_date = list_dates[i] + timedelta(days=1000)
        else:
            to_date = list_dates[i + 1]

        df_filtered_date = df_filtered[(df_filtered["date_of_measurement"] >= from_date) & (df_filtered["date_of_measurement"] < to_date)]
        if len(df_filtered_date) > 0:
            mean_indicator.append(df_filtered_date[indicator].mean())
        else:
            mean_indicator.append(np.nan)
        
    return mean_indicator

def returns_ts_per_node(indicator, global_path, bins_df, list_dates, lines):

    # Load all files together
    all_files = [global_path + "/" +f"{line}.c000.csv" for line in lines]
    cols_of_interest = [indicator, "date_of_measurement", "gtg_neid", "gtg_position"]
    df = pd.DataFrame()
    dfs = []

    # Read and preprocess each file
    for file in all_files:
        df_i = pd.read_csv(file, sep=";", usecols=cols_of_interest)
        df_i = df_i.dropna()
        df_i = df_i.sort_values(["date_of_measurement", "gtg_neid", "gtg_position"])
        df_i["date_of_measurement"] = pd.to_datetime(df_i["date_of_measurement"], format='%Y%m%d').dt.date
        dfs.append(df_i)

    # Concatenate all dataframes
    df = pd.concat(dfs)

    time_series_df = pd.DataFrame(index=bins_df.index, columns=list_dates)
    for i in tqdm(range(len(bins_df)), desc=f"Time Series of {indicator}"):

        row = bins_df.loc[i]

        gtg_row = row.loc["gtg_neid"]
        from_row = row.loc["gtg_position_s"]
        to_row = row.loc["gtg_position_e"]

        df_row = df[(df["gtg_neid"] == gtg_row) & (df["gtg_position"] >= from_row) & (df["gtg_position"] < to_row)]
        

        time_series_df.loc[i,:] = compute_statistics(df_row[["date_of_measurement", indicator]].reset_index(drop=True), list_dates, indicator=indicator)

    return time_series_df

def load_actions_df(lines, path_action):

    usecols = ["gtg_neid", "gtg_position", "line", "line-km", "date", "maintenance_dfa"]

    dfs = []
    all_files_of_path_action = os.listdir(path_action)
    for line in tqdm(lines, desc="Loading actions"):
        file = f"{line}.c000.csv"
        if file not in all_files_of_path_action:
            continue
        df = pd.read_csv(path_action + "/" + file, sep=";", usecols=usecols)
        df["date"] = pd.to_datetime(df["date"], format='%Y%m%d').dt.date

        dfs.append(df)
    all_acts = pd.concat(dfs)

    return all_acts

def compute_action_faster(df_r, dates_list):
    actions = []
    for i in range(len(dates_list) - 1): 
        from_date = dates_list[i]
        to_date = dates_list[i + 1]
        df_filtered = df_r.loc[(df_r["date"] >= from_date) & (df_r["date"] < to_date)]
        actions.append(df_filtered["maintenance_dfa"].unique() if not df_filtered.empty else 0)
    # Last iteration with extended to_date
    actions.append(df_r.loc[df_r["date"] >= dates_list[-1], "maintenance_dfa"].unique() if not df_r.loc[df_r["date"] >= dates_list[-1]].empty else 0)
    return actions


def create_action_time_series_faster(act_df, dates_list, bins_df):
    # Precompute unique gtg_neid values from bins_df
    gtg_neids = bins_df["gtg_neid"].unique()

    # Filter act_df based on gtg_neids
    filtered_act_df = act_df[act_df["gtg_neid"].isin(gtg_neids)]

    time_series_df = pd.DataFrame(index=bins_df.index, columns=dates_list)
    with tqdm(total=len(bins_df), desc="Computing Action Time Series") as pbar: # Initialize tqdm here
        for _, row in bins_df.iterrows():
            df_row = filtered_act_df.loc[
                (filtered_act_df["gtg_neid"] == row["gtg_neid"])
                & (filtered_act_df["gtg_position"] >= row["gtg_position_s"])
                & (filtered_act_df["gtg_position"] < row["gtg_position_e"])
            ]
            time_series_df.loc[row.name] = compute_action_faster(df_row, dates_list)
            pbar.update(1)

    return time_series_df

def create_dict_act(acts_df):

    all_actions = acts_df["maintenance_dfa"].unique()
    dict_actions = {"Other":[], "Re-tamping":[], "Tamping":[], "Renewal":[]}
    for act in all_actions:
        if "A" in act:
            dict_actions["Renewal"].append(act)
        elif (act == "R2") or (act == "R4") :
            dict_actions["Tamping"].append(act)
        elif (act == "R1") or (act == "R41"):
            dict_actions["Re-tamping"].append(act)
        else:
            dict_actions["Other"].append(act)
        
    return dict_actions

def process_actions_ts(actions_ts, dict_act_of_interest=None, acts_df=None):
    
    if dict_act_of_interest is None:
        dict_act_of_interest = create_dict_act(acts_df)
    actions_ts = actions_ts.copy()
    for i in range(len(actions_ts)):
        for j in range(len(actions_ts.columns)):
            if str(actions_ts.iloc[i,j]) != str(0):
                acts = []
                for act in actions_ts.iloc[i,j]:
                    if act in dict_act_of_interest["Re-tamping"]: 
                        acts.append(1)
                    elif act in dict_act_of_interest["Tamping"]:
                        acts.append(2)
                    elif act in dict_act_of_interest["Renewal"]: 
                        acts.append(3)
                    else:
                        acts.append(0)
                actions_ts.iloc[i,j] = np.max(acts) # If there are multiple actions, take the most severe one
            else:
                actions_ts.iloc[i,j] = 0
    
    return actions_ts

def plot_indicator_ts(idx, ts_indicator, actions_ts, list_dates):
    plt.figure(figsize=(12,4))
    plt.plot(ts_indicator.iloc[idx], label="indicator")

    max_ind = np.max(ts_indicator.iloc[idx])
    for i in range(len(actions_ts.iloc[idx])):
        if actions_ts.iloc[idx,i] != 0:
            plt.axvline(x=list_dates[i], color="red", linestyle="--")
            plt.text(list_dates[i],max_ind, actions_ts.iloc[idx,i], rotation=0)

def action_on_delta_ind_per_bin(agg_indicator_per_bin_and_time, agg_action_per_bin_and_time):
    from tqdm import tqdm
    
    delta_ind_per_bin = agg_indicator_per_bin_and_time.diff(axis=1)
    
    all_actions = agg_action_per_bin_and_time.values.flatten()
    all_actions = [act for act in all_actions if act !=0]

    all_actions_types = np.unique(all_actions)
    print("All actions considered: ", all_actions_types)

    # progress bar
    pbar = tqdm(total=len(all_actions_types) * len(agg_action_per_bin_and_time))

    all_inference = {}
    for action in all_actions_types:
        inference_data_action = {-2:[], -1:[], -0:[], 1:[], 2:[], 3:[]}
        for i, row in agg_action_per_bin_and_time.iterrows():
            pbar.update(1)
            for j, act in enumerate(row):
                if act != 0:
                    if action == act:
                        for k in range(-2,4):
                            if j+k >= 0 and j+k < len(row):
                                inference_data_action[k].append(delta_ind_per_bin.iloc[i,j+k])
                pass
        all_inference[action] = inference_data_action
    pbar.close()
    
    return all_inference

def plot_action_on_delta_ind(dict_act_delta, A=0.90, B=0.50, size_per_row=6):

    num_plots = len(dict_act_delta.keys())
    cols = 2
    rows = (num_plots // cols) + (num_plots % cols > 0) 
    fig, axes = plt.subplots(rows, cols, figsize=(12, size_per_row * rows))

    for i, (action, delta_values) in enumerate(dict_act_delta.items()):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        line_A_upper = []
        line_A_lower = []
        line_B_upper = []
        line_B_lower = []
        line_mean = []

        plotted_values = []
        n_samples_per_action = 0
        for key in delta_values.keys():
            list_all_values = delta_values[key]
            list_ = [x for x in list_all_values if str(x) != 'nan']
            n_samples_per_action += len(list_)
            if len(list_) > 0:
                plotted_values.append(key)

                line_A_upper.append(np.percentile(list_, (1-A)/2*100))
                line_A_lower.append(np.percentile(list_, (0.5+A/2)*100))
                line_B_upper.append(np.percentile(list_, (1-B)/2*100))
                line_B_lower.append(np.percentile(list_, (0.5+B/2)*100))
                line_mean.append(np.mean(list_))
        
        ax.scatter(plotted_values, line_mean, label="Mean") 
        ax.fill_between(plotted_values, line_A_lower, line_A_upper, alpha=0.3, label=f"{A} Density")
        ax.fill_between(plotted_values, line_B_lower, line_B_upper, alpha=0.3, label=f"{B} Density") 

        ax.hlines(0, -2, 3, color="black", linestyle="--")

        ax.set_title(f"Action: {action} - {n_samples_per_action} Samples")
        ax.set_xlabel("Delta (Time Steps)") 
        ax.set_ylabel("Value Distribution")
        ax.set_xlim(-2, 3)
        ax.legend()

def get_all_lines(path):
    import os
    all_files = os.listdir(path)
    all_lines = [file.split(".")[0] for file in all_files]
    return all_lines

def find_connections(endpoints_df, location_df, tree, threshold_km=0.1):
    connections = []
    threshold_deg = threshold_km / 111.7  # Approx conversion from km to degrees

    for idx in tqdm(range(endpoints_df.shape[0]), desc="Finding connections"):
        row = endpoints_df.iloc[idx]
        track, position, lat, lon, gtg_pos = row['gtg_neid'], row['position'], row['latitude'], row['longitude'], row['gtg_position']
        point = [lat, lon]
        
        # Query the tree for neighbors within the threshold
        indices = tree.query_ball_point(point, threshold_deg)
        
        closest_distance = float('inf')
        closest_connection = None
        
        for i in indices:
            other_row = location_df.iloc[i]
            other_track = other_row['gtg_neid']
            
            if track == other_track:
                continue

            other_point = [other_row['latitude'], other_row['longitude']]
            distance = np.linalg.norm(np.array(point) - np.array(other_point))
            
            if distance < closest_distance:
                closest_distance = distance
                closest_connection = {
                    'gtg_neid': track,
                    'position': position,
                    'latitude': lat,
                    'longitude': lon,
                    'gtg_position': gtg_pos,
                    'connected_to_track': other_track,
                    'connected_to_latitude': other_row['latitude'],
                    'connected_to_longitude': other_row['longitude'],
                    'connected_to_gtg_pos': other_row['gtg_position'],
                }

        if closest_connection:
            connections.append(closest_connection)
            
    return pd.DataFrame(connections)

def find_endpoints(df):
    # Group by 'gtg_neid'
    grouped = df.groupby('gtg_neid')
    
    # Get the row with the smallest gtg_position for each group
    start_points = grouped.apply(lambda x: x.loc[x['gtg_position'].idxmin()]).reset_index(drop=True)
    start_points['position'] = 'start'
    
    # Get the row with the largest gtg_position for each group
    end_points = grouped.apply(lambda x: x.loc[x['gtg_position'].idxmax()]).reset_index(drop=True)
    end_points['position'] = 'end'
    
    # Combine the start and end points into one DataFrame
    endpoints_df = pd.concat([start_points, end_points], ignore_index=True)
    
    return endpoints_df



def exponential_decay_weights(distances, decay_rate=0.5):
    """Calculate exponential decay weights based on the distances."""
    return np.exp(-decay_rate * distances)

def smooth_with_hard_stops(indic_ts, acts_ts, decay_rate=0.5):
    # Ensure the input dataframes have the same shape
    assert indic_ts.shape == acts_ts.shape, "The dataframes must have the same shape"
    
    # Copy the original indic_ts to create a new dataframe
    new_indic_ts = indic_ts.copy()
    
    # Iterate over each node (row) to process independently
    for node_idx in range(indic_ts.shape[0]):
        actions = acts_ts.iloc[node_idx].values
        indic_values = indic_ts.iloc[node_idx].values
        
        # Identify bins where actions are zero
        zero_bins = []
        current_bin = []
        for t in range(len(actions)):
            if actions[t] == 0:
                current_bin.append(t)
            else:
                if current_bin:
                    zero_bins.append(current_bin)
                    current_bin = []
        if current_bin:
            zero_bins.append(current_bin)
        
        # Calculate and update weighted averages within each bin
        for bin in zero_bins:
            bin_values = indic_values[bin]
            
            for i, t in enumerate(bin):
                # Calculate distances from the current timestep t to all other timesteps in the bin
                distances = np.abs(np.array(bin) - t)
                valid_mask = ~pd.isna(bin_values)
                if valid_mask.sum() == 0:
                    continue  # Skip if all values are NaN
                
                valid_distances = distances[valid_mask]
                valid_values = bin_values[valid_mask]
                
                weights = exponential_decay_weights(valid_distances, decay_rate)
                weighted_avg = np.average(valid_values, weights=weights)
                new_indic_ts.iloc[node_idx, t] = weighted_avg
    
    return new_indic_ts

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity  # Import KernelDensity from sklearn

def analyze_action_effects(actions_ts, indicator_ts, indicator="indicator delta", xlim1=-2, xlim2=1.5, y_size=4):
    """
    Analyze and plot the effect of actions on indicator changes.
    
    Parameters:
    actions_ts (np.ndarray): Time series of actions.
    indicator_ts (np.ndarray): Time series of indicator values.
    
    Returns:
    None
    """
    # Prepare the action time series
    traj_action_train = actions_ts.copy()
    
    # Indicator time series
    traj_fractal_train = indicator_ts.copy()
    
    # Initialize a dictionary to store differences for each action
    differences = {0: [], 1: [], 2: [], 3: []}
    
    # Iterate through each node (trajectory)
    for i in range(traj_action_train.shape[0]):
        for t in range(1, traj_action_train.shape[1]):
            action = traj_action_train[i, t-1]
            fractal_diff = traj_fractal_train[i, t] - traj_fractal_train[i, t-1]
            differences[action].append(fractal_diff)
    
    # Convert lists to numpy arrays for easier statistical computation
    for action in differences:
        differences[action] = np.array(differences[action])
    
    # Calculate statistics for each action
    stats = {}
    for action in [0, 1, 2, 3]:
        if len(differences[action]) == 0:
            continue
        differences[action] = differences[action][~np.isnan(differences[action])]
        mean_diff = np.mean(differences[action])
        median_diff = np.median(differences[action])
        lower_quantile = np.percentile(differences[action], 25)
        upper_quantile = np.percentile(differences[action], 75)
        stats[action] = (mean_diff, median_diff, lower_quantile, upper_quantile)
    
    # Print the statistics
    for action in stats:
        print(f"Count = {len(differences[action])}, Action {action}: Mean = {stats[action][0]}, Median = {stats[action][1]}, 25th Percentile = {stats[action][2]}, 75th Percentile = {stats[action][3]}")
    
    # Plotting the results using KDE
    # Plotting the results using KDE (more general approach)
    plt.figure(figsize=(10, y_size))
    colors = sns.color_palette("deep", 4)
    
    # Choose a kernel (e.g., 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine')
    kernel = 'epanechnikov' 
    # Adjust bandwidth (smoothing parameter) for each action if needed
    bandwidths = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3}  # Example 

    for action in [0, 1, 2, 3]:
        if len(differences[action]) == 0:
            continue
        if len(differences[action]) == 1:
            plt.axvline(stats[action][1], color=colors[action], linestyle='-', linewidth=1.5, label=f'Median (Action {action})')
            continue

        # Kernel Density Estimation using sklearn
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidths.get(action, 0.2))  # Use default if not specified
        kde.fit(differences[action][:, np.newaxis])  # Reshape for sklearn
        x_range = np.linspace(min(differences[action]), max(differences[action]), 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x_range)  # Log density for numerical stability
        kde_values = np.exp(log_dens)
        
        # Plot the KDE
        sns.lineplot(x=x_range[:, 0], y=kde_values, label=f'Action {action}', color=colors[action])
        plt.fill_between(x_range[:, 0], kde_values, color=colors[action], alpha=0.3)

        # Mean and median lines (similar to before)
        mean_height = kde.score_samples(stats[action][0][np.newaxis, np.newaxis])[0]
        median_height = kde.score_samples(stats[action][1][np.newaxis, np.newaxis])[0]
        plt.vlines(stats[action][0], ymin=0, ymax=np.exp(mean_height), color=colors[action], linestyle='--', linewidth=1.5, label=f'Mean (Action {action})')
        plt.vlines(stats[action][1], ymin=0, ymax=np.exp(median_height), color=colors[action], linestyle='-', linewidth=1.5, label=f'Median (Action {action})')

        # # Add rug plot for actions 1 and 2
        # if action in [1, 2,3]:
        #     # Reduce rug density for clarity, if necessary
        #     sns.rugplot(differences[action], height=0.05, color=colors[action], alpha=0.7)

    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--')
    
    # Set the x-axis limits
    plt.xlim(xlim1, xlim2)  

    plt.xlabel('Change in Indicator')
    plt.ylabel('Density')
    plt.title(f'Density of {indicator} After Each Action')
    plt.legend()
    plt.tight_layout()
    plt.show()


def exponential_decay_weights(distances, decay_rate=0.5):
    """Calculate exponential decay weights based on the distances."""
    return np.exp(-decay_rate * distances)

def smooth_with_hard_stops(indic_ts, acts_ts, decay_rate=0.5):
    # Ensure the input dataframes have the same shape
    assert indic_ts.shape == acts_ts.shape, "The dataframes must have the same shape"
    
    # Copy the original indic_ts to create a new dataframe
    new_indic_ts = indic_ts.copy()
    
    # Iterate over each node (row) to process independently
    for node_idx in range(indic_ts.shape[0]):
        actions = acts_ts.iloc[node_idx].values
        indic_values = indic_ts.iloc[node_idx].values
        
        # Identify bins where actions are zero
        zero_bins = []
        current_bin = []
        for t in range(len(actions)):
            if actions[t] == 0:
                current_bin.append(t)
            else:
                if current_bin:
                    zero_bins.append(current_bin)
                    current_bin = []
        if current_bin:
            zero_bins.append(current_bin)
        
        # Calculate and update weighted averages within each bin
        for bin in zero_bins:
            bin_values = indic_values[bin]
            
            for i, t in enumerate(bin):
                # Calculate distances from the current timestep t to all other timesteps in the bin
                distances = np.abs(np.array(bin) - t)
                valid_mask = ~pd.isna(bin_values)
                if valid_mask.sum() == 0:
                    continue  # Skip if all values are NaN
                
                valid_distances = distances[valid_mask]
                valid_values = bin_values[valid_mask]
                
                weights = exponential_decay_weights(valid_distances, decay_rate)
                weighted_avg = np.average(valid_values, weights=weights)
                new_indic_ts.iloc[node_idx, t] = weighted_avg
    
    return new_indic_ts

def analyze_emission_vs_delta(actions_ts, indicator_ts, indicator="indicator delta", frac_smooth=0.3):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    """
    Analyze and plot the relationship between emissions before change and the change in emissions.
    
    Parameters:
    actions_ts (np.ndarray): Time series of actions.
    indicator_ts (np.ndarray): Time series of indicator values.
    
    Returns:
    dict: Dictionary containing the linear model parameters for each action.
    """
    # Prepare the action time series
    traj_action_train = actions_ts.copy()
    traj_action_train[traj_action_train > 2] = 0
    
    # Indicator time series
    traj_indicator_train = indicator_ts.copy()
    
    # Initialize a dictionary to store emissions and deltas for each action
    emissions_before_change = {0: [], 1: [], 2: []}
    deltas = {0: [], 1: [], 2: []}
    
    # Iterate through each trajectory
    for i in range(traj_action_train.shape[0]):
        for t in range(1, traj_action_train.shape[1]):
            action = traj_action_train[i, t-1]
            emission_before = traj_indicator_train[i, t-1]
            delta = traj_indicator_train[i, t] - emission_before
            emissions_before_change[action].append(emission_before)
            deltas[action].append(delta)
    
    # Convert lists to numpy arrays for easier statistical computation
    for action in emissions_before_change:
        emissions_before_change[action] = np.array(emissions_before_change[action])
        deltas[action] = np.array(deltas[action])
    
    # Plotting the results
    plt.figure(figsize=(18, 6))
    
    # Define a color palette
    colors = sns.color_palette("deep", 3)
    
    # Create separate subplots for each action
    for i, action in enumerate([0, 1, 2]):
        if len(emissions_before_change[action]) == 0:
            continue
        
        plt.subplot(1, 3, i + 1)
        plt.scatter(emissions_before_change[action], deltas[action], alpha=0.6, color=colors[action])
        
        # Apply LOWESS smoothing
        smoothed = lowess(deltas[action], emissions_before_change[action], frac=frac_smooth)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='black', linestyle='--')
        
        plt.xlabel('Emissions Before Change')
        plt.ylabel('Change in Emissions')
        plt.title(f'Action {action}')
    
    plt.suptitle(f'Relationship between Emissions Before Change and {indicator}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
