from data_graph_utils import *
import os
import json

class Graph:
    def __init__(self, data_directory, actions_directory, lines, date_period_days=180, limit_gtg_length=1, len_bin=150, threshold_connection_km=0.150, load_init=False):
        
        if load_init:
            self.data_directory = data_directory
            self.actions_directory = actions_directory
            self.lines = lines
            self.date_period_days = date_period_days
            self.limit_gtg_length = limit_gtg_length
            self.len_bin = len_bin
            self.threshold_connection_km = threshold_connection_km
            return
    
        self.len_bin = len_bin
        self.limit_gtg_length = limit_gtg_length
        self.threshold_connection_km = threshold_connection_km        

        self.directory = data_directory
        self.lines = lines
        self.date_period_days = date_period_days
        self.actions_directory = actions_directory

        self.bins_df, s_e_dates = create_bins_df(self.directory, self.lines, limit_gtg_length=limit_gtg_length, len_bin=len_bin)
        self.day_s_date = datetime.strptime(str(int(s_e_dates[0])), '%Y%m%d').date()
        self.day_e_date = datetime.strptime(str(int(s_e_dates[1])), '%Y%m%d').date()

        self.list_dates = choose_dates(self.day_s_date, self.day_e_date, date_period_days)

        actions_df = load_actions_df(self.lines, self.actions_directory)
        full_actions_ts = create_action_time_series_faster(actions_df, self.list_dates, self.bins_df)
        self.actions_ts = process_actions_ts(full_actions_ts, acts_df=actions_df)
        
        self.indicator_ts_dict = {}
        self.endpoints_df, self.connections_df = self.get_endpoints_and_connections(threshold=threshold_connection_km)
        self.adj_matrix = self.get_adjacency_matrix()

        self.actions_ts_adjusted = None

        self.bins_df["lat"], self.bins_df["lon"] =lv95_to_lat_long(self.bins_df["mean_x"], self.bins_df["mean_y"])

    def compute_actions_ts(self):
        actions_df = load_actions_df(self.lines, self.actions_directory)
        full_actions_ts = create_action_time_series_faster(actions_df, self.list_dates, self.bins_df)
        self.actions_ts = process_actions_ts(full_actions_ts, acts_df=actions_df)

    def create_indicator_ts(self, indicator_name, save_it_in_class=True):
        ts_indicator = returns_ts_per_node(indicator_name, self.directory, self.bins_df, self.list_dates, self.lines)
        if save_it_in_class:
            self.save_indicator_ts_to_class(indicator_name, ts_indicator)
            
    def save_indicator_ts_to_class(self, indicator_name, ts_indicator):
        self.indicator_ts_dict[indicator_name] = ts_indicator

    def plot_indicator_ts(self, id_node, indicator_name, new_actions=False):
        if new_actions:
            actions_ts = self.actions_ts_adjusted
        else:
            actions_ts = self.actions_ts
        print(self.bins_df.loc[id_node])
        if indicator_name not in self.indicator_ts_dict.keys():
            print("Indicator ts not saved in this graph class")
        else:
            ts_indicator = self.indicator_ts_dict[indicator_name]
            plot_indicator_ts(id_node, ts_indicator, actions_ts, self.list_dates)

    def return_available_columns(self):
        files_in_directory = os.listdir(self.directory)
        df = pd.read_csv(self.directory + "/"+files_in_directory[0], sep=";", nrows=1)
        return df.columns

    def return_available_indicators(self):
        return self.indicator_ts_dict.keys()
    
    def plot_and_return_effect_of_actions(self, indicator_name, A=0.90,B=0.50,size_per_row=6, return_dict=False):
        indicator_df = self.indicator_ts_dict[indicator_name]
        dict_ = action_on_delta_ind_per_bin(indicator_df, self.actions_ts)
        plot_action_on_delta_ind(dict_, A, B, size_per_row)
        if return_dict:
            return dict_

    def plot_folium_map(self, group_by="gtg_neid", all_lines=False, plot_connections=True, save_path=None):
        import folium
        from folium.plugins import MarkerCluster
        from matplotlib import colormaps as cms
        from matplotlib import colors

        lines = self.lines
        if all_lines:
            lines = get_all_lines(self.directory)

        df_locs = []
        cols_loc = ["LV95_x", "LV95_y", "gtg_neid", "gtg_position", "line", "line-km"]

        for line in tqdm(lines):
            df_i = pd.read_csv(f"{self.directory}/{line}.c000.csv", sep=";", usecols=cols_loc)
            df_locs.append(df_i)

        df_location = pd.concat(df_locs, axis=0)
        df_location = df_location.drop_duplicates()

        lat, long = lv95_to_lat_long(df_location["LV95_x"], df_location["LV95_y"])
        df_location["latitude"] = lat
        df_location["longitude"] = long

        switzerland_map = folium.Map(location=[47.3182, 8.7275], zoom_start=10, tiles='cartodbpositron')

        if group_by == "gtg_neid":
            df_location = df_location.sort_values(["gtg_neid", "gtg_position"])
        elif group_by == "line":
            df_location = df_location.sort_values(["line", "line-km"])
        else: 
            print("Choose a groubpy column: 'gtg_neid' or 'line'")

        def add_line_segments(map_obj, df, group_column, colormap):
            """Adds line segments with hover effect showing the group."""
            unique_groups = df[group_column].unique() 
            color_scale = colormap(np.linspace(0, 1, len(unique_groups))) 

            for group_name, group_data in tqdm(df.groupby(group_column)):
                locations = group_data[['latitude', 'longitude']].values.tolist()

                group_index = np.where(unique_groups == group_name)[0][0]
                color = colors.to_hex(color_scale[group_index])

                line = folium.PolyLine(locations, color=color, weight=2.5, opacity=1)
                tooltip = folium.Tooltip(f"Group: {group_name}")
                line.add_child(tooltip)

                line.add_to(map_obj)
        def add_connection_lines(map_obj, connections_df, location_df):
            # for idx, row in connections_df.iterrows():
            for idx in tqdm(connections_df.index):
                row = connections_df.loc[idx]
                start_lat = row['latitude']
                start_lon = row['longitude']
                end_lat = row['connected_to_latitude']
                end_lon = row['connected_to_longitude']

                line = folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)], color="red", weight=2.5, opacity=0.7)
                tooltip = folium.Tooltip(f"Connection: {row['gtg_neid']} -> {row['connected_to_track']}")
                line.add_child(tooltip)
                
                line.add_to(map_obj)

        colormap = cms.get_cmap('tab20')

        add_line_segments(switzerland_map, df_location, group_by, colormap)
        if plot_connections:
            add_connection_lines(switzerland_map, self.connections_df, df_location)

        if save_path is not None:
            try:
                switzerland_map.save(save_path)
                print(f"Map saved successfully to: {save_path}")
            except Exception as e:
                print(f"Error saving map: {e}")
        
        return switzerland_map

    def plot_folium_map_of_nodes_per_line(self, plot_connections=True, save_path=None):

        if "latitude" not in self.bins_df.columns:
            lat, lon = lv95_to_lat_long(self.bins_df["mean_x"], self.bins_df["mean_y"])
            self.bins_df["latitude"] = lat
            self.bins_df["longitude"] = lon    

        import folium
        from folium.plugins import MarkerCluster
        from matplotlib import colormaps as cms
        from matplotlib import colors

        switzerland_map = folium.Map(location=[47.3182, 8.7275], zoom_start=10, tiles='cartodbpositron')

        def add_line_segments_based_on_adjacency(map_obj, df, adj_matrix, colormap, plot_connection=True):
            """Adds line segments based on the adjacency matrix."""
            unique_groups = df['gtg_neid'].unique()
            color_scale = colormap(np.linspace(0, 1, len(unique_groups)))

            for i, row in df.iterrows():
                start_node_id = i
                start_location = [row['latitude'], row['longitude']]
                start_group = row['gtg_neid']
                group_index = np.where(unique_groups == start_group)[0][0]
                group_color = colors.to_hex(color_scale[group_index])

                connected_nodes = adj_matrix.loc[start_node_id]
                connections = 0  # Counter for connections

                for end_node_id, is_connected in connected_nodes.items():
                    if is_connected:
                        connections += 1
                        end_location = [df.at[end_node_id, 'latitude'], df.at[end_node_id, 'longitude']]
                        end_group = df.at[end_node_id, 'gtg_neid']

                        if start_group != end_group:
                            if plot_connection:
                                tooltip = folium.Tooltip(f"Connection: {start_group} -> {end_group}")
                                color = 'red'
                            else:
                                continue
                        else:
                            color = group_color
                            tooltip = folium.Tooltip(f"Group: {start_group}")

                        line = folium.PolyLine([start_location, end_location], color=color, weight=2.5, opacity=1)
                        line.add_child(tooltip)
                        line.add_to(map_obj)
                
                # If no connections were found, add a CircleMarker for the point
                if connections == 0:
                    folium.CircleMarker(
                        start_location,
                        radius=1, 
                        color=group_color,
                        fill=True,
                        fill_color=group_color,
                        tooltip=folium.Tooltip(f"Group: {start_group}")
                    ).add_to(map_obj)


        # Colormap
        colormap = cms.get_cmap('tab20')

        # Add the line segments to the map
        add_line_segments_based_on_adjacency(switzerland_map, self.bins_df, self.adj_matrix, colormap, plot_connections)

        if save_path is not None:
            try:
                switzerland_map.save(save_path)
                print(f"Map saved successfully to: {save_path}")
            except Exception as e:
                print(f"Error saving map: {e}")

        return switzerland_map
    
    def plot_folium_map_of_nodes(self, plot_connections=True, save_path=None, selected_nodes=None):
        if "latitude" not in self.bins_df.columns:
            lat, lon = lv95_to_lat_long(self.bins_df["mean_x"], self.bins_df["mean_y"])
            self.bins_df["latitude"] = lat
            self.bins_df["longitude"] = lon    

        import folium
        from folium.plugins import MarkerCluster
        from matplotlib import colormaps as cms
        from matplotlib import colors

        switzerland_map = folium.Map(location=[47.3182, 8.7275], zoom_start=10, tiles='cartodbpositron', max_zoom=40)

        if selected_nodes is None:
            return self.plot_folium_map_of_nodes_per_line(plot_connections=plot_connections, save_path=save_path)

        # Precompute and cache unique groups and their colors
        group_color_map = ["green" if i in selected_nodes else "red" for i, row in self.bins_df.iterrows()]

        def add_line_segments_based_on_adjacency(map_obj, df, adj_matrix, plot_connection=True):
            """Adds line segments based on the adjacency matrix."""
            added_endpoints = set()

            for i, row in df.iterrows():
                start_node_id = i
                start_location = [row['latitude'], row['longitude']]
                start_group = row['gtg_neid']
                group_color = group_color_map[i]

                connected_nodes = adj_matrix.loc[start_node_id]
                connections = 0  # Counter for connections

                # Add a CircleMarker for the start location
                folium.CircleMarker(
                    start_location,
                    radius=3,
                    color=group_color,
                    fill=True,
                    fill_color=group_color,
                    tooltip=folium.Tooltip(f"Group: {start_group}"),
                ).add_to(map_obj)

                for end_node_id, is_connected in connected_nodes.items():
                    if is_connected:
                        connections += 1
                        end_location = [df.at[end_node_id, 'latitude'], df.at[end_node_id, 'longitude']]
                        end_group = df.at[end_node_id, 'gtg_neid']

                        if plot_connection:
                            tooltip = folium.Tooltip(f"Connection: {start_group} -> {end_group}")
                            color = 'grey'
                            line = folium.PolyLine([start_location, end_location], color=color, weight=2.5, opacity=1)
                            line.add_child(tooltip)
                            line.add_to(map_obj)

                        # Add a CircleMarker for the end location if not already added
                        if end_node_id not in added_endpoints:
                            end_color = group_color_map[i]
                            folium.CircleMarker(
                                end_location,
                                radius=3,
                                color=end_color,
                                fill=True,
                                fill_color=end_color,
                                tooltip=folium.Tooltip(f"Group: {end_group}"),
                            ).add_to(map_obj)
                            added_endpoints.add(end_node_id)
                    
                # If no connections were found, ensure the start location still has a CircleMarker
                if connections == 0:
                    folium.CircleMarker(
                        start_location,
                        radius=3,
                        color=group_color,
                        fill=True,
                        fill_color=group_color,
                        tooltip=folium.Tooltip(f"Group: {start_group}"),
                    ).add_to(map_obj)

        # Add the line segments to the map
        add_line_segments_based_on_adjacency(switzerland_map, self.bins_df, self.adj_matrix, plot_connections)

        if save_path is not None:
            try:
                switzerland_map.save(save_path)
                print(f"Map saved successfully to: {save_path}")
            except Exception as e:
                print(f"Error saving map: {e}")

        return switzerland_map


    def get_endpoints_and_connections(self, threshold=0.05):
        dfs_pos = []
        usecols = ["gtg_neid", "gtg_position", "LV95_x", "LV95_y"]
        for line in tqdm(self.lines, desc="Files for connections"):
            file = f"{line}.c000.csv"
            df = pd.read_csv(self.directory + "/" + file, sep=";", usecols=usecols)
            dfs_pos.append(df)

        all_positions = pd.concat(dfs_pos)
        all_positions = all_positions.drop_duplicates()

        lat, lon = lv95_to_lat_long(all_positions["LV95_x"], all_positions["LV95_y"])
        all_positions["latitude"] = lat
        all_positions["longitude"] = lon

        endpoints_df = find_endpoints(all_positions)

        from scipy.spatial import cKDTree
        points = np.vstack((all_positions['latitude'], all_positions['longitude'])).T
        tree = cKDTree(points)

        connections_df = find_connections(endpoints_df, all_positions, tree, threshold_km=threshold)
        
        return endpoints_df, connections_df
    
    def get_adjacency_matrix(self):
        node_ids = self.bins_df.index
        adj_matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)

        # Add connections for nodes in the same gtg_neid
        for neid, group in self.bins_df.groupby('gtg_neid'):
            for i, node_i in group.iterrows():
                for j, node_j in group.iterrows():
                    if abs(node_i['gtg_position_e'] - node_j['gtg_position_s']) < 3:
                        adj_matrix.at[i, j] = 1

        # Add connections from connection_df
        for _, row in self.connections_df.iterrows():
            start_node = self.bins_df[(self.bins_df['gtg_neid'] == row['gtg_neid']) & 
                                (self.bins_df['gtg_position_s'] <= row['gtg_position']) & 
                                (self.bins_df['gtg_position_e'] >= row['gtg_position'])].index
            end_node = self.bins_df[(self.bins_df['gtg_neid'] == row['connected_to_track']) & 
                            (self.bins_df['gtg_position_s'] <= row['connected_to_gtg_pos']) & 
                            (self.bins_df['gtg_position_e'] >= row['connected_to_gtg_pos'])].index
            if not start_node.empty and not end_node.empty:
                adj_matrix.at[start_node[0], end_node[0]] = 1

        return adj_matrix
    
    def save_graph(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Graph saved successfully to: {path}")

    def select_subgraph_nodes(self, lat1, lat2, lon1, lon2, t_from, t_to):
        bins_df = self.bins_df
        bins_df["lat"], bins_df["lon"] = lv95_to_lat_long(bins_df["mean_x"], bins_df["mean_y"])

        selected_nodes = bins_df[(bins_df["lat"] > lat1) & (bins_df["lat"] < lat2) & (bins_df["lon"] > lon1) & (bins_df["lon"] < lon2)].index

        t_from_time = self.actions_ts.columns[t_from]
        t_to_time = self.actions_ts.columns[t_to-1]

        self.bins_df = self.bins_df.loc[selected_nodes].reset_index().rename(columns={"node_id":"old_node"})

        self.actions_ts = self.actions_ts.loc[selected_nodes, t_from_time:t_to_time].reset_index(drop=True)

        # if adjusted actions exists, then also filter it
        if self.actions_ts_adjusted is not None:
            self.actions_ts_adjusted = self.actions_ts_adjusted.loc[selected_nodes, t_from_time:t_to_time].reset_index(drop=True)

        for ind in self.indicator_ts_dict.keys():
            self.indicator_ts_dict[ind] = self.indicator_ts_dict[ind].loc[selected_nodes, t_from_time:t_to_time].reset_index(drop=True)

        self.adj_matrix = self.adj_matrix.loc[selected_nodes,selected_nodes].reset_index(drop=True)
        self.adj_matrix.columns = self.adj_matrix.index

    def adjust_actions_ts(self, indicator_name1, indicator_name2, COEF_before=5, COEF_var=10):
        
        # Ensure input DataFrames are copied to avoid modifying the original ones
        new_actions_ts = self.actions_ts.copy()
        actions_ts = self.actions_ts.copy()
        act_to_change = [2,3]

        indicator_ts1 = self.indicator_ts_dict[indicator_name1].copy()
        indicator_ts2 = self.indicator_ts_dict[indicator_name2].copy()

        count_n_changes = 0
        
        for node in actions_ts.index:

            variance_ts1 = indicator_ts1.iloc[node,:].var()
            variance_ts2 = indicator_ts2.iloc[node,:].var()

            for t in range(1, len(actions_ts.columns) - 1):

                action = actions_ts.iloc[node, t]
                
                if action in act_to_change:
                    if action == act_to_change[0]: # When looking at tamping: consider D1 indicator
                        indicator_ts = indicator_ts1
                        variance_ts = variance_ts1
                    else:                           # When looking at renew: consider D2 indicator
                        indicator_ts = indicator_ts2
                        variance_ts = variance_ts2

                    if actions_ts.iloc[node, t-1] == 0:

                        delta_before = indicator_ts.iloc[node, t-1] - indicator_ts.iloc[node, t]
                        delta_after = indicator_ts.iloc[node, t] - indicator_ts.iloc[node, t+1]

                        if delta_before > COEF_var * variance_ts and delta_before > COEF_before * delta_after and delta_before > 0:
                            
                            count_n_changes = count_n_changes + 1

                            new_actions_ts.iloc[node, t-1] = action
                            new_actions_ts.iloc[node, t] = 0  # Reset current timestep action to 0
        
        self.actions_ts_adjusted = new_actions_ts
        return count_n_changes


    def plot_folium_map_of_nodes_colorscale(self, list_nodes, metric, color_scale='viridis', save_path=None):
        if "latitude" not in self.bins_df.columns:
            lat, lon = lv95_to_lat_long(self.bins_df["mean_x"], self.bins_df["mean_y"])
            self.bins_df["latitude"] = lat
            self.bins_df["longitude"] = lon    

        import folium
        from folium.plugins import MarkerCluster
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        import numpy as np

        switzerland_map = folium.Map(location=[47.3182, 8.7275], zoom_start=10, tiles='cartodbpositron', max_zoom=40)

        if list_nodes is None:
            list_nodes = []

        def add_line_segments_based_on_adjacency(map_obj, df, adj_matrix, metric, colormap, plot_connection=True):
            """Adds line segments based on the adjacency matrix."""
            unique_groups = df['gtg_neid'].unique()
            added_endpoints = set()

            norm = colors.Normalize(vmin=metric.min(), vmax=metric.max())
            scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)

            for i, row in df.iterrows():
                start_node_id = i
                start_location = [row['latitude'], row['longitude']]
                metric_value = metric[start_node_id]
                group_color = scalar_map.to_rgba(metric_value, bytes=True)[:3]
                group_color = '#{:02x}{:02x}{:02x}'.format(*group_color)

                connected_nodes = adj_matrix.loc[start_node_id]
                connections = 0  # Counter for connections

                # Add a CircleMarker for the start location
                folium.CircleMarker(
                    start_location,
                    radius=3,
                    color=group_color,
                    fill=True,
                    fill_color=group_color,
                    tooltip=folium.Tooltip(f"Metric: {metric_value}"),
                ).add_to(map_obj)

                for end_node_id, is_connected in connected_nodes.items():
                    if is_connected:
                        connections += 1
                        end_location = [df.at[end_node_id, 'latitude'], df.at[end_node_id, 'longitude']]
                        end_metric_value = metric[end_node_id]
                        end_group_color = scalar_map.to_rgba(end_metric_value, bytes=True)[:3]
                        end_group_color = '#{:02x}{:02x}{:02x}'.format(*end_group_color)

                        if plot_connection:
                            tooltip = folium.Tooltip(f"Connection: {metric_value} -> {end_metric_value}")
                            color = 'grey'
                        else:
                            continue

                        line = folium.PolyLine([start_location, end_location], color=color, weight=2.5, opacity=1)
                        line.add_child(tooltip)
                        line.add_to(map_obj)

                        # Add a CircleMarker for the end location if not already added
                        if end_node_id not in added_endpoints:
                            folium.CircleMarker(
                                end_location,
                                radius=3,
                                color=end_group_color,
                                fill=True,
                                fill_color=end_group_color,
                                tooltip=folium.Tooltip(f"Metric: {end_metric_value}"),
                            ).add_to(map_obj)
                            added_endpoints.add(end_node_id)

                    # If no connections were found, ensure the start location still has a CircleMarker
                    if connections == 0:
                        folium.CircleMarker(
                            start_location,
                            radius=3,
                            color=group_color,
                            fill=True,
                            fill_color=group_color,
                            tooltip=folium.Tooltip(f"Metric: {metric_value}"),
                        ).add_to(map_obj)

        # Colormap
        colormap = cm.get_cmap(color_scale)

        # Add the line segments to the map
        add_line_segments_based_on_adjacency(switzerland_map, self.bins_df, self.adj_matrix.loc[list_nodes,list_nodes], metric, colormap, plot_connection=True)

        if save_path is not None:
            try:
                switzerland_map.save(save_path)
                print(f"Map saved successfully to: {save_path}")
            except Exception as e:
                print(f"Error saving map: {e}")

        return switzerland_map
    

    def save_graph_json(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save DataFrames as CSV
        self.bins_df.to_csv(os.path.join(save_directory, 'bins_df.csv'), index=True)
        self.actions_ts.to_csv(os.path.join(save_directory, 'actions_ts.csv'), index=True)
        self.endpoints_df.to_csv(os.path.join(save_directory, 'endpoints_df.csv'), index=True)
        self.connections_df.to_csv(os.path.join(save_directory, 'connections_df.csv'), index=True)
        self.adj_matrix.to_csv(os.path.join(save_directory, 'adj_matrix.csv'), index=True)
        
        # Save indicator_ts_dict
        for key, df in self.indicator_ts_dict.items():
            df.to_csv(os.path.join(save_directory, f'indicator_ts_{key}.csv'), index=True)
        
        # Save other attributes
        metadata = {
            'len_bin': self.len_bin,
            'limit_gtg_length': self.limit_gtg_length,
            'threshold_connection_km': self.threshold_connection_km,
            'directory': self.directory,
            'lines': self.lines,
            'date_period_days': self.date_period_days,
            'actions_directory': self.actions_directory,
            'day_s_date': self.day_s_date.isoformat(),
            'day_e_date': self.day_e_date.isoformat(),
            'list_dates': [d.isoformat() for d in self.list_dates]
        }
        
        with open(os.path.join(save_directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
    @staticmethod
    def load_graph_json(load_directory):
        # Load metadata
        with open(os.path.join(load_directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Recreate the object with basic attributes
        graph = Graph(
            data_directory=metadata['directory'],
            actions_directory=metadata['actions_directory'],
            lines=metadata['lines'],
            date_period_days=metadata['date_period_days'],
            limit_gtg_length=metadata['limit_gtg_length'],
            len_bin=metadata['len_bin'],
            threshold_connection_km=metadata['threshold_connection_km'],
            load_init=True
        )
        
        # Load DataFrames from CSV
        graph.bins_df = pd.read_csv(os.path.join(load_directory, 'bins_df.csv'), index_col=0)
        graph.actions_ts = pd.read_csv(os.path.join(load_directory, 'actions_ts.csv'), index_col=0)
        graph.endpoints_df = pd.read_csv(os.path.join(load_directory, 'endpoints_df.csv'), index_col=0)
        graph.connections_df = pd.read_csv(os.path.join(load_directory, 'connections_df.csv'), index_col=0)
        graph.adj_matrix = pd.read_csv(os.path.join(load_directory, 'adj_matrix.csv'), index_col=0)
        
        # Load indicator_ts_dict
        indicator_files = [f for f in os.listdir(load_directory) if f.startswith('indicator_ts_')]
        graph.indicator_ts_dict = {}
        for file in indicator_files:
            key = file.split('indicator_ts_')[1].split('.csv')[0]
            graph.indicator_ts_dict[key] = pd.read_csv(os.path.join(load_directory, file), index_col=0)
        
        # Load dates
        graph.day_s_date = datetime.fromisoformat(metadata['day_s_date']).date()
        graph.day_e_date = datetime.fromisoformat(metadata['day_e_date']).date()
        graph.list_dates = [datetime.fromisoformat(d).date() for d in metadata['list_dates']]
                
        return graph

    def load_new_bins_df(self):
        self.new_bins_df, _ = create_bins_df(self.directory, self.lines, limit_gtg_length=self.limit_gtg_length, len_bin=self.len_bin, new=True)
