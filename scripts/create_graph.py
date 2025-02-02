from data_graph_module import Graph
import sys
import os

path = sys.argv[7]
path_action = sys.argv[8]

#####################################################################################################
print("\nChoose indicators and lines: ")
####

lines_str = sys.argv[1].split(",")
if lines_str == ["all"]:
    from data_graph_utils import get_all_lines
    lines = get_all_lines(path=path)
    print("\tLines selected: ", "ALL")
else:
    lines = [int(i) for i in lines_str]
    print("\tLines selected: ", lines)

indicators_str = sys.argv[2].split(",")
print("\tIndicators selected: ", indicators_str)

limit_gtg_length = int(sys.argv[3])
print("\tMinimum_track_length: ", limit_gtg_length)

len_bin = int(sys.argv[4])
print("\tTrack_segment_length: ", len_bin)

threshold_connection_m = float(sys.argv[5])
print("\tMax_distance_connecting_nodes: ", threshold_connection_m)
threshold_connection_km = threshold_connection_m / 1000
threshold_connection_m = int(threshold_connection_m)

graph_path = sys.argv[6]
if graph_path == "auto":
    indicator_base = ["lonle_d1", "lonle_d2", "lonle_d1_std100m", "lonle_d2_std100m"]
    indicator_not_in_base = [i for i in indicators_str if i not in indicator_base]
    if len(indicator_not_in_base) == 0:
        indicator_not_in_base = ["BASE"]
    lines_one_str = "".join([str(i) for i in lines])
    if lines_str == ["all"]:
        lines_one_str = "ALL"
    indicators_one_str = "".join([i for i in indicator_not_in_base])
    graph_path = f"graph__{lines_one_str}lines__{indicators_one_str}indicators__{limit_gtg_length}mintracklength_{len_bin}segmentlength_{threshold_connection_m}thresholdconnection.pkl"

if not graph_path.endswith(".pkl"):
    graph_path += ".pkl"

graph_path = os.path.join("storage_graph", graph_path)
print("\tGraph name: ", graph_path)


#####################################################################################################
print("\nCreating graph...")
####

graph = Graph(data_directory=path, actions_directory=path_action, lines=lines, limit_gtg_length=limit_gtg_length, len_bin=len_bin, threshold_connection_km=threshold_connection_km)


#####################################################################################################
print("\nAdd indicators to graph...")
####

for indicator in indicators_str:
    graph.create_indicator_ts(indicator, save_it_in_class=True)

#####################################################################################################
print("\nSave graph...")
####

graph.save_graph(graph_path)