from pathlib import Path

# Get the path to the current script (my_example.py)
current_file = Path(__file__).resolve()

# Go to project root (assumes "examples" is directly under root)
project_root = current_file.parent.parent

# Define path to results directory
src_dir = project_root / "src"
# src_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

results_dir = project_root / "results" / "Man157_HL3bypassed"
results_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists


import sys
import os
# Navigate relative to the current working directory
sys.path.append(os.path.abspath(src_dir))
import numpy as np
import pandas as pd
from sixdman.core.network import Network
from sixdman.core.band import Band, OpticalParameters
from sixdman.core.planning import PlanningTool
from sixdman.core.optical_result_analyzer import analyse_result
import json
from scipy.io import loadmat

import warnings
warnings.filterwarnings("ignore")

print('Create Network Instance: ...')

# Initialize network
network = Network(topology_name = 'MAN157')

# Load topology from .mat file
network.load_topology(filepath = project_root / 'data' / 'MAN157Nodes.mat', matrixName ='MAN157Nodes')

# Set hierarchical levels
hl_dict = network.define_hierarchy(
    HL1_standalone = [1, 5],
    HL2_standalone = [0, 2, 3, 4],
    HL3_standalone = list(range(6, 39)),
    HL4_standalone = list(range(6, 39)) + list(range(39, 157)), 
    HL4_colocated = [0, 2, 3, 4] + [1, 5]
)

HL4_Standalone = hl_dict['HL4']['standalone']
HL4_colocated = hl_dict['HL4']['colocated']

HL3_Standalone = hl_dict['HL3']['standalone']
HL3_colocated = hl_dict['HL3']['colocated']

HL2_Standalone = hl_dict['HL2']['standalone']
HL2_colocated = hl_dict['HL2']['colocated']

HL1_Standalone = hl_dict['HL1']['standalone']

HL123_Standalone = HL1_Standalone + HL2_Standalone + HL3_Standalone
HL12_Standalone = HL1_Standalone + HL2_Standalone

# Define C-band parameters
c_band_params = OpticalParameters()

# Create C-band instance
c_band = Band(
    name='C',
    start_freq = 190.65, # THz
    end_freq = 196.675, # THz
    opt_params = c_band_params,
    network_instance = network,
    channel_spacing = 0.05 # THz
    )

# Define L-band parameters
l_band_params = OpticalParameters()

# Create L-band instance
l_band = Band(
    name='L',
    start_freq = 184.525, # THz
    end_freq = 190.565, # THz
    opt_params = l_band_params,
    network_instance = network,
    channel_spacing = 0.05 # THz
)

# define C-band and L-band frequency slots
spectrum_C = c_band.calc_spectrum()
spectrum_L = l_band.calc_spectrum()

# concatenate C-band and KL-band to a sigle frequency spectrum
spectrum = np.concatenate((spectrum_C, spectrum_L))

# define total number of frequency slots
num_fslots = 240

f_c_axis = spectrum * 1e12  # Convert to Hz
Pch_dBm = np.arange(-6, -0.9, 0.1)  # Channel power in dBm
num_Ch_mat = np.arange(1, len(spectrum) - 1)  # Channel indices

# Initialize planning tool
planner = PlanningTool(
    network_instance = network,
    bands = [c_band, l_band], 
    period_time = 10)

num_level_process = 4
minimum_hierarchy_level = 4
processing_level_list = [4, 2]

for hierarchy_level in processing_level_list:

    print(f"Processing hierarchy level: {hierarchy_level}")

    match hierarchy_level:

        case 2:
            HL_Standalone = HL2_Standalone
            HL_colocated = HL2_colocated
            HL_lower_Standalone = HL4_Standalone
            HL_up_Standalone = HL1_Standalone
            HL_all = np.concatenate((HL2_Standalone, HL2_colocated))
            capacity_updt_index = 1 # if no bypass scenario this variable is 0
            prev_hierarchy_level = 4
        case 4:
            HL_Standalone = HL4_Standalone
            HL_colocated = HL4_colocated
            HL_lower_Standalone = []
            HL_up_Standalone = HL123_Standalone
            HL_all = np.concatenate((HL4_Standalone, HL4_colocated))
            capacity_updt_index = 2
            prev_hierarchy_level = None

    _, subnetMatrix_HL = network.calculate_subgraph(hierarchy_level, minimum_hierarchy_level)
    HL_connected_nodes = network.find_neighbors(HL_Standalone) - set(HL_lower_Standalone)
    
    file_name = results_dir / f"{network.topology_name}_HL{hierarchy_level}_K_path_attributes.csv"

    if os.path.exists(file_name):

        print(f"Loading K-path attributes of HL{hierarchy_level} ...")

        K_path_attributes_df = pd.read_csv(file_name)
        K_path_attributes_df['links'] = K_path_attributes_df['links'].map(json.loads)
        K_path_attributes_df['nodes'] = K_path_attributes_df['nodes'].map(json.loads)

        # sort dataframes based on num_hops and distance (in order)
        K_path_attributes_df_sorted = K_path_attributes_df.groupby(['src_node'], group_keys = False).apply(lambda x: x.sort_values(['num_hops', 'distance']))
        
        # find disjoint pairs for standalone nodes
        pairs_disjoint = network.land_pair_finder(HL_Standalone, K_path_attributes_df_sorted, num_pairs = 1)

    else:

        print(f"Calculating K-path attributes of HL{hierarchy_level} ...")

        k_paths = 100
        source_not_found = HL_Standalone.copy()
        while len(source_not_found) != 0:

            # define a list to store path attributes for this iteration
            K_path_attributes = []

            # iterate through each "source_not_found" node only
            for src in HL_Standalone:
                for dest in HL_connected_nodes:
                    K_path_attributes = network.calculate_paths(subnetMatrix_HL, K_path_attributes, source = src, target = dest, k = k_paths)

            # Convert K_path_attributes list to a temporary DataFrame
            K_path_attributes_df = pd.DataFrame(K_path_attributes)

            # Optionally save to CSV (if you want to track progress)
            K_path_attributes_df.to_csv(file_name, index=False)

            # Sort by num_hops and distance
            K_path_attributes_df_sorted = K_path_attributes_df.groupby(['src_node'], group_keys=False).apply(
                lambda x: x.sort_values(['num_hops', 'distance'])
            )

            # find disjoint pairs from the sorted full dataset
            pairs_disjoint = network.land_pair_finder(HL_Standalone, K_path_attributes_df_sorted, num_pairs=1)

            # Update source_not_found to exclude newly matched source nodes
            source_not_found = np.setdiff1d(HL_Standalone, pairs_disjoint['src_node'].unique())

            print('Remaining src nodes: ', source_not_found)

            k_paths += 20
    
    file_name = results_dir / f"{network.topology_name}_HL{hierarchy_level}_K_path_attributes_colocated.csv"

    if os.path.exists(file_name):

        print(f"Loading K-path attributes of HL{hierarchy_level} colocated...")

        K_path_attributes_colocated_df = pd.read_csv(file_name)
        K_path_attributes_colocated_df['links'] = K_path_attributes_colocated_df['links'].map(json.loads)
        K_path_attributes_colocated_df['nodes'] = K_path_attributes_colocated_df['nodes'].map(json.loads)

        # sort dataframes based on num_hops and distance (in order)
        K_path_attributes_colocated_df_sorted = K_path_attributes_colocated_df.groupby(['src_node', 'dest_node'], group_keys = False).apply(lambda x: x.sort_values(['num_hops', 'distance']))

    else:

        print(f"Calculating K-path attributes of HL{hierarchy_level} colocated...")
        
        k_paths = 1

        # define a list to store path attributes
        K_path_attributes_colocated = []

        # iterate through each standalone HL4 node
        for src in HL_colocated:
            
            if (hierarchy_level == 3) and (src in HL2_Standalone):
                HL_connected_nodes_col = HL_connected_nodes - set(HL1_Standalone)
            else:
                HL_connected_nodes_col = HL_connected_nodes.copy()

            for dest in HL_connected_nodes:
                if src != dest:
                    K_path_attributes_colocated = network.calculate_paths(subnetMatrix_HL, K_path_attributes_colocated, source = src, target = dest, k = k_paths)

        # Convert K_path_attributes list to dataframe
        K_path_attributes_colocated_df = pd.DataFrame(K_path_attributes_colocated)

        # save dataframe to csv file
        K_path_attributes_colocated_df.to_csv(file_name, index = False)

        # sort dataframes based on num_hops and distance (in order)
        K_path_attributes_colocated_df_sorted = K_path_attributes_colocated_df.groupby(['src_node', 'dest_node'], group_keys = False).apply(lambda x: x.sort_values(['num_hops', 'distance']))

    print('pairs: \n', pairs_disjoint)

    print(f"process link GSMR of HL{hierarchy_level} ...")

    GSNR_opt_link, _, _, _ = c_band.process_link_gsnr(f_c_axis = f_c_axis, 
                                                  Pch_dBm = Pch_dBm, 
                                                  num_Ch_mat = num_Ch_mat,
                                                  spectrum_C = spectrum_C,
                                                  Nspan_array = np.ones(network.all_links.shape[0], dtype=int),
                                                  hierarchy_level = hierarchy_level, 
                                                  minimum_hierarchy_level = minimum_hierarchy_level, 
                                                  result_directory = results_dir)
    
    planner.initialize_planner(num_fslots = num_fslots, 
                               hierarchy_level = hierarchy_level,
                               minimum_hierarchy_level = minimum_hierarchy_level)
    
    if hierarchy_level == minimum_hierarchy_level:

        # Load simulated traffic for HL4 nodes
        mat_data = loadmat(project_root / 'data' / 'HL4_new_data_rate_per_year_Standalone_bypassed.mat')
        planner.lowest_HL_added_traffic_annual_standalone = mat_data['HL4_new_data_rate_per_year_Standalone']

        mat_data = loadmat(project_root / 'data' / 'HL4_new_data_rate_per_year_Colocated_bypassed.mat')
        planner.lowest_HL_added_traffic_annual_colocated = mat_data['HL4_new_data_rate_per_year_Colocated']

        mat_data = loadmat(project_root / 'data' / 'HL4_new_100G_lincense_per_year_bypassed.mat')
        planner.num_100G_licence_annual = mat_data['HL4_new_100G_lincense_per_year']
        
    print(f"running planner for HL{hierarchy_level} ...")

    # run the planner for the current hierarchy level    
    planner.run_planner(hierarchy_level = hierarchy_level,
                        prev_hierarchy_level = prev_hierarchy_level,  
                        pairs_disjoint = pairs_disjoint,
                        kpair_standalone = 1,
                        kpair_colocated = 1,
                        candidate_paths_standalone_df = K_path_attributes_df,
                        candidate_paths_colocated_df = K_path_attributes_colocated_df_sorted,
                        GSNR_opt_link = GSNR_opt_link,
                        minimum_level = minimum_hierarchy_level,
                        node_cap_update_idx = capacity_updt_index, 
                        result_directory = results_dir)

analysing = analyse_result(network, planner, processing_level_list, results_dir)
analysing.plot_link_state()
analysing.plot_FP_usage()
analysing.plot_FP_degree()
analysing.plot_bvt_license()



