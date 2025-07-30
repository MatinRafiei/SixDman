from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .network import Network
from .band import Band
import os


@dataclass
class PlanningTool:
    """
    Main class for 6D-MAN (6-Dimensional Metro-Area Network) planning and optimization.
    
    This tool integrates physical layer modeling with hierarchical network topology
    and supports multi-band optical transmission planning. It is intended for
    use in optical metro and urban transport networks where spectral efficiency,
    topology hierarchy, and multi-band coexistence are jointly considered.

    Attributes:
        network (Network): A reference to the network topology object that contains
            all node, link, and hierarchical information used for planning.
        bands (List[Band]): A list of Band instances, each representing an optical band
            (e.g., C, L, S) with its spectral and physical transmission parameters.
        period_time (int): The time period (in arbitrary units) over which traffic 
            demand aggregation or optimization planning is conducted.
    """
    
    def __init__(self,
                 network_instance: Network,
                 bands: List[Band],
                 period_time: int = 10):
        """
        Initialize the PlanningTool object with a given network topology and bands.

        Args:
            network_instance (Network): The network structure including nodes, links,
                weights, and hierarchical levels.
            bands (List[Band]): The optical bands considered for transmission planning.
                Each Band includes start/end frequencies and physical layer parameters.
            period_time (int, optional): Time granularity (e.g., 10 units) for periodic
                traffic updates or recalculation windows. Default is 10.
        """
        self.network = network_instance
        self.bands = bands
        self.period_time = period_time
        
    def generate_initial_traffic_profile(self,
                                 num_nodes: int,
                                 monteCarlo_steps: int,
                                 min_rate: float,
                                 max_rate: float,
                                 seed: int, 
                                 result_directory) -> np.ndarray:
        """
        Simulate initial traffic demands for each node using Monte Carlo sampling.

        This function estimates the initial traffic capacity (e.g., demand or throughput potential) 
        at each node in the network by generating uniformly distributed random values 
        over several Monte Carlo steps. If results exist, they are loaded from file.

        Args:
            num_nodes (int): Number of nodes to simulate traffic for.
            monteCarlo_steps (int): Number of Monte Carlo iterations to average over.
            min_rate (float): Minimum traffic rate per node.
            max_rate (float): Maximum traffic rate per node.
            seed (int): Random seed to ensure repeatable simulations.
            result_directory (Path): Directory where simulation results are stored or loaded from.

        Returns:
            np.ndarray: Array of estimated traffic capacities per node (averaged over simulations).
        """

        # Define the filename for storing/loading precomputed capacity results
        file_path = result_directory / f"{self.network.topology_name}_HL_capacity_final.npz"

        # Load precomputed capacities if they exist
        if os.path.exists(file_path):
            print("Loading precomputed HL_capacity_final ...")
            data = np.load(file_path)
            self.HL_capacity_final = data['HL_capacity_final']

        else:
            print("Calculate HL_capacity_final ...")

            # Storage for traffic capacity samples across Monte Carlo runs
            random_capacity_storage = []

            for i in range(monteCarlo_steps):
                # Set seed each iteration (to ensure consistent output if seed is constant)
                np.random.seed(seed + i)

                # Generate uniform random capacity for each node
                random_capacity_local = np.random.uniform(min_rate, max_rate, size=num_nodes)

                # Store this realization
                random_capacity_storage.append(random_capacity_local)
            
            # Convert the random_capacity_storage list to numpy array
            random_capacity_storage = np.array(random_capacity_storage)

            # Average traffic capacities across all Monte Carlo simulations
            self.HL_capacity_final = random_capacity_storage.mean(axis=0)

            # Save computed capacity to disk
            np.savez_compressed(file_path, HL_capacity_final=self.HL_capacity_final)

    def simulate_traffic_annual(self,
                                 lowest_hierarchy_dict: dict,
                                 CAGR: int, 
                                 result_directory) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate traffic evolution over multiple years for the lowest hierarchy level nodes.

        This method applies a compound annual growth rate (CAGR) to simulate traffic demands
        on standalone and colocated HL4 nodes. It either loads precomputed values or performs 
        the full calculation. It also estimates the number of required 100G licenses and residual
        capacities per node per year.

        Args:
            lowest_hierarchy_dict (dict): Dictionary containing 'standalone' and 'colocated' node IDs at HL4.
            CAGR (int): Compound Annual Growth Rate (e.g., 0.4 for 40% annual increase).
            result_directory (Path): Directory path for reading/writing the traffic matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Annual added traffic (standalone HL4 nodes)
                - Annual added traffic (colocated HL4 nodes)
                - Annual traffic matrix (standalone HL4 nodes)
                - Annual traffic matrix (colocated HL4 nodes)
        """
        
        # Get node counts
        num_node_standalone = len(lowest_hierarchy_dict['standalone'])
        num_node_colocated = len(lowest_hierarchy_dict['colocated'])
        num_node_total = num_node_standalone + num_node_colocated
        period_time = self.period_time

        # Path for cached results
        file_path = result_directory / f"{self.network.topology_name}_traffic_matrix.npz"

        if os.path.exists(file_path):
            # Load precomputed traffic data
            print("Loading precomputed Traffic Matrix ...")
            data = np.load(file_path)
            added_traffic_annual = data['added_traffic_annual']

            # Initialize first year's 100G license count
            self.num_100G_licence_annual[0, :] = np.ceil(self.HL_capacity_final / 100)

            # Split added traffic into standalone and colocated components
            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, num_node_colocated:]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, 0:num_node_colocated]

        else:
            print("Calculate Traffic Matrix ...")

            # Preallocate data structures for annual metrics
            lowest_HL_traffic_storage_annual = np.empty((period_time, num_node_total))
            total_traffic_annual = np.empty(period_time)
            added_traffic_annual = np.empty((period_time, num_node_total))
            residual_capacity_annual = np.empty((period_time, num_node_total))

            # Initialize first year with current capacity
            lowest_HL_traffic_storage_annual[0, :] = self.HL_capacity_final
            total_traffic_annual[0] = np.sum(self.HL_capacity_final)
            added_traffic_annual[0, :] = self.HL_capacity_final

            # Estimate number of 100G licenses for year 0
            self.num_100G_licence_annual[0, :] = np.ceil(self.HL_capacity_final / 100)
            residual_capacity_annual[0, :] = 100 * self.num_100G_licence_annual[0, :] - self.HL_capacity_final

            # Iterate over years and apply CAGR growth
            for year in range(1, period_time):
                # Apply CAGR to simulate traffic growth
                lowest_HL_traffic_storage_annual[year, :] = (
                    (1 + CAGR) * lowest_HL_traffic_storage_annual[year - 1, :]
                )

                # Compute total network traffic for the year
                total_traffic_annual[year] = np.sum(lowest_HL_traffic_storage_annual[year, :])

                # Compute incremental traffic compared to previous year
                added_traffic_annual[year, :] = (
                    lowest_HL_traffic_storage_annual[year, :] - lowest_HL_traffic_storage_annual[year - 1, :]
                )

                # Update number of 100G licenses needed
                self.num_100G_licence_annual[year, :] = np.ceil(lowest_HL_traffic_storage_annual[year, :] / 100)

                # Calculate residual capacity per node after license allocation
                residual_capacity_annual[year, :] = (
                    100 * self.num_100G_licence_annual[year, :] - lowest_HL_traffic_storage_annual[year, :]
                )

            # Final separation of standalone and colocated traffic data
            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, num_node_colocated:]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, 0:num_node_colocated]

            # Persist computed data to file
            np.savez_compressed(file_path, added_traffic_annual = added_traffic_annual)             
    
    def initialize_planner(self, 
                           num_fslots: int,
                           hierarchy_level: int,
                           minimum_hierarchy_level: int,
                           rolloff: float = 0.1,
                           SR: float = 40 * 1e9,
                           BVT_type: int = 1,
                           Max_bit_rate_BVT: np.ndarray = np.array([400])):
        """
        Initialize planning-related matrices and spectrum parameters.

        This method prepares internal state variables needed for simulating 
        BVT allocations, GSNR calculations, fiber placement tracking, and 
        spectrum usage planning across different optical bands.

        Args:
            num_fslots (int): Number of frequency slots available in the network.
            hierarchy_level (int): Target hierarchy level for current planning.
            minimum_hierarchy_level (int): Minimum hierarchy level for subgraph planning.
            rolloff (float): Rolloff factor for spectral shaping (default: 0.1).
            SR (float): Symbol rate in baud (default: 40 Gbaud).
            BVT_type (int): Identifier for BVT type (default: 1).
            Max_bit_rate_BVT (np.ndarray): Array of supported BVT bitrates in Gbps.
        """

        self.Max_bit_rate_BVT = Max_bit_rate_BVT
        self.BVT_type = BVT_type

        # Get number of HL nodes at given hierarchy level
        num_node_standalone = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone'])
        num_node_colocated = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['colocated'])
        num_links = len(self.network.all_links)
        period_time = self.period_time

        # Generate the planning subgraph
        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_hierarchy_level)

        # Assume fixed-grid channel spacing (from first defined band)
        channel_spacing = self.bands[0].channel_spacing
        self.num_fslots = num_fslots

        # Calculate effective channel bandwidth
        B_ch = SR * (1 + rolloff)

        # Calculate how many frequency slots are needed per BVT
        self.Required_FS_BVT = np.ceil(B_ch / (channel_spacing * 1e12)).astype(int)

        # Yearly tracking of fiber placements across all links
        self.Year_FP = np.zeros((period_time, num_links), dtype=np.int32)

        # Track fiber placement for colocated HL4 nodes specifically
        self.Year_FP_HL_colocated = np.zeros((period_time, num_node_colocated))

        # Residual (unserved) traffic at standalone HL4 nodes
        self.Residual_Throughput_BVT_standalone_HLs = np.zeros((period_time, num_node_standalone))

        # Residual traffic at colocated HL4 nodes
        self.Residual_Throughput_BVT_colocated_HLs = np.zeros((period_time, num_node_colocated))

        # Total number of BVTs deployed per year (across all bands)
        self.HL_BVT_number_all_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))

        # Band-specific BVT deployment tracking
        self.HL_BVT_number_Cband_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))         # Traditional C-band
        self.HL_BVT_number_SuperCband_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))    # Super C-band
        self.HL_BVT_number_SuperCLband_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))   # C + L-band extended

        # Optical spectrum tracking for LSPs (Label Switched Paths)
        max_fps_link = 20  # Maximum number of fiber placements per link
        self.LSP_array = np.zeros((self.num_fslots, num_links, max_fps_link))  # General LSP tracking
        self.LSP_array_Colocated = np.zeros((self.num_fslots, num_node_colocated, max_fps_link))  # Colocated HL tracking

        # Band usage tracking (link-level statistics per year)
        self.num_link_LBand_annual = np.zeros(period_time)
        self.num_link_SupCBand_annual = np.zeros(period_time)
        self.num_link_CBand_annual = np.zeros(period_time)

        # Fiber placement for new subgraph connections
        self.Year_FP_new = np.zeros((period_time, subgraph.number_of_edges()))

        # Effective fiber placement deployment tracking
        self.Total_effective_FP_new_annual = np.zeros(period_time)
        self.Total_effective_FP = np.zeros(period_time)

        # Capacity profile for each node across years and hierarchy levels
        self.node_capacity_profile_array = np.zeros(
            shape=(period_time, self.network.adjacency_matrix.shape[0], minimum_hierarchy_level)
        )

        # GSNR tracking for each year (one object per year)
        self.GSNR_BVT_array = [None] * period_time

        # 10-year GSNR history for HL4-level planning
        self.GSNR_HL4_10Year = []

        # Residual capacity of unused 100G units per node
        self.Residual_100G = np.zeros(self.network.adjacency_matrix.shape[0])

        # Annual 100G license usage tracking per node
        self.num_100G_licence_annual = np.zeros(
            shape=(period_time, self.network.adjacency_matrix.shape[0])
        )

        # Band-specific usage stats across links and years
        self.CBand_usage = np.zeros((self.period_time, num_links), dtype=int)
        self.superCBand_usage = np.zeros((self.period_time, num_links), dtype=int)
        self.superCLBand_usage = np.zeros((self.period_time, num_links), dtype=int)

        # Total traffic flow per link per year
        self.traffic_flow_array = np.zeros((self.period_time, num_links), dtype=float)

        # Stores primary path assignments (-1 means unassigned)
        self.primary_path_storage = -1 * np.ones(
            shape=(self.network.adjacency_matrix.shape[0]), dtype=int
        )


    def spectrum_assignment(self,
                            path_IDx: int,
                            path_type: str,
                            kpair_counter,
                            year: int, 
                            K_path_attributes_df: pd.DataFrame,
                            BVT_number: int,
                            node_IDx: int,
                            node_list: List,
                            GSNR_link: np.ndarray, 
                            LSP_array_pair: np.ndarray, 
                            Year_FP_pair: np.ndarray) -> dict:
        """
        Perform spectrum assignment for a given lightpath.

        This function assigns frequency slots and fiber pairs to a path based on available 
        spectrum and congestion, using a first-fit algorithm. It supports both primary and 
        secondary paths, as well as co-located (intra-node) scenarios.

        Args:
            path_IDx: Index of the selected path from K_path_attributes_df.
            path_type: Type of the path ('primary' or 'secondary').
            kpair_counter: Counter for the current K-shortest path pair being processed.
            year: Planning year for multi-period simulation.
            K_path_attributes_df: DataFrame with attributes of all K-shortest paths.
            BVT_number: Number of BVTs to assign per node.
            node_IDx: Index of the current node being processed.
            node_list: List of all node identifiers.
            GSNR_link: GSNR values per frequency slot per link.
            LSP_array_pair: Spectrum occupancy array [FS, link, fiber pair].
            Year_FP_pair: Annual fiber pair usage record [year, link].
            HL_SubNetwork_links: List of high-level subnetwork links.

        Returns:
            If path_IDx is not None:
                A tuple:
                    - path_info_storage: Dictionary with fiber usage stats, BVT assignment, and costs.
                    - LSP_array_pair: Updated spectrum occupancy.
                    - Year_FP_pair: Updated fiber usage.
            If path_IDx is None (colocated node case):
                - Updated Year_FP_HL_colocated array.
        """
        if path_IDx != None:

            path_info_storage = {}
            
            # determine how many frequency slots (FS) are required for the selected BVT type 
            BVT_required_FS_HL = self.Required_FS_BVT
            
            # initialize counters for BVT allocations in different spectrum bands
            BVT_CBand_count_path = 0
            BVT_superCBand_count_path  = 0
            BVT_superCLBand_count_path  = 0

            # extract the link list for the primary path from K_path_attributes_df
            linkList_path = np.array(K_path_attributes_df.iloc[path_IDx]['links'])

            # extract the number of hops for the primary path from K_path_attributes_df
            numHops_path = K_path_attributes_df.iloc[path_IDx]['num_hops']

            # extract the destination node of the primary path from K_path_attributes_df
            destination_path = int(K_path_attributes_df.iloc[path_IDx]['dest_node'])

            # store path information in dictionary
            path_info_storage['links'] = linkList_path
            path_info_storage['numHops'] = numHops_path

            # initialize FP_counter_links with ones, representing the first available fiber pair for each link 
            FP_counter_links = np.ones(len(linkList_path), dtype = np.int8)

            # store congested links in the primary path
            link_congested_path = np.array(
                        [np.count_nonzero(LSP_array_pair[:, link, FP_counter_links[i] - 1]) for i, link in
                         enumerate(linkList_path)])
            
            # Sort the unique congestion levels in descending order
            unique_sorted_link_congested_primary = np.sort(np.unique(link_congested_path))[::-1]

            # Sort links based on congestion
            linkList_path_sorted = np.concatenate(
                [linkList_path[link_congested_path == congestion] for congestion in
                    unique_sorted_link_congested_primary])

            ###################################################
            #  fiber and spectrum assignment for primary path
            ###################################################

            link_counter = 0 # define a counter for fibers
            f_max_path = np.zeros(BVT_number) # initialize an array for Maximum frequency slot used
            cost_FP_all_BVT_path = np.zeros(BVT_number) # initialize an array for Cost function values
            FP_max_path  = np.zeros(BVT_number) # initialize an array for Maximum fiber pairs assigned

            # iterate through BVTs
            for BVT_counter in range(BVT_number):  
                
                Flag_SA_continue_path = 1 # spectrum assignment continues until an available fiber is found

                while Flag_SA_continue_path: # Note: fiber Pair assignment is done based on first fit

                    PST_path = np.zeros(self.num_fslots) # PST_parimary is a binary vector that will store whether each Frequency Slot is occupied or available

                    for FS in range(self.num_fslots): # iterate through each frequency slot
                        
                        vector_state_FS = np.empty(len(linkList_path_sorted))  # vector_state_FS will contain one value per link, indicating whether the slot is free (0) or used (1) on a certain link

                        for link_idx in range(len(linkList_path_sorted)): # check the status of the current frequency slot (FS) for each link

                            vector_state_FS[link_idx] = LSP_array_pair[FS, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] # LSP_array contain a number for each FS in each link 

                        if any(vector_state_FS): # check that there is any link that use that frequecy slot or not
                            PST_path[FS] = 1
                        else:
                            PST_path[FS] = 0

                    
                    FS_count = 0 # keep track of the number of contiguous free slots
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1) # PST_vector_aux stores differences in spectrum occupancy
                    flag_First_Fit = 1 # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen
                    FS_path = [] # stores the selected frequency slots
                    
                    if np.any(PST_vector_aux):

                        startIndex = np.where(PST_vector_aux < 0)[0] # find the first index that 0 changes to 1 (start of free block)                   
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1 # find the first index that 1 changes to 0 (end of free block)
                        duration = endIndex - startIndex + 1 # compute the length of each contiguous free block

                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0] # search for exactly matching free blocks
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0] # search for the first block that match

                        if Exact_Fit.size > 0: # if Exact_Fit is found select the first exact-fit slot and assigns it
                            
                            FS_count = duration[Exact_Fit[0]] # select the first available exact-fit slot
                            b_1 = np.arange(startIndex[Exact_Fit[0]], startIndex[Exact_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0
    
                        elif First_Fit.size > 0 and flag_First_Fit: # if no Exact-Fit, use First-Fit
                            
                            FS_count = duration[First_Fit[0]] # select the first available larger slot
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
      
                    if FS_count >= BVT_required_FS_HL: # if enough contiguous slots are found, the assignment proceeds

                        GSNR_BVT1 = [0]

                        for link_idx in range(len(linkList_path_sorted)):
                            
                            if path_type == 'primary':
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = (node_list[node_IDx] + 1) # update LSP_array_pair to reflect the new assignment
                            elif path_type == 'secondary':
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = -(node_list[node_IDx] + 1) # update LSP_array_pair to reflect the new assignment with a negative identifier

                            GSNR_BVT1 += (10 ** (GSNR_link[link_idx, FS_path] / 10)) ** -1 # compute GSNR 
                                                    
                        Flag_SA_continue_path = 0 # stop searching for more slots
                        
                        for link_counter_local in range(len(FP_counter_links)):

                            Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]] =  max(Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]], FP_counter_links[link_counter_local]) # update the Year_FP_pair to record spectrum usage for each link

                        cost_FP_all_BVT_path[BVT_counter] = np.dot(FP_counter_links, self.network.weights_array[linkList_path_sorted]) # calculate the cost of assigning fiber pairs for the BVT_counter-th BVT

                        if FS_path[-1] <= 95: # The final frequency slot used (FS_path(end)) determines the spectrum band
                            BVT_CBand_count_path += 2
                            band_used = 0  # C-band
                        elif 96 <= FS_path[-1] <= 119:
                            BVT_superCBand_count_path += 2
                            band_used = 1  # superC
                        else:
                            BVT_superCLBand_count_path += 2
                            band_used = 2  # superCL

                        for link_id in linkList_path_sorted: # Store the frequency of specific band usage for each link in this year
                            if band_used == 0:
                                self.CBand_usage[year - 1, link_id] += 1
                            elif band_used == 1:
                                self.superCBand_usage[year - 1, link_id] += 1
                            elif band_used == 2:
                                self.superCLBand_usage[year - 1, link_id] += 1
     
                    else: # If no suitable spectrum was found, move to the next FP link

                        link_counter = (link_counter + 1) % len(FP_counter_links)
                        FP_counter_links[link_counter - 1] += 1

                f_max_path[BVT_counter] = max(FS_path) # store the highest frequency slot index used for this BVT
                FP_max_path[BVT_counter] = max(FP_counter_links) # record the maximum fiber pair index used

                # compute the GSNR for the assigned BVT
                if path_type == 'primary': 
                    self.GSNR_BVT_Kpair_BVTnum_primary[kpair_counter, BVT_counter] =  10 * np.log10(GSNR_BVT1[0] ** -1)
                    self.HL_dest_prim[kpair_counter] = destination_path
                elif path_type == 'secondary':
                    self.GSNR_BVT_Kpair_BVTnum_secondary[kpair_counter, BVT_counter] = 10 * np.log10(GSNR_BVT1[0] ** -1)
                    self.HL_dest_scnd[kpair_counter] = destination_path


            path_info_storage['cost_FP'] = cost_FP_all_BVT_path
            path_info_storage['f_max'] = f_max_path
            path_info_storage['FP_max'] = FP_max_path
            path_info_storage['BVT_CBand_count'] = BVT_CBand_count_path
            path_info_storage['BVT_superCBand_count'] = BVT_superCBand_count_path
            path_info_storage['BVT_superCLBand_count'] = BVT_superCLBand_count_path
            path_info_storage['LSP_array_pair'] = LSP_array_pair
            path_info_storage['Year_FP_pair'] = Year_FP_pair

            return path_info_storage, LSP_array_pair, Year_FP_pair
        
        else: # if path_IDX is None

            FP_counter_links = 0  # define a counter for fibers
            BVT_required_FS_HL = self.Required_FS_BVT # determine how many frequency slots (FS) are required for the selected BVT type 

            for BVT_counter in range(BVT_number): # iterate through BVTs
                
                Flag_SA_continue_path = 1 # spectrum assignment continues until an available fiber is found

                while Flag_SA_continue_path: # Note: fiber Pair assignment is done based on first fit

                    PST_path = self.LSP_array_Colocated[:, node_IDx, FP_counter_links].T # PST_path is a binary vector that will store whether each Frequency Slot is occupied or available
                    FS_count = 0 # keep track of the number of contiguous free slots
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1) # PST_vector_aux stores differences in spectrum occupancy

                    flag_First_Fit = 1 # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen 
                    FS_path = [] # stores the selected frequency slots
                    
                    if np.any(PST_vector_aux != 0):

                        startIndex = np.where(PST_vector_aux < 0)[0] # find the first index that 0 changes to 1 (start of free block)
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1 # find the first index that 1 changes to 0 (end of free block)     
                        duration = endIndex - startIndex + 1 # compute the length of each contiguous free block

                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0] # search for exactly matching free blocks
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0] # search for the first block that match

                        if Exact_Fit.size > 0: # if Exact_Fit is found select the first exact-fit slot and assigns it

                            FS_count = duration[Exact_Fit[0]]
                            b_1 = np.arange(startIndex[Exact_Fit[0]], endIndex[Exact_Fit[0]] + 1)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0

                        elif First_Fit.size > 0 and flag_First_Fit: # if no Exact-Fit, use First-Fit

                            FS_count = duration[First_Fit[0]]
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                                     
                    if FS_count >= BVT_required_FS_HL: # if enough contiguous slots are found, the assignment proceeds

                        self.LSP_array_Colocated[FS_path, node_IDx, FP_counter_links] = 1 # update LSP_array_pair to reflect the new assignment
                        Flag_SA_continue_path = 0 # stop searching for more slots
                        self.Year_FP_HL_colocated[year - 1, node_IDx] =  max(self.Year_FP_HL_colocated[year - 1, node_IDx], FP_counter_links + 1) # update the Year_FP_pair to record spectrum usage for each link
                       
                        if FS_path[-1] <= 95: # The final frequency slot used (FS_primary(end)) determines the spectrum band
                            self.HL_BVT_number_Cband_annual[year - 1] += 2
                        elif 96 <= FS_path[-1] <= 119:
                            self.HL_BVT_number_SuperCband_annual[year - 1]  += 2
                        else:
                            self.HL_BVT_number_SuperCLband_annual[year - 1]  += 2
                    
                    
                    else: # If no slots are available, increment the fiber pair counter and retry
                        FP_counter_links = FP_counter_links + 1

            return self.Year_FP_HL_colocated

    def update_hl_node_degrees(self, 
                            hierarchy_level: dict,
                            Year_FP: np.ndarray) -> np.ndarray:
        """
        Update and track the average node degree of High-Level (HL) nodes over the planning period.

        Args:
            HL_dict (dict): Dictionary containing HL network information.
                            Required key:
                                - 'standalone': np.ndarray of HL node IDs.
            Year_FP (np.ndarray): A 2D array (years × links) indicating the fiber pair allocation
                                per link for each year.

        Returns:
            np.ndarray: Array of average HL node degrees for each year.
        """
        HL_Standalone = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone'] # Extract standalone HL nodes
        HL_degrees = self.network.get_node_degrees(HL_Standalone) # Get initial node degrees for HL nodes (degree per node)
        degree_node_all_topo_HL_final = HL_degrees.copy() # Create a copy to track node degrees evolution across years

        degree_number_HLs = np.zeros(self.period_time) # Initialize array to store average node degree for each year
        degree_number_HLs[0] = np.mean(HL_degrees[:, 1]) # Record initial average node degree (baseline for year 1)

        # Loop through each year (starting from year 2, since year 1 is baseline)
        for year in range(2, self.period_time + 1):

            # Iterate through each link in the network
            for link_counter in range(len(self.network.all_links)):

                # Check if the fiber pair allocation changed between this year and the previous year
                if Year_FP[year - 1, link_counter] != Year_FP[year - 2, link_counter]:
                    src_node = self.network.all_links[link_counter, 0]
                    dest_node = self.network.all_links[link_counter, 1]

                    # Update degree for source node if it is an HL node
                    if src_node in HL_Standalone:
                        indices = np.where(HL_Standalone == src_node)[0]
                        degree_node_all_topo_HL_final[indices, 1] += (
                            Year_FP[year - 1, link_counter] - Year_FP[year - 2, link_counter]
                        )

                    # Update degree for destination node if it is an HL node
                    if dest_node in HL_Standalone:
                        indices = np.where(HL_Standalone == dest_node)[0]
                        degree_node_all_topo_HL_final[indices, 1] += (
                            Year_FP[year - 1, link_counter] - Year_FP[year - 2, link_counter]
                        )

            degree_number_HLs[year - 1] = np.mean(degree_node_all_topo_HL_final[:, 1]) # Compute average HL node degree for the current year
 
        self.degree_number_HLs = degree_number_HLs  # Save result for further use

    def calculate_BVT_usage(self) -> dict:
        """
        Calculate cumulative BVT (Bandwidth Variable Transceiver) counts and 100G licenses for each year.

        This function accumulates the number of BVTs (for all bands: C, SuperC, L) and 100G licenses 
        across the planning period and stores them as instance attributes.

        Returns:
            dict: A dictionary containing yearly cumulative counts for:
                - HL_All_100G_lincense
                - HL_BVTNum_All
                - HL_BVTNum_CBand
                - HL_BVTNum_SuperCBand
                - HL_BVTNum_LBand
        """
        period_time = self.period_time

        # Initialize arrays to track cumulative counts
        HL_All_100G_lincense = np.zeros(period_time)  # Total 100G licenses across years
        HL_BVTNum_All = np.zeros(period_time)         # Total BVT count (all bands)
        HL_BVTNum_CBand = np.zeros(period_time)       # Total BVT count (C-band only)
        HL_BVTNum_SuperCBand = np.zeros(period_time)  # Total BVT count (Super C-band only)
        HL_BVTNum_LBand = np.zeros(period_time)       # Total BVT count (L-band only)

        # Loop through each year and calculate cumulative values
        for year_num in range(period_time):

            # Aggregate BVT counts up to the current year
            HL_BVTNum_All[year_num] = np.sum(self.HL_BVT_number_all_annual[:year_num + 1])
            HL_BVTNum_CBand[year_num] = np.sum(self.HL_BVT_number_Cband_annual[:year_num + 1])
            HL_BVTNum_SuperCBand[year_num] = np.sum(self.HL_BVT_number_SuperCband_annual[:year_num + 1])
            HL_BVTNum_LBand[year_num] = np.sum(self.HL_BVT_number_SuperCLband_annual[:year_num + 1])

            # Calculate cumulative 100G license usage (4 licenses per unit)
            if year_num > 0:
                HL_All_100G_lincense[year_num] = (
                    np.sum(4 * self.num_100G_licence_annual[year_num, :]) 
                    + HL_All_100G_lincense[year_num - 1]
                )
            else:
                HL_All_100G_lincense[year_num] = np.sum(4 * self.num_100G_licence_annual[year_num, :])

        # Store results as instance attributes
        self.HL_All_100G_lincense = HL_All_100G_lincense
        self.HL_BVTNum_CBand = HL_BVTNum_CBand
        self.HL_BVTNum_SuperCBand = HL_BVTNum_SuperCBand
        self.HL_BVTNum_LBand = HL_BVTNum_LBand
        self.HL_BVTNum_All = HL_BVTNum_All


    def save_network_results(self,
                         hierarchy_level: int,
                         minimum_hierarchy_level: int,
                         result_directory):
        """
        Save network analysis results for a given hierarchy level to compressed NPZ files.

        This function calculates the subgraph for the given hierarchy level, determines the
        corresponding link indices, and saves several result files including BVT information,
        link-level usage, node capacity profiles, and traffic data.

        Args:
            hierarchy_level (int): The current hierarchy level being analyzed.
            minimum_hierarchy_level (int): The minimum hierarchy level considered for subgraph generation.
            result_directory (Path): Directory where the output files will be saved.

        Saves:
            - {topology_name}_HL{hierarchy_level}_bvt_info.npz
            - {topology_name}_HL{hierarchy_level}_link_info.npz
            - {topology_name}_HL{hierarchy_level}_node_capacity_profile_array.npz
        """    

        # Generate the subgraph for the given hierarchy level
        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_hierarchy_level)

        # Extract HL link indices (filter from all links)
        HL_subnet_links = np.array(list(subgraph.edges(data='weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis=2), axis=1)
        HL_links_indices = np.where(mask)[0]

        # Calculate degree per domain (each link adds 2 degrees: one per endpoint)
        HL_CDegree_Domain = 2 * self.num_link_CBand_annual
        HL_SuperCDegree_Domain = 2 * self.num_link_SupCBand_annual
        HL_LDegree_Domain = 2 * self.num_link_LBand_annual

        # Save BVT-related information
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz',
                            HL_All_100G_lincense=self.HL_All_100G_lincense,
                            HL_BVTNum_All=self.HL_BVTNum_All,
                            HL_BVTNum_CBand=self.HL_BVTNum_CBand,
                            HL_BVTNum_SuperCBand=self.HL_BVTNum_SuperCBand,
                            HL_BVTNum_LBand=self.HL_BVTNum_LBand)

        # Save link-related information and usage statistics
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_link_info.npz',
                            HL_links_indices=HL_links_indices,
                            num_link_CBand_annual=self.num_link_CBand_annual,
                            num_link_SupCBand_annual=self.num_link_SupCBand_annual,
                            num_link_LBand_annual=self.num_link_LBand_annual,
                            HL_CDegree_Domain=HL_CDegree_Domain,
                            HL_SuperCDegree_Domain=HL_SuperCDegree_Domain,
                            HL_LDegree_Domain=HL_LDegree_Domain,
                            Total_effective_FP_new_annual=self.Total_effective_FP_new_annual,
                            HL_FPNum=self.Year_FP_new,
                            HL_FPNumCo=self.Year_FP_HL_colocated,
                            degree_number_HLs=self.degree_number_HLs,
                            CBand_usage=self.CBand_usage,
                            superCBand_usage=self.superCBand_usage,
                            superCLBand_usage=self.superCLBand_usage,
                            traffic_flow_array=self.traffic_flow_array,
                            primary_paths=self.primary_path_storage)

        # Save node capacity profile
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_node_capacity_profile_array.npz',
                            node_capacity_profile_array=self.node_capacity_profile_array)


        
    def run_planner(self, 
                    hierarchy_level: int,
                    prev_hierarchy_level: int,
                    pairs_disjoint: pd.DataFrame,
                    kpair_standalone: int,
                    kpair_colocated: int,
                    candidate_paths_standalone_df: pd.DataFrame,
                    candidate_paths_colocated_df: pd.DataFrame,
                    GSNR_opt_link: np.ndarray,
                    minimum_level: int,
                    node_cap_update_idx: int, 
                    result_directory) -> float:
        """
        Executes the hierarchical planning algorithm for the given hierarchy level.

        Args:
            HL_dict (dict): Dictionary containing standalone and colocated HL node IDs.
            pairs_disjoint (pd.DataFrame): List of disjoint node pairs for path computation.
            kpair_standalone (int): Number of K-shortest paths for standalone HL nodes.
            kpair_colocated (int): Number of K-shortest paths for colocated HL nodes.
            candidate_paths_standalone_df (pd.DataFrame): Candidate paths for standalone HL pairs.
            candidate_paths_colocated_df (pd.DataFrame): Candidate paths for colocated HL pairs.
            GSNR_opt_link (np.ndarray): Link GSNR values.
            prev_hierarchy_level (int): Previous hierarchy level.
            hierarchy_level (int): Current hierarchy level.
            minimum_level (int): Minimum HL level to consider for FP continuity.
            node_cap_update_idx (int): Index of node capacity vector to update.
            result_directory (Path): Directory to save result files.

        Returns:
            float: Total cost of the generated network design for the current
        """
        HL_standalone = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone']
        HL_colocated = self.network.hierarchical_levels[f"HL{hierarchy_level}"]['colocated']

        subgraph, _ = self.network.compute_hierarchy_subgraph(hierarchy_level, minimum_level)
        HL_subnet_links = np.array(list(subgraph.edges(data = 'weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis=2), axis=1)
        HL_links_indices = np.where(mask)[0]

        # GSNR to reduce when storing GSNR values per year in the calculations
        if hierarchy_level == 4 or hierarchy_level == 5: 
            reduce_GSNR_year = 1.5 
        elif hierarchy_level == 3:
             reduce_GSNR_year = 2
        elif hierarchy_level == 2:
             reduce_GSNR_year = 5.5 

        period_time = self.period_time

        # array for saving destinations of standalone nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_standalone_dest_profile = np.zeros(shape = (period_time, len(HL_standalone), 2), dtype = np.int8)

        # array for saving destinations of colocated nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_colocated_dest_profile = np.zeros(shape = (period_time, len(HL_colocated)), dtype = np.int8)

        # Define maximum number of fiber pairs per link
        FP_max_num = 20
        
        for year in range(1 , period_time + 1):

            print('Processing Year: ', year)

            if hierarchy_level == minimum_level:
                    # Create node_capacity_profile array in the minimum hierarchy level
                    node_capacity_profile = np.zeros(shape = (len(HL_colocated) + len(HL_standalone), minimum_level))
            else:
                    # Load node_capacity_profile array of previous hierarchy level
                    node_capacity_profile_array_prev_hl = np.load(result_directory /  f"{self.network.topology_name}_HL{prev_hierarchy_level}_node_capacity_profile_array.npz")['node_capacity_profile_array']
                    node_capacity_profile = node_capacity_profile_array_prev_hl[year - 1, :, :]

                    # Calculate number of 100G licence of each year
                    self.num_100G_licence_annual[year - 1, :] = np.ceil(0.01 * (node_capacity_profile[:, node_cap_update_idx + 1] - self.Residual_100G))
                    self.Residual_100G += 100 * self.num_100G_licence_annual[year - 1, :] - node_capacity_profile[:, node_cap_update_idx + 1]

                    # Load different band usage for previous hierarchy level
                    self.CBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['CBand_usage']
                    self.superCBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['superCBand_usage']
                    self.superCLBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['superCLBand_usage']
                    
                    # Load traffic_flow_array for previous hierarchy level
                    self.traffic_flow_array = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['traffic_flow_array']
                    
                    # Load primary_path_storage for previous hierarchy level (use for calculate latency of primary paths)
                    self.primary_path_storage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['primary_paths']

            #######################################################
            # Part 1: Spectrum assignment for standalone HLs
            #######################################################

            
            GSNR_BVT_per_year = [] # tracks signal quality (GSNR) per BVT

            for node_idx in range(len(HL_standalone)): # Iterate through standalone nodes
                
                # get traffic demand for this node in this year
                if hierarchy_level == minimum_level:
                    HL_needed_traffic = self.lowest_HL_added_traffic_annual_standalone[year - 1, node_idx]
                else:
                    HL_needed_traffic = node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]
                
                
                if year != 1: # if it isnt the first year                   
                    HL_pure_throughput_to_assign = HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx] # subtract residual throughput (unallocated traffic from previous years)
                
                else: # if it is the first year
                    HL_pure_throughput_to_assign = HL_needed_traffic
                    
                if hierarchy_level == minimum_level:
                    node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1] = HL_needed_traffic # store traffic capacity assigned to current node
                
                #################
                # BVT selection 
                #################
                if HL_pure_throughput_to_assign > 0:
                
                    # calculate the number of BVTs needed to handle the assigned throughput
                    BVT_number  = int(np.ceil(HL_pure_throughput_to_assign / self.Max_bit_rate_BVT[self.BVT_type - 1]))
                    
                    # update BVT allocation tracking, multiplying by 4
                    self.HL_BVT_number_all_annual[year - 1, self.BVT_type - 1] += 4 * BVT_number

                    ##############################################################
                    # Routing, MF, spectrum, L-band, and new fiber assignment 
                    ##############################################################

                    # Extract the first precomputed K-shortest paths for the current standalone node 
                    candidate_path_pair = pairs_disjoint[pairs_disjoint['src_node'] == HL_standalone[node_idx]]

                    # Calculate the number of LAND pairs for this standalone node
                    num_K_pair_final = self.network.calc_num_pair(pairs_disjoint_df = pairs_disjoint)
                    num_kpairs = min(num_K_pair_final[node_idx], kpair_standalone) # the num_kpairs for this standalone node is minimum of available k_pair and minimum allowed pairs

                    # Initialize the GSNR_BVT arrays
                    self.GSNR_BVT_Kpair_BVTnum_primary = np.zeros((num_kpairs, BVT_number))
                    self.GSNR_BVT_Kpair_BVTnum_secondary = np.zeros((num_kpairs, BVT_number))

                    # Initialize the cost function matrix with infinity values for each metric (f_max, N_hop, cost, GSNR, FP_max)
                    cost_func = np.full((num_kpairs, 5), np.inf)

                    # keep track of spectrum assignments across different bands
                    HL_BVT_CBand_count_Kpair = np.zeros(num_kpairs)
                    HL_BVT_SuperCBand_count_Kpair = np.zeros(num_kpairs)
                    HL_BVT_SuperCLBand_count_Kpair = np.zeros(num_kpairs)

                    # storage for LSP_arrays
                    LSP_array_pair_storage = []

                    # storage for Year_FP
                    Year_FP_pair_storage = []

                    # storage for paths
                    paths_storage = []

                    primary_path_storage_array_standalone = []

                    self.HL_dest_prim = np.zeros(num_kpairs)
                    self.HL_dest_scnd = np.zeros(num_kpairs)

                    for final_K_pair_counter in range(num_kpairs): # Iterate through LAND_pairs

                        # track Label Switched Paths (LSPs) for allocated routes
                        LSP_array_pair = self.LSP_array.copy()

                        # define a variable to track frequency slots (FS) occupied per year
                        Year_FP_pair = self.Year_FP.copy()

                        # Spectrum assignment of primary path
                        primary_path_IDX = int(candidate_path_pair.iloc[final_K_pair_counter]['primary_path_IDx'])
                        primary_info_dict, LSP_array_pair, Year_FP_pair = self.spectrum_assignment(path_IDx = primary_path_IDX,
                                                                                                   path_type = 'primary',
                                                                                                   kpair_counter = final_K_pair_counter,
                                                                                                   year = year, 
                                                                                                   K_path_attributes_df = candidate_paths_standalone_df,
                                                                                                   BVT_number = BVT_number,
                                                                                                   node_IDx = node_idx,
                                                                                                   node_list = HL_standalone,
                                                                                                   GSNR_link = GSNR_opt_link,
                                                                                                   LSP_array_pair = LSP_array_pair, 
                                                                                                   Year_FP_pair = Year_FP_pair)
                        
                        # Spectrum assignment of secondary path
                        secondary_path_IDX = int(candidate_path_pair.iloc[final_K_pair_counter]['secondary_path_IDx'])
                        secondary_info_dict, LSP_array_pair, Year_FP_pair = self.spectrum_assignment(path_IDx = secondary_path_IDX,
                                                                                                     path_type = 'secondary',
                                                                                                     kpair_counter = final_K_pair_counter,
                                                                                                     year = year, 
                                                                                                     K_path_attributes_df = candidate_paths_standalone_df,
                                                                                                     BVT_number = BVT_number,
                                                                                                     node_IDx = node_idx,
                                                                                                     node_list = HL_standalone,
                                                                                                     GSNR_link = GSNR_opt_link,
                                                                                                     LSP_array_pair = LSP_array_pair, 
                                                                                                     Year_FP_pair = Year_FP_pair)
                        
                        # Calculate the first cost metric, representing the maximum frequency slot (FS) usage on both primary and secondary paths
                        cost_func[final_K_pair_counter, 0] = max(primary_info_dict['f_max']) + max(secondary_info_dict['f_max'])

                        # Add the number of hops for both primary and secondary paths 
                        cost_func[final_K_pair_counter, 1] = primary_info_dict['numHops'] + secondary_info_dict['numHops']

                        # Reflect the total resource usage considering frequency slots and link lengths
                        cost_func[final_K_pair_counter, 2] = max(primary_info_dict['cost_FP']) + max(secondary_info_dict['cost_FP'])

                        # Placeholder for GSNR cost metric - Initialized with inf 
                        cost_func[final_K_pair_counter, 3] = np.inf

                        # Indicate the maximum frequency path indices used for primary and secondary paths
                        cost_func[final_K_pair_counter, 4] = max(primary_info_dict['FP_max']) + max(secondary_info_dict['FP_max'])

                        # record how many BVTs in different bands are used for the current K-shortest path pair
                        HL_BVT_CBand_count_Kpair[final_K_pair_counter] = primary_info_dict['BVT_CBand_count'] + secondary_info_dict['BVT_CBand_count']
                        HL_BVT_SuperCBand_count_Kpair[final_K_pair_counter] = primary_info_dict['BVT_superCBand_count'] + secondary_info_dict['BVT_superCBand_count']
                        HL_BVT_SuperCLBand_count_Kpair[final_K_pair_counter] = primary_info_dict['BVT_superCLBand_count'] + secondary_info_dict['BVT_superCLBand_count']

                        # save the label-switched path (LSP) and frequency path (FP) arrays for further evaluation
                        LSP_array_pair_storage.append(LSP_array_pair.copy())
                        Year_FP_pair_storage.append(Year_FP_pair.copy())
                        pair_links_tuple = (primary_info_dict['links'], secondary_info_dict['links'])
                        paths_storage.append(pair_links_tuple)
                        primary_path_storage_array_standalone.append(primary_path_IDX)


                    # #################### Pair Selection ####################

                    # Sort feasible path pairs based on cost function [5 1 2 3 4] in ascending order
                    index_feasible_pair = np.lexsort((cost_func[:, 1], cost_func[:, 2], cost_func[:, 0],
                                                  cost_func[:, 4], cost_func[:, 3]))  # Sort using lexsort

                    # select the best path pair after sorting
                    self.LSP_array =  LSP_array_pair_storage[index_feasible_pair[0]]
                    self.Year_FP =  Year_FP_pair_storage[index_feasible_pair[0]]

                    # record the primary and secondary destinations for the selected path
                    HL_standalone_dest_profile[year -1, node_idx, 0] = self.HL_dest_prim[index_feasible_pair[0]]
                    HL_standalone_dest_profile[year -1, node_idx, 1] = self.HL_dest_scnd[index_feasible_pair[0]]

                    # update yearly BVT usage counts based on selected path
                    self.HL_BVT_number_Cband_annual[year - 1] += HL_BVT_CBand_count_Kpair[index_feasible_pair[0]]
                    self.HL_BVT_number_SuperCband_annual[year - 1] += HL_BVT_SuperCBand_count_Kpair[index_feasible_pair[0]]
                    self.HL_BVT_number_SuperCLband_annual[year - 1] += HL_BVT_SuperCLBand_count_Kpair[index_feasible_pair[0]]
                
                    # record GSNR for the selected path across all BVTs
                    GSNR_BVT_per_year.extend(
                        np.concatenate([
                            self.GSNR_BVT_Kpair_BVTnum_primary[index_feasible_pair[0], :],
                            self.GSNR_BVT_Kpair_BVTnum_secondary[index_feasible_pair[0], :]
                        ])
                    )

                    # Store traffic flow through each link in this year
                    best_pair_links_tuple = paths_storage[index_feasible_pair[0]]
                    for links_arr in best_pair_links_tuple:
                        for link in links_arr:
                            self.traffic_flow_array[year - 1, link] += HL_needed_traffic

                    # Store primary path IDx for calculating latency
                    self.primary_path_storage[HL_standalone[node_idx]] = primary_path_storage_array_standalone[index_feasible_pair[0]]

                if year > 1 and (hierarchy_level == minimum_level or HL_needed_traffic != 0):

                    # check if the required HL4 traffic exceeds the residual BVT throughput from the previous year
                    if HL_needed_traffic > self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx]:

                        # alculate the residual throughput for the current year after allocating BVT resources:
                        # - Take the previous year's residual throughput.
                        # - Add the throughput assigned to the BVT (rounded up to the nearest integer multiple of Max_bit_rate_BVT).
                        # - Subtract the needed HL4 traffic.
                        self.Residual_Throughput_BVT_standalone_HLs[year - 1, node_idx] = self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx] + \
                        np.ceil(HL_pure_throughput_to_assign / self.Max_bit_rate_BVT[self.BVT_type - 1]) * self.Max_bit_rate_BVT[[self.BVT_type - 1]] - HL_needed_traffic
                    
                        # update the destination node capacity profile: add half of the newly assigned traffic (minus previous residual throughput) to the destination node.
                        # primary destination last year
                        node_capacity_profile[HL_standalone_dest_profile[year - 2, node_idx, 0], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx]

                        # secondary destination last year
                        node_capacity_profile[HL_standalone_dest_profile[year - 2, node_idx, 1], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx]

                        # primary destination this year
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx])

                        # secondary destination this year
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx])

                    # if residual capacity is enough, just subtracts the required traffic from the existing capacity
                    else:
                        
                        # deduct the required HL4 traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_standalone_HLs[year - 1, node_idx] = self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx] - HL_needed_traffic
                
                        # maintain the same destination profile as the previous year (no change in destination node).
                        # primary destination
                        HL_standalone_dest_profile[year - 1, node_idx, 0] = HL_standalone_dest_profile[year - 2, node_idx, 0]

                        # secondary destination
                        HL_standalone_dest_profile[year - 1, node_idx, 1] = HL_standalone_dest_profile[year - 2, node_idx, 1]
                
                        # add half of the needed traffic to the source node's allocated capacity.
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * HL_needed_traffic
                
                        # add the other half of the needed traffic to the destination node's allocated capacity.
                        node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * HL_needed_traffic
                
                # if this is the first year
                elif hierarchy_level == minimum_level or HL_needed_traffic != 0:

                    # initialize the residual throughput for the BVT:
                    # - Calculate the number of BVTs needed by dividing the traffic by the max BVT bit rate (rounding up).
                    # - Compute the leftover capacity after allocating the BVT.
                    self.Residual_Throughput_BVT_standalone_HLs[0, node_idx] = np.ceil(HL_needed_traffic / self.Max_bit_rate_BVT[self.BVT_type - 1]) * self.Max_bit_rate_BVT[self.BVT_type - 1] - HL_needed_traffic
                
                    # update source node capacity: add half of the node's original capacity (from the capacity profile) to the allocated capacity.
                    node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 0], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]
                
                    # update destination node capacity: add the remaining half of the node's original capacity to the destination node's allocated capacity.
                    node_capacity_profile[HL_standalone_dest_profile[year - 1, node_idx, 1], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]

            #######################################################
            # Part 2: Spectrum assignment for colocated HLs
            #######################################################

            # Initialize the cost function matrix with infinity values for each metric (f_max, N_hop, cost, GSNR, FP_max)
            cost_func = np.inf * np.ones(shape = (1, 5))

            max_path_secondary = candidate_paths_colocated_df.groupby('src_node').size().to_numpy()

            for node_idx in range(len(HL_colocated)): # Iterate through colocated nodes
               
                # get traffic demand for this node in this year
                if hierarchy_level == minimum_level:
                    HL_needed_traffic = self.lowest_HL_added_traffic_annual_colocated[year - 1, node_idx]
                else:
                    HL_needed_traffic = node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]
                
                if year != 1: # if it isnt the first year
                    # subtract residual throughput (unallocated traffic from previous years)
                    HL_pure_throughput_to_assign = HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx]
                else: # if it is the first year    
                    HL_pure_throughput_to_assign = HL_needed_traffic
                
                if hierarchy_level == minimum_level:
                    # store traffic capacity assigned to current node
                    node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1] = HL_needed_traffic
                
                #################
                # BVT selection 
                #################
                if HL_pure_throughput_to_assign > 0:
                
                    # calculate the number of BVTs needed to handle the assigned throughput
                    BVT_number  = int(np.ceil(HL_pure_throughput_to_assign / self.Max_bit_rate_BVT[self.BVT_type - 1]))
                    
                    # update BVT allocation tracking, multiplying by 4
                    self.HL_BVT_number_all_annual[year - 1, self.BVT_type - 1] += 4 * BVT_number

                    ##############################################################
                    # Routing, MF, spectrum, L-band, and new fiber assignment 
                    ##############################################################

                    # Spectrum assignment for primary path
                    Year_FP_HL_colocated = self.spectrum_assignment(path_IDx = None,
                                                                    path_type = None,
                                                                    kpair_counter = None,
                                                                    year = year, 
                                                                    K_path_attributes_df = candidate_paths_colocated_df,
                                                                    BVT_number = BVT_number,
                                                                    node_IDx = node_idx,
                                                                    node_list = HL_colocated,
                                                                    GSNR_link = GSNR_opt_link,
                                                                    LSP_array_pair = None, 
                                                                    Year_FP_pair = None)
                    
                    self.Year_FP_HL_colocated = Year_FP_HL_colocated
                    
                    # Calculate the number of LAND pairs for this colocated node
                    num_kpairs = int(min(max_path_secondary[node_idx], kpair_colocated))
                    cost_func = np.full((num_kpairs, 5), np.inf)  # Initialize cost function with infinity

                    self.HL_dest_scnd = np.zeros(num_kpairs)

                    # keep track of spectrum assignments across different bands
                    HL_BVT_CBand_count_Kpair = np.zeros(num_kpairs)
                    HL_BVT_SuperCBand_count_Kpair = np.zeros(num_kpairs)
                    HL_BVT_SuperCLBand_count_Kpair = np.zeros(num_kpairs)

                    # storage for LSP_arrays
                    LSP_array_pair_storage = []

                    # storage for Year_FP
                    Year_FP_pair_storage = []

                    self.GSNR_BVT_Kpair_BVTnum_secondary = np.zeros((num_kpairs, BVT_number))

                    for final_K_pair_counter in range(num_kpairs): # Iterate through all candidate LAND_pairs

                        # track Label Switched Paths (LSPs) for allocated routes
                        LSP_array_pair = self.LSP_array.copy()

                        # define a variable to track frequency slots (FS) occupied per year
                        Year_FP_pair = self.Year_FP.copy()

                        # Spectrum assignment for secondary path
                        secondary_path_IDX = candidate_paths_colocated_df[candidate_paths_colocated_df['src_node'] == HL_colocated[node_idx]].head(1).index[0]
                        secondary_info_dict, LSP_array_pair, Year_FP_pair = self.spectrum_assignment(path_IDx = secondary_path_IDX,
                                                                                                     path_type = 'secondary',
                                                                                                     kpair_counter = final_K_pair_counter,
                                                                                                     year = year, 
                                                                                                     K_path_attributes_df = candidate_paths_colocated_df,
                                                                                                     BVT_number = BVT_number,
                                                                                                     node_IDx = node_idx,
                                                                                                     node_list = HL_colocated,
                                                                                                     GSNR_link = GSNR_opt_link,
                                                                                                     LSP_array_pair = LSP_array_pair, 
                                                                                                     Year_FP_pair = Year_FP_pair)
                        
                        # Calculate the first cost metric, representing the maximum frequency slot (FS) usage on both primary and secondary paths
                        cost_func[final_K_pair_counter, 0] = max(secondary_info_dict['f_max'])

                        # Add the number of hops for both primary and secondary paths 
                        cost_func[final_K_pair_counter, 1] = secondary_info_dict['numHops']

                        # Reflect the total resource usage considering frequency slots and link lengths
                        cost_func[final_K_pair_counter, 2] = max(secondary_info_dict['cost_FP'])

                        # Placeholder for GSNR cost metric - Initialized with inf 
                        cost_func[0, 3] = np.inf

                        # Indicate the maximum frequency path indices used for primary and secondary paths
                        cost_func[final_K_pair_counter, 4] = max(secondary_info_dict['FP_max'])


                        # record how many BVTs in different bands are used for the current K-shortest path pair
                        HL_BVT_CBand_count_Kpair[final_K_pair_counter] = secondary_info_dict['BVT_CBand_count']
                        HL_BVT_SuperCBand_count_Kpair[final_K_pair_counter] = secondary_info_dict['BVT_superCBand_count']
                        HL_BVT_SuperCLBand_count_Kpair[final_K_pair_counter] = secondary_info_dict['BVT_superCLBand_count']


                        # save the label-switched path (LSP) and frequency path (FP) arrays for further evaluation
                        LSP_array_pair_storage.append(LSP_array_pair.copy())
                        Year_FP_pair_storage.append(Year_FP_pair.copy())

                    # #################### Pair Selection ####################

                    # Sort feasible path pairs based on cost function [5 1 2 3 4] in ascending order
                    index_feasible_pair = np.lexsort((cost_func[:, 1], cost_func[:, 2], cost_func[:, 0],
                                                  cost_func[:, 4], cost_func[:, 3]))  # Sort using lexsort

                    # select the best path pair after sorting
                    self.LSP_array =  LSP_array_pair_storage[index_feasible_pair[0]]
                    self.Year_FP =  Year_FP_pair_storage[index_feasible_pair[0]]

                    # record the secondary destinations for the selected path
                    HL_colocated_dest_profile[year -1, node_idx] = self.HL_dest_scnd[index_feasible_pair[0]]

                    # update yearly BVT usage counts based on selected path
                    self.HL_BVT_number_Cband_annual[year - 1] += HL_BVT_CBand_count_Kpair[index_feasible_pair[0]]
                    self.HL_BVT_number_SuperCband_annual[year - 1] += HL_BVT_SuperCBand_count_Kpair[index_feasible_pair[0]]
                    self.HL_BVT_number_SuperCLband_annual[year - 1] += HL_BVT_SuperCLBand_count_Kpair[index_feasible_pair[0]]
                
                    # record GSNR for the selected path across all BVTs  
                    GSNR_BVT_per_year.extend(self.GSNR_BVT_Kpair_BVTnum_secondary[index_feasible_pair[0]])

                if year > 1 and (hierarchy_level == minimum_level or HL_needed_traffic != 0):

                    # check if the required HL4 traffic exceeds the residual BVT throughput from the previous year
                    if HL_needed_traffic > self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx]:

                        # alculate the residual throughput for the current year after allocating BVT resources:
                        # - Take the previous year's residual throughput.
                        # - Add the throughput assigned to the BVT (rounded up to the nearest integer multiple of Max_bit_rate_BVT).
                        # - Subtract the needed HL4 traffic.
                        self.Residual_Throughput_BVT_colocated_HLs[year - 1, node_idx] = self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx] + \
                        np.ceil(HL_pure_throughput_to_assign / self.Max_bit_rate_BVT[self.BVT_type - 1]) * self.Max_bit_rate_BVT[self.BVT_type - 1] - HL_needed_traffic

                        # update the source node capacity profile: add half of the previous year's residual throughput to the source node's allocated capacity.
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx]

                        # update the source node capacity profile: add half of the previous year's residual throughput to the source node's allocated capacity.
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx])
                        
                        # update the destination node capacity profile: add half of the newly assigned traffic (minus previous residual throughput) to the destination node.
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * (self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx])
                                            
                        # update the destination node capacity profile: add half of the newly assigned traffic (minus previous residual throughput) to the destination node.
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * (HL_needed_traffic - self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx])


                    # if the needed traffic is less than or equal to the previous year's residual throughput
                    else:
                        
                        # deduct the required HL4 traffic from the previous year's residual throughput.
                        self.Residual_Throughput_BVT_colocated_HLs[year - 1, node_idx] = self.Residual_Throughput_BVT_colocated_HLs[year - 2, node_idx] - HL_needed_traffic
                
                        # maintain the same destination profile as the previous year (no change in destination node).
                        HL_colocated_dest_profile[year - 1, node_idx] = HL_colocated_dest_profile[year - 2, node_idx]
                
                        # add half of the needed traffic to the source node's allocated capacity.
                        node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * HL_needed_traffic
                
                        # add the other half of the needed traffic to the destination node's allocated capacity.
                        node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * HL_needed_traffic

                # if this is the first year
                elif hierarchy_level == minimum_level or HL_needed_traffic != 0:

                    # initialize the residual throughput for the BVT:
                    # - Calculate the number of BVTs needed by dividing the traffic by the max BVT bit rate (rounding up).
                    # - Compute the leftover capacity after allocating the BVT.
                    self.Residual_Throughput_BVT_colocated_HLs[0, node_idx] = np.ceil(HL_needed_traffic / self.Max_bit_rate_BVT[self.BVT_type - 1]) * self.Max_bit_rate_BVT[self.BVT_type - 1] - HL_needed_traffic
                
                    # update source node capacity: add half of the node's original capacity (from the capacity profile) to the allocated capacity.
                    node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]
                
                    # update destination node capacity: add the remaining half of the node's original capacity to the destination node's allocated capacity.
                    node_capacity_profile[HL_colocated_dest_profile[year - 1, node_idx], node_cap_update_idx] += 0.5 * node_capacity_profile[HL_colocated[node_idx], node_cap_update_idx + 1]

            ######################################################################
            # Update Frequency Plans (FP) and Degree Counters for Each Year
            ######################################################################
            
            if year > 1:

                #  update Frequency Plan (FP) for HL4 SubNetwork Links
                for link_idx in range(len(HL_subnet_links)):

                        # If the FP for the current year and link is not established (i.e., equals zero) inherit the FP from the previous year for continuity.
                        if hierarchy_level == minimum_level and self.Year_FP[year - 2, HL_links_indices[link_idx]] == 0 and self.Year_FP[year - 1, HL_links_indices[link_idx]] == 0:
                            self.Year_FP[year - 1, HL_links_indices[link_idx]] = self.Year_FP[year - 2, HL_links_indices[link_idx]]
                        elif self.Year_FP[year - 1, HL_links_indices[link_idx]] == 0:
                            self.Year_FP[year - 1, HL_links_indices[link_idx]] = self.Year_FP[year - 2, HL_links_indices[link_idx]]


                # update Frequency Plan (FP) for HL4 Co-located Links
                for node_idx in range(len(HL_colocated)):

                    # If the FP for the current year and link is not established (i.e., equals zero) inherit the FP from the previous year for continuity.
                    if hierarchy_level == minimum_level and self.Year_FP_HL_colocated[year - 1, node_idx] == 0 and self.Year_FP_HL_colocated[year - 2, node_idx] == 0:
                        self.Year_FP_HL_colocated[year - 1, node_idx] = self.Year_FP_HL_colocated[year - 2, node_idx]
                    elif self.Year_FP_HL_colocated[year - 1, node_idx] == 0:
                        self.Year_FP_HL_colocated[year - 1, node_idx] = self.Year_FP_HL_colocated[year - 2, node_idx]

            ##################################################
            # Calculate Total Effective Frequency Plan (FP)
            ##################################################

            # compute the weighted total FP for the current year: 
            # - First term: Weighted sum of FP across all links using provided link weights.
            # - Second term: Contribution from co-located HL4 links is multiplied by zeros, effectively ignoring them.
            self.Total_effective_FP[year - 1] = 2 * np.dot(self.Year_FP[year - 1, :], self.network.weights_array.T) + \
                0 * 2 * np.dot(self.Year_FP_HL_colocated[year - 1, :], 0.5 * np.ones(len(HL_colocated)))

            # Save Node Capacity Profile for Current Year
            self.node_capacity_profile_array[year - 1] = node_capacity_profile


            #############################################################
            # Frequency Plan (FP) Calculation for HL4 SubNetwork Links
            #############################################################

            # loop over each link in the HL4 SubNetwork
            for link_idx in range(len(HL_subnet_links)):

                # loop over each Frequency Plan (FP) counter (assumed 20 possible FPs per link)
                for FP_counter in range(FP_max_num):

                    # initialize flag to check if an FP has been counted for this link in this iteration
                    FP_flag = 0

                    # check for L-Band Link Utilization (indices 121 to end in LSP_array) --- if any element in the specified LSP_array slice is non-zero, it indicates L-Band usage.
                    if np.any(self.LSP_array[120:, HL_links_indices[link_idx], FP_counter] != 0):

                        # increment the L-Band link count for the current year
                        self.num_link_LBand_annual[year - 1] += 1
            
                        # Set flag indicating an FP was used for this link
                        FP_flag = 1
            
                        # update the FP usage count for the link in the current year
                        self.Year_FP_new[year - 1, link_idx] += 1
            
                        # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                        self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]

                    # check for Super C-Band Link Utilization (indices 96 to 119 in LSP_array)
                    if np.any(self.LSP_array[96:120, HL_links_indices[link_idx], FP_counter]) != 0:

                        # increment the L-Band link count for the current year
                        self.num_link_SupCBand_annual[year - 1] += 1

                        # if no FP has been counted yet for this link:
                        if FP_flag == 0:
                            
                            # Set flag indicating an FP was used for this link
                            FP_flag = 1
            
                            # update the FP usage count for the link in the current year
                            self.Year_FP_new[year - 1, link_idx] += 1
                
                            # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                            self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]

                    
                    # check for C-Band Link Utilization (indices 0 to 95 in LSP_array)
                    if any(self.LSP_array[:96, HL_links_indices[link_idx], FP_counter]) != 0:

                        # increment the L-Band link count for the current year
                        self.num_link_CBand_annual[year - 1] += 1

                        # if no FP has been counted yet for this link:
                        if FP_flag == 0:
                            
                            # update the FP usage count for the link in the current year
                            self.Year_FP_new[year - 1, link_idx] += 1
                
                            # add to the total effective FP with a weight factor (multiplied by 2 for bidirectional consideration)
                            self.Total_effective_FP_new_annual[year - 1] += 2 * self.network.weights_array[HL_links_indices[link_idx]]


            # store GSNR values for the current year after subtracting 1.5 dB penalty into a cell array.
            self.GSNR_BVT_array[year - 1] = np.array(GSNR_BVT_per_year) - reduce_GSNR_year

            # append the adjusted GSNR values to the overall 10-year GSNR array.
            self.GSNR_HL4_10Year.append(np.array(GSNR_BVT_per_year) - reduce_GSNR_year)

        # Updating Node degress based on frequency plan
        self.update_hl_node_degrees(hierarchy_level = hierarchy_level,
                                    Year_FP = self.Year_FP)
        
        # BVT license count tracking over the simulation period
        self.calculate_BVT_usage()
        
        # save all simulation results
        self.save_network_results(hierarchy_level = hierarchy_level, 
                        minimum_hierarchy_level = minimum_level, 
                        result_directory = result_directory)

