from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .network import Network
from .band import Band
import os


@dataclass
class PlanningTool:
    """Main class for 6D-MAN network planning and optimization.
    
    This class integrates network topology and band specifications to perform
    multi-band planning and optimization for optical metro-urban networks.
    """
    
    def __init__(self,
                 network_instance: Network,
                 bands: List[Band],
                 period_time: int = 10):
        """Initialize PlanningTool instance.
        
        Args:
            network: Network instance containing topology and hierarchical levels
            bands: List of Band instances for multi-band operation
            constraints: Planning constraints for optimization
        """
        self.network = network_instance
        self.bands = bands
        self.period_time = period_time
        
    def simulate_traffic_initial(self,
                                 num_nodes: int,
                                 monteCarlo_steps: int,
                                 min_rate: float,
                                 max_rate: float,
                                 seed: int, 
                                 result_directory) -> np.ndarray:
        """Perform network-wide optimization.

        Args:
            num_nodes: number of nodes to simulate traffic for
            monteCarlo_steps: number of steps of Monte-Carlo simulation for traffic aggregation
            min_rate: minimum rate of a node
            max_rate: maximum rate of a node
            seed: numer for set random seed to make our random numbers constant in different runs
        
        Returns:
            numpy array containing a rate for each node
        """
        file_path = result_directory / f"{self.network.topology_name}_HL_capacity_final.npz"

        if os.path.exists(file_path):

            print("Loading precomputed HL_capacity_final ...")
            data = np.load(file_path)
            self.HL_capacity_final = data['HL_capacity_final']

        else:
            
            print("Calculate HL_capacity_final ...")
            # storage for random capacities in each Monte Carlo step
            random_capacity_storage = []

            for i in range(monteCarlo_steps):

                # set random seed to make our random numbers constant in different runs
                np.random.seed(seed)

                # create random capacity for each HL4 node (standalone & colocated) uniformly distributed
                random_capacity_local = np.random.uniform(min_rate, max_rate, size = num_nodes)
                random_capacity_storage.append(random_capacity_local)

            random_capacity_storage = np.array(random_capacity_storage)

            # average over Monte Carlo steps to find a final capacity for each HL4 node
            self.HL_capacity_final = random_capacity_storage.mean(axis = 0)

            np.savez_compressed(file_path,
                            HL_capacity_final = self.HL_capacity_final)


    
    def simulate_traffic_annual(self,
                                 lowest_hierarchy_dict: dict,
                                 CAGR: int, 
                                 result_directory) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform network-wide optimization.

        Args:
            lowest_hierarchy_dict: dictionary of lowest hierarchy level nodes (close to access) 
            CAGR: Compounded Annual Growth Rate (CAGR)
        
        Returns:
            Tupple containing four numpy arrays
            ----
                traffic storage of lowest level standalone HL nodes
                traffic storage of lowest level colocated HL nodes
                added traffic lowest level standalone HL nodes
                added traffic lowest level colocated HL nodes
            ----
        """
        num_node_standalone = len(lowest_hierarchy_dict['standalone'])
        num_node_colocated = len(lowest_hierarchy_dict['colocated'])
        num_node_total = num_node_standalone + num_node_colocated
        period_time = self.period_time

        file_path = result_directory / f"{self.network.topology_name}_traffic_matrix.npz"

        if os.path.exists(file_path):

            print("Loading precomputed Traffic Matrix ...")
            data = np.load(file_path)
            added_traffic_annual = data['added_traffic_annual']

            # calculate the number of 100G licences for the first year for each HL4 node
            self.num_100G_licence_annual[0, :] = np.ceil(self.HL_capacity_final / 100)

            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, num_node_colocated:]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, 0:num_node_colocated]
        
        else:

            print("Calculate Traffic Matrix ...")
            lowest_HL_traffic_storage_annual = np.empty(shape = (period_time, num_node_total))
            total_traffic_annual = np.empty(shape = (period_time))
            added_traffic_annual = np.empty(shape = (period_time, num_node_total))
            residual_capacity_annual = np.empty(shape = (period_time, num_node_total))

            # set the capacity of HL4 nodes for first year
            lowest_HL_traffic_storage_annual[0, :] = self.HL_capacity_final

            # calculate total traffic at the fist year
            total_traffic_annual[0] = np.sum(self.HL_capacity_final)

            # calculate added traffic in compare of last year
            added_traffic_annual[0, :] = self.HL_capacity_final

            # calculate the number of 100G licences for the first year for each HL4 node
            self.num_100G_licence_annual[0, :] = np.ceil(self.HL_capacity_final / 100)

            # calculate the residual capcacity of 100G licences for each HL4 node at the first year
            residual_capacity_annual[0, :] = 100 * self.num_100G_licence_annual[0, :] - lowest_HL_traffic_storage_annual[0, :]

            for year in range(1, period_time):

                # 40% increase in the value of traffic per HL4 node in related to the last yeat
                lowest_HL_traffic_storage_annual[year, :] = (1 + CAGR) * lowest_HL_traffic_storage_annual[year - 1, :]

                # calculate total traffic of this year
                total_traffic_annual[year] = np.sum(lowest_HL_traffic_storage_annual[year, :])

                # calcualte the added traffic in compare of last year
                added_traffic_annual[year, :] = lowest_HL_traffic_storage_annual[year, :] - lowest_HL_traffic_storage_annual[year - 1, :]

                # calculate the number of 100G licences for this year for each HL4 node
                self.num_100G_licence_annual[year, :] = np.ceil(lowest_HL_traffic_storage_annual[year, :] / 100)

                # calculate the residual capcacity of 100G licences for each HL4 node at this year
                residual_capacity_annual[year, :] = 100 * self.num_100G_licence_annual[year, :] - lowest_HL_traffic_storage_annual[year, :]

            # separating Standalone and Co-located HL4 Traffic Data
            self.lowest_HL_added_traffic_annual_standalone = added_traffic_annual[:, num_node_colocated:]
            self.lowest_HL_added_traffic_annual_colocated = added_traffic_annual[:, 0:num_node_colocated]

            np.savez_compressed(file_path,
                            added_traffic_annual = added_traffic_annual)
                
    
    
    def initialize_planner(self, 
                           num_fslots: int,
                           hierarchy_level: int,
                           minimum_hierarchy_level: int,
                           rolloff: float = 0.1,
                           SR: float = 40 * 1e9,
                           BVT_type: int = 1,
                           Max_bit_rate_BVT: np.ndarray = np.array([400])):
        """Calculate transmission metrics for a path using specific band.
        
        Args:
            HL_dict: dictionary that containing specific HL nodes
            rolloff: rolloff factor
            SR: Symbol Rate (baud rate)
            Max_bit_rate_BVT: Maximum bit rate supported per BVT (Gbps)

        Returns:

        """
        self.Max_bit_rate_BVT = Max_bit_rate_BVT
        self.BVT_type = BVT_type



        num_node_standalone = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['standalone'])
        num_node_colocated = len(self.network.hierarchical_levels[f"HL{hierarchy_level}"]['colocated'])
        num_links = len(self.network.all_links)
        period_time = self.period_time
            
        subgraph, _ = self.network.calculate_subgraph(hierarchy_level, minimum_hierarchy_level)

        # define channel spacing (fix grid)
        channel_spacing = self.bands[0].channel_spacing

        self.num_fslots = num_fslots

        # calculates the actual channel bandwidth
        B_ch = SR * (1 + rolloff)

        # compute required frequency slots per BVT
        self.Required_FS_BVT = np.ceil(B_ch / (channel_spacing * 1e12)).astype(int)

        # define a matrix storing Fiber Placement (FP) status for each network edge over time  
        self.Year_FP = np.zeros((period_time, num_links), dtype = np.int32)

        # similar to Year_FP, but specifically for HL4 colocated nodes
        self.Year_FP_HL_colocated =  np.zeros((period_time, num_node_colocated))
    
        # store unallocated bandwidth (residual traffic) for Standalone HL4s 
        self.Residual_Throughput_BVT_standalone_HLs = np.zeros((period_time, num_node_standalone))

        # store unallocated bandwidth (residual traffic) for Co-located HL4s
        self.Residual_Throughput_BVT_colocated_HLs = np.zeros((period_time, num_node_colocated))

        # Total number of BVTs deployed across all years
        self.HL_BVT_number_all_annual = np.zeros((period_time, len(Max_bit_rate_BVT)))

        # track the number of BVTs deployed for different spectrum bands each year
        self.HL_BVT_number_Cband_annual = np.zeros((period_time, len(Max_bit_rate_BVT))) # Traditional optical band
        self.HL_BVT_number_SuperCband_annual = np.zeros((period_time, len(Max_bit_rate_BVT))) # Extended C-band for higher capacity
        self.HL_BVT_number_SuperCLband_annual = np.zeros((period_time, len(Max_bit_rate_BVT))) # Includes L-band for extreme capacity scaling

        # LSP (Label Switched Path) arrays track optical paths over num_FS and links  
        max_fps_link = 20
        self.LSP_array = np.zeros((self.num_fslots, num_links, max_fps_link)) # track spectrum allocation for all links
        self.LSP_array_Colocated = np.zeros((self.num_fslots, num_node_colocated, max_fps_link)) # Specifically tracks lightpaths for co-located HL4s

        # track how many optical links are deployed per year in different bands
        self.num_link_LBand_annual = np.zeros(period_time)
        self.num_link_SupCBand_annual = np.zeros(period_time)
        self.num_link_CBand_annual = np.zeros(period_time)

        # track how many optical links are deployed per year in different bands
        self.Year_FP_new =  np.zeros((period_time, subgraph.number_of_edges()))

        # track the total new fiber placements for each year
        self.Total_effective_FP_new_annual = np.zeros(period_time)

        self.Total_effective_FP = np.zeros(period_time)
        self.node_capacity_profile_array = np.zeros(shape = (period_time, self.network.adjacency_matrix.shape[0], minimum_hierarchy_level))

        # store GSNR values for different BVTs over multiple years
        self.GSNR_BVT_array = [None] * period_time

        # store GSNR data for HL4 nodes over 10 years
        self.GSNR_HL4_10Year = []

        self.Residual_100G = np.zeros(self.network.adjacency_matrix.shape[0])

        self.num_100G_licence_annual = np.zeros(shape = (period_time, self.network.adjacency_matrix.shape[0]))

        # Arrays to count usage
        self.CBand_usage = np.zeros((self.period_time, num_links), dtype=int)
        self.superCBand_usage = np.zeros((self.period_time, num_links), dtype=int)
        self.superCLBand_usage = np.zeros((self.period_time, num_links), dtype=int)

        self.traffic_flow_array = np.zeros((self.period_time, num_links), dtype=float)

        self.primary_path_storage =  -1 * np.ones(shape = (self.network.adjacency_matrix.shape[0]), dtype=int)



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
                            Year_FP_pair: np.ndarray,
                            HL_SubNetwork_links: np.ndarray) -> dict:
        """Calculate total cost for a path solution.
        
        Args:
            path: Path dictionary containing nodes, links, and distances
            metrics: Dictionary of calculated metrics
            
        Returns:
            Total cost value
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
            link_congested_path = np.zeros(len(linkList_path))

            # # iterate through each link in the primary path
            # for link_idx in range(len(linkList_path)):

            #     # calc congestion as the number of nonzero elements in LSP_array for each link in the primary path
            #     link_congested_path[link_idx] = np.count_nonzero(LSP_array_pair[:, linkList_path[link_idx], FP_counter_links[link_idx] - 1])
            
            # # sort the primary path links based on congestion value in descending order
            # sorted_indices = np.argsort(link_congested_path)[::-1]
            # linkList_path_sorted = np.array(linkList_path)[sorted_indices]

            link_congested_primary = np.array(
                        [np.count_nonzero(LSP_array_pair[:, link, FP_counter_links[i] - 1]) for i, link in
                         enumerate(linkList_path)])
            
            # Sort the unique congestion levels in descending order
            unique_sorted_link_congested_primary = np.sort(np.unique(link_congested_primary))[::-1]

            # Sort links based on congestion
            linkList_path_sorted = np.concatenate(
                [linkList_path[link_congested_primary == congestion] for congestion in
                    unique_sorted_link_congested_primary])

            ###################################################
            #  fiber and spectrum assignment for primary path
            ###################################################

            # define a counter for fibers
            link_counter = 0
            
            # initialize an array for Maximum frequency slot used
            f_max_path = np.zeros(BVT_number) 
            
            # initialize an array for Cost function values
            cost_FP_all_BVT_path = np.zeros(BVT_number)  

            # initialize an array for Maximum fiber pairs assigned
            FP_max_path  = np.zeros(BVT_number)

            # iterate through BVTs
            for BVT_counter in range(BVT_number):
                
                # spectrum assignment continues until an available fiber is found
                Flag_SA_continue_path = 1

                # Note: fiber Pair assignment is done based on first fit
                while Flag_SA_continue_path:

                    # PST_parimary is a binary vector that will store whether each Frequency Slot is occupied or available
                    PST_path = np.zeros(self.num_fslots)

                    # iterate through each frequency slot
                    for FS in range(self.num_fslots):
                        
                        # vector_state_FS will contain one value per link, indicating whether the slot is free (0) or used (1) on a certain link
                        vector_state_FS = np.empty(len(linkList_path_sorted))

                        # check the status of the current frequency slot (FS) for each link
                        for link_idx in range(len(linkList_path_sorted)):

                            # LSP_array contain a number for each FS in each link 
                            vector_state_FS[link_idx] = LSP_array_pair[FS, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] 

                        # check that there is any link that use that frequecy slot or not
                        if any(vector_state_FS):
                            PST_path[FS] = 1
                        else:
                            PST_path[FS] = 0

                    # keep track of the number of contiguous free slots
                    FS_count = 0

                    # PST_vector_aux stores differences in spectrum occupancy
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1)

                    # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen 
                    flag_First_Fit = 1

                    # stores the selected frequency slots
                    FS_path = []
                    
                    if np.any(PST_vector_aux):

                        # find the first index that 0 changes to 1 (start of free block)
                        startIndex = np.where(PST_vector_aux < 0)[0]

                        # find the first index that 1 changes to 0 (end of free block)
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1

                        # compute the length of each contiguous free block
                        duration = endIndex - startIndex + 1

                        # search for exactly matching free blocks
                        # Exact_Fit = np.atleast_1d(duration == BVT_required_FS_HL).nonzero()[0]
                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0]
                        
                        # search for the first block that match
                        # First_Fit = np.atleast_1d(duration > BVT_required_FS_HL).nonzero()[0]
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0]

                        # if Exact_Fit is found select the first exact-fit slot and assigns it
                        if Exact_Fit.size > 0:
                            
                            # select the first available exact-fit slot
                            FS_count = duration[Exact_Fit[0]]
                            b_1 = np.arange(startIndex[Exact_Fit[0]], startIndex[Exact_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0

                        # if no Exact-Fit, use First-Fit
                        elif First_Fit.size > 0 and flag_First_Fit:
                            
                            # select the first available larger slot
                            FS_count = duration[First_Fit[0]]
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]

                            
                    print('FS_path', FS_path)

                    # if enough contiguous slots are found, the assignment proceeds
                    if FS_count >= BVT_required_FS_HL:

                        GSNR_BVT1 = [0]

                        for link_idx in range(len(linkList_path_sorted)):
                            
                            if path_type == 'primary':

                                # update LSP_array_pair to reflect the new assignment
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = node_list[node_IDx] + 1

                            elif path_type == 'secondary':

                                # update LSP_array_pair to reflect the new assignment with a negative identifier
                                LSP_array_pair[FS_path, linkList_path_sorted[link_idx], FP_counter_links[link_idx] - 1] = -(node_list[node_IDx] + 1)
                            
                            # print('FS_path', FS_path)

                            # compute GSNR 
                            GSNR_BVT1 += (10 ** (GSNR_link[link_idx, FS_path] / 10)) ** -1
                                                    
                        # stop searching for more slots
                        Flag_SA_continue_path = 0
                        
                        for link_counter_local in range(len(FP_counter_links)):

                            # update the Year_FP_pair to record spectrum usage for each link
                            Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]] =  max(Year_FP_pair[year - 1, linkList_path_sorted[link_counter_local]], FP_counter_links[link_counter_local])

                        # calculate the cost of assigning fiber pairs for the BVT_counter-th BVT
                        cost_FP_all_BVT_path[BVT_counter] = np.dot(FP_counter_links, self.network.weights_array[linkList_path_sorted])

                        # The final frequency slot used (FS_path(end)) determines the spectrum band
                        if FS_path[-1] <= 95:
                            BVT_CBand_count_path += 2 # The +2 accounts for dual polarization usage in optical networks
                            band_used = 0  # C-band
                        elif 96 <= FS_path[-1] <= 119:
                            BVT_superCBand_count_path += 2
                            band_used = 1  # superC
                        else:
                            BVT_superCLBand_count_path += 2
                            band_used = 2  # superCL

                        for link_id in linkList_path_sorted:
                            if band_used == 0:
                                self.CBand_usage[year - 1, link_id] += 1
                            elif band_used == 1:
                                self.superCBand_usage[year - 1, link_id] += 1
                            elif band_used == 2:
                                self.superCLBand_usage[year - 1, link_id] += 1

                    
                     # If no suitable spectrum was found, move to the next FP link
                    else:
                        # # move to the next fiber pair
                        # link_counter += 1

                        # # if link_counter exceeds the number of available fibers, reset it to 1
                        # if link_counter > len(FP_counter_links):
                        #     link_counter  = 0 # Reset link counter

                        # # increase the fiber pair counter for the selected fiber
                        # FP_counter_links[link_counter - 1] += 1

                        link_counter = (link_counter + 1) % len(FP_counter_links)
                        FP_counter_links[link_counter - 1] += 1

                    # store the highest frequency slot index used for this BVT
                    # help in tracking the last occupied frequency slot for future allocations
                if FS_path == []:
                    print('!!!!!!!!!!!!!!')
                    print('node_IDx: ', node_IDx)
                f_max_path[BVT_counter] = max(FS_path)

                # record the maximum fiber pair index used
                # useful for managing fiber resources
                FP_max_path[BVT_counter] = max(FP_counter_links)

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
        
        # if path_IDX is None
        else:

            # define a counter for fibers
            FP_counter_links = 0

            # determine how many frequency slots (FS) are required for the selected BVT type 
            BVT_required_FS_HL = self.Required_FS_BVT
            
            # iterate through BVTs
            for BVT_counter in range(BVT_number):
                
                # spectrum assignment continues until an available fiber is found
                Flag_SA_continue_path = 1

                # Note: fiber Pair assignment is done based on first fit
                while Flag_SA_continue_path:

                    # PST_path is a binary vector that will store whether each Frequency Slot is occupied or available
                    PST_path = self.LSP_array_Colocated[:, node_IDx, FP_counter_links].T

                    # keep track of the number of contiguous free slots
                    FS_count = 0

                    # PST_vector_aux stores differences in spectrum occupancy
                    PST_vector_aux = np.diff(np.concatenate(([1], PST_path, [1])), n = 1)

                    # this flag ensures that if exact-fit slots aren’t found, the first available larger slot is chosen 
                    flag_First_Fit = 1

                    # stores the selected frequency slots
                    FS_path = []
                    
                    if np.any(PST_vector_aux != 0):

                        # find the first index that 0 changes to 1 (start of free block)
                        startIndex = np.where(PST_vector_aux < 0)[0]

                        # find the first index that 1 changes to 0 (end of free block)
                        endIndex = np.where(PST_vector_aux > 0)[0] - 1

                        # compute the length of each contiguous free block
                        duration = endIndex - startIndex + 1

                        # search for exactly matching free blocks
                        Exact_Fit = np.where(duration == BVT_required_FS_HL)[0]
                        
                        # search for the first block that match
                        First_Fit = np.where(duration > BVT_required_FS_HL)[0]

                        # if Exact_Fit is found select the first exact-fit slot and assigns it
                        if Exact_Fit.size > 0:

                            FS_count = duration[Exact_Fit[0]]
                            b_1 = np.arange(startIndex[Exact_Fit[0]], endIndex[Exact_Fit[0]] + 1)
                            FS_path = b_1[:BVT_required_FS_HL]
                            flag_First_Fit = 0


                        # if no Exact-Fit, use First-Fit
                        elif First_Fit.size > 0 and flag_First_Fit:

                            # select the first available larger slot
                            FS_count = duration[First_Fit[0]]
                            b_1 = np.arange(startIndex[First_Fit[0]],
                                            startIndex[First_Fit[0]] + BVT_required_FS_HL)
                            FS_path = b_1[:BVT_required_FS_HL]
                            
                    print('FS_path', FS_path)

                    # if enough contiguous slots are found, the assignment proceeds
                    if FS_count >= BVT_required_FS_HL:

                        # update LSP_array_pair to reflect the new assignment
                        self.LSP_array_Colocated[FS_path, node_IDx, FP_counter_links] = 1

                        # stop searching for more slots
                        Flag_SA_continue_path = 0
                        
                        # update the Year_FP_pair to record spectrum usage for each link
                        self.Year_FP_HL_colocated[year - 1, node_IDx] =  max(self.Year_FP_HL_colocated[year - 1, node_IDx], FP_counter_links + 1)

                        # calculate the cost of assigning fiber pairs for the BVT_counter-th BVT
                        # The final frequency slot used (FS_primary(end)) determines the spectrum band
                        if FS_path[-1] <= 95:

                            # The +2 accounts for dual polarization usage in optical networks
                            self.HL_BVT_number_Cband_annual[year - 1] += 2

                        elif 96 <= FS_path[-1] <= 119:
                            self.HL_BVT_number_SuperCband_annual[year - 1]  += 2
                        
                        else:
                            self.HL_BVT_number_SuperCLband_annual[year - 1]  += 2
                    
                    # If no slots are available, increment the fiber pair counter and retry
                    else:
                        
                        FP_counter_links = FP_counter_links + 1

            return self.Year_FP_HL_colocated

    def update_node_degree(self, 
                            HL_dict: dict,
                            Year_FP: np.ndarray) -> np.ndarray:
        """Calculate total cost for a path solution.
        
        Args:
            path: Path dictionary containing nodes, links, and distances
            metrics: Dictionary of calculated metrics
            
        Returns:
            Total cost value
        """
        HL_Standalone = HL_dict['standalone']

        # calculate node degree of different HLx
        HL_degrees = self.network.get_node_degrees(HL_Standalone)

        # initialize node degree tracking with initial topology degrees.
        degree_node_all_topo_HL_final = HL_degrees.copy()

        degree_number_HLs = np.zeros(self.period_time)

        # store the initial average node degree.
        degree_number_HLs[0] = np.mean(HL_degrees[:, 1])

        # for year in range(2, 10):

        #     for link_idx in range(len(self.network.all_links)):
                
        #         if Year_FP[year - 1, link_idx] != Year_FP[year - 2, link_idx]:
                    
        #             src_node = self.network.all_links[link_idx, 0]
        #             dest_node = self.network.all_links[link_idx, 1]

        #             # update source node degree if it's in the standalone node list.
        #             if src_node in HL_Standalone:
                        
        #                 indices = np.where(degree_node_all_topo_HL_final[:, 0] == src_node)[0]               
                        
        #                 degree_node_all_topo_HL_final[indices, 1] += (Year_FP[year - 1, link_idx] - Year_FP[year - 2, link_idx])
                        
        #             # update destination node degree if it's in the standalone node list.
        #             if dest_node in HL_Standalone:

        #                 indices = np.where(degree_node_all_topo_HL_final[:, 0] == dest_node)[0]

        #                 # print('index:', indices)
        #                 degree_node_all_topo_HL_final[indices, 1] += (Year_FP[year - 1, link_idx] - Year_FP[year - 2, link_idx])

        #     degree_number_HLs[year - 1] = np.mean(degree_node_all_topo_HL_final[:, 1])


        for year in range(2, self.period_time + 1):

            for link_counter in range(len(self.network.all_links)):

                if Year_FP[year - 1, link_counter] != Year_FP[year - 2, link_counter]:
                    src_node = self.network.all_links[link_counter, 0]
                    dest_node = self.network.all_links[link_counter, 1]

                    # Check if s_node is in HL_StandAlone
                    if src_node in HL_Standalone:
                        # Find indices where HL_StandAlone equals s_node
                        indices = np.where(HL_Standalone == src_node)[0]

                        # Update degree_node_all_topo_HL_final at those indices
                        degree_node_all_topo_HL_final[indices, 1] += Year_FP[year - 1, link_counter] - Year_FP[
                            year - 2, link_counter]

                    # Check if t_node is in HL_StandAlone
                    if dest_node in HL_Standalone:
                        # Find indices where HL_StandAlone equals t_node
                        indices = np.where(HL_Standalone == dest_node)[0]

                        # Update degree_node_all_topo_HL_final at those indices
                        degree_node_all_topo_HL_final[indices, 1] += Year_FP[year - 1, link_counter] - Year_FP[
                            year - 2, link_counter]

            degree_number_HLs[year - 1] = np.mean(degree_node_all_topo_HL_final[:, 1])

        self.degree_number_HLs = degree_number_HLs

    def BVT_count(self) -> dict:
        """Calculate total cost for a path solution.
        
        Args:
            path: Path dictionary containing nodes, links, and distances
            metrics: Dictionary of calculated metrics
            
        Returns:
            Total cost value
        """
        period_time = self.period_time

        HL_All_100G_lincense = np.zeros(period_time)
        HL_BVTNum_All = np.zeros(period_time)
        HL_BVTNum_CBand = np.zeros(period_time)
        HL_BVTNum_SuperCBand = np.zeros(period_time)
        HL_BVTNum_LBand = np.zeros(period_time)

        for year_num in range(period_time):

            HL_BVTNum_All[year_num] = np.sum(self.HL_BVT_number_all_annual[:year_num + 1])
            HL_BVTNum_CBand[year_num] = np.sum(self.HL_BVT_number_Cband_annual[:year_num + 1])
            HL_BVTNum_SuperCBand[year_num] = np.sum(self.HL_BVT_number_SuperCband_annual[:year_num + 1])
            HL_BVTNum_LBand[year_num] = np.sum(self.HL_BVT_number_SuperCLband_annual[:year_num + 1])

            if year_num > 0:
                HL_All_100G_lincense[year_num] = np.sum(4 * self.num_100G_licence_annual[year_num, :]) + HL_All_100G_lincense[year_num - 1]
            else:
                HL_All_100G_lincense[year_num] = np.sum(4 * self.num_100G_licence_annual[year_num, :])
        
        
        self.HL_All_100G_lincense = HL_All_100G_lincense
        self.HL_BVTNum_CBand = HL_BVTNum_CBand
        self.HL_BVTNum_SuperCBand = HL_BVTNum_SuperCBand
        self.HL_BVTNum_LBand = HL_BVTNum_LBand
        self.HL_BVTNum_All = HL_BVTNum_All

    def save_files(self,
                   hierarchy_level: int,
                   minimum_hierarchy_level: int, 
                   result_directory):
        """Calculate total cost for a path solution.
        
        Args:
            path: Path dictionary containing nodes, links, and distances
            metrics: Dictionary of calculated metrics
            
        Returns:
            Total cost value
        """            
        subgraph, _ = self.network.calculate_subgraph(hierarchy_level, minimum_hierarchy_level)
        
        HL_subnet_links = np.array(list(subgraph.edges(data = 'weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis = 2), axis = 1)
        HL_links_indices = np.where(mask)[0]

        HL_CDegree_Domain = 2 * self.num_link_CBand_annual
        HL_SuperCDegree_Domain = 2 * self.num_link_SupCBand_annual
        HL_LDegree_Domain = 2 * self.num_link_LBand_annual

        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz',
                            HL_All_100G_lincense = self.HL_All_100G_lincense,
                            HL_BVTNum_All = self.HL_BVTNum_All,
                            HL_BVTNum_CBand = self.HL_BVTNum_CBand,
                            HL_BVTNum_SuperCBand = self.HL_BVTNum_SuperCBand,
                            HL_BVTNum_LBand = self.HL_BVTNum_LBand)
        
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_link_info.npz',
                            HL_links_indices = HL_links_indices,
                            num_link_CBand_annual = self.num_link_CBand_annual,
                            num_link_SupCBand_annual = self.num_link_SupCBand_annual,
                            num_link_LBand_annual = self.num_link_LBand_annual,
                            HL_CDegree_Domain = HL_CDegree_Domain,
                            HL_SuperCDegree_Domain = HL_SuperCDegree_Domain,
                            HL_LDegree_Domain = HL_LDegree_Domain,
                            Total_effective_FP_new_annual = self.Total_effective_FP_new_annual,
                            HL_FPNum = self.Year_FP_new,
                            HL_FPNumCo = self.Year_FP_HL_colocated,
                            degree_number_HLs = self.degree_number_HLs, 
                            CBand_usage = self.CBand_usage, 
                            superCBand_usage = self.superCBand_usage,
                            superCLBand_usage = self.superCLBand_usage, 
                            traffic_flow_array = self.traffic_flow_array, 
                            primary_paths = self.primary_path_storage)
        
        np.savez_compressed(result_directory / f'{self.network.topology_name}_HL{hierarchy_level}_node_capacity_profile_array.npz',
                            node_capacity_profile_array = self.node_capacity_profile_array)

        
    def run_planner(self, 
                    HL_dict: Dict, 
                    pairs_disjoint: pd.DataFrame,
                    kpair_standalone: int,
                    kpair_colocated: int,
                    candidate_paths_standalone_df: pd.DataFrame,
                    candidate_paths_colocated_df: pd.DataFrame,
                    GSNR_opt_link: np.ndarray,
                    prev_hierarchy_level: int,
                    hierarchy_level: int,
                    minimum_level: int,
                    node_cap_update_idx: int, 
                    result_directory) -> float:
        """Calculate total cost for a path solution.
        
        Args:
            path: Path dictionary containing nodes, links, and distances
            metrics: Dictionary of calculated metrics
            
        Returns:
            Total cost value
        """
        HL_standalone = HL_dict['standalone']
        HL_colocated = HL_dict['colocated']

        subgraph, _ = self.network.calculate_subgraph(hierarchy_level, minimum_level)
        HL_subnet_links = np.array(list(subgraph.edges(data = 'weight')))
        mask = np.any(np.all(self.network.all_links[:, None] == HL_subnet_links, axis=2), axis=1)
        HL_links_indices = np.where(mask)[0]

        # GSNR to reduce when storing GSNR values per year in the calculations
        if hierarchy_level == 4 or hierarchy_level == 5: 
            reduce_GSNR_year = 1.5  # 1.5 HL4, 2 HL3, 5.5 HL2
        elif hierarchy_level == 3:
             reduce_GSNR_year = 2
        elif hierarchy_level == 2:
             reduce_GSNR_year = 5.5 

        period_time = self.period_time

        # array for saving destinations of standalone nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_standalone_dest_profile = np.zeros(shape = (period_time, len(HL_standalone), 2), dtype = np.int8)

        # array for saving destinations of colocated nodes in each year, in the third dimension 0 is for primary destination and 1 is for secondary destination
        HL_colocated_dest_profile = np.zeros(shape = (period_time, len(HL_colocated)), dtype = np.int8)
        
        for year in range(1 , period_time + 1):

            # store traffic capacity assigned to each node, in the third dimension 0 is for primary SNR and 1 is for secondary SNR
            # print('year: ', year)
            if year == 6:
                print('hello')

            if hierarchy_level == minimum_level:
                    node_capacity_profile = np.zeros(shape = (len(HL_colocated) + len(HL_standalone), minimum_level))
            else:
                    node_capacity_profile_array_prev_hl = np.load(result_directory /  f"{self.network.topology_name}_HL{prev_hierarchy_level}_node_capacity_profile_array.npz")['node_capacity_profile_array']
                    node_capacity_profile = node_capacity_profile_array_prev_hl[year - 1, :, :]

                    self.num_100G_licence_annual[year - 1, :] = np.ceil(0.01 * (node_capacity_profile[:, node_cap_update_idx + 1] - self.Residual_100G))
                    self.Residual_100G += 100 * self.num_100G_licence_annual[year - 1, :] - node_capacity_profile[:, node_cap_update_idx + 1]

                    self.CBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['CBand_usage']
                    self.superCBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['superCBand_usage']
                    self.superCLBand_usage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['superCLBand_usage']
                    self.traffic_flow_array = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['traffic_flow_array']
                    self.primary_path_storage = np.load(result_directory /  f'{self.network.topology_name}_HL{prev_hierarchy_level}_link_info.npz')['primary_paths']

            #######################################################
            # Part 1: Spectrum assignment for standalone HLs
            #######################################################

            # tracks signal quality (GSNR) per BVT
            GSNR_BVT_per_year = []

            for node_idx in range(len(HL_standalone)):
                
                if node_idx == 53:
                    print(node_idx) 

                # get traffic demand for this node in this year
                if hierarchy_level == minimum_level:
                    HL_needed_traffic = self.lowest_HL_added_traffic_annual_standalone[year - 1, node_idx]
                else:
                    HL_needed_traffic = node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1]
                
                
                if year != 1: # if it isnt the first year

                    # subtract residual throughput (unallocated traffic from previous years)
                    HL_pure_throughput_to_assign = HL_needed_traffic - self.Residual_Throughput_BVT_standalone_HLs[year - 2, node_idx]
                else: # if it is the first year
                    HL_pure_throughput_to_assign = HL_needed_traffic
                    
                if hierarchy_level == minimum_level:
                    # store traffic capacity assigned to current node
                    node_capacity_profile[HL_standalone[node_idx], node_cap_update_idx + 1] = HL_needed_traffic
                
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

                    # extract the first precomputed K-shortest paths for the current standalone HL4 
                    candidate_path_pair = pairs_disjoint[pairs_disjoint['src_node'] == HL_standalone[node_idx]]

                    num_K_pair_final = self.network.calc_num_pair(pairs_disjoint_df = pairs_disjoint)
                    num_kpairs = min(num_K_pair_final[node_idx], kpair_standalone)

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

                    for final_K_pair_counter in range(num_kpairs):

                        if final_K_pair_counter == 68:
                            print('rrrr')

                        # track Label Switched Paths (LSPs) for allocated routes
                        LSP_array_pair = self.LSP_array.copy()

                        # define a variable to track frequency slots (FS) occupied per year
                        Year_FP_pair = self.Year_FP.copy()

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
                                                                                                   Year_FP_pair = Year_FP_pair, 
                                                                                                   HL_SubNetwork_links = HL_links_indices)
                        
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
                                                                                                     Year_FP_pair = Year_FP_pair,
                                                                                                     HL_SubNetwork_links = HL_links_indices)
                        
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

                    best_pair_links_tuple = paths_storage[index_feasible_pair[0]]
                    for links_arr in best_pair_links_tuple:
                        for link in links_arr:
                            self.traffic_flow_array[year - 1, link] += HL_needed_traffic


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


            print('end of standalone')
            #######################################################
            # Part 2: Spectrum assignment for colocated HLs
            #######################################################

            # Initialize the cost function matrix with infinity values for each metric (f_max, N_hop, cost, GSNR, FP_max)
            cost_func = np.inf * np.ones(shape = (1, 5))

            max_path_secondary = candidate_paths_colocated_df.groupby('src_node').size().to_numpy()

            for node_idx in range(len(HL_colocated)):
                
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
                                                                    Year_FP_pair = None, 
                                                                    HL_SubNetwork_links = HL_links_indices)
                    
                    self.Year_FP_HL_colocated = Year_FP_HL_colocated
                    
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

                    for final_K_pair_counter in range(num_kpairs):

                        # track Label Switched Paths (LSPs) for allocated routes
                        LSP_array_pair = self.LSP_array.copy()

                        # define a variable to track frequency slots (FS) occupied per year
                        Year_FP_pair = self.Year_FP.copy()

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
                                                                                                     Year_FP_pair = Year_FP_pair, 
                                                                                                     HL_SubNetwork_links = HL_links_indices)
                        
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
                for FP_counter in range(20):

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

            # # check if any GSNR value (with a 2 dB margin subtracted) falls below the target SNR for 64-QAM.
            # if any(GSNR_BVT_per_year - 2 < target_SNR_dB_64):
            #     print('error')

            # append the adjusted GSNR values to the overall 10-year GSNR array.
            self.GSNR_HL4_10Year.append(np.array(GSNR_BVT_per_year) - reduce_GSNR_year)


        # Updating Node degress based on frequency plan
        self.update_node_degree(HL_dict = HL_dict,
                                Year_FP = self.Year_FP)
        
        # BVT license count tracking over the simulation period
        self.BVT_count()
        
        # save all simulation results
        self.save_files(hierarchy_level = hierarchy_level, 
                        minimum_hierarchy_level = minimum_level, 
                        result_directory = result_directory)

