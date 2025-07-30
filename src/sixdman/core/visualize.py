from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
from .network import Network
import matplotlib.pyplot as plt
from itertools import accumulate

class analyse_result:
    """A class representing the optical network topology and its properties.
    
    This class handles the network topology, hierarchical levels, and path computation
    for the 6D-MAN planning tool.
    
    Attributes:
        graph (nx.Graph): NetworkX graph representing the network topology
        hierarchical_levels (Dict): Dictionary containing nodes in each hierarchical level
        num_spans (List): each element of this list is number of spans of specific link in the topology
    """
    
    def __init__(self,
                 network_instance: Network,
                 period_time: int, 
                 processing_level_list: List, 
                 result_directory):
        """Initialize Network instance.
        
        Args:
            num_spans: A list containing of the number of spans per link in the topology
        """
        self.network = network_instance
        self.period_time = period_time
        self.processing_level_list = processing_level_list
        self.result_directory = result_directory
        
    def load_data(self):
        """Load network topology from a .mat file.
        
        Args:
            filepath: Path to the .mat file containing network topology
            matrixName: name of the adjacancy matrix as a MATLAB variable
        """
        link_data = {}
        bvt_data = {}
        for hierarchy_level in self.processing_level_list:

            link_data_path = self.result_directory / f"{self.network.topology_name}_HL{hierarchy_level}_link_info.npz"
            bvt_data_path = self.result_directory / f"{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz"
            
            try:
                        
                hl_link_data = np.load(link_data_path)
                link_data[f"HL{hierarchy_level}"] = hl_link_data
                self.link_data = link_data

            except Exception as e:
                raise IOError(f"Failed to load data from {link_data_path}: {str(e)}")

            try:
                hl_BVT_data = np.load(bvt_data_path)
                bvt_data[f"HL{hierarchy_level}"] = hl_BVT_data  
                self.bvt_data = bvt_data

            except Exception as e:
                raise IOError(f"Failed to load data from {bvt_data_path}: {str(e)}")
        
            
    def plot_link_state(self, 
                        splitter: List,
                        save_flag: int, 
                        save_suffix: str = "",
                        flag_plot: int = 1):
        """Set the hierarchical levels of nodes in the network with flexible HLx input.
        
        Args:
            kwargs: Any number of HLx_standalone and/or HLx_colocated lists, e.g.,
                    HL1_standalone=[...], HL2_colocated=[...], etc.
        Returns:
            dict: hierarchical_levels dictionary
        """
        self.load_data()
        link_state_HL_partisioned = np.empty(shape = (self.period_time, 0))
        for hierarchy_level in self.processing_level_list:
            link_state_HL_partisioned = np.hstack((link_state_HL_partisioned, self.link_data[f"HL{hierarchy_level}"]['HL_FPNum']))


        year = np.arange(1, self.period_time + 1)  # From 1 to 10 inclusive

        if flag_plot == 1:

            plt.figure(figsize=(7, 5))
            plt.title("ALL HLs: link state profile")

            # Display the image
            plt.imshow(link_state_HL_partisioned.T, aspect = 'auto', interpolation = 'none', origin = 'upper')

            plt.xlabel("Year")
            plt.ylabel("Link index")

            # Set color limits (caxis equivalent)
            plt.clim(0, 3)
            plt.colorbar(label='State')

            # Set axis limits (MATLAB axis([x1 x2 y1 y2]))
            plt.xlim(0.5, self.period_time + 0.5)
            plt.xticks(np.arange(1, 11))  # So ticks are labeled 1 to 10

            # Add grid
            plt.grid(True)

            splitter_converted = list(accumulate(splitter))

            for i in range(len(splitter_converted)):
                plt.plot(year, splitter_converted[i] * np.ones_like(year), 'k--', linewidth=1)

            plt.tight_layout()

            if save_flag:
                plt.savefig(self.result_directory / f"{self.network.topology_name}_Link_State{save_suffix}.png",
                            dpi=300, bbox_inches='tight')
                    
            plt.show()

        else:
            return link_state_HL_partisioned

    def plot_FP_usage(self,
                      save_flag: int, 
                      save_suffix: str = "",
                      flag_plot: int = 1):
        """Set the hierarchical levels of nodes in the network with flexible HLx input.
        
        Args:
            kwargs: Any number of HLx_standalone and/or HLx_colocated lists, e.g.,
                    HL1_standalone=[...], HL2_colocated=[...], etc.
        Returns:
            dict: hierarchical_levels dictionary
        """
        self.load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:

            # Plotting
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total FP Usage [km] and [number of FP]")

            # Right Y-axis: Linear scale for degrees
            ax2 = ax1.twinx()

        # Default fallback labels and markers
        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']

        # Cumulative Fiber Pair usage in km
        km_total = 0
        fp_total = 0
        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            fp = np.sum(self.link_data[f"HL{hierarchy_level}"]['HL_FPNum'], axis = 1)

            if flag_plot == 1:
                ax1.semilogy(year, km, 
                            f"b-{default_markers[hierarchy_level]}", 
                            label = f"HL{hierarchy_level}s[km]", linewidth = 1.5)
                ax2.plot(year, fp, 
                        f"r-.{default_markers[hierarchy_level]}", 
                        label = f'HL{hierarchy_level}s[#]', linewidth = 1.5)
            km_total += km
            fp_total += fp

        if flag_plot == 1:
            ax1.semilogy(year, km_total, 
                            f"b-+", 
                            label='Total[km]', linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color='blue')
            ax1.set_xlabel("Year")
            ax1.tick_params(axis='y', labelcolor='blue')


            ax2.plot(year, fp_total, 
                    'r-+', label = 'Total[#]', linewidth = 1.5)
            ax2.set_ylabel("Cumulative Nodal Degree Number", color = 'red')
            ax2.tick_params(axis = 'y', labelcolor = 'red')
            ax2.set_ylim(0, 250) # # Set right Y-axis limits and ticks (linear)
            ax2.set_yticks(np.arange(0, 251, 50))

            # Grid, legend, and layout
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.result_directory / f"{self.network.topology_name}_FP_Usage{save_suffix}.png", 
                            dpi=300, bbox_inches='tight')
                
            plt.show()

    def plot_FP_degree(self,
                       save_flag: int, 
                       save_suffix: str = "",
                       flag_plot: int = 1):
        """Set the hierarchical levels of nodes in the network with flexible HLx input.
        
        Args:
            kwargs: Any number of HLx_standalone and/or HLx_colocated lists, e.g.,
                    HL1_standalone=[...], HL2_colocated=[...], etc.
        Returns:
            dict: hierarchical_levels dictionary
        """
        self.load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:

            # Plotting
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("FP [km] and Degree")

            # Right Y-axis: Linear scale for degrees
            ax2 = ax1.twinx()

        # Default fallback labels and markers
        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']

        # Cumulative Fiber Pair usage in km
        km_total = 0
        deg_c = 0
        deg_superc = 0
        deg_l = 0
        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            ax1.semilogy(year, km, 
                         f"b-{default_markers[hierarchy_level]}", 
                         label = f"HL{hierarchy_level}s[km]", linewidth = 1.5) if flag_plot == 1 else None
            km_total += km
            deg_c += self.link_data[f"HL{hierarchy_level}"]['HL_CDegree_Domain']
            deg_superc += self.link_data[f"HL{hierarchy_level}"]['HL_SuperCDegree_Domain']
            deg_l += self.link_data[f"HL{hierarchy_level}"]['HL_LDegree_Domain']

        self.deg_pure_c = deg_c - deg_superc - deg_l
        self.deg_c = deg_c
        self.deg_superc = deg_superc
        self.deg_l = deg_l

        if flag_plot == 1:

            ax1.semilogy(year, km_total, 
                            f"b-+", 
                            label='Total[km]', linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color='blue')
            ax1.set_xlabel("Year")
            ax1.tick_params(axis='y', labelcolor='blue')


            ax2.set_ylabel("Cumulative Nodal Degree Number", color='red')
            ax2.plot(year, deg_c, 
                    'r-.h', label = 'C-Band-Degree[#]', linewidth = 1.5)
            ax2.plot(year, deg_superc, 
                    'r-.*', label = 'SupC-Band-Degree[#]', linewidth = 1.5)
            ax2.plot(year, deg_l, 
                    'r-.s', label = 'L-Band-Degree[#]', linewidth = 1.5)
            ax2.tick_params(axis = 'y', labelcolor = 'red')
            # Set right Y-axis limits and ticks (linear)
            ax2.set_ylim(0, 800)
            ax2.set_yticks(np.arange(0, 801, 100))

            # Grid, legend, and layout
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.result_directory / f"{self.network.topology_name}_FP_Degree{save_suffix}.png", 
                            dpi=300, bbox_inches='tight')
                
            plt.show()


    def plot_bvt_license(self, 
                         save_flag: int, 
                         save_suffix: str = "",
                         flag_plot: int = 1):
        """Set the hierarchical levels of nodes in the network with flexible HLx input.
        
        Args:
            kwargs: Any number of HLx_standalone and/or HLx_colocated lists, e.g.,
                    HL1_standalone=[...], HL2_colocated=[...], etc.
        Returns:
            dict: hierarchical_levels dictionary
        """
        self.load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:
            # --- Plotting ---
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total BVT and 100G-License")

            # Right Y-axis: Linear scale for degrees
            ax2 = ax1.twinx()

        # Cumulative Fiber Pair usage in km
        All_BVT_CBand = 0
        All_BVT_SuperC = 0
        All_BVT_L = 0
        Total_License = 0
        for hierarchy_level in self.processing_level_list:

            All_BVT_CBand += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_CBand']
            All_BVT_SuperC += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_SuperCBand']
            All_BVT_L += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_LBand']

            Total_License += self.bvt_data[f"HL{hierarchy_level}"]['HL_All_100G_lincense']

        self.All_BVT_CBand = All_BVT_CBand
        self.All_BVT_SuperC = All_BVT_SuperC
        self.All_BVT_L = All_BVT_L

        Total_BVT = All_BVT_CBand + All_BVT_SuperC + All_BVT_L
        self.CBand_100G_License = (All_BVT_CBand / Total_BVT) * Total_License
        self.SupCBand_100G_License = (All_BVT_SuperC / Total_BVT) * Total_License
        self.LBand_100G_License = (All_BVT_L / Total_BVT) * Total_License
        Total_100G_License = self.CBand_100G_License + self.SupCBand_100G_License + self.LBand_100G_License

        if flag_plot == 1:
            # Left Y-axis (BVT)
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Cumulative BVT Number", color='blue')
            ax1.semilogy(year, All_BVT_CBand, 'b->', label='C-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_SuperC, 'b-o', label='SupC-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_L, 'b-s', label='L-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, Total_BVT, 'b-+', label='Total-BVT[#]', linewidth=1.5)
            ax1.tick_params(axis = 'y', labelcolor = 'blue')

            ax2.set_ylabel("Cumulative 100G - License Number", color='red')
            ax2.plot(year, self.CBand_100G_License, 'r-.>', label='C-Band-100GL[#]', linewidth=1.5)
            ax2.plot(year, self.SupCBand_100G_License, 'r-.o', label='SupC-Band-100GL', linewidth=1.5)
            ax2.plot(year, self.LBand_100G_License, 'r-.s', label='L-Band-100GL', linewidth=1.5)
            ax2.plot(year, Total_100G_License, 'r-.+', label='Total-100GL', linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor='red')

            # Legend & Grid
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            ax1.grid(True)
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.result_directory / f"{self.network.topology_name}_BVT_License{save_suffix}.png",
                            dpi=300, bbox_inches='tight')
                
            plt.show()

            

    def calc_cost(self, 
                  save_flag: int,
                  save_suffix: str = "", 
                  C_100GL: float = 1,
                  C_MCS: float = 0.7,
                  C_RoB: float = 1.9,
                  C_IRU: float = 0.5) -> dict:
        """Load network topology from a .mat file.
        
        Args:
            filepath: Path to the .mat file containing network topology
            matrixName: name of the adjacancy matrix as a MATLAB variable
        """
        cost_dict = {}

        # --- Fiber Pair Usage [km] ---
        self.load_data()
        self.plot_FP_degree(flag_plot = 0, save_flag = 0)
        self.plot_bvt_license(flag_plot = 0, save_flag = 0)

        All_FP_km = 0
        for hierarchy_level in self.processing_level_list:
            All_FP_km += self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']

        OPEX = C_IRU * All_FP_km
        cost_dict['OPEX'] = OPEX

        # --- CAPEX Initialization ---
        Capex_RoB_C = np.zeros(10)
        Capex_RoB_SupC = np.zeros(10)
        Capex_RoB_L = np.zeros(10)

        Capex_MCS_C = np.zeros(10)
        Capex_MCS_SupC = np.zeros(10)
        Capex_MCS_L = np.zeros(10)

        Capex_100GL_Cband = np.zeros(10)
        Capex_100GL_SupCBand = np.zeros(10)
        Capex_100GL_LBand = np.zeros(10)

        for y in range(self.period_time):
            if y > 0:
                Capex_RoB_C[y] = (self.deg_pure_c[y] - self.deg_pure_c[y - 1]) * C_RoB
                Capex_RoB_SupC[y] = (self.deg_superc[y] - self.deg_superc[y - 1]) * C_RoB * (1 + 0.1 * (1 - 0.1) ** (y + 1))
                Capex_RoB_L[y] = (self.deg_l[y] - self.deg_l[y-1]) * C_RoB * (1 + 0.2 * (1 - 0.1) ** (y + 1))

                Capex_MCS_C[y] = ((self.All_BVT_CBand[y] - self.All_BVT_CBand[y - 1]) / 16) * C_MCS
                Capex_MCS_SupC[y] = ((self.All_BVT_SuperC[y] - self.All_BVT_SuperC[y - 1]) / 16) * C_MCS * (1 + 0.1 * (1 - 0.1) ** (y + 1))
                Capex_MCS_L[y] = ((self.All_BVT_L[y] - self.All_BVT_L[y - 1]) / 16) * C_MCS * (1 + 0.2 * (1 - 0.1) ** (y + 1))

                Capex_100GL_Cband[y] = (self.CBand_100G_License[y] - 0 * self.CBand_100G_License[y-1]) * C_100GL
                Capex_100GL_SupCBand[y] = (self.SupCBand_100G_License[y] - 0 * self.SupCBand_100G_License[y-1]) * C_100GL * (1 + 0.1 * (1 - 0.1)**(y+1))
                Capex_100GL_LBand[y] = (self.LBand_100G_License[y] - 0 * self.LBand_100G_License[y-1]) * C_100GL * (1 + 0.2 * (1 - 0.1)**(y+1))
            else:
                Capex_RoB_C[y] = self.deg_pure_c[y] * C_RoB
                Capex_RoB_SupC[y] = self.deg_superc[y] * C_RoB * 1.1
                Capex_RoB_L[y] = self.deg_l[y] * C_RoB * 1.2

                Capex_MCS_C[y] = self.All_BVT_CBand[y] / 16 * C_MCS
                Capex_MCS_SupC[y] = self.All_BVT_SuperC[y] / 16 * C_MCS * 1.1
                Capex_MCS_L[y] = self.All_BVT_L[y] / 16 * C_MCS * 1.2

                Capex_100GL_Cband[y] = self.CBand_100G_License[y] * C_100GL
                Capex_100GL_SupCBand[y] = self.SupCBand_100G_License[y] * C_100GL * 1.1
                Capex_100GL_LBand[y] = self.LBand_100G_License[y] * C_100GL * 1.2

        # Final CAPEX vectors
        Capex_RoB = Capex_RoB_C + Capex_RoB_SupC + Capex_RoB_L
        Capex_MCS = Capex_MCS_C + Capex_MCS_SupC + Capex_MCS_L
        Capex_100GL = Capex_100GL_Cband + Capex_100GL_SupCBand + Capex_100GL_LBand
        CAPEX = Capex_RoB + Capex_MCS + Capex_100GL

        cost_dict['Capex_RoB'] = Capex_RoB
        cost_dict['Capex_MCS'] = Capex_MCS
        cost_dict['Capex_100GL'] = Capex_100GL
        cost_dict['CAPEX'] = CAPEX

        cost_df = pd.DataFrame(cost_dict)

        if save_flag:
            cost_df.to_csv(self.result_directory / f"{self.network.topology_name}_cost_analyse{save_suffix}.csv", index = False)

        return cost_df
    

    def calc_latency(self,  
        primary_paths: np.ndarray,
        processing_level_list: list,
        save_flag: int, 
        save_suffix: str = ""):
        """
        Generalized version of path distance computation across any hierarchical processing levels.

        Parameters:
            primary_paths: dict
                Maps node ID to row index in corresponding level DataFrame.
            processing_level_list: list of ints
                The order of hierarchy levels to follow (e.g., [4, 3, 2]).

        Returns:
            np.ndarray: total latency (micro second) per HL4 node.
        """
        minimum_hierarchy_level = processing_level_list[0]
        minimum_HL_nodes = self.network.hierarchical_levels[f'HL{minimum_hierarchy_level}']['standalone']
        paths_km = np.zeros(len(minimum_HL_nodes))

        for idx, node_id in enumerate(minimum_HL_nodes):
            current_node = node_id

            for hierarchy_level in processing_level_list:
                df = pd.read_csv(self.result_directory / f"{self.network.topology_name}_HL{hierarchy_level}_K_path_attributes.csv")
                nodes = self.network.hierarchical_levels[f'HL{hierarchy_level}']['standalone']

                if current_node not in nodes:
                    continue

                path_idx = primary_paths[current_node]
                distance = df.iloc[path_idx]['distance']
                dest_node = df.iloc[path_idx]['dest_node']

                paths_km[idx] += distance
                current_node = dest_node  # Move to next-level node

        latency = paths_km * 5 + 200 * len(processing_level_list)

        if save_flag:
            np.savez_compressed(self.result_directory / f"{self.network.topology_name}_latency{save_suffix}.npz",
                            latency = latency)

        return latency

