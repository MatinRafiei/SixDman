from typing import List
import numpy as np
import pandas as pd
from .network import Network
import matplotlib.pyplot as plt
from itertools import accumulate

class analyse_result:
    """"
    A class for analyzing and managing results from hierarchical optical network simulations.

    This class supports loading network planning results for different hierarchy levels,
    including link-level and transceiver-level (BVT) metrics.

    Attributes:
        network (Network): An instance of the Network class defining topology and metadata.
        period_time (int): Duration (in years or time units) for planning or simulation.
        processing_level_list (List[int]): List of hierarchy levels to be analyzed (e.g., [2, 3, 4]).
        save_directory (Path): Directory containing result .npz files for each hierarchy level.
        link_data (Dict[str, np.lib.npyio.NpzFile]): Dictionary containing link-level data for each HL.
        bvt_data (Dict[str, np.lib.npyio.NpzFile]): Dictionary containing transceiver (BVT) data for each HL.
    """
    
    def __init__(self,
                 network_instance: Network,
                 period_time: int, 
                 processing_level_list: List, 
                 save_directory):
        """
        Initialize the AnalyseResult object.

        Args:
            network_instance (Network): The optical network structure.
            period_time (int): Simulation or planning time period (e.g., 10 years).
            processing_level_list (List[int]): Hierarchy levels to process (e.g., [2, 3, 4]).
            save_directory (Path): Path to the directory where .npz result files are stored.

        Example:
        --------
        >>> from sixdman.core.optical_result_analyzer import analyse_result
        >>> analyser = analyse_result(
        ...     net, # Network instance
        ...     10, # Planning period time in years
        ...     processing_level_list, # List of hierarchy levels to analyze
        ...     results_dir # Directory where results are saved
        ... )
        """
        self.network = network_instance
        self.period_time = period_time
        self.processing_level_list = processing_level_list
        self.save_directory = save_directory
        self.link_data = {}
        self.bvt_data = {}
        
    def _load_data(self):

        """
        Load link and BVT result data for each hierarchy level.

        This method reads .npz files generated for each HL level and stores the parsed
        data in the class attributes `link_data` and `bvt_data`.

        Raises:
            IOError: If any expected file is missing or unreadable.
        """
        link_data = {}
        bvt_data = {}
        for hierarchy_level in self.processing_level_list:

            # Construct expected file paths
            link_data_path = self.save_directory / f"{self.network.topology_name}_HL{hierarchy_level}_link_info.npz"
            bvt_data_path = self.save_directory / f"{self.network.topology_name}_HL{hierarchy_level}_bvt_info.npz"
            
            # Load link information
            try:
                        
                hl_link_data = np.load(link_data_path)
                link_data[f"HL{hierarchy_level}"] = hl_link_data
                self.link_data = link_data

            except Exception as e:
                raise IOError(f"Failed to load data from {link_data_path}: {str(e)}")

            # Load BVT (transceiver) information
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
        """
        Plot or return the evolution of link states (Frequency Plan numbers) across all hierarchy levels over time.

        This function creates a heatmap-style visualization of link states (FP numbers) over a multi-year period,
        where each row corresponds to a specific link and each column to a simulation year. Vertical dashed lines
        separate different hierarchical levels using the provided splitter.

        Args:
            splitter (List): A list indicating how many links are in each hierarchy level (used for dashed separators).
            save_flag (int): If 1, the plot will be saved to disk.
            save_suffix (str, optional): Custom suffix for the saved file name. Default is "".
            flag_plot (int, optional): If 1, show the plot; otherwise, return the data. Default is 1.

        Returns:
            np.ndarray: If flag_plot == 0, returns the link state matrix of shape (years, total_links).
                        Each element corresponds to the FP number assigned to a specific link in a given year.

        Example:
        ---------
        >>> analyser.plot_link_state(
        ...     splitter, # List of integers indicating the number of links per hierarchy level
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     ave_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )
        """
        self._load_data()

        # Initialize an empty array to hold FP numbers across all hierarchy levels and years
        link_state_HL_partisioned = np.empty(shape=(self.period_time, 0))

        # Concatenate link FP number arrays from all hierarchy levels
        for hierarchy_level in self.processing_level_list:
            link_state_HL_partisioned = np.hstack((
                link_state_HL_partisioned,
                self.link_data[f"HL{hierarchy_level}"]['HL_FPNum']
            ))

        year = np.arange(1, self.period_time + 1)  # Year range for x-axis

        if flag_plot == 1:
            # Plot the link state profile
            plt.figure(figsize=(7, 5))
            plt.title("ALL HLs: link state profile")

            # Show the heatmap image of link state transitions
            plt.imshow(link_state_HL_partisioned.T,
                    aspect='auto',
                    interpolation='none',
                    origin='upper')

            plt.xlabel("Year")
            plt.ylabel("Link index")

            # Set color limits (0 = unused, 1–3 = different link states)
            plt.clim(0, 3)
            plt.colorbar(label='State')

            # Define x-axis limits and ticks
            plt.xlim(0.5, self.period_time + 0.5)
            plt.xticks(np.arange(1, 11))  # Adjust as needed if period_time != 10

            # Add grid to the plot
            plt.grid(True)

            # Convert splitter into cumulative index positions to separate HL groups
            splitter_converted = list(accumulate(splitter))

            # Plot dashed lines between hierarchical levels
            for i in range(len(splitter_converted)):
                plt.plot(year,
                        splitter_converted[i] * np.ones_like(year),
                        'k--',
                        linewidth=1)

            plt.tight_layout()

            # Save the plot if requested
            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_Link_State{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()

        else:
            # Return the raw matrix if no plot is requested
            return link_state_HL_partisioned

    def plot_FP_usage(self,
                      save_flag: int, 
                      save_suffix: str = "",
                      flag_plot: int = 1):
        """
        Plot and optionally save the Fiber Pair (FP) usage over time across all hierarchy levels.

        This function visualizes:
            - The cumulative FP usage in kilometers (left y-axis, log scale)
            - The total number of FPs (right y-axis, linear scale)

        Args:
            save_flag (int): If 1, saves the plot to the result directory.
            save_suffix (str): Optional suffix to append to the saved filename.
            flag_plot (int): If 1, display the plot; if 0, skip plotting.

        Returns:
            None

        Example:
        --------
        >>> analyser.plot_FP_usage(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()  # Load link data for all hierarchy levels
        year = np.arange(1, self.period_time + 1)  # e.g., years 1 to 10

        if flag_plot == 1:
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total FP Usage [km] and [number of FP]")
            ax2 = ax1.twinx()  # Create secondary y-axis for FP count

        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']  # Marker styles per HL

        km_total = 0
        fp_total = 0
        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            fp = np.sum(self.link_data[f"HL{hierarchy_level}"]['HL_FPNum'], axis=1)

            if flag_plot == 1:
                # Left Y-axis: FP usage in km (log scale)
                ax1.semilogy(year, km,
                            f"b-{default_markers[hierarchy_level]}",
                            label=f"HL{hierarchy_level}s [km]", linewidth=1.5)
                # Right Y-axis: FP count (linear scale)
                ax2.plot(year, fp,
                        f"r-.{default_markers[hierarchy_level]}",
                        label=f"HL{hierarchy_level}s [#]", linewidth=1.5)

            km_total += km
            fp_total += fp

        if flag_plot == 1:
            # Plot total across all HLs
            ax1.semilogy(year, km_total, "b-+", label="Total [km]", linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color="blue")
            ax1.set_xlabel("Year")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2.plot(year, fp_total, "r-+", label="Total [#]", linewidth=1.5)
            ax2.set_ylabel("Cumulative Nodal Degree Number", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0, 250)
            ax2.set_yticks(np.arange(0, 251, 50))

            # Add grid and combined legend
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_FP_Usage{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()

    def plot_FP_degree(self,
                       save_flag: int, 
                       save_suffix: str = "",
                       flag_plot: int = 1):
        """
        Plot and optionally save the cumulative Fiber Pair (FP) usage and Band degree types.

        This function visualizes:
            - Total FP usage in kilometers (left Y-axis, log scale)
            - Nodal degrees for C-band, SuperC-band, and L-band (right Y-axis, linear scale)

        Args:
            save_flag (int): If 1, saves the plot as an image to the save_directory.
            save_suffix (str): Optional suffix to append to the saved filename.
            flag_plot (int): If 1, plot will be displayed. If 0, no plot is shown.

        Returns:
            None

        Example:
        --------
        >>> analyser.plot_FP_degree(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("FP [km] and Degree")
            ax2 = ax1.twinx()  # Secondary axis for nodal degree plots

        default_markers = ['d', 'p', 's', 'o', '>', 'h', '*', 'x']

        km_total = 0
        deg_c = 0
        deg_superc = 0
        deg_l = 0

        for hierarchy_level in self.processing_level_list:
            km = self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']
            if flag_plot == 1:
                ax1.semilogy(year, km,
                            f"b-{default_markers[hierarchy_level]}",
                            label=f"HL{hierarchy_level}s [km]", linewidth=1.5)
            km_total += km

            deg_c += self.link_data[f"HL{hierarchy_level}"]['HL_CDegree_Domain']
            deg_superc += self.link_data[f"HL{hierarchy_level}"]['HL_SuperCDegree_Domain']
            deg_l += self.link_data[f"HL{hierarchy_level}"]['HL_LDegree_Domain']

        # Store for external use if needed
        self.deg_pure_c = deg_c - deg_superc - deg_l
        self.deg_c = deg_c
        self.deg_superc = deg_superc
        self.deg_l = deg_l

        if flag_plot == 1:
            ax1.semilogy(year, km_total,
                        "b-+",
                        label="Total [km]", linewidth=1.5)
            ax1.set_ylabel("Cumulative Fiber Pair Usage [Km]", color="blue")
            ax1.set_xlabel("Year")
            ax1.tick_params(axis="y", labelcolor="blue")

            # Right Y-axis: Degrees
            ax2.set_ylabel("Cumulative Nodal Degree Number", color="red")
            ax2.plot(year, deg_c, 'r-.h', label='C-Band-Degree [#]', linewidth=1.5)
            ax2.plot(year, deg_superc, 'r-.*', label='SupC-Band-Degree [#]', linewidth=1.5)
            ax2.plot(year, deg_l, 'r-.s', label='L-Band-Degree [#]', linewidth=1.5)
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(0, 800)
            ax2.set_yticks(np.arange(0, 801, 100))

            # Combined legend and final layout
            ax1.grid(True)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.tight_layout()

            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_FP_Degree{save_suffix}.png",
                            dpi=300, bbox_inches='tight')

            plt.show()


    def plot_bvt_license(self, 
                         save_flag: int, 
                         save_suffix: str = "",
                         flag_plot: int = 1):
        """
        Plot and optionally save the cumulative BVT usage and 100G license allocation over time.

        This function visualizes:
            - The cumulative count of BVTs (Bandwidth Variable Transponders) for each band: 
            C-Band, SuperC-Band, and L-Band (left Y-axis, log scale).
            - The corresponding allocation of 100G licenses proportionally distributed across 
            the bands (right Y-axis, linear scale).

        Args:
            save_flag (int): If 1, saves the plot as an image to the save_directory.
            save_suffix (str): Optional suffix for the saved image filename.
            flag_plot (int): If 1, displays the plot; if 0, skips plotting.

        Returns:
            None

        Example:
        --------
        >>> analyser.plot_bvt_license(
        ...     save_flag = 0, # If 1, save the plot; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... )

        """
        self._load_data()
        year = np.arange(1, self.period_time + 1)

        if flag_plot == 1:
            # Initialize figure and axis
            fig, ax1 = plt.subplots(figsize=(7, 5))
            fig.suptitle("Total BVT and 100G-License")
            ax2 = ax1.twinx()  # Right-side Y-axis

        # Initialize cumulative BVT and license counters
        All_BVT_CBand = 0
        All_BVT_SuperC = 0
        All_BVT_L = 0
        Total_License = 0

        # Accumulate values across all processing levels
        for hierarchy_level in self.processing_level_list:
            All_BVT_CBand += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_CBand']
            All_BVT_SuperC += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_SuperCBand']
            All_BVT_L += self.bvt_data[f"HL{hierarchy_level}"]['HL_BVTNum_LBand']
            Total_License += self.bvt_data[f"HL{hierarchy_level}"]['HL_All_100G_lincense']

        # Save accumulated results for external access
        self.All_BVT_CBand = All_BVT_CBand
        self.All_BVT_SuperC = All_BVT_SuperC
        self.All_BVT_L = All_BVT_L

        # Calculate total BVTs and proportional 100G license allocation
        Total_BVT = All_BVT_CBand + All_BVT_SuperC + All_BVT_L
        self.CBand_100G_License = (All_BVT_CBand / Total_BVT) * Total_License
        self.SupCBand_100G_License = (All_BVT_SuperC / Total_BVT) * Total_License
        self.LBand_100G_License = (All_BVT_L / Total_BVT) * Total_License
        Total_100G_License = (
            self.CBand_100G_License +
            self.SupCBand_100G_License +
            self.LBand_100G_License
        )

        if flag_plot == 1:
            # --- Plot BVT Counts on Left Y-axis ---
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Cumulative BVT Number", color='blue')
            ax1.semilogy(year, All_BVT_CBand, 'b->', label='C-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_SuperC, 'b-o', label='SupC-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, All_BVT_L, 'b-s', label='L-Band-BVT[#]', linewidth=1.5)
            ax1.semilogy(year, Total_BVT, 'b-+', label='Total-BVT[#]', linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor='blue')

            # --- Plot 100G Licenses on Right Y-axis ---
            ax2.set_ylabel("Cumulative 100G - License Number", color='red')
            ax2.plot(year, self.CBand_100G_License, 'r-.>', label='C-Band-100GL[#]', linewidth=1.5)
            ax2.plot(year, self.SupCBand_100G_License, 'r-.o', label='SupC-Band-100GL', linewidth=1.5)
            ax2.plot(year, self.LBand_100G_License, 'r-.s', label='L-Band-100GL', linewidth=1.5)
            ax2.plot(year, Total_100G_License, 'r-.+', label='Total-100GL', linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor='red')

            # --- Finalize plot ---
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            ax1.grid(True)
            plt.tight_layout()

            # Save if requested
            if save_flag:
                plt.savefig(self.save_directory / f"{self.network.topology_name}_BVT_License{save_suffix}.png",
                            dpi=300, bbox_inches='tight')
            plt.show()

            

    def calc_cost(self, 
                  save_flag: int,
                  save_suffix: str = "", 
                  C_100GL: float = 1,
                  C_MCS: float = 0.7,
                  C_RoB: float = 1.9,
                  C_IRU: float = 0.5) -> dict:
        """
        Compute OPEX and CAPEX values for network deployment over time.

        This function evaluates capital and operational expenditures based on 
        nodal degree evolution and bandwidth resource demands (e.g., BVTs, licenses).

        Args:
            save_flag (int): Whether to save the output CSV file (1) or not (0).
            save_suffix (str): Optional suffix for the saved file name.
            C_100GL (float): Unit cost of a 100G license [default = 1].
            C_MCS (float): Unit cost of a modular chassis switch [default = 0.7].
            C_RoB (float): Unit cost of a router or BVT-based node [default = 1.9].
            C_IRU (float): Cost per km of IRU fiber pair usage [default = 0.5].

        Returns:
            dict: A dictionary containing vectors for OPEX and CAPEX components.

        Example:
        ---------
        >>> analyser.calc_cost(
        ...     save_flag = 0, # If 1, save the results; if 0, do not save
        ...     save_suffix = "_NoBypass" # Optional suffix for the saved file name
        ... ) # Returns a dictionary with OPEX and CAPEX components

        """

        cost_dict = {}

        # --- Load data and compute usage from prior functions ---
        self._load_data()
        self.plot_FP_degree(flag_plot=0, save_flag=0)
        self.plot_bvt_license(flag_plot=0, save_flag=0)

        # --- Aggregate all Fiber Pair usage across hierarchy levels ---
        All_FP_km = 0
        for hierarchy_level in self.processing_level_list:
            All_FP_km += self.link_data[f"HL{hierarchy_level}"]['Total_effective_FP_new_annual']

        # --- Compute Operational Expenditure (OPEX) ---
        OPEX = C_IRU * All_FP_km
        cost_dict['OPEX'] = OPEX

        # --- Initialize CAPEX components for each technology and band ---
        Capex_RoB_C = np.zeros(10)
        Capex_RoB_SupC = np.zeros(10)
        Capex_RoB_L = np.zeros(10)

        Capex_MCS_C = np.zeros(10)
        Capex_MCS_SupC = np.zeros(10)
        Capex_MCS_L = np.zeros(10)

        Capex_100GL_Cband = np.zeros(10)
        Capex_100GL_SupCBand = np.zeros(10)
        Capex_100GL_LBand = np.zeros(10)

        # --- Loop over time period to compute annual CAPEX ---
        for y in range(self.period_time):
            if y > 0:
                # Compute annual increment for RoB cost
                Capex_RoB_C[y] = (self.deg_pure_c[y] - self.deg_pure_c[y - 1]) * C_RoB
                Capex_RoB_SupC[y] = (self.deg_superc[y] - self.deg_superc[y - 1]) * C_RoB * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_RoB_L[y] = (self.deg_l[y] - self.deg_l[y - 1]) * C_RoB * (1 + 0.2 * (1 - 0.1)**(y + 1))

                # Compute annual increment for MCS cost (per 16 BVT units)
                Capex_MCS_C[y] = ((self.All_BVT_CBand[y] - self.All_BVT_CBand[y - 1]) / 16) * C_MCS
                Capex_MCS_SupC[y] = ((self.All_BVT_SuperC[y] - self.All_BVT_SuperC[y - 1]) / 16) * C_MCS * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_MCS_L[y] = ((self.All_BVT_L[y] - self.All_BVT_L[y - 1]) / 16) * C_MCS * (1 + 0.2 * (1 - 0.1)**(y + 1))

                # Compute annual increment for 100G License cost
                Capex_100GL_Cband[y] = (self.CBand_100G_License[y]) * C_100GL
                Capex_100GL_SupCBand[y] = (self.SupCBand_100G_License[y]) * C_100GL * (1 + 0.1 * (1 - 0.1)**(y + 1))
                Capex_100GL_LBand[y] = (self.LBand_100G_License[y]) * C_100GL * (1 + 0.2 * (1 - 0.1)**(y + 1))
            else:
                # Initial year costs
                Capex_RoB_C[y] = self.deg_pure_c[y] * C_RoB
                Capex_RoB_SupC[y] = self.deg_superc[y] * C_RoB * 1.1
                Capex_RoB_L[y] = self.deg_l[y] * C_RoB * 1.2

                Capex_MCS_C[y] = self.All_BVT_CBand[y] / 16 * C_MCS
                Capex_MCS_SupC[y] = self.All_BVT_SuperC[y] / 16 * C_MCS * 1.1
                Capex_MCS_L[y] = self.All_BVT_L[y] / 16 * C_MCS * 1.2

                Capex_100GL_Cband[y] = self.CBand_100G_License[y] * C_100GL
                Capex_100GL_SupCBand[y] = self.SupCBand_100G_License[y] * C_100GL * 1.1
                Capex_100GL_LBand[y] = self.LBand_100G_License[y] * C_100GL * 1.2

        # --- Aggregate CAPEX components ---
        Capex_RoB = Capex_RoB_C + Capex_RoB_SupC + Capex_RoB_L
        Capex_MCS = Capex_MCS_C + Capex_MCS_SupC + Capex_MCS_L
        Capex_100GL = Capex_100GL_Cband + Capex_100GL_SupCBand + Capex_100GL_LBand
        CAPEX = Capex_RoB + Capex_MCS + Capex_100GL

        cost_dict['Capex_RoB'] = Capex_RoB
        cost_dict['Capex_MCS'] = Capex_MCS
        cost_dict['Capex_100GL'] = Capex_100GL
        cost_dict['CAPEX'] = CAPEX

        # --- Convert results to DataFrame and optionally save ---
        cost_df = pd.DataFrame(cost_dict)

        if save_flag:
            cost_df.to_csv(
                self.save_directory / f"{self.network.topology_name}_cost_analyse{save_suffix}.csv", 
                index=False
            )

        return cost_df
    

    def calc_latency(self,  
        primary_paths: np.ndarray,
        processing_level_list: list,
        save_flag: int, 
        save_suffix: str = ""):
        
        """
        Compute end-to-end latency across hierarchical processing levels.
        
        This function computes the cumulative latency from each lowest-level (e.g., HL4) 
        node to its destination, traversing a set of hierarchy levels (e.g., HL4 → HL3 → HL2).
        It reads precomputed path attributes (distance and destination node) from disk 
        and accumulates fiber delay plus processing overhead per level.

        Args:
            primary_paths (np.ndarray): 
                An array mapping node IDs to row indices in their corresponding 
                HLx_K_path_attributes.csv files.
            processing_level_list (list of int): 
                A list of hierarchy levels to traverse (e.g., [4, 3, 2]).
            save_flag (int): 
                If True (1), saves the latency array to a compressed .npz file.
            save_suffix (str): 
                Optional suffix added to the filename when saving.

        Returns:
            np.ndarray: 
                Array of total latency values in microseconds, one per HL4 node.

        Example:
        ---------
        >>> analyser.calc_latency(
        ...     primary_paths = planner.primary_path_storage, # storage of primary paths
        ...     processing_level_list = processing_level_list, # List of hierarchy levels to process
        ...     save_flag = 0 # If 1, save the results; if 0, do not save
        ... )
        """
        # HLx hierarchy is traversed top-down; HL4 assumed to be the lowest
        minimum_hierarchy_level = processing_level_list[0]
        minimum_HL_nodes = self.network.hierarchical_levels[f'HL{minimum_hierarchy_level}']['standalone']

        # Initialize total distance array (in km)
        paths_km = np.zeros(len(minimum_HL_nodes))

        for idx, node_id in enumerate(minimum_HL_nodes):
            current_node = node_id  # Start from HL4 (or other base-level)

            for hierarchy_level in processing_level_list:
                # Load precomputed path attributes for current level
                df = pd.read_csv(
                    self.save_directory / f"{self.network.topology_name}_HL{hierarchy_level}_K_path_attributes.csv"
                )
                nodes = self.network.hierarchical_levels[f'HL{hierarchy_level}']['standalone']

                # Skip if current node is not in this level
                if current_node not in nodes:
                    continue

                # Look up path from current node
                path_idx = primary_paths[current_node]
                distance = df.iloc[path_idx]['distance']
                dest_node = df.iloc[path_idx]['dest_node']

                # Accumulate distance and update node for next level
                paths_km[idx] += distance
                current_node = dest_node

        # Convert distance to latency: 5 µs/km fiber delay + 200 µs processing delay per level
        latency = paths_km * 5 + 200 * len(processing_level_list)

        # Optional: Save to disk in compressed format
        if save_flag:
            np.savez_compressed(
                self.save_directory / f"{self.network.topology_name}_latency{save_suffix}.npz",
                latency=latency
            )

        return latency

