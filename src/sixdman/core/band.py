from typing import Dict, List, Optional, Union
import numpy as np
import os
from dataclasses import dataclass, field
from scipy.special import erfcinv
from .network import Network

@dataclass
class OpticalParameters:
    """Class for storing optical transmission parameters.
    
    Attributes:
        beta_2: Group velocity dispersion (s²/m)
        beta_3: Third-order dispersion coefficient (s³/m)
        alpha_db: Attenuation coefficient (dB/km)
        gamma: Nonlinear parameter (1/W·m)
        F_dB: Noise figure (dB)
    """
    
    h_plank: float = 6.626e-34  # Planck's constant (J s)
    target_ber: float = 1e-2  # Target bit error rate
    phi_MFL: np.ndarray = field(default_factory=lambda: -1 * np.array([1, 1, 2 / 3, 17 / 25, 69 / 100, 13 / 21]))

    # System Parameters
    epsilon: int = 0  # Auxiliary variable
    beta_3: float = 0.14e-39  # Third-order dispersion parameter (s^3/m)
    Cr: float = 0.028 / 1e3 / 1e12  # Chromatic dispersion coefficient (1/(m·Hz^2))

    # Fiber Parameters
    alpha_db: float = 0.2  # Fiber attenuation in dB/km
    alpha_norm: float = field(init=False)
    beta_2: float = -21.7e-27  # Second-order dispersion (s^2/m)
    gama: float = 1.21e-3  # Nonlinear parameter (1/(W·m))
    L_eff_a: float = field(init=False)
    F_C: float = field(default_factory=lambda: 10 ** 0.45)  # Noise figure for C-band (6 dB)
    F_L: float = field(default_factory=lambda: 10 ** 0.5)   # Noise figure for L-band (6 dB)

    # System Transmission Parameters
    Rs_mat: float = 40e9  # Symbol rate (Baud)
    MFL: np.ndarray = field(default_factory=lambda: np.arange(1, 7))  # Modulation format levels
    rof: float = 0.1  # Roll-off factor
    B_ch_mat: float = field(init=False)
    B_ch: float = field(init=False)

    def __post_init__(self):
        # Computed fields
        self.alpha_norm = self.alpha_db / (10 * np.log10(np.exp(1)) * 1e3)
        self.L_eff_a = 1 / self.alpha_norm
        self.B_ch_mat = self.Rs_mat * (1 + self.rof)
        self.B_ch = self.B_ch_mat

class Band:
    """Class representing an optical transmission band with its characteristics.
    
    This class handles the multi-band specifications and calculations for optical
    transmission parameters in the 6D-MAN planning tool.
    """
    
    def __init__(self, 
                 name: str,
                 start_freq: float,
                 end_freq: float,
                 opt_params: OpticalParameters,
                 network_instance: Network,
                 channel_spacing: float = 0.05):  # 50 GHz (0.05 THz) default spacing
        
        """Initialize Band instance.
        
        Args:
            name: Name identifier for the band (e.g., 'C', 'L', 'S')
            start_freq: start frequency of the band (THz)
            end_freq: end frequency of the band (THz)
            opt_params: Optical transmission parameters for this band
            channel_spacing: Channel spacing in THz (default: 0.05 THz)
        """
        self.name = name
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.opt_params = opt_params
        self.network = network_instance
        self.channel_spacing = channel_spacing
        self.calc_spectrum()  # Calculate frequency grid upon initialization
                       
    def calc_spectrum(self):
        """Calculate derived optical parameters."""
                
        # Calculate frequency grid
        return  np.flip(np.arange(self.start_freq, self.end_freq, step = self.channel_spacing))
    
    def process_link_gsnr(self,
                          f_c_axis: np.ndarray, 
                          Pch_dBm: np.ndarray, 
                          num_Ch_mat: np.ndarray, 
                          spectrum_C: np.ndarray,
                          Nspan_array: np.ndarray, 
                          hierarchy_level: int,
                          minimum_hierarchy_level: int, 
                          result_directory):
        """
        Function to process a single link in parallel.
        Receives extra parameters from the main function using `functools.partial()`.
        """
        file_name = result_directory / f'{self.network.topology_name}_process_GSNR_HL{hierarchy_level}.npz'

        if os.path.exists(file_name):

            print("Loading precomputed link GSNR analysis")
            data = np.load(file_name)
            return data['GSNR_opt_link'], data['Throughput_link_per_power_mat'], data['Throughput_link_at_optimumpower_mat'], data['power_opt_link']
        
        else:

            print("Process Link GSNR: .....")

            beta_2 = self.opt_params.beta_2
            beta_3 = self.opt_params.beta_3
            alpha_norm = self.opt_params.alpha_norm
            B_ch = self.opt_params.B_ch
            Cr = self.opt_params.Cr
            h_plank = self.opt_params.h_plank
            F_C = self.opt_params.F_C
            F_L = self.opt_params.F_L
            gama = self.opt_params.gama
            epsilon = self.opt_params.epsilon
            phi_MFL = self.opt_params.phi_MFL

            # Compute fixed target SNR values based on target BER
            target_SNR_dB = [
                10 * np.log10(1 * (erfcinv(2 * self.opt_params.target_ber)) ** 2),
                10 * np.log10(2 * (erfcinv(2 * self.opt_params.target_ber)) ** 2),
                10 * np.log10((14 / 3) * (erfcinv(1.5 * self.opt_params.target_ber)) ** 2),
                10 * np.log10(10 * (erfcinv((8 / 3) * self.opt_params.target_ber)) ** 2),
                10 * np.log10(2 * (erfcinv(np.log2(32) * self.opt_params.target_ber / 2 / (1 - 1 / np.sqrt(32)))) ** 2 * (32 - 1) / 3),
                10 * np.log10(2 * (erfcinv(np.log2(64) * self.opt_params.target_ber / 2 / (1 - 1 / np.sqrt(64)))) ** 2 * (64 - 1) / 3)
            ]

            fc = np.median(f_c_axis)  # Median frequency
            weights_Lspan = self.network.weights_array / Nspan_array
            alpha_norm_bar = alpha_norm
            A = alpha_norm + alpha_norm_bar  # Auxiliary parameter

            subgraph, _ = self.network.calculate_subgraph(hierarchy_level, minimum_hierarchy_level)
            HL_links = np.array(list(subgraph.edges(data = 'weight')))
            mask = np.any(np.all(self.network.all_links[:, None] == HL_links, axis=2), axis=1)
            HL_links_indices = np.where(mask)[0]

            # Initialize capacity and SNR matrices
            Capacity_throuput_list_all_Chs_per_ML = np.zeros(len(Pch_dBm))
            snr_mat = np.zeros((len(Pch_dBm), len(num_Ch_mat)))
            
            Throughput_link_per_power_mat = np.zeros((len(HL_links_indices), len(Pch_dBm)))
            Throughput_link_at_optimumpower_mat = np.zeros((len(HL_links_indices)))
            GSNR_opt_link = np.zeros((len(HL_links_indices), len(num_Ch_mat)))
            power_opt_link = np.zeros(len(HL_links_indices))
            
            for link_counter_HL in range(len(HL_links_indices)):

                print(f"Processing link {link_counter_HL} out of {len(HL_links_indices)}")

                length_span = weights_Lspan[HL_links_indices[link_counter_HL]] * 1e3
                n_s = Nspan_array[HL_links_indices[link_counter_HL]]
            
                for power_counter in range(len(Pch_dBm)):
                    flag = 0  # Reset flag for checking conditions
                    Ptx = 10 ** ((Pch_dBm[power_counter] - 30) / 10)
                    p_cut = Ptx  # Initialize power cut
                    p_k = Ptx  # Initialize power level
                    P_tot = len(num_Ch_mat) * Ptx  # Compute total power

                    # Initialize arrays to store noise and SNR values
                    P_NLI_span = np.zeros(len(num_Ch_mat))  # Non-linear interference power
                    P_NLI_span_db = np.zeros(len(num_Ch_mat))  # NLI power in dB
                    eta_NLI_span_db = np.zeros(len(num_Ch_mat))  # Efficiency of NLI in dB
                    P_ASE_span_db = np.zeros(len(num_Ch_mat))  # Amplified spontaneous emission noise in dB
                    SNR_CUT = np.zeros(len(num_Ch_mat))  # Signal-to-noise ratio per channel

                    # Process each frequency channel
                    for num_fcut in range(len(num_Ch_mat)):
                        f_cut = f_c_axis[num_fcut]  # Extract the frequency value
                        phi_cut = 1.5 * np.pi ** 2 * (beta_2 + 2 * beta_3 * (f_cut - fc))  # Compute phase shift
                        T_cut = (A - P_tot * Cr * (f_cut - fc)) ** 2  # Compute transmission coefficient
                        T_tilda = -P_tot * Cr * (f_cut - fc) / alpha_norm_bar  # Compute auxiliary parameter
                        G_ASE = (1 + T_tilda) * np.exp(-alpha_norm * length_span) - T_tilda * np.exp(
                            - (alpha_norm + alpha_norm_bar) * length_span)  # ASE Gain
                        # n_s = 1  # Number of spans
                        eta_total = 0  # Initialize non-linear coefficient

                        if n_s == 1:
                            n_bar = 0  # TODO: ASM n_bar is always 0 ??
                        else:
                            n_bar = n_s

                        # Define modulation format level (MFL) value
                        MFL = 6

                        # Create a mask to distinguish between C-band and L-band frequencies
                        C_band_mask = num_fcut <= len(spectrum_C) - 1

                        # Compute ASE noise power based on the gain and frequency band
                        P_ASE_span = n_s * B_ch * h_plank * f_cut * (1 / G_ASE) * (
                            F_C if (num_fcut <= len(spectrum_C) - 1) else F_L)

                        # Define frequency components for further calculations
                        f_k = f_c_axis[:len(num_Ch_mat)]  # Extract frequency range
                        match_f_cut = f_k == f_cut  # Boolean mask for matching frequencies and apply either eta_SCI or eta_XCI

                        # Compute frequency-dependent parameters
                        delta_f = f_k - f_cut
                        phi = -4 * np.pi ** 2 * (beta_2 + np.pi * beta_3 * (f_cut + f_k - 2 * fc)) * length_span
                        T_k = (A - P_tot * Cr * (f_k - fc)) ** 2
                        phi_cut_k = -2 * np.pi ** 2 * delta_f * (beta_2 + np.pi * beta_3 * (f_cut + f_k - 2 * fc))
                        # Compute the first integral component for nonlinear interference calculation
                        part_1 = ((T_k - alpha_norm ** 2) / alpha_norm) * np.arctan((phi_cut_k * B_ch) / alpha_norm) + \
                                (((A ** 2 - T_k) / A) * np.arctan((phi_cut_k * B_ch) / A))

                        # Initialize nonlinear interference coefficients
                        eta_SCI = np.zeros(len(num_Ch_mat))
                        eta_XCI = np.zeros(len(num_Ch_mat))
                        eta_XCI_MFL = np.zeros(len(num_Ch_mat))

                        # Compute self-channel interference (SCI) for matched frequencies
                        # mainly caused by self-phase modulation (SPM) and intrachannel interactions
                        eta_SCI[match_f_cut] = (4 / 9) * (gama / B_ch) ** 2 * ((np.pi * n_s ** (1 + epsilon)) / (
                                phi_cut * alpha_norm_bar * (2 * alpha_norm + alpha_norm_bar))) * (
                                                    (T_cut - alpha_norm ** 2) / alpha_norm) * np.arcsinh(
                            (phi_cut * B_ch ** 2) / (np.pi * alpha_norm)) + (
                                                    ((A ** 2 - T_cut) / A) * np.arcsinh((phi_cut * B_ch ** 2) / (np.pi * A)))

                        # eta efficiency factors
                        # Compute cross-channel interference (XCI) for non-matching frequencies (adjacent channels)
                        #  primarily caused by cross-phase modulation (XPM) and four-wave mixing (FWM).
                        eta_XCI[~match_f_cut] = ((32 / 27) * (p_k / p_cut) ** 2 * (gama ** 2 / B_ch) *
                                                (((n_s + (5 / 6) * phi_MFL[MFL - 1]) /
                                                (phi_cut_k * alpha_norm_bar *
                                                    (2 * alpha_norm + alpha_norm_bar))) * part_1))[~match_f_cut]

                        # Compute modulation format loss-induced XCI (XCI_MFL)
                        # modulation format affects the interference propagation, particularly in flexible-grid networks.
                        eta_XCI_MFL[~match_f_cut] = ((32 / 27) * (p_k / Ptx) ** 2 * (gama ** 2 / B_ch) * (5 / 3) * (
                                ((phi_MFL[MFL - 1] * n_bar * np.pi * T_k) /
                                (np.abs(phi) * B_ch ** 2 * alpha_norm ** 2 * A ** 2)) *
                                ((2 * np.abs(delta_f) - B_ch) * np.log((2 * np.abs(delta_f) - B_ch) /
                                                                    (2 * np.abs(delta_f) + B_ch)) + 2 * B_ch)))[~match_f_cut]
                        # Total efficiency factor
                        eta_total = np.sum(eta_SCI + eta_XCI + eta_XCI_MFL)

                        # Compute nonlinear interference (NLI) power per span
                        P_NLI_span[num_fcut] = Ptx ** 3 * eta_total  # NLI power grows cubically with input power
                        # Convert NLI power to dB scale
                        P_NLI_span_db[num_fcut] = 10 * np.log10(Ptx ** 3 * eta_total)  # Convert to logarithmic scale
                        # Convert NLI efficiency to dB scale
                        eta_NLI_span_db[num_fcut] = 10 * np.log10(eta_total)  # Logarithmic representation of efficiency
                        # Convert ASE noise power to dB scale
                        P_ASE_span_db[num_fcut] = 10 * np.log10(P_ASE_span)  # Amplified Spontaneous Emission (ASE) noise in dB
                        # Compute signal-to-noise ratio (SNR) for each channel
                        # ASM: in this case the contribution of the NLI to the
                        SNR_CUT[num_fcut] = 10 * np.log10(
                            (p_cut - 0 * P_NLI_span[num_fcut]) / (P_ASE_span + P_NLI_span[num_fcut]))  # SNR Calculation

                        # ASM: Check if [5] 64-QAM or [1] BPSK should be used
                        if SNR_CUT[num_fcut] < target_SNR_dB[5]:
                            flag = 1
                            print("It is less than expected SNR! ", SNR_CUT[num_fcut])
                            # break

                        if SNR_CUT[num_fcut] < 0:
                            flag = 1
                            print("It is negative! ", SNR_CUT[num_fcut], ": ", length_span, ":", length_span)
                            # break

                        if np.iscomplex(SNR_CUT[num_fcut]):
                            raise ValueError("Complex SNR detected")
                        if flag == 1:
                            break

                    if flag == 0:
                        # Compute the total achievable capacity (in Tbps) for the given power level
                        # (2 * B_ch) to account for both polarizations
                        Capacity_throuput_list_all_Chs_per_ML[power_counter] = (2 * B_ch) * np.sum(
                            np.log2(1 + 10 ** (SNR_CUT / 10))) * 1e-12

                    # Store the SNR for all the channels of this power level into the SNR matrix
                    snr_mat[power_counter, :] = SNR_CUT

                # The achievable capacity for the link is stored in the matrix
                Throughput_link_per_power_mat[link_counter_HL, :] = Capacity_throuput_list_all_Chs_per_ML

                # Tracking the best throughput per link based on the optimal power setting.
                # Stores the power level that maximizes throughput for this link
                Throughput_link_at_optimumpower_mat[link_counter_HL] = Capacity_throuput_list_all_Chs_per_ML[
                    np.argmax(Capacity_throuput_list_all_Chs_per_ML)]
                
                # Finds the optimal power setting that maximizes throughput for this link
                power_opt_link[link_counter_HL] = Pch_dBm[np.argmax(Capacity_throuput_list_all_Chs_per_ML)]

                # Finds and stores the GSNR at the optimal power setting / best throughput for this link
                GSNR_opt_link[link_counter_HL, :] = snr_mat[np.argmax(Capacity_throuput_list_all_Chs_per_ML), :]

                Throughput_link_per_power_mat[link_counter_HL, :] = Capacity_throuput_list_all_Chs_per_ML
                GSNR_opt_link[link_counter_HL, :] = snr_mat[np.argmax(Capacity_throuput_list_all_Chs_per_ML), :]

            np.savez_compressed(file_name,
                    GSNR_opt_link=GSNR_opt_link,
                    Throughput_link_per_power_mat=Throughput_link_per_power_mat,
                    Throughput_link_at_optimumpower_mat=Throughput_link_at_optimumpower_mat,
                    power_opt_link=power_opt_link)
            
            return GSNR_opt_link, Throughput_link_per_power_mat, Throughput_link_at_optimumpower_mat, power_opt_link
