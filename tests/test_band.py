import pytest
import numpy as np
from sixdman.core.band import Band, OpticalParameters

@pytest.fixture
def c_band_params():
    """Fixture providing typical C-band parameters."""
    return OpticalParameters(
        beta_2=-21.7e-27,  # s²/m
        beta_3=0.14e-39,   # s³/m
        alpha_db=0.2,      # dB/km
        gamma=1.21e-3,     # 1/W·m
        F_dB=6.0          # dB
    )

def test_band_initialization(c_band_params):
    band = Band(
        name='C',
        center_frequency=193.5e12,  # 193.5 THz
        bandwidth=4.4e12,           # 4.4 THz
        opt_params=c_band_params
    )
    
    assert band.name == 'C'
    assert band.num_channels == 88  # 4.4 THz / 50 GHz
    assert len(band.frequencies) == band.num_channels

def test_snr_calculation(c_band_params):
    band = Band(
        name='C',
        center_frequency=193.5e12,
        bandwidth=4.4e12,
        opt_params=c_band_params
    )
    
    # Test SNR calculation for typical parameters
    power = 1e-3  # 0 dBm
    distance = 100e3  # 100 km
    num_spans = 2
    
    snr = band.calculate_snr(power, distance, num_spans)
    
    assert snr['snr_db'] > 0
    assert snr['snr_total_db'] < snr['snr_db']  # Total SNR should include nonlinear effects
    assert snr['P_ASE'] > 0
    assert snr['P_NL'] > 0

def test_required_snr(c_band_params):
    band = Band(
        name='C',
        center_frequency=193.5e12,
        bandwidth=4.4e12,
        opt_params=c_band_params
    )
    
    # Test required SNR for different modulation formats
    snr_qpsk = band.get_required_snr(4)    # QPSK
    snr_16qam = band.get_required_snr(16)  # 16-QAM
    snr_64qam = band.get_required_snr(64)  # 64-QAM
    
    assert snr_64qam > snr_16qam > snr_qpsk  # Higher order modulations need higher SNR

# Add more tests...