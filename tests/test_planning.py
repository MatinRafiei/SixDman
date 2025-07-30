import pytest
import numpy as np
from sixdman.core.network import Network
from sixdman.core.band import Band, OpticalParameters
from sixdman.core.planning import PlanningTool, PlanningConstraints

@pytest.fixture
def sample_network():
    network = Network()
    # Create a simple test network
    edges = [
        (0, 1, 10), (1, 2, 20), (0, 2, 35),
        (2, 3, 15), (3, 4, 25), (1, 4, 40)
    ]
    for u, v, w in edges:
        network.graph.add_edge(u, v, weight=w)
        
    # Set hierarchical levels
    network.set_hierarchical_levels(
        hl1_standalone=[0],
        hl2_standalone=[1, 2],
        hl3_standalone=[3],
        hl4_standalone=[4]
    )
    return network

@pytest.fixture
def c_band():
    params = OpticalParameters(
        beta_2=-21.7e-27,
        beta_3=0.14e-39,
        alpha_db=0.2,
        gamma=1.21e-3,
        F_dB=6.0
    )
    return Band(
        name='C',
        center_frequency=193.5e12,
        bandwidth=4.4e12,
        opt_params=params
    )

@pytest.fixture
def planning_constraints():
    return PlanningConstraints(
        max_power_per_channel=2e-3,  # 2 mW
        min_power_per_channel=0.1e-3,  # 0.1 mW
        target_ber=1e-2,
        min_osnr=20.0,
        max_nonlinear_phase=0.1
    )

def test_planning_tool_initialization(sample_network, c_band, planning_constraints):
    tool = PlanningTool(
        network=sample_network,
        bands=[c_band],
        constraints=planning_constraints
    )
    
    assert tool.network is sample_network
    assert 'C' in tool.bands
    assert tool.constraints is planning_constraints

def test_network_optimization(sample_network, c_band, planning_constraints):
    tool = PlanningTool(
        network=sample_network,
        bands=[c_band],
        constraints=planning_constraints
    )
    
    results = tool.optimize_network()
    
    assert 'paths' in results
    assert 'band_assignments' in results
    assert 'modulation_formats' in results
    assert 'power_levels' in results

def test_path_recommendations(sample_network, c_band, planning_constraints):
    tool = PlanningTool(
        network=sample_network,
        bands=[c_band],
        constraints=planning_constraints
    )
    
    recommendations = tool.get_path_recommendations(0, 4)
    
    assert len(recommendations) > 0
    assert 'path' in recommendations[0]
    assert 'band' in recommendations[0]
    assert 'modulation_format' in recommendations[0]

# Add more tests...