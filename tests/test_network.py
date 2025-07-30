import pytest
import numpy as np
from src.sixdman.core.network import Network

numspans = np.ones(100)

def test_network_initialization():
    network = Network(numspans)
    assert len(network.graph.nodes) == 0
    assert len(network.hierarchical_levels['HL1']['standalone']) == 0

def test_hierarchical_levels():
    network = Network(numspans)
    hl1 = [1, 5]
    hl2 = [0, 2, 3, 4]
    hl3 = list(range(6, 39))
    hl4 = list(range(39, 157))
    
    network.set_hierarchical_levels(hl1, hl2, hl3, hl4)
    
    assert network.hierarchical_levels['HL1']['standalone'] == hl1
    assert network.hierarchical_levels['HL2']['colocated'] == hl1
    assert set(network.hierarchical_levels['HL3']['colocated']) == set(hl1 + hl2)

def test_path_calculation():
    network = Network(numspans)
    # Create a simple test graph
    edges = [(0, 1, 10), (1, 2, 20), (0, 2, 35)]
    for u, v, w in edges:
        network.graph.add_edge(u, v, weight=w)
    
    paths = network.calculate_paths(0, 2, k=2)
    assert len(paths) == 2
    assert paths[0]['distance'] == 30  # 
    assert paths[1]['distance'] == 35  # second shortest path
    assert paths[0]['nodes'] == [0, 1, 2]  # first path
    assert paths[1]['nodes'] == [0, 2]  # second path    

# Add more tests...