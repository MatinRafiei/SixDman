from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
from scipy.sparse.csgraph import yen
from scipy.sparse import csr_matrix

class Network:
    """A class representing the optical network topology and its properties.
    
    This class handles the network topology, hierarchical levels, and path computation
    for the 6D-MAN planning tool.
    
    Attributes:
        graph (nx.Graph): NetworkX graph representing the network topology
        hierarchical_levels (Dict): Dictionary containing nodes in each hierarchical level
        num_spans (List): each element of this list is number of spans of specific link in the topology
    """
    
    def __init__(self, 
                 topology_name: str):
        """Initialize Network instance.
        
        Args:
            num_spans: A list containing of the number of spans per link in the topology
        """
        self.graph = nx.Graph()
        self.hierarchical_levels = {
            'HL1': {'standalone': [], 'colocated': []},
            'HL2': {'standalone': [], 'colocated': []},
            'HL3': {'standalone': [], 'colocated': []},
            'HL4': {'standalone': [], 'colocated': []}
        }
        self.topology_name = topology_name
        
    def load_topology(self, filepath: str, matrixName: str) -> nx.Graph:
        """Load network topology from a .mat file.
        
        Args:
            filepath: Path to the .mat file containing network topology
            matrixName: name of the adjacancy matrix as a MATLAB variable
        """
        try:
            mat_data = loadmat(filepath)
            self.adjacency_matrix = mat_data[matrixName]
            if self.adjacency_matrix is None:
                raise ValueError("Could not find network topology matrix in .mat file")
                
            # Convert to upper triangular to avoid duplicate edges
            self.adjacency_matrix = np.triu(self.adjacency_matrix)
            
            # Create NetworkX graph from adjacency matrix
            self.graph = nx.from_numpy_array(self.adjacency_matrix)
            self.all_links = np.array(list(self.graph.edges(data = 'weight')))

            # calculate the weights array of network graph 
            self.weights_array = self.all_links[:, 2]
    
        except Exception as e:
            raise IOError(f"Failed to load topology from {filepath}: {str(e)}")    
            
    def set_hierarchical_levels(self, **kwargs) -> dict:
        """Set the hierarchical levels of nodes in the network with flexible HLx input.
        
        Args:
            kwargs: Any number of HLx_standalone and/or HLx_colocated lists, e.g.,
                    HL1_standalone=[...], HL2_colocated=[...], etc.
        Returns:
            dict: hierarchical_levels dictionary
        """
        self.hierarchical_levels = {}
        colocated_accum = []

        # Find all unique HLx levels from the keys
        levels = set()
        for key in kwargs:
            if key.endswith('_standalone'):
                levels.add(key[:-11])
            elif key.endswith('_colocated'):
                levels.add(key[:-9])
            else:
                levels.add(key)

        # Sort levels for consistent order (HL1, HL2, ...)
        for hl in sorted(levels):
            standalone = kwargs.get(f"{hl}_standalone", [])
            colocated = kwargs.get(f"{hl}_colocated", colocated_accum.copy())
            self.hierarchical_levels[hl] = {
                'standalone': standalone,
                'colocated': sorted(colocated)
            }
            # Only accumulate if colocated wasn't set by user
            if f"{hl}_colocated" not in kwargs:
                colocated_accum += standalone

        return self.hierarchical_levels
    
    def calculate_paths(self,
                       subnetMatrix: np.ndarray,
                       paths: List,
                       source: int,
                       target: int,
                       k: int = 20) -> List[Dict]:
        """Calculate k-shortest paths between source and target nodes.
        
        Args:
            subnetMatrix: adjacency matrix of subnet
            paths: list of all candidate paths
            source: Source node ID
            target: Target node ID
            k: Number of paths to compute (default: 20)
            
        Returns:
            paths: List of dictionaries containing path information (updates version of input paths)
        """
        all_links = self.all_links
        # Convert subnetMatrix to scipy sparse matrix for Yen's algorithm
        graph_sparse = csr_matrix(subnetMatrix)
        
        # Calculate k shortest paths
        distances, predecessors = yen(
            csgraph = graph_sparse,
            source = source,
            sink = target,
            K = k,
            directed = False,
            return_predecessors = True
        )
        
        for i, distance in enumerate(distances):
            if distance == np.inf:
                continue
                
            # Reconstruct path from predecessors
            path = self._reconstruct_path(predecessors, i, source, target)
            if not path:
                continue
                
            # Calculate path properties
            links = self.links_in_path(path)
            
            paths.append({
                'src_node': source,
                'dest_node': target,
                'nodes': path,
                'links': links,
                'distance': distance,
                'num_hops': len(path) - 1
            })
            
        return paths
    
    def _reconstruct_path(self,
                         predecessors: np.ndarray,
                         path_index: int,
                         source: int,
                         target: int) -> List[int]:
        """Reconstruct path from predecessors matrix.
        
        Args:
            predecessors: Predecessors matrix from Yen's algorithm
            path_index: Index of the path to reconstruct
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs in the path
        """
        path = []
        node = target
        while node != -9999 and node != source:
            path.append(node)
            node = predecessors[path_index, node]
        if node != -9999:
            path.append(source)
            return path[::-1]
        return []
    
    def links_in_path(self, path: List[int]) -> List[int]:
        """Get list of link indices in a path, trying forward then reverse direction.

        Args:
            path: List of node IDs in the path

        Returns:
            List of indices into all_links representing links
        """
        all_links = self.all_links  # shape: (num_links, 2)
        links_array = []

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]

            # Try forward direction
            link_idx = np.where((all_links[:, 0] == src) & (all_links[:, 1] == dst))[0]

            if link_idx.size == 0:
                # Try reverse direction
                link_idx = np.where((all_links[:, 0] == dst) & (all_links[:, 1] == src))[0]

            if link_idx.size == 0:
                raise ValueError(f"No link found between {src} and {dst} in either direction.")

            links_array.append(link_idx[0])  # Use the first match

        return links_array
    
    # def links_in_path(self, all_links, path: List[int]) -> List[Tuple[int, int]]:
    #     """Get list of links in a path.
        
    #     Args:
    #         path: List of node IDs in the path
            
    #     Returns:
    #         List of (source, target) tuples representing links
    #     """

    #     all_links = self.all_links

    #     links_array = []
    #     for i in range(len(path)):
    #         if i != len(path) - 1:
    #             link_idx = np.where(((all_links[:, 0] == path[i]) & (all_links[:, 1] == path[i + 1])) | ((all_links[:, 0] == path[i + 1]) & (all_links[:, 1] == path[i])))[0]
    #             links_array.append(link_idx[0])

    #     return links_array
    
    def get_node_degrees(self, nodes: List[int]) -> Dict[int, int]:
        """Get the degree of specified nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Dictionary mapping node IDs to their degrees
        """
        return np.array(self.graph.degree(nodes))
    
    def calculate_subgraph(self,
                           hierarchy_level: int, 
                           minimum_hierarchy_level: int) -> Tuple[nx.Graph, np.ndarray]:
        """Calculate subgraph induced by specified nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            NetworkX graph representing the subgraph
            netCostMatrix_subgraph: Cost matrix of the subgraph
        """
        hierarchy_node = self.hierarchical_levels[f"HL{hierarchy_level}"]['standalone']

        lower_hierarchy_node = []
        for hl in range(hierarchy_level + 1, minimum_hierarchy_level + 1):
            lower_hierarchy_node.extend(self.hierarchical_levels[f"HL{hl}"]['standalone'])
        
        lower_hierarchy_node = np.array(lower_hierarchy_node)

        # Create a subgraph containing only edges where at least one node is in nodes
        edges_in_subgraph = []
        for u, v in self.graph.edges:
            if (u in hierarchy_node) and (v not in lower_hierarchy_node):
                edges_in_subgraph.append((u, v))
            elif (v in hierarchy_node) and (u not in lower_hierarchy_node):
                edges_in_subgraph.append((u, v))

        # calculate the adjacency matrix of the subgraphS
        netCostMatrix_subgraph = np.zeros_like(self.adjacency_matrix)
        for edge in edges_in_subgraph:
            netCostMatrix_subgraph[edge[0], edge[1]] = self.adjacency_matrix[edge[0], edge[1]]
        
        # calculate subnet graphs
        subgraph = nx.from_numpy_array(netCostMatrix_subgraph)

        netCostMatrix_subgraph = np.where(netCostMatrix_subgraph == 0, np.inf, netCostMatrix_subgraph)
        
        return subgraph, netCostMatrix_subgraph
    
    def find_neighbors(self, nodes: List[int]) -> Tuple[nx.Graph, np.ndarray]:
        """Calculate neighbors of list of nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            List of node IDs of all connected nodes
        """

        # define set to avoid duplicates
        connected_nodes = set() 
        for node in nodes:
            connected_nodes.update(self.graph.neighbors(node))

        # Remove the target nodes themselves from the result
        connected_nodes -= set(nodes)

        return connected_nodes
    
    def land_pair_finder(self, src_list: List[int], candidate_paths_sorted: pd.DataFrame, num_pairs: int) -> pd.DataFrame:
        """
        Calculate the candidate link & node disjoint (LAND) pair.

        Args:
            src_list: List of source node IDs
            candidate_paths_sorted: DataFrame of candidate paths sorted by some metric
            num_pairs: Number of disjoint pairs to select per source

        Returns:
            DataFrame with selected primary and secondary path indices
        """

        results = []

        # Preprocessing: store the index once to avoid repeated use of iterrows
        candidate_paths_sorted = candidate_paths_sorted.reset_index()

        for node in src_list:
            node_df = candidate_paths_sorted[candidate_paths_sorted['src_node'] == node]
            used_secondary_idxs = set()
            pair_counter = 0

            # Precompute all rows as lists for speed
            node_records = node_df.to_dict('records')

            for i, primary in enumerate(node_records):
                if pair_counter >= num_pairs:
                    break

                primary_idx = primary['index']
                dest_primary = primary['dest_node']
                nodes_primary = set(primary['nodes'])
                links_primary = set(primary['links'])

                # Skip primary paths already used as secondary
                if primary_idx in used_secondary_idxs:
                    continue

                # Filter once outside the inner loop
                secondary_candidates = [
                    (j, secondary) for j, secondary in enumerate(node_records)
                    if secondary['dest_node'] != dest_primary and secondary['index'] not in used_secondary_idxs
                ]

                for _, secondary in secondary_candidates:
                    nodes_secondary = set(secondary['nodes'])
                    links_secondary = set(secondary['links'])

                    # Disjoint condition: only source node in common (assumed to be first element)
                    common_nodes = nodes_primary & nodes_secondary
                    common_links = links_primary & links_secondary

                    if len(common_nodes) == 1 and len(common_links) == 0:
                        # Store result
                        results.append([
                            primary_idx,
                            secondary['index'],
                            secondary['num_hops'],
                            secondary['distance'],
                            node
                        ])
                        used_secondary_idxs.add(secondary['index'])
                        pair_counter += 1
                        break  # Only one valid secondary per primary

        # Convert to DataFrame
        standalone_path_df = pd.DataFrame(results, columns=[
            'primary_path_IDx',
            'secondary_path_IDx',
            'numHops_secondary',
            'distance_secondary',
            'src_node'
        ])

        return standalone_path_df
        
    def calc_num_pair(self, 
                      pairs_disjoint_df: pd.DataFrame):

        """calculate the candidate link & node disjoint (LAND) pair.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            List of node IDs of all connected nodes
        """
        return pairs_disjoint_df.groupby('src_node')['primary_path_IDx'].count().to_numpy()

    def print_info(self) -> None:

        """Print basic information about the network."""
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Hierarchical levels: {self.hierarchical_levels}")
        print(f"All links: {self.all_links}")