"""
Graph Coloring Problem Solver using QAOA

Problem Definition:
Assign colors to graph vertices such that no adjacent vertices share the same color,
using minimum number of colors.

Mathematical Formulation (QUBO):
Binary variables: x_{i,k} = 1 if vertex i is assigned color k, else 0

Objective: Minimize edge conflicts
Cost = Σ_{(i,j)∈E} Σ_k x_{i,k} * x_{j,k}

Constraint (with penalty M):
Each vertex has exactly one color: Σ_k x_{i,k} = 1 for all i

Total QUBO:
H = Conflict_Term + M * Constraint_Penalties

Applications:
- Task scheduling
- Register allocation
- Frequency assignment
- Sudoku solving
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Set, Tuple
from qaoa_core import QAOAOptimizer
import logging

logger = logging.getLogger(__name__)

class GraphColoringSolver:
    """
    QAOA-based Graph Coloring Solver with proper QUBO encoding
    """
    
    def __init__(self, graph: nx.Graph, num_colors: int, p_layers: int = 2):
        """
        Initialize Graph Coloring solver
        
        Args:
            graph: Input graph (NetworkX)
            num_colors: Number of colors to use
            p_layers: Number of QAOA layers
        """
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.num_colors = num_colors
        self.p_layers = p_layers
        
        # Store edges first (needed by _build_qubo_matrix)
        self.edges = list(graph.edges())
        
        # One-hot encoding: n vertices × k colors = n*k qubits
        # x_{i,k} means vertex i has color k
        self.num_qubits = self.num_vertices * self.num_colors
        self.encoding = 'one-hot'
        
        # Penalty parameter for constraint violations
        # Should be larger than number of edges
        self.penalty = 10.0 * len(self.edges) if len(self.edges) > 0 else 10.0
        
        # Build QUBO matrix
        self.qubo_matrix = self._build_qubo_matrix()
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
        logger.info(f"Graph Coloring initialized: {self.num_vertices} vertices, {self.num_colors} colors, {self.num_qubits} qubits")
    
    def _build_qubo_matrix(self) -> np.ndarray:
        """
        Build QUBO matrix for Graph Coloring
        
        Returns:
            QUBO matrix Q
        """
        n = self.num_vertices
        k = self.num_colors
        size = n * k
        Q = np.zeros((size, size))
        
        # Helper function to get qubit index for x_{i,c}
        def idx(vertex, color):
            return vertex * k + color
        
        # 1. Objective: Minimize edge conflicts
        # For each edge (u,v), penalize if both have same color
        # Σ_{(u,v)∈E} Σ_c x_{u,c} * x_{v,c}
        for u, v in self.edges:
            for color in range(k):
                u_idx = idx(u, color)
                v_idx = idx(v, color)
                
                # Add penalty for same color assignment
                if u_idx == v_idx:
                    Q[u_idx][u_idx] += 1.0
                else:
                    Q[u_idx][v_idx] += 0.5
                    Q[v_idx][u_idx] += 0.5
        
        # 2. Constraint: Each vertex has exactly one color
        # (Σ_c x_{i,c} - 1)² for each vertex i
        M = self.penalty
        
        for vertex in range(n):
            for color1 in range(k):
                i1 = idx(vertex, color1)
                # Linear term: -Σ x
                Q[i1][i1] += M * (-1)
                # Quadratic terms
                for color2 in range(color1 + 1, k):
                    i2 = idx(vertex, color2)
                    Q[i1][i2] += M
                    Q[i2][i1] += M
        
        return Q
    
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for Graph Coloring using QUBO matrix
        
        H penalizes:
        1. Adjacent vertices with same color (conflicts)
        2. Invalid colorings (vertex with multiple/no colors)
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        n = self.num_qubits
        
        # Apply diagonal terms
        for i in range(n):
            if self.qubo_matrix[i][i] != 0:
                qc.rz(2 * gamma * self.qubo_matrix[i][i], i)
        
        # Apply off-diagonal terms (ZZ interactions)
        for i in range(n):
            for j in range(i + 1, n):
                if self.qubo_matrix[i][j] != 0:
                    qc.rzz(2 * gamma * self.qubo_matrix[i][j], i, j)
    
    def decode_coloring(self, bitstring: str) -> Tuple[Dict[int, int], bool]:
        """
        Decode bitstring to vertex coloring with validity check
        
        Args:
            bitstring: Binary string
            
        Returns:
            (coloring dict mapping vertex -> color, is_valid)
        """
        coloring = {}
        is_valid = True
        k = self.num_colors
        
        # One-hot encoding: each vertex has k bits
        for vertex in range(self.num_vertices):
            start = vertex * k
            end = start + k
            
            if end <= len(bitstring):
                vertex_bits = bitstring[start:end]
                # Find assigned colors (where bit is 1)
                assigned_colors = [c for c, bit in enumerate(vertex_bits) if bit == '1']
                
                if len(assigned_colors) == 1:
                    # Valid: exactly one color
                    coloring[vertex] = assigned_colors[0]
                elif len(assigned_colors) == 0:
                    # No color assigned: assign color 0 as default
                    coloring[vertex] = 0
                    is_valid = False
                else:
                    # Multiple colors: take first one
                    coloring[vertex] = assigned_colors[0]
                    is_valid = False
            else:
                coloring[vertex] = 0
                is_valid = False
        
        return coloring, is_valid
    
    def is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Check if coloring is valid (no adjacent vertices with same color)
        
        Args:
            coloring: Vertex -> color mapping
            
        Returns:
            True if valid
        """
        for u, v in self.edges:
            if coloring.get(u, -1) == coloring.get(v, -1):
                return False
        return True
    
    def count_conflicts(self, coloring: Dict[int, int]) -> int:
        """
        Count number of edge conflicts (adjacent vertices with same color)
        
        Args:
            coloring: Vertex -> color mapping
            
        Returns:
            Number of conflicts
        """
        conflicts = 0
        for u, v in self.edges:
            if coloring.get(u, -1) == coloring.get(v, -1):
                conflicts += 1
        return conflicts
    
    def _calculate_cost(self, bitstring: str) -> float:
        """
        Calculate cost for a bitstring (conflicts + constraint penalties)
        
        Used during QAOA optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Total cost
        """
        coloring, is_valid = self.decode_coloring(bitstring)
        conflicts = self.count_conflicts(coloring)
        
        # Add penalty for invalid colorings
        if not is_valid:
            conflicts += self.penalty
        
        return float(conflicts)
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function (number of conflicts + penalties)
        
        Args:
            bitstring: Binary string
            
        Returns:
            Number of conflicts with penalties
        """
        return self._calculate_cost(bitstring)
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 100) -> Dict:
        """
        Solve Graph Coloring using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving Graph Coloring: {self.num_vertices} vertices, {self.num_colors} colors")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best valid coloring
        best_coloring = None
        best_conflicts = float('inf')
        is_valid = False
        best_bitstring = None
        
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:20]:
            coloring, valid = self.decode_coloring(bitstring)
            conflicts = self.count_conflicts(coloring)
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_coloring = coloring
                is_valid = (conflicts == 0) and valid
                best_bitstring = bitstring
                
                if is_valid:
                    break
        
        # Classical comparison (greedy coloring)
        classical_coloring = self._greedy_coloring()
        classical_conflicts = self.count_conflicts(classical_coloring)
        
        # Compute chromatic number bound
        chromatic_bound = self._compute_chromatic_bound()
        
        # Convergence analysis
        convergence = self.qaoa.analyze_convergence()
        
        # Build coloring string for display
        coloring_string = ", ".join([f"V{v}:C{c}" for v, c in sorted(best_coloring.items())]) if best_coloring else "No solution"
        
        # Convert all numpy types to Python native types for JSON serialization
        def to_native(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(item) for item in obj]
            return obj
        
        result = {
            'success': bool(opt_result['success'] and is_valid),
            'coloring': {int(k): int(v) for k, v in best_coloring.items()} if best_coloring else {},
            'coloring_string': coloring_string,
            'num_conflicts': int(best_conflicts),
            'is_valid': bool(is_valid),
            'best_bitstring': best_bitstring,
            'optimal_params': [float(x) for x in opt_result['optimal_params']],
            'iterations': int(opt_result['iterations']),
            'convergence_analysis': to_native(convergence),
            'classical_comparison': {
                'classical_coloring': {int(k): int(v) for k, v in classical_coloring.items()},
                'classical_conflicts': int(classical_conflicts),
                'improvement': int(classical_conflicts - best_conflicts)
            },
            'graph_metrics': {
                'num_vertices': int(self.num_vertices),
                'num_edges': int(len(self.edges)),
                'num_colors': int(self.num_colors),
                'chromatic_bound': int(chromatic_bound),
                'encoding': self.encoding,
                'num_qubits': int(self.num_qubits),
                'p_layers': int(self.p_layers),
                'penalty_parameter': float(self.penalty)
            }
        }
        
        return result
    
    def _greedy_coloring(self) -> Dict[int, int]:
        """
        Greedy graph coloring algorithm
        
        Returns:
            Vertex -> color mapping
        """
        coloring = {}
        
        for vertex in self.graph.nodes():
            # Find colors of neighbors
            neighbor_colors = {coloring.get(neighbor) for neighbor in self.graph.neighbors(vertex)
                             if neighbor in coloring}
            
            # Assign first available color
            for color in range(self.num_colors):
                if color not in neighbor_colors:
                    coloring[vertex] = color
                    break
            else:
                # No color available, assign last color (conflict)
                coloring[vertex] = self.num_colors - 1
        
        return coloring
    
    def _compute_chromatic_bound(self) -> int:
        """
        Compute upper bound on chromatic number
        
        Returns:
            Upper bound (max degree + 1)
        """
        if self.graph.number_of_nodes() == 0:
            return 0
        
        max_degree = max(dict(self.graph.degree()).values()) if self.graph.number_of_edges() > 0 else 1
        return max_degree + 1
    
    @staticmethod
    def generate_random_graph(num_vertices: int, edge_probability: float = 0.3) -> nx.Graph:
        """
        Generate random graph for testing
        
        Args:
            num_vertices: Number of vertices
            edge_probability: Probability of edge creation
            
        Returns:
            Random graph
        """
        return nx.gnp_random_graph(num_vertices, edge_probability)
