"""
Traveling Salesman Problem (TSP) Solver using QAOA

Problem Definition:
Find the shortest route visiting all cities exactly once and returning to start.

Mathematical Formulation (QUBO):
Binary variables: x_{i,j} = 1 if city i is visited at position j, else 0

Objective: Minimize total distance
Cost = Σ_{i,j,k} d_{ij} * x_{i,k} * x_{j,k+1}

Constraints (with penalty M):
1. Each city visited exactly once: Σ_j x_{i,j} = 1 for all i
2. Each position has exactly one city: Σ_i x_{i,j} = 1 for all j

Total QUBO:
H = Distance_Term + M * Constraint_Penalties
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qaoa_core import QAOAOptimizer
import logging
import itertools

logger = logging.getLogger(__name__)

class TSPSolver:
    """
    QAOA-based Traveling Salesman Problem Solver with proper QUBO encoding
    """
    
    def __init__(self, distance_matrix: np.ndarray, p_layers: int = 2):
        """
        Initialize TSP solver
        
        Args:
            distance_matrix: n x n matrix of distances between cities
            p_layers: Number of QAOA layers
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.p_layers = p_layers
        
        # Standard TSP encoding: n cities × n positions = n² qubits
        # x_{i,j} means city i is at position j
        self.num_qubits = self.num_cities * self.num_cities
        
        # Penalty parameter for constraint violations
        # Should be larger than max distance to prioritize feasibility
        self.penalty = 10.0 * np.max(distance_matrix)
        
        # Build QUBO matrix
        self.qubo_matrix = self._build_qubo_matrix()
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
        logger.info(f"TSP Solver initialized: {self.num_cities} cities, {self.num_qubits} qubits")
    
    def _build_qubo_matrix(self) -> np.ndarray:
        """
        Build QUBO matrix for TSP
        
        Returns:
            QUBO matrix Q where objective = x^T Q x
        """
        n = self.num_cities
        size = n * n
        Q = np.zeros((size, size))
        
        # Helper function to get qubit index for x_{i,j}
        def idx(city, position):
            return city * n + position
        
        # 1. Distance objective: Minimize tour length
        # Add distance between consecutive positions
        for pos in range(n):
            next_pos = (pos + 1) % n  # Circular tour
            for city_i in range(n):
                for city_j in range(n):
                    if city_i != city_j:
                        # If city_i at pos and city_j at next_pos, add distance
                        i_idx = idx(city_i, pos)
                        j_idx = idx(city_j, next_pos)
                        # QUBO: add to diagonal if i==j, off-diagonal otherwise
                        if i_idx == j_idx:
                            Q[i_idx][i_idx] += self.distance_matrix[city_i][city_j]
                        else:
                            Q[i_idx][j_idx] += self.distance_matrix[city_i][city_j] / 2
                            Q[j_idx][i_idx] += self.distance_matrix[city_i][city_j] / 2
        
        # 2. Constraint penalties
        M = self.penalty
        
        # Constraint 1: Each city appears exactly once
        # (Σ_j x_{i,j} - 1)² for each city i
        for city in range(n):
            # Quadratic expansion: (Σ x - 1)² = Σ x² + Σ Σ x_i x_j - 2 Σ x + 1
            # x² = x for binary, so: Σ x + 2 Σ Σ_{i<j} x_i x_j - 2 Σ x + 1
            # = 2 Σ Σ_{i<j} x_i x_j - Σ x + 1
            for pos1 in range(n):
                i1 = idx(city, pos1)
                # Linear term: -Σ x
                Q[i1][i1] += M * (-1)
                # Quadratic terms: 2 * x_i * x_j for different positions
                for pos2 in range(pos1 + 1, n):
                    i2 = idx(city, pos2)
                    Q[i1][i2] += M
                    Q[i2][i1] += M
        
        # Constraint 2: Each position has exactly one city
        # (Σ_i x_{i,j} - 1)² for each position j
        for pos in range(n):
            for city1 in range(n):
                i1 = idx(city1, pos)
                # Linear term
                Q[i1][i1] += M * (-1)
                # Quadratic terms
                for city2 in range(city1 + 1, n):
                    i2 = idx(city2, pos)
                    Q[i1][i2] += M
                    Q[i2][i1] += M
        
        return Q
    
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for TSP using QUBO matrix
        
        The Hamiltonian H = Σ_{i,j} Q_{ij} Z_i Z_j encodes:
        1. Distance minimization between consecutive cities
        2. Constraint penalties for valid tours
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        n = self.num_qubits
        
        # Apply diagonal terms (single-qubit rotations)
        for i in range(n):
            if self.qubo_matrix[i][i] != 0:
                qc.rz(2 * gamma * self.qubo_matrix[i][i], i)
        
        # Apply off-diagonal terms (two-qubit interactions)
        for i in range(n):
            for j in range(i + 1, n):
                if self.qubo_matrix[i][j] != 0:
                    # ZZ interaction using RZZ gate
                    qc.rzz(2 * gamma * self.qubo_matrix[i][j], i, j)
    
    def decode_solution(self, bitstring: str) -> Tuple[List[int], bool]:
        """
        Decode bitstring to tour using x_{i,j} encoding
        
        Args:
            bitstring: Binary string from quantum measurement
            
        Returns:
            (tour as list of city indices, is_valid)
        """
        n = self.num_cities
        
        # Parse bitstring into x_{i,j} matrix
        x = np.zeros((n, n), dtype=int)
        for city in range(n):
            for pos in range(n):
                idx = city * n + pos
                if idx < len(bitstring):
                    x[city][pos] = int(bitstring[idx])
        
        # Extract tour from x matrix
        tour = []
        is_valid = True
        
        # For each position, find which city is assigned
        for pos in range(n):
            cities_at_pos = [city for city in range(n) if x[city][pos] == 1]
            
            if len(cities_at_pos) == 1:
                tour.append(cities_at_pos[0])
            elif len(cities_at_pos) == 0:
                # No city at this position - take first unassigned
                unassigned = set(range(n)) - set(tour)
                if unassigned:
                    tour.append(min(unassigned))
                is_valid = False
            else:
                # Multiple cities at same position - take first
                tour.append(cities_at_pos[0])
                is_valid = False
        
        # Check if all cities are visited
        if len(set(tour)) != n:
            is_valid = False
        
        return tour, is_valid
    
    def _calculate_cost(self, bitstring: str) -> float:
        """
        Calculate cost for a bitstring (tour cost + constraint penalties)
        
        This is used during QAOA optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Total cost (distance + penalties)
        """
        tour, is_valid = self.decode_solution(bitstring)
        
        # Compute tour distance
        distance = self.compute_tour_cost(tour)
        
        # Add penalty for invalid tours
        if not is_valid:
            distance += self.penalty
        
        return distance
    
    def compute_tour_cost(self, tour: List[int]) -> float:
        """
        Compute total tour distance
        
        Args:
            tour: List of city indices in visit order
            
        Returns:
            Total distance
        """
        if len(tour) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(tour)):
            next_i = (i + 1) % len(tour)  # Circular tour
            if tour[i] < len(self.distance_matrix) and tour[next_i] < len(self.distance_matrix):
                cost += self.distance_matrix[tour[i]][tour[next_i]]
        
        return cost
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function for optimization (used by QAOA)
        
        Args:
            bitstring: Binary string
            
        Returns:
            Tour cost with penalties for constraint violations
        """
        return self._calculate_cost(bitstring)
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 50) -> Dict:
        """
        Solve TSP using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving TSP for {self.num_cities} cities with {self.num_qubits} qubits")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best valid solution
        best_tour = None
        best_cost = float('inf')
        is_valid = False
        best_bitstring = None
        
        # Try top solutions
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:20]:
            tour, valid = self.decode_solution(bitstring)
            cost = self.compute_tour_cost(tour)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                is_valid = valid
                best_bitstring = bitstring
                
                if valid:
                    break  # Found valid solution
        
        # Compute classical bound (greedy nearest neighbor)
        classical_tour, classical_cost = self._greedy_nearest_neighbor()
        
        # Optimal solution (brute force for small instances)
        optimal_cost = self._compute_optimal() if self.num_cities <= 7 else classical_cost
        
        # Convergence analysis
        convergence = self.qaoa.analyze_convergence()
        
        # Build tour string for display
        tour_string = " → ".join(map(str, best_tour)) + f" → {best_tour[0]}" if best_tour else "No solution"
        
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
            'best_tour': [int(x) for x in best_tour] if best_tour else [],
            'tour_string': tour_string,
            'tour_cost': float(best_cost),
            'is_valid': bool(is_valid),
            'best_bitstring': best_bitstring,
            'optimal_params': [float(x) for x in opt_result['optimal_params']],
            'iterations': int(opt_result['iterations']),
            'convergence_analysis': to_native(convergence),
            'classical_comparison': {
                'classical_tour': [int(x) for x in classical_tour],
                'classical_cost': float(classical_cost),
                'optimal_cost': float(optimal_cost),
                'qaoa_approximation_ratio': float(best_cost / optimal_cost) if optimal_cost > 0 else 1.0
            },
            'problem_info': {
                'num_cities': int(self.num_cities),
                'num_qubits': int(self.num_qubits),
                'p_layers': int(self.p_layers),
                'penalty_parameter': float(self.penalty)
            }
        }
        
        return result
    
    def _greedy_nearest_neighbor(self) -> Tuple[List[int], float]:
        """
        Greedy nearest neighbor heuristic
        
        Returns:
            (tour, cost)
        """
        unvisited = set(range(self.num_cities))
        tour = [0]  # Start from city 0
        unvisited.remove(0)
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        cost = self.compute_tour_cost(tour)
        return tour, cost
    
    def _compute_optimal(self) -> float:
        """
        Compute optimal solution via brute force (only for small instances)
        
        Returns:
            Optimal tour cost
        """
        if self.num_cities > 7:
            return float('inf')
        
        cities = list(range(self.num_cities))
        min_cost = float('inf')
        
        for perm in itertools.permutations(cities[1:]):
            tour = [0] + list(perm)
            cost = self.compute_tour_cost(tour)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    @staticmethod
    def generate_random_instance(num_cities: int, max_distance: int = 100) -> np.ndarray:
        """s
        Generate random TSP instance
        
        Args:
            num_cities: Number of cities
            max_distance: Maximum distance between cities
            
        Returns:
            Distance matrix
        """
        # Random city coordinates
        coords = np.random.rand(num_cities, 2) * max_distance
        
        # Compute Euclidean distances
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        return distance_matrix
