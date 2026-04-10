"""
Vehicle Routing Problem (VRP) Solver using QAOA

Problem Definition:
Optimize routes for multiple vehicles to serve customers from a depot.

Simplified VRP Formulation:
Binary variables: x_{c,v} = 1 if customer c is assigned to vehicle v

Objective: Minimize total distance traveled by all vehicles
Cost = Σ_v Σ_{c∈route_v} (distance from depot + route distance)

Constraints (with penalty M):
1. Each customer assigned to exactly one vehicle: Σ_v x_{c,v} = 1 for all c
2. Vehicle capacity: Σ_{c: x_{c,v}=1} demand_c ≤ capacity_v for all v

Simplified Approach:
- Fixed depot (node 0)
- Customer assignment to vehicles (not full routing)
- Greedy TSP within each vehicle's customers

Applications:
- Delivery optimization
- Fleet management
- Logistics planning
"""

import numpy as np
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qaoa_core import QAOAOptimizer
import logging

logger = logging.getLogger(__name__)

class VRPSolver:
    """
    QAOA-based Vehicle Routing Problem Solver (Simplified)
    """
    
    def __init__(self, distance_matrix: np.ndarray, demands: List[float],
                 vehicle_capacity: float, num_vehicles: int, p_layers: int = 2):
        """
        Initialize VRP solver
        
        Args:
            distance_matrix: (n+1) x (n+1) matrix (depot + n customers)
            demands: Customer demands (excluding depot)
            vehicle_capacity: Vehicle capacity
            num_vehicles: Number of vehicles
            p_layers: Number of QAOA layers
        """
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.num_customers = len(demands)
        self.p_layers = p_layers
        
        # Encoding: customer × vehicle assignment
        # x_{c,v} means customer c assigned to vehicle v
        self.num_qubits = self.num_customers * num_vehicles
        
        # Penalty for constraint violations
        max_dist = np.max(distance_matrix)
        self.penalty = 100.0 * max_dist
        
        # Build QUBO matrix
        self.qubo_matrix = self._build_qubo_matrix()
        
        # Initialize QAOA
        self.qaoa = QAOAOptimizer(self.num_qubits, p_layers)
        
        logger.info(f"VRP Solver initialized: {self.num_customers} customers, {num_vehicles} vehicles, {self.num_qubits} qubits")
    
    def _build_qubo_matrix(self) -> np.ndarray:
        """
        Build QUBO matrix for simplified VRP (customer-vehicle assignment)
        
        Returns:
            QUBO matrix Q
        """
        n_c = self.num_customers
        n_v = self.num_vehicles
        size = n_c * n_v
        Q = np.zeros((size, size))
        
        # Helper function to get qubit index for x_{c,v}
        def idx(customer, vehicle):
            return customer * n_v + vehicle
        
        # 1. Objective: Minimize distance (approximate with depot distances)
        # For each customer-vehicle pair, add depot-customer distance
        for customer in range(n_c):
            # Distance from depot (node 0) to customer (node customer+1)
            depot_dist = self.distance_matrix[0][customer + 1]
            for vehicle in range(n_v):
                i = idx(customer, vehicle)
                Q[i][i] += depot_dist
        
        # 2. Constraint: Each customer assigned to exactly one vehicle
        # (Σ_v x_{c,v} - 1)² for each customer c
        M = self.penalty
        
        for customer in range(n_c):
            for v1 in range(n_v):
                i1 = idx(customer, v1)
                # Linear term
                Q[i1][i1] += M * (-1)
                # Quadratic terms
                for v2 in range(v1 + 1, n_v):
                    i2 = idx(customer, v2)
                    Q[i1][i2] += M
                    Q[i2][i1] += M
        
        # 3. Capacity constraint penalties (soft)
        # Penalize if vehicle capacity is exceeded
        capacity_penalty = 0.5 * M
        for vehicle in range(n_v):
            for c1 in range(n_c):
                for c2 in range(c1 + 1, n_c):
                    # If both customers assigned to same vehicle, add demand penalty
                    i1 = idx(c1, vehicle)
                    i2 = idx(c2, vehicle)
                    combined_demand = self.demands[c1] + self.demands[c2]
                    
                    if combined_demand > self.vehicle_capacity:
                        # Strong penalty for capacity violation
                        Q[i1][i2] += capacity_penalty
                        Q[i2][i1] += capacity_penalty
        
        return Q
    
    def cost_hamiltonian(self, qc: QuantumCircuit, gamma) -> None:
        """
        Apply cost Hamiltonian for VRP using QUBO matrix
        
        H_C encodes:
        1. Distance minimization (depot to customers)
        2. Customer assignment constraints
        3. Capacity constraint penalties
        
        Args:
            qc: Quantum circuit
            gamma: Cost parameter
        """
        n = self.num_qubits
        
        # Apply diagonal terms
        for i in range(n):
            if self.qubo_matrix[i][i] != 0:
                qc.rz(2 * gamma * self.qubo_matrix[i][i], i)
        
        # Apply off-diagonal terms
        for i in range(n):
            for j in range(i + 1, n):
                if self.qubo_matrix[i][j] != 0:
                    qc.rzz(2 * gamma * self.qubo_matrix[i][j], i, j)
    
    def decode_solution(self, bitstring: str) -> Tuple[List[List[int]], bool]:
        """
        Decode bitstring to vehicle routes with validity check
        
        Args:
            bitstring: Binary string
            
        Returns:
            (routes as list of customer lists per vehicle, is_valid)
        """
        routes = [[] for _ in range(self.num_vehicles)]
        is_valid = True
        n_v = self.num_vehicles
        
        # Parse customer-vehicle assignments
        for customer in range(self.num_customers):
            assigned_vehicles = []
            for vehicle in range(n_v):
                idx = customer * n_v + vehicle
                if idx < len(bitstring) and bitstring[idx] == '1':
                    assigned_vehicles.append(vehicle)
            
            if len(assigned_vehicles) == 1:
                # Valid: exactly one vehicle
                routes[assigned_vehicles[0]].append(customer + 1)  # +1 for depot offset
            elif len(assigned_vehicles) == 0:
                # No assignment: assign to vehicle 0
                routes[0].append(customer + 1)
                is_valid = False
            else:
                # Multiple assignments: take first
                routes[assigned_vehicles[0]].append(customer + 1)
                is_valid = False
        
        return routes, is_valid
    
    def compute_route_cost(self, routes: List[List[int]]) -> Tuple[float, bool, List[float]]:
        """
        Compute total route cost and check feasibility
        
        Args:
            routes: List of routes (customer lists per vehicle)
            
        Returns:
            (total_cost, is_feasible, vehicle_loads)
        """
        total_cost = 0.0
        is_feasible = True
        vehicle_loads = []
        
        for route in routes:
            if not route:
                vehicle_loads.append(0.0)
                continue
            
            # Check capacity
            route_demand = sum(self.demands[customer - 1] for customer in route)
            vehicle_loads.append(route_demand)
            
            if route_demand > self.vehicle_capacity:
                is_feasible = False
            
            # Compute route distance: depot -> customers -> depot
            # Simplified: depot to each customer and back
            route_cost = 0.0
            for customer in route:
                route_cost += 2 * self.distance_matrix[0][customer]  # Round trip
            
            total_cost += route_cost
        
        # Penalty for infeasibility
        if not is_feasible:
            total_cost *= 2.0
        
        return total_cost, is_feasible, vehicle_loads
    
    def _calculate_cost(self, bitstring: str) -> float:
        """
        Calculate cost for a bitstring (route cost + penalties)
        
        Used during QAOA optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Total cost
        """
        routes, is_valid = self.decode_solution(bitstring)
        cost, is_feasible, _ = self.compute_route_cost(routes)
        
        # Add penalty for invalid assignments
        if not is_valid:
            cost += self.penalty
        
        return cost
    
    def cost_function(self, bitstring: str) -> float:
        """
        Cost function for optimization
        
        Args:
            bitstring: Binary string
            
        Returns:
            Route cost with penalties
        """
        return self._calculate_cost(bitstring)
    
    def solve(self, method: str = 'COBYLA', max_iter: int = 50) -> Dict:
        """
        Solve VRP using QAOA
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Solution dictionary
        """
        logger.info(f"Solving VRP: {self.num_customers} customers, {self.num_vehicles} vehicles")
        
        # Create QAOA circuit
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian)
        
        # Optimize
        opt_result = self.qaoa.optimize(circuit, self.cost_function, method, max_iter)
        
        # Get solution probabilities
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Find best feasible solution
        best_routes = None
        best_cost = float('inf')
        is_feasible = False
        is_valid = False
        best_bitstring = None
        best_loads = []
        
        for bitstring, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]:
            routes, valid = self.decode_solution(bitstring)
            cost, feasible, loads = self.compute_route_cost(routes)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = routes
                is_feasible = feasible
                is_valid = valid
                best_bitstring = bitstring
                best_loads = loads
                
                if feasible and valid:
                    break  # Found valid feasible solution
        
        # Classical comparison (greedy)
        classical_routes, classical_cost = self._greedy_solution()
        
        # Convergence analysis
        convergence = self.qaoa.analyze_convergence()
        
        # Build routes string
        routes_string = " | ".join([
            f"V{v}: [" + "→".join(map(str, route)) + "]" 
            for v, route in enumerate(best_routes) if route
        ]) if best_routes else "No solution"
        
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
            'success': bool(opt_result['success'] and is_feasible and is_valid),
            'routes': [[int(x) for x in route] for route in best_routes] if best_routes else [],
            'routes_string': routes_string,
            'total_cost': float(best_cost),
            'is_feasible': bool(is_feasible),
            'is_valid': bool(is_valid),
            'vehicle_loads': [float(load) for load in best_loads],
            'best_bitstring': best_bitstring,
            'optimal_params': [float(x) for x in opt_result['optimal_params']],
            'iterations': int(opt_result['iterations']),
            'convergence_analysis': to_native(convergence),
            'classical_comparison': {
                'classical_routes': [[int(x) for x in route] for route in classical_routes],
                'classical_cost': float(classical_cost),
                'improvement': float((classical_cost - best_cost) / classical_cost * 100) if classical_cost > 0 else 0.0
            },
            'problem_info': {
                'num_customers': int(self.num_customers),
                'num_vehicles': int(self.num_vehicles),
                'vehicle_capacity': float(self.vehicle_capacity),
                'num_qubits': int(self.num_qubits),
                'p_layers': int(self.p_layers),
                'penalty_parameter': float(self.penalty)
            }
        }
        
        return result
    
    def _greedy_solution(self) -> Tuple[List[List[int]], float]:
        """
        Greedy VRP solution (nearest neighbor with capacity check)
        
        Returns:
            (routes, total_cost)
        """
        routes = [[] for _ in range(self.num_vehicles)]
        route_demands = [0.0] * self.num_vehicles
        unassigned = set(range(1, self.num_customers + 1))
        
        while unassigned:
            for vehicle in range(self.num_vehicles):
                if not unassigned:
                    break
                
                # Find nearest unassigned customer that fits
                current = routes[vehicle][-1] if routes[vehicle] else 0
                
                feasible_customers = [
                    c for c in unassigned 
                    if route_demands[vehicle] + self.demands[c - 1] <= self.vehicle_capacity
                ]
                
                if not feasible_customers:
                    continue
                
                nearest = min(feasible_customers, 
                            key=lambda c: self.distance_matrix[current][c])
                
                routes[vehicle].append(nearest)
                route_demands[vehicle] += self.demands[nearest - 1]
                unassigned.remove(nearest)
        
        cost, _, _ = self.compute_route_cost(routes)
        return routes, cost
    
    @staticmethod
    def generate_random_instance(num_customers: int, num_vehicles: int,
                                capacity: float = 100.0, max_distance: int = 100) -> Tuple:
        """
        Generate random VRP instance
        
        Returns:
            (distance_matrix, demands)
        """
        # Random coordinates (depot + customers)
        coords = np.random.rand(num_customers + 1, 2) * max_distance
        
        # Distance matrix
        distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        # Random demands (ensure solvable)
        demands = np.random.uniform(10, capacity / num_vehicles, num_customers)
        
        return distance_matrix, demands.tolist()
