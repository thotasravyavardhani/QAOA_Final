import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from scipy.optimize import minimize
from typing import List, Dict, Callable, Optional, Any
import time

# ==============================================================================
# 1. CORE QAOA OPTIMIZER LOGIC (qaoa_core.py adaptation)
# ==============================================================================

class QAOAOptimizer:
    
    def __init__(self, num_qubits: int, p_layers: int = 1):
        self.num_qubits = num_qubits
        self.p_layers = p_layers
        self.backend = Aer.get_backend('qasm_simulator')
        self.qaoa_params = [Parameter(f'p_{i}') for i in range(2 * p_layers)] 
        self.optimization_history = []
        self.best_cost = float('inf')
        self.best_params = None
        
    def create_qaoa_circuit(self, cost_hamiltonian_func: Callable, 
                           initial_angles: Optional[List[float]] = None) -> QuantumCircuit:
        
        qr = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # WARM START / STANDARD INITIALIZATION
        if initial_angles is not None:
            for i in range(self.num_qubits):
                if i < len(initial_angles):
                    qc.ry(initial_angles[i], i)
            qc.barrier(label='Warm_Start_Init')
        else:
            qc.h(range(self.num_qubits))
            qc.barrier(label='Init_Hadamard')
        
        # QAOA layers
        for layer in range(self.p_layers):
            gamma = self.qaoa_params[2 * layer]
            beta = self.qaoa_params[2 * layer + 1]
            cost_hamiltonian_func(qc, gamma)
            qc.rx(2 * beta, range(self.num_qubits)) 
        
        qc.measure_all()
        return qc

    def compute_expectation(self, counts: Dict[str, int], cost_function: Callable) -> float:
        total_counts = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            probability = count / total_counts
            cost = cost_function(bitstring)
            expectation += probability * cost
        
        return expectation

    def optimize(self, circuit: QuantumCircuit, cost_function: Callable,
                method: str = 'COBYLA', max_iter: int = 100) -> Dict:
        
        self.optimization_history = []
        self.best_cost = float('inf')
        self.best_params = None

        def objective_function(params):
            param_dict = dict(zip(self.qaoa_params, params))
            bound_circuit = circuit.assign_parameters(param_dict)
            
            job = self.backend.run(bound_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            expectation = self.compute_expectation(counts, cost_function)
            
            self.optimization_history.append({'cost': expectation})
            
            if expectation < self.best_cost:
                self.best_cost = expectation
                self.best_params = params.copy()
            
            # Print status to console (helpful for long runs)
            if len(self.optimization_history) % 10 == 0 or len(self.optimization_history) == 1:
                print(f"  > Iteration {len(self.optimization_history):<3}: Current Cost = {expectation:.4f}")
            
            return expectation
        
        initial_params = self._get_initial_params(strategy='informed')
        
        result = minimize(
            objective_function,
            initial_params,
            method=method,
            options={'maxiter': max_iter}
        )
        
        optimal_params_list = result.x.tolist() if isinstance(result.x, np.ndarray) else list(result.x)
        
        return {
            'success': result.success,
            'optimal_params': optimal_params_list,
            'optimal_cost': result.fun,
            'iterations': getattr(result, 'nfev', len(self.optimization_history)),
        }
    
    def _get_initial_params(self, strategy: str = 'informed') -> np.ndarray:
        if strategy == 'informed':
            betas = np.linspace(0, np.pi/4, self.p_layers)
            gammas = np.linspace(0, np.pi/2, self.p_layers)
            initial_params = np.empty(2 * self.p_layers)
            initial_params[0::2] = betas
            initial_params[1::2] = gammas
            return initial_params
        return np.random.uniform(0, 2*np.pi, 2 * self.p_layers)
    
    def get_solution_probabilities(self, circuit: QuantumCircuit, 
                                   params: np.ndarray, shots: int = 8192) -> Dict:
        param_dict = dict(zip(self.qaoa_params, params))
        bound_circuit = circuit.assign_parameters(param_dict)
        job = self.backend.run(bound_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        probabilities = {k: v/shots for k, v in counts.items()}
        return probabilities


# ==============================================================================
# 2. MAX-CUT SOLVER LOGIC (maxcut_solver.py adaptation)
# ==============================================================================

class MaxCutSolver:
    
    def __init__(self, graph: nx.Graph, p_layers: int = 1):
        self.graph = graph
        self.num_vertices = graph.number_of_nodes()
        self.p_layers = p_layers
        self.qaoa = QAOAOptimizer(self.num_vertices, p_layers)
        self.edges = list(graph.edges())
        self.edge_weights = nx.get_edge_attributes(graph, 'weight')
        for edge in self.edges:
            if edge not in self.edge_weights:
                self.edge_weights[edge] = 1.0

    def cost_hamiltonian(self, qc: QuantumCircuit, gamma: Parameter) -> None:
        """Applies RZZ gates for the Max-Cut cost term."""
        for edge in self.edges:
            i, j = edge
            weight = self.edge_weights.get(edge, 1.0)
            qc.rzz(2 * gamma * weight, i, j) 

    def compute_cut_value(self, bitstring: str) -> float:
        """Calculates the cut value for a given binary partition."""
        cut_value = 0.0
        partition = [int(b) for b in bitstring]
        for edge in self.edges:
            u, v = edge
            weight = self.edge_weights.get(edge, 1.0)
            if partition[u] != partition[v]:
                cut_value += weight
        return cut_value

    def cost_function(self, bitstring: str) -> float:
        """Cost function (to be minimized) is the negative cut value."""
        return -self.compute_cut_value(bitstring)

    def _compute_classical_bound(self) -> float:
        """Calculates the classical greedy bound for comparison."""
        partition = [0] * self.num_vertices
        for v in range(self.num_vertices):
            cut_0 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 1)
            cut_1 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 0)
            partition[v] = 1 if cut_1 > cut_0 else 0
        
        return self.compute_cut_value("".join(map(str, partition)))

    def _classical_warm_start_angles(self) -> List[float]:
        """Generates Ry angles from the greedy classical solution."""
        partition = [0] * self.num_vertices
        for v in range(self.num_vertices):
            cut_0 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 1)
            cut_1 = sum(self.edge_weights.get((v, u), self.edge_weights.get((u, v), 1.0)) 
                       for u in self.graph.neighbors(v) if partition[u] == 0)
            partition[v] = 1 if cut_1 > cut_0 else 0
            
        initial_angles = np.array(partition) * np.pi
        return initial_angles.tolist()

    def solve(self, initialization_strategy: str = 'standard', max_iter: int = 100) -> Dict:
        
        initial_angles = None
        if initialization_strategy == 'warm-start':
            initial_angles = self._classical_warm_start_angles()
            
        circuit = self.qaoa.create_qaoa_circuit(self.cost_hamiltonian, initial_angles=initial_angles)
        
        opt_result = self.qaoa.optimize(circuit, self.cost_function, max_iter=max_iter)
        
        probs = self.qaoa.get_solution_probabilities(circuit, opt_result['optimal_params'])
        
        # Get best partition from measurement results
        best_bitstring = max(probs.items(), key=lambda x: x[1])[0]
        best_cut_value = self.compute_cut_value(best_bitstring)
        
        classical_bound = self._compute_classical_bound()
        
        return {
            'strategy': initialization_strategy,
            'cut_value': best_cut_value,
            'approximation_ratio': best_cut_value / classical_bound if classical_bound > 0 else 0,
            'iterations': opt_result['iterations'],
            'optimal_cost': opt_result['optimal_cost'],
            'best_partition': best_bitstring,
        }

# ==============================================================================
# 3. MAIN EXECUTION (Demonstration)
# ==============================================================================

if __name__ == '__main__':
    # Define the problem graph from your paper (6-node graph)
    # Vertices 0-5
    # Edges: (0,1), (0,5), (1,2), (1,3), (2,4), (3,5), (4,5)
    
    N = 6
    TEST_EDGES = [(0, 1), (0, 5), (1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    MAX_ITER = 50 # Reduce iterations for fast console test
    P_LAYERS = 2 # Use p=2 layers for better approximation

    # 1. Create the NetworkX Graph
    G = nx.Graph()
    G.add_edges_from(TEST_EDGES)
    
    print("="*60)
    print(f"QAOA Console Verification (N={N} Nodes, P={P_LAYERS} Layers)")
    print(f"Max-Cut Edges: {TEST_EDGES}")
    print("="*60)

    # --- BENCHMARK 1: STANDARD QAOA ---
    solver_std = MaxCutSolver(G, p_layers=P_LAYERS)
    start_time = time.time()
    
    print("\n--- Running STANDARD QAOA (Hadamard Init) ---")
    result_std = solver_std.solve(initialization_strategy='standard', max_iter=MAX_ITER)
    time_std = time.time() - start_time

    # --- BENCHMARK 2: WARM-START QAOA ---
    solver_ws = MaxCutSolver(G, p_layers=P_LAYERS)
    start_time = time.time()
    
    print("\n--- Running WARM-START QAOA (Greedy Init) ---")
    result_ws = solver_ws.solve(initialization_strategy='warm-start', max_iter=MAX_ITER)
    time_ws = time.time() - start_time

    # --- FINAL COMPARISON REPORT ---
    print("\n\n" + "="*60)
    print("FINAL QAOA COMPARISON REPORT")
    print("="*60)
    
    classical_bound = solver_std._compute_classical_bound()
    print(f"Classical Greedy Bound: {classical_bound}")
    
    print("\n| Strategy        | Cut Value | AR (%) | Iterations | Time (s) |")
    print("|-----------------|-----------|--------|------------|----------|")
    
    print(f"| Standard QAOA   | {result_std['cut_value']:<9.2f} | {result_std['approximation_ratio']*100:<6.1f} | {result_std['iterations']:<10} | {time_std:<8.2f} |")
    print(f"| Warm-Start QAOA | {result_ws['cut_value']:<9.2f} | {result_ws['approximation_ratio']*100:<6.1f} | {result_ws['iterations']:<10} | {time_ws:<8.2f} |")
    print("-" * 60)
    
    if result_ws['approximation_ratio'] > result_std['approximation_ratio']:
        print(f"\n✅ WS-QAOA SUCCESS: Achieved a higher Approximation Ratio!")
    elif result_ws['iterations'] < result_std['iterations']:
        print(f"\n✅ WS-QAOA SUCCESS: Converged faster ({result_std['iterations']} vs {result_ws['iterations']} iterations).")
    else:
        print("\nNOTE: Results are stable but require further iteration/optimization for a clear advantage.")