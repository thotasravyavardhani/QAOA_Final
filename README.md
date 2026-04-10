# QAOA Optimization Platform 🚀

## Quantum Approximate Optimization Algorithm for Solving NP-Hard Problems

### 📚 Project Overview

This is a comprehensive **full-stack research platform** implementing the **Quantum Approximate Optimization Algorithm (QAOA)** for solving combinatorial optimization problems. Designed for academic research, Scopus publication, and final year projects.

---

## 🎯 Key Features

### ✅ Implemented Problems
1. **Max-Cut** - Graph partitioning to maximize edges between sets (✅ Full Implementation)
2. **Traveling Salesman Problem (TSP)** - Finding shortest routes (Backend Ready)
3. **Vehicle Routing Problem (VRP)** - Multi-vehicle route optimization (Backend Ready)
4. **Graph Coloring** - Vertex coloring with constraints (Backend Ready)

### 🔬 Research Features
- **Multi-layer QAOA** (p = 1 to 5 layers)
- **Parameter optimization** with classical feedback (COBYLA, SLSQP, Nelder-Mead)
- **Performance analysis** and convergence metrics
- **Quantum-classical comparison** benchmarks
- **Scalability testing** (2-20 qubits)
- **Experiment tracking** and history management

### 💻 Technical Stack
- **Quantum Framework**: Qiskit 1.3.1 (IBM)
- **Backend**: FastAPI (Python 3.11)
- **Frontend**: React 19 with Tailwind CSS
- **Database**: MongoDB
- **Scientific Computing**: NumPy, SciPy, NetworkX, Matplotlib

---

## 🚀 Quick Start

### Services Running
- **Frontend**: Port 3000 (Auto-reloading)
- **Backend**: Port 8001 (Auto-reloading)
- **API Docs**: http://localhost:8001/docs

### Test the API
```bash
curl -X POST http://localhost:8001/api/maxcut/solve \
  -H "Content-Type: application/json" \
  -d '{
    "num_vertices": 4,
    "edges": [[0,1], [1,2], [2,3], [0,3]],
    "p_layers": 2,
    "max_iter": 50
  }'
```

---

## 📖 Mathematical Foundation

### QAOA Circuit Structure
```
|\u03c8(\u03b2, \u03b3)\u27e9 = U(B, \u03b2_p)U(C, \u03b3_p)...U(B, \u03b2_1)U(C, \u03b3_1)|s\u27e9
```

**Components:**
- `U(C, \u03b3) = e^(-i\u03b3C)` - Cost unitary
- `U(B, \u03b2) = e^(-i\u03b2B)` - Mixer unitary
- `p` - Number of QAOA layers (circuit depth)

---

## 📊 API Endpoints

### Main Solvers
- `POST /api/maxcut/solve` - Solve Max-Cut problem
- `POST /api/tsp/solve` - Solve TSP
- `POST /api/vrp/solve` - Solve VRP
- `POST /api/graph-coloring/solve` - Solve Graph Coloring

### Utilities
- `POST /api/generate/random` - Generate random problem instances
- `GET /api/experiments` - Get experiment history
- `GET /api/stats` - Get platform statistics
- `GET /api/health` - Health check

---

## 🎓 For Academic/Research Use

### Key Contributions
1. **Complete QAOA Implementation** for 4 NP-hard problems
2. **Comparative Analysis** with classical algorithms
3. **Scalability Study** across different problem sizes
4. **Parameter Optimization** strategies
5. **Convergence Analysis** and performance metrics

### Documentation Included
- ✅ Mathematical formulations
- ✅ Algorithm implementations
- ✅ Performance benchmarks
- ✅ API documentation
- ✅ Research references
- ✅ Usage examples

---

## 📚 Research References

1. **Farhi et al.** (2014) - "A Quantum Approximate Optimization Algorithm"
2. **Hadfield et al.** (2019) - "QAOA to Quantum Alternating Operator Ansatz"
3. **Crooks** (2018) - "Performance on Maximum Cut Problem"

---

## 👥 Contributors

**Vignan's Nirula Institute of Technology and Science for Women**
- Kilaru Srikanth
- Thota Sravya Vardhani
- Attluri Udaya Lakshmi
- Kota Mounika
- Thotakura Gnana Prasuna

---

**Built with ❤️ for Quantum Computing Research**
