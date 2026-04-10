from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone
import numpy as np
import networkx as nx

# Import QAOA solvers
from maxcut_solver import MaxCutSolver
from tsp_solver import TSPSolver
from vrp_solver import VRPSolver
from graph_coloring_solver import GraphColoringSolver

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="QAOA Optimization API", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Helper Functions ====================

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ==================== Models ====================

class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    problem_type: str
    problem_instance: Dict[str, Any]
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ExperimentCreate(BaseModel):
    problem_type: str
    problem_instance: Dict[str, Any]
    parameters: Dict[str, Any]
    results: Dict[str, Any]

# Problem-specific request models
class MaxCutRequest(BaseModel):
    num_vertices: int = Field(ge=2, le=10)
    edges: List[List[int]]
    edge_weights: Optional[Dict[str, float]] = None
    p_layers: int = Field(default=2, ge=1, le=5)
    max_iter: int = Field(default=100, ge=10, le=500)
    method: str = Field(default="COBYLA")
    initialization_strategy: str = Field(default="standard")


class TSPRequest(BaseModel):
    distance_matrix: List[List[float]]
    p_layers: int = Field(default=2, ge=1, le=5)
    max_iter: int = Field(default=50, ge=10, le=200)
    method: str = Field(default="COBYLA")
    
    def model_post_init(self, __context):
        # Enforce TSP limit: max 4 cities
        if len(self.distance_matrix) > 4:
            raise ValueError("TSP limited to maximum 4 cities for quantum simulation feasibility")

class VRPRequest(BaseModel):
    distance_matrix: List[List[float]]
    demands: List[float]
    vehicle_capacity: float
    num_vehicles: int = Field(ge=1, le=2)
    p_layers: int = Field(default=2, ge=1, le=5)
    max_iter: int = Field(default=50, ge=10, le=200)
    method: str = Field(default="COBYLA")
    
    def model_post_init(self, __context):
        # Enforce VRP limit: max 3 customers, 2 vehicles
        if len(self.demands) > 3:
            raise ValueError("VRP limited to maximum 3 customers for quantum simulation feasibility")

class GraphColoringRequest(BaseModel):
    num_vertices: int = Field(ge=2, le=4)
    edges: List[List[int]]
    num_colors: int = Field(ge=2, le=3)
    p_layers: int = Field(default=2, ge=1, le=5)
    max_iter: int = Field(default=100, ge=10, le=500)
    method: str = Field(default="COBYLA")


class RandomInstanceRequest(BaseModel):
    problem_type: str
    size: int = Field(ge=2, le=10)
    additional_params: Optional[Dict[str, Any]] = None

# ==================== Routes ====================

@api_router.get("/")
async def root():
    return {
        "message": "QAOA Optimization API",
        "version": "1.0.0",
        "problems": ["maxcut", "tsp", "vrp", "graph_coloring"],
        "documentation": "/docs"
    }

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "quantum_backend": "simulator"
        }
    }

# ==================== Max-Cut ====================

@api_router.post("/maxcut/solve")
async def solve_maxcut(request: MaxCutRequest):
    """
    Solve Max-Cut problem using QAOA
    """
    try:
        logger.info(f"Solving Max-Cut: {request.num_vertices} vertices. Strategy: {request.initialization_strategy}")
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(request.num_vertices))
        G.add_edges_from(request.edges)
        
        # Add edge weights
        if request.edge_weights:
            for edge_str, weight in request.edge_weights.items():
                u, v = map(int, edge_str.split('-'))
                G[u][v]['weight'] = weight
        
        # Solve
        solver = MaxCutSolver(G, request.p_layers)
        result = solver.solve(
            method=request.method, 
            max_iter=request.max_iter,
            # --- PASS NEW PARAMETER ---
            initialization_strategy=request.initialization_strategy 
        )
        
        # Convert to serializable format
        result = convert_to_serializable(result)
        
        # Store experiment
        experiment = ExperimentCreate(
            problem_type="maxcut",
            problem_instance={
                "num_vertices": request.num_vertices,
                "edges": request.edges,
                "edge_weights": request.edge_weights
            },
            parameters={
                "p_layers": request.p_layers, 
                "max_iter": request.max_iter, 
                "initialization_strategy": request.initialization_strategy
            },
            results=result
        )
        
        await store_experiment(experiment)
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving Max-Cut: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ==================== TSP ====================

@api_router.post("/tsp/solve")
async def solve_tsp(request: TSPRequest):
    """
    Solve Traveling Salesman Problem using QAOA
    """
    try:
        logger.info(f"Solving TSP: {len(request.distance_matrix)} cities")
        
        distance_matrix = np.array(request.distance_matrix)
        
        # Solve
        solver = TSPSolver(distance_matrix, request.p_layers)
        result = solver.solve(method=request.method, max_iter=request.max_iter)
        
        # Convert to serializable format
        result = convert_to_serializable(result)
        
        # Store experiment
        experiment = ExperimentCreate(
            problem_type="tsp",
            problem_instance={
                "num_cities": len(request.distance_matrix),
                "distance_matrix": request.distance_matrix
            },
            parameters={"p_layers": request.p_layers, "max_iter": request.max_iter},
            results=result
        )
        
        await store_experiment(experiment)
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving TSP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== VRP ====================

@api_router.post("/vrp/solve")
async def solve_vrp(request: VRPRequest):
    """
    Solve Vehicle Routing Problem using QAOA
    """
    try:
        logger.info(f"Solving VRP: {len(request.distance_matrix)-1} customers, {request.num_vehicles} vehicles")
        
        distance_matrix = np.array(request.distance_matrix)
        
        # Solve
        solver = VRPSolver(
            distance_matrix, 
            request.demands, 
            request.vehicle_capacity,
            request.num_vehicles,
            request.p_layers
        )
        result = solver.solve(method=request.method, max_iter=request.max_iter)
        
        # Convert to serializable format
        result = convert_to_serializable(result)
        
        # Store experiment
        experiment = ExperimentCreate(
            problem_type="vrp",
            problem_instance={
                "num_customers": len(request.demands),
                "num_vehicles": request.num_vehicles,
                "vehicle_capacity": request.vehicle_capacity
            },
            parameters={"p_layers": request.p_layers, "max_iter": request.max_iter},
            results=result
        )
        
        await store_experiment(experiment)
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving VRP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Graph Coloring ====================

@api_router.post("/graph-coloring/solve")
async def solve_graph_coloring(request: GraphColoringRequest):
    """
    Solve Graph Coloring problem using QAOA
    """
    try:
        logger.info(f"Solving Graph Coloring: {request.num_vertices} vertices, {request.num_colors} colors")
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(request.num_vertices))
        G.add_edges_from(request.edges)
        
        # Solve
        solver = GraphColoringSolver(G, request.num_colors, request.p_layers)
        result = solver.solve(method=request.method, max_iter=request.max_iter)
        
        # Convert to serializable format
        result = convert_to_serializable(result)
        
        # Store experiment
        experiment = ExperimentCreate(
            problem_type="graph_coloring",
            problem_instance={
                "num_vertices": request.num_vertices,
                "edges": request.edges,
                "num_colors": request.num_colors
            },
            parameters={"p_layers": request.p_layers, "max_iter": request.max_iter},
            results=result
        )
        
        await store_experiment(experiment)
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving Graph Coloring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Random Instance Generation ====================

@api_router.post("/generate/random")
async def generate_random_instance(request: RandomInstanceRequest):
    """
    Generate random problem instance
    """
    try:
        problem_type = request.problem_type.lower()
        size = request.size
        params = request.additional_params or {}
        
        if problem_type == "maxcut":
            edge_prob = params.get("edge_probability", 0.5)
            weighted = params.get("weighted", False)
            
            G = MaxCutSolver.generate_random_graph(size, edge_prob, weighted)
            
            return {
                "num_vertices": size,
                "edges": list(G.edges()),
                "edge_weights": {f"{u}-{v}": G[u][v]['weight'] 
                               for u, v in G.edges() 
                               if 'weight' in G[u][v]}
            }
            
        elif problem_type == "tsp":
            max_distance = params.get("max_distance", 100)
            distance_matrix = TSPSolver.generate_random_instance(size, max_distance)
            
            return {
                "num_cities": size,
                "distance_matrix": distance_matrix.tolist()
            }
            
        elif problem_type == "vrp":
            num_vehicles = params.get("num_vehicles", max(2, size // 3))
            capacity = params.get("capacity", 100.0)
            
            distance_matrix, demands = VRPSolver.generate_random_instance(
                size, num_vehicles, capacity
            )
            
            return {
                "num_customers": size,
                "num_vehicles": num_vehicles,
                "vehicle_capacity": capacity,
                "distance_matrix": distance_matrix.tolist(),
                "demands": demands
            }
            
        elif problem_type == "graph_coloring":
            edge_prob = params.get("edge_probability", 0.3)
            
            G = GraphColoringSolver.generate_random_graph(size, edge_prob)
            
            # Estimate chromatic number
            max_degree = max(dict(G.degree()).values()) if G.number_of_edges() > 0 else 1
            suggested_colors = min(max_degree + 1, 5)
            
            return {
                "num_vertices": size,
                "edges": list(G.edges()),
                "suggested_colors": suggested_colors,
                "num_edges": G.number_of_edges()
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown problem type: {problem_type}")
            
    except Exception as e:
        logger.error(f"Error generating random instance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Experiments ====================

async def store_experiment(experiment: ExperimentCreate):
    """Store experiment in database"""
    try:
        experiment_obj = ExperimentRecord(
            problem_type=experiment.problem_type,
            problem_instance=experiment.problem_instance,
            parameters=experiment.parameters,
            results=experiment.results
        )
        
        doc = experiment_obj.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        
        await db.experiments.insert_one(doc)
        logger.info(f"Stored experiment: {experiment_obj.id}")
        
    except Exception as e:
        logger.error(f"Error storing experiment: {str(e)}")

@api_router.get("/experiments", response_model=List[ExperimentRecord])
async def get_experiments(limit: int = 20, problem_type: Optional[str] = None):
    """
    Get experiment history
    """
    try:
        query = {}
        if problem_type:
            query["problem_type"] = problem_type
        
        experiments = await db.experiments.find(query, {"_id": 0}).sort(
            "timestamp", -1
        ).to_list(limit)
        
        # Convert timestamps
        for exp in experiments:
            if isinstance(exp['timestamp'], str):
                exp['timestamp'] = datetime.fromisoformat(exp['timestamp'])
        
        return experiments
        
    except Exception as e:
        logger.error(f"Error fetching experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """
    Get specific experiment
    """
    try:
        experiment = await db.experiments.find_one({"id": experiment_id}, {"_id": 0})
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if isinstance(experiment['timestamp'], str):
            experiment['timestamp'] = datetime.fromisoformat(experiment['timestamp'])
        
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """
    Delete experiment
    """
    try:
        result = await db.experiments.delete_one({"id": experiment_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {"message": "Experiment deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Statistics ====================

@api_router.get("/stats")
async def get_statistics():
    """
    Get overall statistics
    """
    try:
        total_experiments = await db.experiments.count_documents({})
        
        # Count by problem type
        pipeline = [
            {"$group": {"_id": "$problem_type", "count": {"$sum": 1}}}
        ]
        problem_counts = await db.experiments.aggregate(pipeline).to_list(None)
        
        return {
            "total_experiments": total_experiments,
            "by_problem_type": {item["_id"]: item["count"] for item in problem_counts},
            "available_problems": ["maxcut", "tsp", "vrp", "graph_coloring"]
        }
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
