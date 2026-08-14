from typing import Any, Callable, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from .env_systems import (
        WebShopEnvironment,
        BrowseCompPlusEnvironment,
        TravelPlannerEnvironment,
        MathEnvironment,
    )
except ImportError:
    from env_systems import (
        WebShopEnvironment,
        BrowseCompPlusEnvironment,
        TravelPlannerEnvironment,
        MathEnvironment,
    )

# ---------- Request schemas ----------
class InitializeRequest(BaseModel):
    task_id: str
    env_name: str
    env_config: Dict[str, Any] = {}


class ResetRequest(BaseModel):
    task_id: str
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: str
    action: Any
    ground_truth: Optional[Any] = None
    need_judge: bool = False

    class Config:
        extra = "allow"


class GetObservationRequest(BaseModel):
    task_id: str


class CloseRequest(BaseModel):
    task_id: str


# ---------- Environment factories ----------
ENV_FACTORIES: Dict[str, Callable] = {}
if WebShopEnvironment is not None:
    ENV_FACTORIES["webshop"] = WebShopEnvironment
if BrowseCompPlusEnvironment is not None:
    ENV_FACTORIES["browsecomp-plus"] = BrowseCompPlusEnvironment
if TravelPlannerEnvironment is not None:
    ENV_FACTORIES["travel_planner"] = TravelPlannerEnvironment
if MathEnvironment is not None:
    ENV_FACTORIES["math"] = MathEnvironment
    ENV_FACTORIES["phys"] = MathEnvironment #formal reasoning environment is shared between math and phys.
    ENV_FACTORIES["formal_reasoning_math"] = MathEnvironment
    ENV_FACTORIES["formal_reasoning_phys"] = MathEnvironment


# ---------- FastAPI app ----------
app = FastAPI(title="Environment Server")

# Store active environments
ENVIRONMENTS: Dict[str, Any] = {}


def _get_env(task_id: str):
    """Get environment for a task_id."""
    env = ENVIRONMENTS.get(task_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Environment not found for task_id: {task_id}")
    return env


@app.post("/env/initialize")
def initialize(req: InitializeRequest):
    """Initialize an environment for a task."""
    if req.task_id in ENVIRONMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Environment already exists for task_id: {req.task_id}"
        )
    
    factory = ENV_FACTORIES.get(req.env_name)
    if factory is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported environment: {req.env_name}. Available: {list(ENV_FACTORIES.keys())}"
        )
    
    # Create the environment
    env = factory(config=req.env_config)
    ENVIRONMENTS[req.task_id] = {
        "env": env,
        "env_name": req.env_name,
    }
    
    return {
        "status": "ok",
        "task_id": req.task_id,
        "env_name": req.env_name,
        "message": f"Environment '{req.env_name}' initialized for task '{req.task_id}'",
    }


@app.post("/env/reset")
def reset(req: ResetRequest):
    """Reset an environment and get initial observation."""
    env_data = _get_env(req.task_id)
    env = env_data["env"]
    
    observation = env.reset(seed=req.seed)
    
    return {
        "status": "ok",
        "task_id": req.task_id,
        "observation": observation,
    }


@app.post("/env/step")
def step(req: StepRequest):
    """Execute an action in the environment."""
    env_data = _get_env(req.task_id)
    env = env_data["env"]

    extra_kwargs = req.dict(
        exclude={"task_id", "action", "ground_truth", "need_judge"},
        exclude_none=True,
    )

    # Keep backward compatibility with older envs returning (observation, reward, info)
    # while allowing newer envs to return (observation, reward, done, info).
    result = env.step(
        req.action,
        ground_truth=req.ground_truth,
        need_judge=req.need_judge,
        **extra_kwargs,
    )

    if not isinstance(result, tuple):
        raise HTTPException(status_code=500, detail="Environment step() must return a tuple")
    if len(result) == 4:
        observation, reward, done, info = result
    elif len(result) == 3:
        observation, reward, info = result
        done = getattr(env, "_done", False)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Environment step() returned unsupported tuple length: {len(result)}",
        )
    if info is None:
        info = {}

    return {
        "status": "ok",
        "task_id": req.task_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/env/get_observation")
def get_observation(req: GetObservationRequest):
    """Get current observation without stepping."""
    env_data = _get_env(req.task_id)
    env = env_data["env"]
    
    observation = env.get_observation()
    
    return {
        "status": "ok",
        "task_id": req.task_id,
        "observation": observation,
    }


@app.post("/env/close")
def close(req: CloseRequest):
    """Close and cleanup an environment."""
    env_data = _get_env(req.task_id)
    env = env_data["env"]

    close_warning = None
    try:
        env.close()
    except Exception as exc:
        close_warning = f"{type(exc).__name__}: {exc}"
    finally:
        ENVIRONMENTS.pop(req.task_id, None)

    response = {
        "status": "ok",
        "task_id": req.task_id,
        "message": f"Environment closed for task '{req.task_id}'",
    }
    if close_warning is not None:
        response["warning"] = f"Environment cleanup raised but was ignored: {close_warning}"
    return response


@app.get("/env/list")
def list_environments():
    """List all active environments."""
    return {
        "status": "ok",
        "environments": [
            {"task_id": task_id, "env_name": data["env_name"]}
            for task_id, data in ENVIRONMENTS.items()
        ],
    }


@app.get("/env/available")
def available_environments():
    """List available environment types."""
    return {
        "status": "ok",
        "available_environments": list(ENV_FACTORIES.keys()),
    }


if __name__ == "__main__":
    # Run on a different port than memory server (8001 instead of 8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
