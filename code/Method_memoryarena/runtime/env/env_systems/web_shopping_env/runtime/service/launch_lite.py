"""
Lightweight launcher for Combo WebShop tasks.

This launcher creates a minimal environment with only a few products,
avoiding the heavy load time of the full WebShop dataset.
"""

import argparse
import os
import logging
import time
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel


class WebshopEnvServerLite:
    """
    Lightweight WebShop environment server for combo tasks.

    Key differences from the full server:
    - Loads ALL products (1.1M) for complete search coverage
    - Does NOT load items_ins_v2.json (avoiding 11M human goals)
    - Uses limited synthetic goals (1000) generated from products
    - Faster startup time (~2-3 minutes vs 10+ minutes)
    - Much lower memory usage (~5GB vs 15GB)
    """

    def __init__(self, limit_goals: int = 1000, share_catalog: bool = False) -> None:
        self._max_id = 0
        self.env = {}
        self.ls = []
        self.sz = 8000
        self.now = -1
        self.limit_goals = limit_goals
        self.share_catalog = share_catalog
        self._shared_server = None
        print(f"[Lite Server] Initializing:")
        print(f"  - Products: ALL (1,181,436)")
        print(f"  - Goals: Limited to {limit_goals} (synthetic, no items_ins_v2.json)")
        print(f"  - Expected time: 2-3 minutes...")

    def create(self) -> int:
        env_idx = self._max_id
        import random
        import time

        random.seed(time.time())
        idx = random.randint(0, 48950076)
        print(f"-------Env {idx} created (Lite mode)--------")

        if len(self.env) == self.sz:
            self.now = self.now + 1
            if self.now == self.sz:
                self.now = 0
            return self.ls[self.now]

        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": 0,
            "limit_goals": self.limit_goals,
            "session_prefix": f"env{env_idx}_",
        }
        items_file = os.getenv("MEMORYARENA_WEBSHOP_ITEMS_FILE")
        if items_file:
            env_kwargs["file_path"] = items_file
        from .web_agent_site.envs import WebAgentTextEnv

        if self.share_catalog and self._shared_server is not None:
            env_kwargs["server"] = self._shared_server
        self.env[idx] = WebAgentTextEnv(**env_kwargs)
        if self.share_catalog and self._shared_server is None:
            self._shared_server = self.env[idx].server
        self.env[idx].reset()
        self._max_id += 1
        self.ls.append(idx)
        return idx

    def step(self, env_idx, action: str):
        return self.env[env_idx].step(action)

    def get_available_actions(self, env_idx):
        return self.env[env_idx].get_available_actions()

    def get_image(self, env_idx):
        return self.env[env_idx].get_image()

    def get_instruction_text(self, env_idx):
        return self.env[env_idx].get_instruction_text()

    def observation(self, env_idx):
        return self.env[env_idx].observation

    def state(self, env_idx):
        return self.env[env_idx].state

    def reset(self, env_idx, session_id):
        return self.env[env_idx].reset(session=session_id)

    def close(self, env_idx):
        env = self.env.pop(env_idx, None)
        if env is None:
            return False
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            if env_idx in self.ls:
                self.ls.remove(env_idx)
        return True

    def __del__(self):
        for idx in self.ls:
            env = self.env.get(idx)
            close_fn = getattr(env, "close", None) if env is not None else None
            if callable(close_fn):
                close_fn()
            print(f"-------Env {idx} closed--------")


class StepQuery(BaseModel):
    env_idx: int
    action: str


class ResetQuery(BaseModel):
    env_idx: int
    session_id: Optional[int] = None


class CloseQuery(BaseModel):
    env_idx: int


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: Optional[dict] = None


class AvailableActionsResponse(BaseModel):
    has_search_bar: bool
    clickables: List[str]


class StateResponse(BaseModel):
    url: str
    html: str
    instruction_text: str


webshop_env_server_lite = WebshopEnvServerLite(limit_goals=1000)
app = FastAPI(debug=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.2f} seconds"
    )
    return response


@app.get("/", response_model=str)
async def generate_ok():
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    return list(webshop_env_server_lite.env.keys())


@app.get("/catalog_status")
async def catalog_status():
    server_ids = {
        id(env.server)
        for env in webshop_env_server_lite.env.values()
    }
    return {
        "share_catalog": webshop_env_server_lite.share_catalog,
        "env_count": len(webshop_env_server_lite.env),
        "server_object_count": len(server_ids),
        "catalog_loaded": webshop_env_server_lite._shared_server is not None,
    }


@app.get("/environment_status")
async def environment_status():
    return {
        str(env_id): {
            "session": env.session,
            "url": env.browser.current_url,
        }
        for env_id, env in webshop_env_server_lite.env.items()
    }


@app.post("/create", response_model=int)
async def create():
    return webshop_env_server_lite.create()


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    state, reward, done, info = webshop_env_server_lite.step(
        step_query.env_idx,
        step_query.action,
    )
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/available_actions", response_model=AvailableActionsResponse)
def get_available_actions(env_idx: int):
    res = webshop_env_server_lite.get_available_actions(env_idx)
    return AvailableActionsResponse(
        has_search_bar=res["has_search_bar"],
        clickables=res["clickables"],
    )


@app.get("/instruction_text", response_model=str)
def get_instruction_text(env_idx: int):
    return webshop_env_server_lite.get_instruction_text(env_idx)


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return webshop_env_server_lite.observation(env_idx)


@app.get("/state", response_model=StateResponse)
def get_state(env_idx: int):
    url, html, instruction_text = webshop_env_server_lite.state(env_idx)
    return StateResponse(url=url, html=html, instruction_text=instruction_text)


@app.post("/reset", response_model=Tuple[str, None])
def reset(reset_query: ResetQuery):
    return webshop_env_server_lite.reset(reset_query.env_idx, reset_query.session_id)


@app.post("/close")
def close(close_query: CloseQuery):
    return {"closed": webshop_env_server_lite.close(close_query.env_idx)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch WebShop Lite Server")
    parser.add_argument("--port", type=int, default=36001, help="Port to run server on")
    parser.add_argument(
        "--limit_goals",
        type=int,
        default=1000,
        help="Number of goals to generate (default: 1000, vs 11M in full server)",
    )
    parser.add_argument(
        "--webshop-data-root",
        type=str,
        default=None,
        help="External data root containing WebShop data files, e.g. data/shopping.",
    )
    parser.add_argument(
        "--share-catalog",
        action="store_true",
        help=(
            "Share immutable product, goal, and search-index objects across "
            "independent environment instances. Session state remains isolated."
        ),
    )
    args = parser.parse_args()

    if args.webshop_data_root:
        os.environ["MEMORYARENA_WEBSHOP_DATA_ROOT"] = args.webshop_data_root

    webshop_env_server_lite.limit_goals = args.limit_goals
    webshop_env_server_lite.share_catalog = args.share_catalog

    print("=" * 60)
    print("WebShop Lite Server for Combo Tasks")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Products: ALL (1,181,436) - full search coverage")
    print(f"Goals: {args.limit_goals} (synthetic, no items_ins_v2.json)")
    print(f"Shared immutable catalog: {args.share_catalog}")
    if args.webshop_data_root:
        print(f"External data root: {args.webshop_data_root}")
    print("Memory saved: ~10GB (avoiding items_ins_v2.json)")
    print("Expected startup time: 2-3 minutes (vs 10+ minutes)")
    print("=" * 60)
    print("\nStarting server...")
    print("Note: First environment creation will load all products (~2 min)")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
