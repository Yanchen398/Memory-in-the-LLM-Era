import requests
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_session(session: Optional[requests.Session] = None) -> requests.Session:
    if session is not None:
        return session
    retry = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    new_session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    new_session.mount("http://", adapter)
    new_session.mount("https://", adapter)
    return new_session


class EnvironmentClient:
    """Client for interacting with hosted environments."""

    def __init__(
        self,
        task_id: str,
        env_name: str,
        base_url: str = "http://0.0.0.0:8005",
        session: Optional[requests.Session] = None,
        timeout: int = 300,
        env_config: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = str(task_id)
        self.env_name = env_name
        self.base_url = base_url.rstrip("/")
        self.session = _build_session(session)
        self.timeout = timeout
        
        # Initialize the environment
        self._post(
            "/env/initialize",
            {
                "task_id": self.task_id,
                "env_name": self.env_name,
                "env_config": env_config or {},
            },
        )

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and get initial observation."""
        data = self._post(
            "/env/reset",
            {
                "task_id": self.task_id,
                "seed": seed,
            },
        )
        return data["observation"]

    def step(
        self,
        action: Any,
        ground_truth: Any = None,
        need_judge: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute an action and get the result."""
        payload = {
            "task_id": self.task_id,
            "action": action,
            "need_judge": need_judge,
        }
        if ground_truth is not None:
            payload["ground_truth"] = ground_truth
        payload.update(kwargs)

        data = self._post("/env/step", payload)
        return {
            "observation": data["observation"],
            "reward": data.get("reward"),
            "done": data.get("done", False),
            "info": data.get("info") or {},
        }

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation without stepping."""
        data = self._post(
            "/env/get_observation",
            {"task_id": self.task_id},
        )
        return data["observation"]

    def close(self) -> Dict[str, Any]:
        """Close and cleanup the environment."""
        return self._post(
            "/env/close",
            {"task_id": self.task_id},
        )

    def _post(self, path: str, payload: dict) -> dict:
        response = self.session.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()