from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseEnvironment(ABC):
    """Base class for all environments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._current_observation = None
        self._done = False

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        pass

    @abstractmethod
    def step(
        self,
        action: Any,
        ground_truth: Any = None,
        need_judge: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action in the math environment."""
        """
        needs to return observation, reward, info
        reward is llm judge in each subquery (optional): return observation, None, info
        reward may be necessary in the final subtask: return observation, reward, info
        """
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation without stepping."""
        pass

    @abstractmethod
    def close(self):
        """Cleanup environment resources."""
        pass
