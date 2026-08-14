from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Base class for all agents with required abstract methods."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the base agent.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature

   
    @abstractmethod
    def build_memory_entry(
        self,
        task: str,
        action: str,
        observation: Dict[str, Any],
        reward: Optional[float] = None,
    ) -> str:
        """
        Build a memory entry from the agent's experience.
        
        This method MUST be implemented by each agent subclass.
        
        Args:
            task: The task that was being performed
            action: The action that was taken
            observation: The observation received
            reward: Optional reward received
            
        Returns:
            str: Formatted memory entry to store
        """
        pass
    @abstractmethod
    def act(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an action based on task, observation, and memory.
        
        This is a default implementation that can be overridden.
        
        Args:
        prompt: str which should includes
            task: The current task
            memory_context: memory context extracted from the memory system. (Your memory system should have all history with previous task, actions/trajectories, and observations.)
            
        Returns:
            str: The LLM agent output (action  to take)
        """
        raise NotImplementedError("Subclasses should implement act() to return an action.")

   