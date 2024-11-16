# magnetic_one_langgraph/agents/base_agent.py

import logging
from typing import Dict, Any
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
from datetime import datetime
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents in the workflow."""
    
    def __init__(self, settings: Settings, description: str):
        """Initialize the base agent.
        
        Args:
            settings: Application settings
            description: A description of what the agent does
        """
        self.settings = settings
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def process(self, state: AgentState) -> Dict[str, Any]:
        """Process the current state."""
        self.logger.info(f"Processing in {self.__class__.__name__}")
        return await self._process(state)
        
    def add_task_result(self, state: AgentState, task_name: str, result: Any) -> None:
        """Add a task result to the state's task ledger."""
        state.task_ledger[task_name] = result
        
    @abstractmethod
    async def _process(self, state: AgentState) -> Dict[str, Any]:
        """Process method to be implemented by child classes."""
        pass
