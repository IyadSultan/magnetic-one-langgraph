# magnetic_one_langgraph/core/state.py

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime

class AgentState(BaseModel):
    """State management for the agent system."""
    messages: List[str] = Field(default_factory=list)
    task_ledger: Dict[str, Any] = Field(default_factory=dict)
    task_plan: List[Union[str, Tuple[str, str]]] = Field(default_factory=list)
    counter: int = Field(default=0)
    final_report: Optional[str] = Field(default=None)
    task_complete: bool = Field(default=False)
    current_agent: str = Field(default="OrchestratorAgent")
    next_agent: Optional[str] = Field(default=None)
    workflow_complete: bool = Field(default=False)
    visited_agents: List[Dict[str, Any]] = Field(default_factory=list)

    def update_current_agent(self, agent_name: str) -> None:
        """Update current agent and track visit."""
        self.counter += 1
        self.current_agent = agent_name
        self.visited_agents.append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "task_count": len([k for k in self.task_ledger.keys() 
                             if k not in ['status', 'progress', 'workflow_start', 'workflow_end']])
        })

    def update_next_agent(self, agent_name: str) -> None:
        """Update next agent."""
        self.next_agent = agent_name

    def add_message(self, message: str) -> None:
        """Add a message to the state."""
        self.messages.append(message)

    def update_task_ledger(self, task: str, data: Any) -> None:
        """Update task ledger with new data."""
        self.task_ledger[task] = {
            "data": data,
            "status": "completed",
            "agent": self.current_agent,
            "timestamp": datetime.now().isoformat()
        }

    def calculate_progress(self) -> float:
        """Calculate current progress percentage."""
        completed_tasks = len([k for k in self.task_ledger.keys() 
                             if k not in ['status', 'progress', 'workflow_start', 'workflow_end']])
        return (completed_tasks / 4) * 100  # 4 total tasks

    def mark_complete(self) -> None:
        """Mark workflow as complete."""
        self.workflow_complete = True
        self.task_complete = True
        self.task_ledger["workflow_end"] = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
