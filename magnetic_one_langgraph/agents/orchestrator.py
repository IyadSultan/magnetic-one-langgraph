# magnetic_one_langgraph/agents/orchestrator.py

import logging
from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.config.settings import Settings

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the workflow."""
    
    def __init__(self, settings: Settings):
        """Initialize the orchestrator agent.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings=settings, description="Task orchestration agent")
        self.task_sequence = [
            ("web_research", "WebSurferAgent"),
            ("file_analysis", "FileSurferAgent"),
            ("code_generation", "CoderAgent"),
            ("code_execution", "TerminalAgent"),
            ("final_review", "FinalReviewAgent")
        ]
    
    async def _process(self, state: AgentState) -> Dict[str, Any]:
        """Process the current state and determine next steps."""
        try:
            if state.task_complete:
                return state
                
            # Update progress based on completed tasks
            completed_tasks = state.task_ledger.keys()
            total_tasks = len(self.task_sequence)
            progress = (len(completed_tasks) / total_tasks) * 100
            
            logger.info(f"Progress: {progress}%")
            logger.info(f"Completed tasks: {list(completed_tasks)}")
            
            # Find next task
            for task_name, agent_name in self.task_sequence:
                if task_name not in completed_tasks:
                    state.next_agent = agent_name
                    logger.info(f"Next task: {task_name} -> {agent_name}")
                    break
            else:
                state.task_complete = True
                
            return state
                
        except Exception as e:
            logger.error(f"Orchestrator processing error: {str(e)}")
            raise
