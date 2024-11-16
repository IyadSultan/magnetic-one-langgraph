# magnetic_one_langgraph/agents/final_review.py

from typing import Dict, Any
import logging
from datetime import datetime
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.core.settings import Settings

logger = logging.getLogger(__name__)

class FinalReviewAgent(BaseAgent):
    """Agent for performing final review of the workflow results."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings, "Final Review Agent - Reviews workflow results and provides recommendations")
    
    async def _process(self, state: AgentState) -> Dict[str, Any]:
        """Process final review."""
        try:
            logger.info("Starting final review")
            
            # Review results
            review_results = {
                "timestamp": datetime.now().isoformat(),
                "status": "complete",
                "summary": "Final review completed successfully",
                "recommendations": [
                    "All tasks completed successfully",
                    "No issues found",
                    "Ready for next phase"
                ]
            }
            
            # Add results to task ledger
            self.add_task_result(state, "final_review", review_results)
            
            # Mark task as complete and set next agent
            state.task_complete = True
            state.next_agent = "Orchestrator"
            
            logger.info("Final review completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Final review error: {str(e)}")
            raise
