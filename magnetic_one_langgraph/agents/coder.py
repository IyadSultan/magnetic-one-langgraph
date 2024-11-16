# magnetic_one_langgraph/agents/coder.py

import logging
from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState

logger = logging.getLogger(__name__)

class CoderAgent(BaseAgent):
    """Agent responsible for code generation tasks."""
    
    def __init__(self, settings: Settings):
        """Initialize the coder agent.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings=settings, description="Code generation agent")
    
    async def _process(self, state: AgentState) -> Dict[str, Any]:
        """Process code generation task."""
        try:
            logger.info("Starting code generation")
            
            # Simulate code generation (replace with actual implementation)
            code_results = {
                "code": "print('Hello, World!')",
                "language": "python",
                "description": "Sample generated code",
                "dependencies": ["numpy", "pandas"],
                "instructions": [
                    "Install dependencies using pip",
                    "Run the script using Python 3.x"
                ]
            }
            
            # Add results to task ledger
            self.add_task_result(state, "code_generation", code_results)
            
            # Set next agent
            state.next_agent = "Orchestrator"
            
            logger.info("Code generation completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            raise
