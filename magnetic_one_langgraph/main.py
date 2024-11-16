# magnetic_one_langgraph/main.py

import asyncio
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.core.graph import create_workflow_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_agent_state(step: dict) -> Optional[AgentState]:
    """Convert step to AgentState if possible."""
    try:
        if not isinstance(step, dict):
            return None
        return AgentState(**step)
    except Exception as e:
        logger.error(f"Error converting step to AgentState: {e}")
        return None

def extract_tasks(state: AgentState) -> list:
    """Extract completed tasks from state."""
    return [k for k in state.task_ledger.keys() 
            if k not in ['status', 'progress', 'workflow_start', 'workflow_end', 'workflow_summary']]

async def main():
    """Main entry point."""
    try:
        # Load environment variables
        load_dotenv()

        # Initialize settings
        settings = Settings()

        # Create workflow graph
        workflow = create_workflow_graph(settings)

        # Example task
        task = """
        Analyze the current market trends in the AI industry with focus on:
        1. Latest developments in LLM technologies
        2. Major market players and their strategies
        3. Investment patterns and funding rounds
        4. Potential market opportunities and risks
        """

        # Initialize state
        initial_state = AgentState(
            messages=[task],
            task_ledger={
                "status": "initialized",
                "progress": 0,
                "workflow_start": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "started"
                }
            },
            task_plan=[],
            counter=0,
            current_agent="OrchestratorAgent",
            visited_agents=[]
        )

        # Process workflow
        step_counter = 0
        final_state = None
        
        logger.info("\nStarting workflow execution...")
        logger.info("=" * 50)

        # Stream workflow execution
        async for step in workflow.astream(initial_state.model_dump()):
            step_counter += 1
            
            # Convert step to state
            current_state = get_agent_state(step)
            if not current_state:
                logger.warning(f"Unable to process step {step_counter}")
                continue
                
            # Log progress
            logger.info(f"\nStep {step_counter}:")
            logger.info(f"Current Agent: {current_state.current_agent}")
            logger.info(f"Completed Tasks: {extract_tasks(current_state)}")
            logger.info(f"Progress: {current_state.task_ledger.get('progress', 0)}%")
            
            # Track messages
            if current_state.messages:
                logger.info(f"Latest Message: {current_state.messages[-1]}")
            
            # Check for completion
            if step.get("final", False) and current_state.final_report:
                final_state = current_state
                break
                
            # Safety check
            if step_counter >= 50:
                logger.warning("Maximum steps reached!")
                break

        # Handle completion
        logger.info("\nWorkflow execution completed")
        logger.info("=" * 50)
        
        if final_state and final_state.final_report:
            logger.info("\nFinal Report:")
            logger.info("=" * 80)
            logger.info(final_state.final_report)
            logger.info("=" * 80)
            logger.info("\nWorkflow completed successfully!")
        else:
            logger.warning("\nWorkflow completed without final report!")
            if final_state:
                logger.info("Final state:")
                logger.info("-" * 40)
                logger.info(f"Agent: {final_state.current_agent}")
                logger.info(f"Tasks: {extract_tasks(final_state)}")
                logger.info(f"Progress: {final_state.task_ledger.get('progress', 0)}%")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
