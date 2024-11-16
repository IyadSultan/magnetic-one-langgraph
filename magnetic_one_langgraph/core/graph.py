# magnetic_one_langgraph/core/graph.py

import logging
from datetime import datetime
from typing import Dict, Any, Annotated, Union, Literal
from langgraph.graph import END, StateGraph
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.agents.orchestrator import OrchestratorAgent
from magnetic_one_langgraph.agents.web_surfer import WebSurferAgent
from magnetic_one_langgraph.agents.file_surfer import FileSurferAgent
from magnetic_one_langgraph.agents.coder import CoderAgent
from magnetic_one_langgraph.agents.terminal import TerminalAgent
from magnetic_one_langgraph.agents.final_review import FinalReviewAgent
from magnetic_one_langgraph.config.settings import Settings

logger = logging.getLogger(__name__)

async def handle_final_review(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final report."""
    try:
        if isinstance(state, dict):
            state = AgentState(**state)
        
        logger.info("Generating final report")
        
        # Get completed tasks
        completed_tasks = [k for k in state.task_ledger.keys() 
                         if k not in ['status', 'progress', 'workflow_start', 'workflow_end', 'workflow_summary']]
                         
        # Create task details section
        task_details = []
        for task in completed_tasks:
            task_details.append(f"- {task}:")
            task_details.append(f"  {str(state.task_ledger[task].get('data', 'No data'))}")
        
        # Create agent activity section
        agent_visits = []
        for visit in state.visited_agents:
            agent_visits.append(f"- {visit['agent']} at {visit['timestamp']}")
            
        # Create final report
        sections = [
            "Workflow Summary",
            "---------------",
            f"Initial Task: {state.messages[0] if state.messages else 'No initial task'}",
            "",
            f"Completed Tasks ({len(completed_tasks)}/4):",
            *[f"- {task}: {state.task_ledger[task]['status']}" for task in completed_tasks],
            "",
            "Task Details:",
            *task_details,
            "",
            "Agent Activity:",
            *agent_visits,
            "",
            "Progress Messages:",
            *[f"- {msg}" for msg in state.messages],
            "",
            "Workflow Status:",
            f"- Progress: {state.task_ledger.get('progress', 0)}%",
            f"- Status: {state.task_ledger.get('status', 'unknown')}",
            f"- Start Time: {state.task_ledger.get('workflow_start', {}).get('timestamp', 'unknown')}",
            f"- End Time: {datetime.now().isoformat()}"
        ]
        
        final_report = "\n".join(sections)
        
        # Update state directly
        state.final_report = final_report
        state.current_agent = "FinalReview"
        state.workflow_complete = True
        state.task_complete = True
        
        # Prepare result for return
        result = state.model_dump()
        result["final"] = True
        
        logger.info("Final report generated")
        return result
        
    except Exception as e:
        logger.error(f"Error in final review: {str(e)}")
        raise

def create_workflow_graph(settings: Settings) -> StateGraph:
    """Create and configure the workflow graph"""
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Initialize all agents with settings
    agents = {
        "Orchestrator": OrchestratorAgent(settings),
        "WebSurfer": WebSurferAgent(settings),
        "FileSurfer": FileSurferAgent(settings),
        "Coder": CoderAgent(settings),
        "Terminal": TerminalAgent(settings),
        "FinalReview": FinalReviewAgent(settings)
    }
    
    # Add all nodes
    for name, agent in agents.items():
        workflow.add_node(name, agent.process)
    
    # Define conditional edges from Orchestrator
    def get_next_step(state: Union[Dict[str, Any], AgentState]) -> Dict[str, float]:
        """Get probabilities for next steps."""
        if isinstance(state, dict):
            state = AgentState(**state)
            
        # Initialize all probabilities to 0
        probabilities = {
            "WebSurfer": 0.0,
            "FileSurfer": 0.0,
            "Coder": 0.0,
            "Terminal": 0.0,
            "FinalReview": 0.0
        }
        
        # Set probability for next step
        if state.task_complete:
            probabilities["FinalReview"] = 1.0
        elif state.next_agent:
            probabilities[state.next_agent] = 1.0
        else:
            probabilities["WebSurfer"] = 1.0
            
        return probabilities
    
    # Add conditional edges from Orchestrator
    workflow.add_conditional_edges(
        "Orchestrator",
        get_next_step,
        {
            "WebSurfer": "WebSurfer",
            "FileSurfer": "FileSurfer",
            "Coder": "Coder",
            "Terminal": "Terminal",
            "FinalReview": "FinalReview"
        }
    )
    
    # Add edges from agents back to Orchestrator
    for name in ["WebSurfer", "FileSurfer", "Coder", "Terminal"]:
        workflow.add_edge(name, "Orchestrator")
    
    # Add final edge
    workflow.add_edge("FinalReview", END)
    
    # Set entry point
    workflow.set_entry_point("Orchestrator")
    
    return workflow.compile()