from typing import Dict, List, TypedDict, Any
from langgraph.graph import StateGraph, END
import sys
import time

class State(TypedDict):
    """Simple state for workflow management"""
    messages: List[str]
    status: str
    complete: bool
    steps: List[str]

def print_with_flush(message: str):
    """Print message and flush output"""
    print(message)
    sys.stdout.flush()

def process_task(state: State) -> Dict[str, Any]:
    """Process a single task step"""
    # Update state
    messages = state["messages"].copy()
    steps = state["steps"].copy()
    step_number = len(steps)
    
    # Create new message
    new_message = f"Step {step_number}: Processing task..."
    messages.append(new_message)
    steps.append(new_message)
    
    # Print status
    print_with_flush(f"🔄 {new_message}")
    time.sleep(1)
    
    # Check completion
    is_complete = len(steps) >= 5
    if is_complete:
        print_with_flush("✅ Processing complete!")
    
    # Create updated state
    return {
        "messages": messages,
        "steps": steps,
        "status": "complete" if is_complete else "processing",
        "complete": is_complete
    }

def should_continue(state: State) -> str:
    """Determine if processing should continue"""
    return "end" if state["complete"] else "continue"

def create_workflow() -> Any:
    """Create the workflow graph"""
    workflow = StateGraph(State)
    workflow.add_node("process", process_task)
    workflow.set_entry_point("process")
    
    workflow.add_conditional_edges(
        "process",
        should_continue,
        {
            "continue": "process",
            "end": END
        }
    )
    
    return workflow.compile()

def print_summary(steps: List[str]) -> None:
    """Print workflow summary"""
    print_with_flush("\n" + "="*50)
    print_with_flush("📋 Workflow Summary:")
    print_with_flush("="*50)
    print_with_flush("\nSteps completed:")
    
    for idx, step in enumerate(steps, 1):
        print_with_flush(f"{idx}. {step}")

def run_workflow(description: str) -> None:
    """Run the workflow"""
    try:
        # Print header
        print_with_flush("\n" + "="*50)
        print_with_flush(f"🚀 Starting Workflow: {description}")
        print_with_flush("="*50 + "\n")
        
        # Initialize state
        initial_state = State(
            messages=[],
            steps=[],
            status="starting",
            complete=False
        )
        
        # Create workflow
        workflow = create_workflow()
        steps_completed = []
        
        # Process steps
        for current_state in workflow.stream(initial_state):
            if isinstance(current_state, dict):
                # Extract actual state from process wrapper
                state_data = current_state.get("process", {})
                if "steps" in state_data:
                    steps_completed = state_data["steps"]
        
        # Print summary
        if steps_completed:
            print_summary(steps_completed)
        else:
            print_with_flush("\n❌ No workflow steps recorded")
            
    except Exception as e:
        print_with_flush(f"\n❌ Error occurred: {str(e)}")
        raise
    finally:
        print_with_flush("\n" + "="*50)

def main():
    """Main entry point"""
    try:
        task = "How to make cheesecake"
        run_workflow(task)
    finally:
        print("Press Enter to exit...")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()