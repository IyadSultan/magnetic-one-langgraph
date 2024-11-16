import os
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

load_dotenv()

# Initialize LLM with ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster/cheaper processing
    temperature=0.7  # Allow some creativity in responses
)

class AgentState(BaseModel):
    """State information for the agent system"""
    messages: List[str] = Field(default_factory=list)
    task_ledger: Dict[str, str] = Field(default_factory=dict)
    task_plan: List[Tuple[str, str]] = Field(default_factory=list)
    counter: int = Field(default=0)
    final_report: Optional[str] = Field(default=None)
    task_complete: bool = Field(default=False)
    current_agent: str = Field(default="Orchestrator")
    next_agent: str = Field(default="Orchestrator")
    llm_insights: Dict[str, Any] = Field(default_factory=dict)

def create_task_ledger() -> Dict[str, str]:
    """Create initial task ledger."""
    return {
        "task_id": "001",
        "known_facts": "Initial task facts",
        "guesses": "Potential unknown elements"
    }

def generate_task_plan() -> List[Tuple[str, str]]:
    """Generate initial task plan."""
    return [
        ("WebSurfer", "Perform web search"),
        ("FileSurfer", "Read from files"),
        ("Coder", "generate code to analyze the data"),
        ("ComputerTerminal", "Execute the code")
    ]

def get_llm_response(content: str, task_type: str) -> str:
    """Get LLM analysis for any task type."""
    prompts = {
        "planning": """
        Analyze this task and create a detailed plan:
        {content}
        
        Focus on:
        1. Key objectives
        2. Required steps
        3. Potential challenges
        4. Success criteria
        
        Provide a structured response with clear sections.
        """,
        
        "web_research": """
        Create a web research strategy for:
        {content}
        
        Include:
        1. Key areas to investigate
        2. Important data points to collect
        3. Specific trends to analyze
        4. Metrics to track
        
        Provide specific, actionable research points.
        """,
        
        "data_analysis": """
        Design a request for data analysis code for:
        {content}
        
        Include:
        1. data to analyze
        2. request for code to analyze the data
        3. visualizations to create
        4. insights to draw
        
        Provide concrete analysis steps and expectations.
        """,
        
        "coder": """You are a helpful AI assistant.
        Solve tasks using your coding and language skills.
        In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
            1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
            2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
        Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
        When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
        If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.

        Here is the task: {content}""",
        
        "implementation": """
        Develop an implementation strategy for:
        {content}
        
        Include:
        1. Key implementation steps
        2. Resource requirements
        3. Timeline considerations
        4. Success metrics
        
        Provide actionable implementation guidance.
        """,
        
        "summary": """
        Create a comprehensive summary of this task and findings:
        {content}
        
        Include:
        1. Key findings
        2. Important insights
        3. Recommendations
        4. Next steps
        
        Provide a clear, actionable summary.
        """
    }
    
    try:
        prompt = ChatPromptTemplate.from_template(prompts[task_type])
        chain = prompt | llm
        response = chain.invoke({"content": content})
        return response.content
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return f"Unable to generate {task_type} analysis. Using fallback process."

def orchestrator_agent(state: AgentState) -> Dict:
    """General-purpose orchestrator for any task."""
    try:
        print("\n🎯 Orchestrator Analysis")
        print("-" * 40)
        
        # Initialize if needed
        if not state.task_ledger:
            # Get task-specific planning from LLM
            planning_output = get_llm_response(state.messages[0], "planning")
            
            state.task_ledger = create_task_ledger()
            state.task_plan = generate_task_plan()
            state.messages.append(planning_output)
            print(planning_output)
        
        # Rest of the function remains the same
        if state.task_complete:
            next_agent = "FinalReview"
        elif state.task_plan:
            next_task = state.task_plan.pop(0)
            next_agent = next_task[0]
            state.counter = 0
        else:
            next_agent = "FinalReview"
        
        state.current_agent = next_agent
        state.next_agent = next_agent
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": next_agent
        }
    except Exception as e:
        print(f"Error in orchestrator_agent: {str(e)}")
        raise

def web_surfer_agent(state: AgentState) -> Dict:
    """Web research agent using Tavily search API."""
    try:
        print("\n🌐 Web Research Analysis")
        print("-" * 40)
        
        from tavily import TavilyClient
        
        # Initialize Tavily client with API key from environment
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Extract search query from first message
        search_query = state.messages[0]
        
        # Perform web search using Tavily
        search_result = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )
        
        # Format search results
        research_output = "Web Search Results:\n\n"
        for result in search_result['results']:
            research_output += f"Title: {result['title']}\n"
            research_output += f"Content: {result['content']}\n"
            research_output += f"URL: {result['url']}\n\n"
        
        state.messages.append("Web research completed via Tavily.")
        state.task_ledger["web_data"] = research_output
        state.next_agent = "Orchestrator"
        
        print(research_output)
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in web_surfer_agent: {str(e)}")
        raise

def file_surfer_agent(state: AgentState) -> Dict:
    """General-purpose data analysis agent."""
    try:
        print("\n📁 Data Analysis")
        print("-" * 40)
        
        # Get task-specific data analysis approach
        analysis_output = get_llm_response(state.messages[0], "data_analysis")
        
        state.messages.append("Data analysis completed.")
        state.task_ledger["file_data"] = analysis_output
        state.next_agent = "Orchestrator"
        
        print(analysis_output)
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in file_surfer_agent: {str(e)}")
        raise


def coder_agent(state: AgentState) -> Dict:
    """write python code to analyze the data"""
    def _extract_execution_request(markdown_text: str) -> Union[Tuple[str, str], None]:
        pattern = r"```(\w+)\n(.*?)\n```"
        # Search for the pattern in the markdown text
        match = re.search(pattern, markdown_text, re.DOTALL)
        # Extract the language and code block if a match is found
        if match:
            return (match.group(1), match.group(2))
        return None

    try:
        print("\n💻 Code generation")
        print("-" * 40)
        
        # Get task-specific technical analysis
        response = get_llm_response(state.messages[0], "coder")
        code = _extract_execution_request(response)
        
        if code is None:
            technical_output = "No code block found in response"
        else:
            technical_output = code[1]
        
        state.messages.append("Technical analysis completed.")
        state.task_ledger["code_analysis"] = technical_output
        state.next_agent = "Orchestrator"
        
        print(technical_output)
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in coder_agent: {str(e)}")
        raise

def computer_terminal_agent(state: AgentState) -> Dict:
    """General-purpose implementation agent."""
    def _sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]

    python_repl = PythonREPL()

    try:
        print("\n⚡ Implementation Analysis")
        print("-" * 40)
        
        # Get task-specific implementation strategy
        implementation_output = get_llm_response(state.messages[0], "implementation")
        print(state.task_ledger.get("code_analysis"))
        # Execute Python code if present
        code = state.task_ledger.get("code_analysis")
        if code:
            print(f"Executing code: {code}")
            try:
                execution_result = python_repl.run(code)
                implementation_output += f"\nCode Execution Result:\n{execution_result}"
            except Exception as e:
                implementation_output += f"\nCode Execution Error: {str(e)}"
        
        state.messages.append("Implementation analysis completed.")
        state.task_ledger["execution_result"] = implementation_output
        state.next_agent = "Orchestrator"
        
        print(implementation_output)
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in computer_terminal_agent: {str(e)}")
        raise

def finalize_task(state: AgentState) -> Dict:
    """General-purpose task finalizer."""
    try:
        print("\n📊 Final Analysis")
        print("-" * 40)
        
        # Combine all insights for final summary
        task_summary = f"""
Task Overview:
{state.messages[0]}

Analysis Results:
---------------
{state.task_ledger.get('web_data', '')}
{state.task_ledger.get('file_data', '')}
{state.task_ledger.get('code_analysis', '')}
{state.task_ledger.get('execution_result', '')}
"""
        
        # Get task-specific final summary
        final_output = get_llm_response(task_summary, "summary")
        
        state.final_report = final_output
        state.messages.append("Final analysis completed.")
        state.task_complete = True
        state.next_agent = END
        
        print(final_output)
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "next": END
        }
    except Exception as e:
        print(f"Error in finalize_task: {str(e)}")
        raise

def create_workflow_graph() -> StateGraph:
    """Create and configure the workflow graph."""
    try:
        print("DEBUG: Creating workflow graph")
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("Orchestrator", orchestrator_agent)
        workflow.add_node("WebSurfer", web_surfer_agent)
        workflow.add_node("FileSurfer", file_surfer_agent)
        workflow.add_node("Coder", coder_agent)
        workflow.add_node("ComputerTerminal", computer_terminal_agent)
        workflow.add_node("FinalReview", finalize_task)
        
        # Add conditional edges from Orchestrator
        workflow.add_conditional_edges(
            "Orchestrator",
            lambda x: x.next_agent,
            {
                "WebSurfer": "WebSurfer",
                "FileSurfer": "FileSurfer",
                "Coder": "Coder",
                "ComputerTerminal": "ComputerTerminal",
                "FinalReview": "FinalReview"
            }
        )
        
        # Add edges back to Orchestrator
        workflow.add_edge("WebSurfer", "Orchestrator")
        workflow.add_edge("FileSurfer", "Orchestrator")
        workflow.add_edge("Coder", "Orchestrator")
        workflow.add_edge("ComputerTerminal", "Orchestrator")
        
        # Add start edge
        workflow.add_edge(START, "Orchestrator")
        
        return workflow.compile()
    except Exception as e:
        print(f"DEBUG: Error in create_workflow_graph: {str(e)}")
        raise

def run_task_system(task_description: str):
    """Run the task system with enhanced output."""
    try:
        # Header
        print("\n" + "="*50)
        print("🚀 Starting Market Analysis System")
        print("="*50)
        
        print(f"\n📋 Task Description:")
        print("-" * 20)
        print(task_description)
        print("\n" + "="*50)
        
        # Initialize state
        initial_state = AgentState(
            messages=[f"Task started: {task_description}"]
        )
        
        # Create and run workflow
        workflow = create_workflow_graph()
        execution_log = []
        
        for step_dict in workflow.stream(initial_state):
            if "__end__" not in step_dict:
                current_agent = step_dict.get("current_agent", "Unknown")
                messages = step_dict.get("messages", [])
                
                if messages:
                    latest_message = messages[-1]
                    print(f"\n👉 {current_agent} Update:")
                    print("-" * 40)
                    print(latest_message)
                    
                    # Log execution
                    execution_log.append({
                        "agent": current_agent,
                        "action": latest_message,
                        "insights": step_dict.get("llm_insights", {})
                    })
                
                # Display final report
                if step_dict.get("final_report"):
                    print("\n" + "="*50)
                    print("📊 FINAL ANALYSIS REPORT")
                    print("="*50)
                    print(step_dict["final_report"])
                    
                    # Display execution summary
                    print("\n" + "="*50)
                    print("📈 Analysis Process Summary")
                    print("="*50)
                    for entry in execution_log:
                        print(f"\n{entry['agent']} Contribution:")
                        print("-" * 30)
                        print(f"Action: {entry['action']}")
                        if entry['insights']:
                            print("\nInsights Generated:")
                            for k, v in entry['insights'].items():
                                if isinstance(v, dict):
                                    print(f"  {k}:")
                                    for sub_k, sub_v in v.items():
                                        print(f"    - {sub_k}: {sub_v}")
                                elif isinstance(v, list):
                                    print(f"  {k}:")
                                    for item in v:
                                        print(f"    - {item}")
                                else:
                                    print(f"  - {k}: {v}")
                    break
                
    except Exception as e:
        print(f"Error in run_task_system: {str(e)}")
        raise

if __name__ == "__main__":
    task = """
    Analyze the current market trends in the AI industry with focus on:
    1. Latest developments in LLM technologies
    2. Major market players and their strategies
    3. Investment patterns and funding rounds
    4. Potential market opportunities and risks
    
    Provide specific insights and actionable recommendations.
    """
    run_task_system(task)