# Combined Python and HTML files
# Generated from directory: C:\Users\isult\OneDrive\Documents\magnetic-one-langgraph\magnetic_one_langgraph
# Total files found: 24



# Contents from: .\__init__.py


# Contents from: .\agents\__init__.py


# Contents from: .\agents\base_agent.py
# agents/base_agent.py

import logging
from typing import Dict, Any
from magnetic_one_langgraph.core.state import AgentState
from datetime import datetime

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, description: str):
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state and return updated state."""
        try:
            # Convert dict to AgentState if needed
            if isinstance(state, dict):
                state = AgentState(**state)
                
            # Update agent info
            agent_name = self.__class__.__name__
            state.update_current_agent(agent_name)
            self.logger.info(f"Processing in {agent_name}")
            
            # Process state
            state = await self._process(state)
            
            # Add completion message
            state.add_message(f"{agent_name} processing complete")
            
            # Convert back to dict
            result = state.model_dump()
            
            # Add final flag if workflow is complete
            if state.workflow_complete:
                result["final"] = True
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            raise
            
    async def _process(self, state: AgentState) -> AgentState:
        """Main processing logic - must be implemented by subclasses."""
        raise NotImplementedError
        
    def add_task_result(self, state: AgentState, task_name: str, result: Any) -> None:
        """Add task result to state ledger."""
        state.update_task_ledger(task_name, result)
        state.add_message(f"{task_name} completed successfully")
        self.logger.info(f"Added result for {task_name}")

# Contents from: .\agents\coder.py
# agents/coder.py

from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
import logging

logger = logging.getLogger(__name__)

class CoderAgent(BaseAgent):
    """Agent for code generation."""
    
    def __init__(self, settings: Settings):
        super().__init__(description="Code generation agent")
        self.settings = settings
        
    async def _process(self, state: AgentState) -> AgentState:
        """Process code generation task."""
        try:
            logger.info("Starting code generation")
            
            # Simulate code generation
            generated_code = {
                "code": """
def analyze_data():
    print("Analyzing data...")
    return "Analysis complete"
""",
                "language": "python",
                "purpose": "Data analysis function"
            }
            
            # Add results to task ledger
            self.add_task_result(state, "code_generation", generated_code)
            state.next_agent("OrchestratorAgent")
            
            logger.info("Code generation completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            raise

# Contents from: .\agents\file_surfer.py
# agents/file_surfer.py

from typing import Dict, Any, List
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FileSurferAgent(BaseAgent):
    """Agent for file analysis."""
    
    def __init__(self, settings: Settings):
        super().__init__(description="File analysis agent")
        self.settings = settings
        
    async def _process(self, state: AgentState) -> AgentState:
        """Process file analysis task."""
        try:
            logger.info("Starting file analysis")
            
            # Simulate file analysis (replace with actual implementation)
            analysis_results = {
                "files_analyzed": ["sample1.txt", "sample2.txt"],
                "summary": "Sample file analysis results",
                "findings": [
                    {"file": "sample1.txt", "content": "Content analysis 1"},
                    {"file": "sample2.txt", "content": "Content analysis 2"}
                ]
            }
            
            # Add results to task ledger
            self.add_task_result(state, "file_analysis", analysis_results)
            state.next_agent("OrchestratorAgent")
            
            logger.info("File analysis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"File analysis error: {str(e)}")
            raise

# Contents from: .\agents\orchestrator.py
# agents/orchestrator.py

from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
import logging

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent for coordinating tasks."""

    def __init__(self, settings: Settings):
        super().__init__(description="Task orchestration agent")
        self.settings = settings
        self._tasks = [
            ("web_research", "WebSurferAgent"),
            ("file_analysis", "FileSurferAgent"),
            ("code_generation", "CoderAgent"),
            ("code_execution", "TerminalAgent")
        ]

    async def _process(self, state: AgentState) -> AgentState:
        """Process the current state and determine next steps."""
        try:
            # Calculate progress
            completed_tasks = [task for task, _ in self._tasks if task in state.task_ledger]
            progress = (len(completed_tasks) / len(self._tasks)) * 100
            
            # Update progress in ledger
            state.task_ledger['progress'] = progress
            logger.info(f"Progress: {progress}%")
            logger.info(f"Completed tasks: {completed_tasks}")

            # Find next incomplete task
            for task_name, agent_name in self._tasks:
                if task_name not in state.task_ledger:
                    logger.info(f"Next task: {task_name} -> {agent_name}")
                    state.update_next_agent(agent_name)
                    return state

            # All tasks complete
            logger.info("All tasks complete")
            state.workflow_complete = True
            state.task_complete = True
            state.task_ledger['status'] = 'completed'

            # Add workflow summary
            self.add_task_result(state, "workflow_summary", {
                "completed_tasks": completed_tasks,
                "total_tasks": len(self._tasks),
                "progress": progress,
                "status": "completed"
            })

            return state

        except Exception as e:
            logger.error(f"Error in orchestrator: {str(e)}")
            raise

    def get_current_progress(self, state: AgentState) -> float:
        """Calculate current progress."""
        completed = len([task for task, _ in self._tasks if task in state.task_ledger])
        return (completed / len(self._tasks)) * 100

# Contents from: .\agents\terminal.py
# agents/terminal.py

from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
import logging

logger = logging.getLogger(__name__)

class TerminalAgent(BaseAgent):
    """Agent for code execution."""
    
    def __init__(self, settings: Settings):
        super().__init__(description="Code execution agent")
        self.settings = settings
        
    async def _process(self, state: AgentState) -> AgentState:
        """Process code execution task."""
        try:
            logger.info("Starting code execution")
            
            # Simulate code execution
            execution_result = {
                "status": "success",
                "output": "Analysis complete\n",
                "execution_time": "0.1s"
            }
            
            # Add results to task ledger
            self.add_task_result(state, "code_execution", execution_result)
            state.next_agent("OrchestratorAgent")
            
            logger.info("Code execution completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            raise

# Contents from: .\agents\web_surfer.py
# agents/web_surfer.py

from typing import Dict, Any
from magnetic_one_langgraph.agents.base_agent import BaseAgent
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
import logging

logger = logging.getLogger(__name__)

class WebSurferAgent(BaseAgent):
    """Agent for web research."""
    
    def __init__(self, settings: Settings):
        super().__init__(description="Web research agent")
        self.settings = settings
        
    async def _process(self, state: AgentState) -> AgentState:
        """Process web research task."""
        try:
            query = state.messages[0]
            logger.info(f"Processing search query: {query}")
            
            # Simulate search results
            results = {
                "query": query,
                "data": [
                    {
                        "title": "AI Market Analysis 2024",
                        "summary": "Comprehensive analysis of AI market trends"
                    },
                    {
                        "title": "LLM Technology Overview",
                        "summary": "Latest developments in language models"
                    }
                ]
            }
            
            # Add results to task ledger
            self.add_task_result(state, "web_research", results)
            logger.info("Web research completed successfully")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in web research: {str(e)}")
            raise

# Contents from: .\combine.py
import os

def get_files_recursively(directory, extensions):
    """
    Recursively get all files with specified extensions from directory and subdirectories
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    return file_list

def combine_files(output_file, file_list):
    """
    Combine contents of all files in file_list into output_file
    """
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for fname in file_list:
            # Add a header comment to show which file's contents follow
            outfile.write(f"\n\n# Contents from: {fname}\n")
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
            except Exception as e:
                outfile.write(f"# Error reading file {fname}: {str(e)}\n")

def main():
    # Define the base directory (current directory in this case)
    base_directory = "."
    output_file = 'combined.py'
    extensions = ('.py', '.html')

    # Remove output file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception as e:
            print(f"Error removing existing {output_file}: {str(e)}")
            return

    # Get all files recursively
    all_files = get_files_recursively(base_directory, extensions)
    
    # Sort files by extension and then by name
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x))

    # Add a header to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# Combined Python and HTML files\n")
        outfile.write(f"# Generated from directory: {os.path.abspath(base_directory)}\n")
        outfile.write(f"# Total files found: {len(all_files)}\n\n")

    # Combine all files
    combine_files(output_file, all_files)
    
    print(f"Successfully combined {len(all_files)} files into {output_file}")
    print("Files processed:")
    for file in all_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()

# Contents from: .\config\__init__.py


# Contents from: .\config\logging_config.py
# config/logging_config.py
import logging.config
from pathlib import Path

def setup_logging(log_dir: Path) -> None:
    """Configure logging for the application."""
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": log_dir / "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": log_dir / "error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "ERROR"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file", "error_file"],
                "level": "INFO",
                "propagate": True
            },
            "langgraph": {
                "handlers": ["console", "file", "error_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)


# Contents from: .\config\settings.py
# config/settings.py

from typing import Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Required API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    TAVILY_API_KEY: str = Field(..., description="Tavily API key")
    
    # System Settings
    DEBUG: bool = Field(False, description="Debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    MAX_CONCURRENT_TASKS: int = Field(5, description="Maximum concurrent tasks")
    ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")
    
    # LLM Settings
    LLM_MODEL: str = Field("gpt-4", description="LLM model name")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(2000, description="Maximum tokens for LLM")
    
    # Agent Settings
    MAX_ROUNDS: int = Field(20, description="Maximum interaction rounds")
    MAX_TIME: int = Field(3600, description="Maximum execution time in seconds")
    MAX_FILE_SIZE: int = Field(1048576, description="Maximum file size in bytes")
    MAX_MEMORY: int = Field(536870912, description="Maximum memory usage in bytes")
    
    # Project Paths
    PROJECT_ROOT: Path = Field(default=Path(__file__).parent.parent)
    LOG_DIR: Path = Field(default=Path(__file__).parent.parent / "logs")
    DATA_DIR: Path = Field(default=Path(__file__).parent.parent / "data")
    
    # Derived Settings
    @property
    def LLM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "model": self.LLM_MODEL,
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
        }
    
    @property
    def AGENT_SETTINGS(self) -> Dict[str, Dict[str, Any]]:
        return {
            "orchestrator": {
                "max_rounds": self.MAX_ROUNDS,
                "max_time": self.MAX_TIME,
                "handle_messages_concurrently": False,
                "max_retries": 3
            },
            "web_surfer": {
                "max_results": 3,
                "search_depth": "regular",
                "timeout": 30,
                "max_retries": 3,
                "allowed_domains": [],
                "blocked_domains": []
            },
            "file_surfer": {
                "max_file_size": self.MAX_FILE_SIZE,
                "supported_formats": [".txt", ".py", ".json", ".md", ".yaml", ".yml"],
                "max_files": 10,
                "timeout": 30
            },
            "coder": {
                "max_code_length": 1000,
                "supported_languages": ["python", "javascript", "shell"],
                "timeout": 30,
                "max_complexity": 10,
                "style_check": True
            },
            "terminal": {
                "timeout": 30,
                "max_memory": self.MAX_MEMORY,
                "allowed_modules": ["numpy", "pandas", "sklearn"],
                "blocked_modules": ["os", "subprocess", "sys"],
                "max_processes": 1
            }
        }
    
    @property
    def SYSTEM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "max_concurrent_tasks": self.MAX_CONCURRENT_TASKS,
            "task_timeout": self.MAX_TIME,
            "enable_metrics": self.ENABLE_METRICS,
            "metrics_interval": 60,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Contents from: .\core\__init__.py


# Contents from: .\core\graph.py
# core/graph.py

from typing import Dict, Any
from langgraph.graph import END, StateGraph
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.agents.orchestrator import OrchestratorAgent
from magnetic_one_langgraph.agents.web_surfer import WebSurferAgent
from magnetic_one_langgraph.agents.file_surfer import FileSurferAgent
from magnetic_one_langgraph.agents.coder import CoderAgent
from magnetic_one_langgraph.agents.terminal import TerminalAgent
from magnetic_one_langgraph.config.settings import Settings
import logging
from datetime import datetime

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
        
        # Return final state
        final_state = {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "visited_agents": state.visited_agents,
            "counter": state.counter,
            "current_agent": "FinalReview",
            "workflow_complete": True,
            "task_complete": True,
            "final_report": final_report,
            "final": True
        }
        
        logger.info("Final report generated")
        return final_state
        
    except Exception as e:
        logger.error(f"Error in final review: {str(e)}")
        raise

def next_step(state: Dict[str, Any]) -> str:
    """Determine next step in workflow."""
    try:
        if isinstance(state, dict):
            state = AgentState(**state)
            
        logger.info(f"Current agent: {state.current_agent}")
        
        # Handle completion
        if state.workflow_complete or state.task_complete:
            if state.current_agent == "FinalReview":
                logger.info("Workflow complete, ending")
                return END
            logger.info("Moving to final review")
            return "FinalReview"
            
        # Handle regular transitions
        if state.current_agent != "OrchestratorAgent":
            logger.info("Returning to Orchestrator")
            return "OrchestratorAgent"
            
        if state.next_agent:
            logger.info(f"Moving to {state.next_agent}")
            return state.next_agent
            
        return "OrchestratorAgent"
        
    except Exception as e:
        logger.error(f"Error in next_step: {str(e)}")
        return "FinalReview"

def create_workflow_graph(settings: Settings) -> StateGraph:
    """Create workflow graph."""
    try:
        workflow = StateGraph(AgentState)
        
        # Initialize agents
        agents = {
            "OrchestratorAgent": OrchestratorAgent(settings),
            "WebSurferAgent": WebSurferAgent(settings),
            "FileSurferAgent": FileSurferAgent(settings),
            "CoderAgent": CoderAgent(settings),
            "TerminalAgent": TerminalAgent(settings)
        }
        
        # Add nodes
        for name, agent in agents.items():
            workflow.add_node(name, agent.process)
            
        # Add final review node
        workflow.add_node("FinalReview", handle_final_review)
        
        # Add regular agent edges
        for agent_name in agents:
            if agent_name != "OrchestratorAgent":
                workflow.add_edge(agent_name, "OrchestratorAgent")
        
        # Add orchestrator edges
        workflow.add_conditional_edges(
            "OrchestratorAgent",
            next_step,
            {
                "WebSurferAgent": "WebSurferAgent",
                "FileSurferAgent": "FileSurferAgent",
                "CoderAgent": "CoderAgent",
                "TerminalAgent": "TerminalAgent",
                "FinalReview": "FinalReview"
            }
        )
        
        # Add final review edge
        workflow.add_conditional_edges(
            "FinalReview",
            lambda x: END,
            {END: END}
        )
        
        # Set entry point
        workflow.set_entry_point("OrchestratorAgent")
        
        return workflow.compile()
        
    except Exception as e:
        logger.error(f"Error creating workflow graph: {str(e)}")
        raise

# Contents from: .\core\prompts.py
SYSTEM_PROMPTS = {
    "orchestrator": """You are an orchestrator agent responsible for coordinating task execution.
Your role is to:
1. Analyze incoming tasks
2. Create execution plans
3. Assign tasks to appropriate agents
4. Monitor progress
5. Ensure task completion

Consider available agents and their capabilities when planning.""",

    "web_surfer": """You are a web research agent capable of gathering information from the internet.
Your role is to:
1. Analyze research requirements
2. Perform targeted searches
3. Validate information sources
4. Extract relevant data
5. Summarize findings

Focus on accuracy and relevance of gathered information.""",

    "file_surfer": """You are a file analysis agent capable of processing and analyzing files.
Your role is to:
1. Read and parse files
2. Extract relevant information
3. Analyze content patterns
4. Generate insights
5. Provide structured analysis

Ensure thorough analysis and data validation.""",

    "coder": """You are a code generation agent capable of creating and modifying code.
Your role is to:
1. Analyze coding requirements
2. Generate appropriate code
3. Include error handling
4. Add documentation
5. Validate code quality

Follow best practices and ensure code safety.""",

    "terminal": """You are a code execution agent responsible for running and monitoring code.
Your role is to:
1. Validate code safety
2. Set up execution environment
3. Monitor execution
4. Handle errors
5. Report results

Prioritize safe and controlled execution."""
}

# Contents from: .\core\settings.py
# config/settings.py

from typing import Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration."""

    # Required API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    TAVILY_API_KEY: str = Field(..., description="Tavily API key")

    # System Settings
    DEBUG: bool = Field(False, description="Debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    MAX_CONCURRENT_TASKS: int = Field(5, description="Maximum concurrent tasks")
    ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")

    # LLM Settings
    LLM_MODEL: str = Field("gpt-4", description="LLM model name")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(2000, description="Maximum tokens for LLM")

    # Agent Settings
    MAX_ROUNDS: int = Field(20, description="Maximum interaction rounds")
    MAX_TIME: int = Field(3600, description="Maximum execution time in seconds")
    MAX_FILE_SIZE: int = Field(1048576, description="Maximum file size in bytes")
    MAX_MEMORY: int = Field(536870912, description="Maximum memory usage in bytes")

    # Project Paths
    PROJECT_ROOT: Path = Field(default=Path(__file__).parent.parent.resolve())
    LOG_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Derived Settings
    @property
    def LLM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "model": self.LLM_MODEL,
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
        }

    @property
    def AGENT_SETTINGS(self) -> Dict[str, Dict[str, Any]]:
        return {
            "orchestrator": {
                "max_rounds": self.MAX_ROUNDS,
                "max_time": self.MAX_TIME,
                "handle_messages_concurrently": False,
                "max_retries": 3,
            },
            "web_surfer": {
                "max_results": 5,
                "search_depth": "advanced",
                "timeout": 30,
                "max_retries": 3,
                "allowed_domains": [],
                "blocked_domains": [],
            },
            "file_surfer": {
                "max_file_size": self.MAX_FILE_SIZE,
                "supported_formats": [".txt", ".py", ".json", ".md", ".yaml", ".yml"],
                "max_files": 10,
                "timeout": 30,
            },
            "coder": {
                "max_code_length": 1000,
                "supported_languages": ["python", "javascript", "shell"],
                "timeout": 30,
                "max_complexity": 10,
                "style_check": True,
            },
            "terminal": {
                "timeout": 30,
                "max_memory": self.MAX_MEMORY,
                "allowed_modules": ["numpy", "pandas", "sklearn"],
                "blocked_modules": ["os", "subprocess", "sys"],
                "max_processes": 1,
            },
        }

    @property
    def SYSTEM_SETTINGS(self) -> Dict[str, Any]:
        return {
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "max_concurrent_tasks": self.MAX_CONCURRENT_TASKS,
            "task_timeout": self.MAX_TIME,
            "enable_metrics": self.ENABLE_METRICS,
            "metrics_interval": 60,
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Contents from: .\core\state.py
# core/state.py

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

# Contents from: .\main.py
# main.py

import asyncio
import logging
from dotenv import load_dotenv
from magnetic_one_langgraph.config.settings import Settings
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.core.graph import create_workflow_graph
from datetime import datetime
from typing import Optional

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

# Contents from: .\setup.py
# setup.py

from setuptools import setup, find_packages

setup(
    name="magnetic_one_langgraph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph==0.2.48",
        "langchain==0.3.7",
        "langchain-core==0.3.19",
        "langchain-openai==0.2.8",
        "python-dotenv==1.0.1",
        "tavily-python==0.5.0",
        "numpy==1.26.4",
        "pydantic==2.9.2",
        "pydantic-settings>=2.0",  # Added pydantic-settings
        "openai==1.54.4",
        "tiktoken==0.8.0",
        "httpx==0.27.2",
        "aiohttp==3.11.2",
        "aiofiles==23.2.1",        # If using aiofiles in your code
    ],
)


# Contents from: .\tests\test_agents.py
# tests/test_agents.py

import pytest
from magnetic_one_langgraph.core.state import AgentState
from magnetic_one_langgraph.agents.orchestrator import OrchestratorAgent
from magnetic_one_langgraph.agents.web_surfer import WebSurferAgent
from magnetic_one_langgraph.agents.file_surfer import FileSurferAgent
from magnetic_one_langgraph.agents.coder import CoderAgent
from magnetic_one_langgraph.agents.terminal import TerminalAgent
from magnetic_one_langgraph.config.settings import Settings

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def initial_state():
    return AgentState(
        messages=["Test task description"]
    )

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_initialization(self, settings, initial_state):
        orchestrator = OrchestratorAgent(settings)
        state = await orchestrator._process(initial_state)
        
        assert state is not None
        assert isinstance(state, AgentState)
        assert state.messages
        assert state.task_plan

    @pytest.mark.asyncio
    async def test_task_planning(self, settings, initial_state):
        orchestrator = OrchestratorAgent(settings)
        state = await orchestrator._process(initial_state)
        
        assert state.task_plan is not None
        assert len(state.task_plan) > 0

class TestWebSurfer:
    @pytest.mark.asyncio
    async def test_web_search(self, settings, initial_state):
        web_surfer = WebSurferAgent(settings)
        state = await web_surfer._process(initial_state)
        
        assert state is not None
        assert "web_research" in state.task_ledger

class TestCoder:
    @pytest.mark.asyncio
    async def test_code_generation(self, settings, initial_state):
        coder = CoderAgent(settings)
        initial_state.messages.append("Generate a function to calculate fibonacci numbers")
        state = await coder._process(initial_state)
        
        assert state is not None
        assert "code_generation" in state.task_ledger

    def test_code_validation(self, settings):
        coder = CoderAgent(settings)
        safe_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        unsafe_code = """
import os
os.system('rm -rf /')
"""
        assert coder._validate_code(safe_code) is not None
        assert coder._validate_code(unsafe_code) is None

class TestTerminal:
    @pytest.mark.asyncio
    async def test_code_execution(self, settings, initial_state):
        terminal = TerminalAgent(settings)
        initial_state.task_ledger["code_generation"] = {
            "code": "print('Hello, World!')",
            "language": "python"
        }
        
        state = await terminal._process(initial_state)
        
        assert state is not None
        assert "code_execution" in state.task_ledger
        assert state.task_ledger["code_execution"]["result"]["success"]


# Contents from: .\tests\test_utilities.py
# tests/test_utilities.py
import pytest
from ..utils.metrics import MetricsCollector
from ..utils.validation import CodeValidator

class TestMetricsCollector:
    def test_metrics_collection(self):
        collector = MetricsCollector()
        
        collector.add_metric(
            "test_metric",
            {"value": 1.0},
            {"test": True}
        )
        
        summary = collector.get_metrics_summary()
        assert "test_metric" in summary
        assert summary["test_metric"]["count"] == 1
        
    def test_trend_analysis(self):
        collector = MetricsCollector()
        
        # Add increasing trend
        for i in range(5):
            collector.add_metric(
                "test_metric",
                {"value": i}
            )
            
        summary = collector.get_metrics_summary()
        assert summary["test_metric"]["trends"]["value"] == "increasing"

class TestCodeValidator:
    def test_code_validation(self):
        validator = CodeValidator()
        
        safe_code = """
def greet(name):
    return f"Hello, {name}!"
"""
        
        unsafe_code = """
import os
os.system('rm -rf /')
"""
        
        assert validator.validate_code(safe_code, "python")["is_valid"]
        assert not validator.validate_code(unsafe_code, "python")["is_valid"]
        
    def test_code_quality(self):
        validator = CodeValidator()
        
        code = """
# This is a test function
def test_function():
    # Initialize value
    x = 0
    
    # Loop and increment
    for i in range(10):
        x += i
        
    return x
"""
        
        quality_metrics = validator._analyze_code_quality(code)
        assert quality_metrics["complexity"] > 0
        assert quality_metrics["documentation_ratio"] > 0
        assert quality_metrics["style_score"] > 0

if __name__ == "__main__":
    pytest.main([__file__])

# Contents from: .\utils\__init__.py
# main.py
import asyncio
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .core.graph import create_workflow_graph
from .core.state import AgentState
from .config.settings import Settings
from .agents.orchestrator import OrchestratorAgent
from .agents.web_surfer import WebSurferAgent
from .agents.file_surfer import FileSurferAgent
from .agents.coder import CoderAgent
from .agents.terminal import TerminalAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LangGraphSystem:
    """Main system class for LangGraph implementation."""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = {
            "orchestrator": OrchestratorAgent(self.settings),
            "web_surfer": WebSurferAgent(self.settings),
            "file_surfer": FileSurferAgent(self.settings),
            "coder": CoderAgent(self.settings),
            "terminal": TerminalAgent(self.settings)
        }
        
        # Create workflow
        self.workflow = create_workflow_graph()
        
    async def run_task(self, task_description: str) -> Dict[str, Any]:
        """Run a task through the system."""
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[f"Task started: {task_description}"]
            )
            
            # Execute workflow
            result = {"status": "running"}
            async for step in self.workflow.astream(initial_state):
                try:
                    if step is None:
                        continue
                        
                    # Update result
                    result = {
                        "status": "completed" if step.get("task_complete") else "in_progress",
                        "messages": step.get("messages", []),
                        "final_report": step.get("final_report"),
                        "error_count": step.get("error_count", {})
                    }
                    
                    # Check for completion
                    if step.get("task_complete"):
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error in workflow step: {str(e)}")
                    result["status"] = "error"
                    result["error"] = str(e)
                    break
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Error running task: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

async def main():
    """Main entry point."""
    try:
        system = LangGraphSystem()
        
        # Example task
        task = """
        Analyze the current market trends in the AI industry with focus on:
        1. Latest developments in LLM technologies
        2. Major market players and their strategies
        3. Investment patterns and funding rounds
        4. Potential market opportunities and risks
        
        Provide specific insights and actionable recommendations.
        """
        
        result = await system.run_task(task)
        print(f"Task completed with status: {result['status']}")
        
        if result.get("final_report"):
            print("\nFinal Report:")
            print("=" * 50)
            print(result["final_report"])
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

# Contents from: .\utils\metrics.py


# Contents from: .\utils\validation.py
# utils/validation.py
from typing import Optional, Dict, Any
import ast
import re

class CodeValidator:
    """Validate and analyze code safety and quality."""
    
    def __init__(self):
        self.unsafe_patterns = [
            r"os\s*\.\s*system",
            r"subprocess",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"open\s*\(",
            r"write\s*\(",
            r"delete\s*\(",
            r"remove\s*\("
        ]
        
    def validate_code(self, code: str, language: str) -> Optional[Dict[str, Any]]:
        """Validate code safety and quality."""
        try:
            # Basic syntax check
            if language.lower() in ['python', 'py']:
                ast.parse(code)
                
            # Check for unsafe patterns
            for pattern in self.unsafe_patterns:
                if re.search(pattern, code):
                    return None
                    
            # Analyze code quality
            quality_metrics = self._analyze_code_quality(code)
            
            return {
                "is_valid": True,
                "quality_metrics": quality_metrics
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e)
            }
            
    def _analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        lines = code.split('\n')
        
        return {
            "line_count": len(lines),
            "complexity": self._calculate_complexity(code),
            "documentation_ratio": self._calculate_documentation_ratio(lines),
            "style_score": self._calculate_style_score(lines)
        }
        
    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity."""
        # Simple complexity measure based on control structures
        complexity = 0
        control_structures = ['if', 'for', 'while', 'try', 'with']
        
        for line in code.split('\n'):
            if any(struct in line for struct in control_structures):
                complexity += 1
                
        return complexity
        
    def _calculate_documentation_ratio(self, lines: List[str]) -> float:
        """Calculate ratio of documentation to code."""
        doc_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return doc_lines / len(lines) if lines else 0
        
    def _calculate_style_score(self, lines: List[str]) -> float:
        """Calculate code style score."""
        style_score = 100.0
        
        # Check line length
        long_lines = sum(1 for line in lines if len(line) > 79)
        style_score -= (long_lines / len(lines)) * 20
        
        # Check indentation
        bad_indentation = sum(1 for line in lines if line.startswith(' ') and not line.startswith('    '))
        style_score -= (bad_indentation / len(lines)) * 20
        
        return max(0, style_score)