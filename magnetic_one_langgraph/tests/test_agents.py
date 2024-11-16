# magnetic_one_langgraph/tests/test_agents.py

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
    return Settings(
        OPENAI_API_KEY="test",
        TAVILY_API_KEY="test"
    )

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

    @pytest.mark.asyncio
    async def test_task_planning(self, settings, initial_state):
        orchestrator = OrchestratorAgent(settings)
        state = await orchestrator._process(initial_state)
        
        assert state.task_ledger is not None

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

class TestTerminal:
    @pytest.mark.asyncio
    async def test_code_execution(self, settings, initial_state):
        terminal = TerminalAgent(settings)
        initial_state.task_ledger["code_generation"] = {
            "data": {
                "code": "print('Hello, World!')",
                "language": "python"
            }
        }
        
        state = await terminal._process(initial_state)
        
        assert state is not None
        assert "code_execution" in state.task_ledger
        assert state.task_ledger["code_execution"]["data"]["status"] == "success"
