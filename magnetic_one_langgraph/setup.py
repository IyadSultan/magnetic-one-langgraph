# magnetic_one_langgraph/setup.py

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
        "pydantic-settings>=2.0",
        "openai==1.54.4",
        "tiktoken==0.8.0",
        "httpx==0.27.2",
        "aiohttp==3.11.2",
        "aiofiles==23.2.1",
    ],
)
