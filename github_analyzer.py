import os
from typing import Dict, List, Tuple, Optional, Any
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import glob
import re
from datetime import datetime
import json

# Load environment variables and initialize LLM
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-3.5-turbo" for faster, cheaper analysis
    temperature=0,  # Keep it deterministic
)

class RepositoryAnalysisState(BaseModel):
    """State for repository analysis system"""
    messages: List[str] = Field(default_factory=list)
    task_ledger: Dict[str, str] = Field(default_factory=dict)
    task_plan: List[Tuple[str, str]] = Field(default_factory=list)
    counter: int = Field(default=0)
    final_report: Optional[str] = Field(default=None)
    task_complete: bool = Field(default=False)
    current_agent: str = Field(default="Orchestrator")
    next_agent: str = Field(default="Orchestrator")
    analysis_results: Dict[str, Any] = Field(default_factory=dict)

def get_llm_analysis(content: str, analysis_type: str) -> Dict:
    """Get LLM analysis for specific content."""
    prompts = {
        "code": """Analyze the following code and provide insights about:
        1. Code quality and maintainability
        2. Potential improvements
        3. Best practices followed or missing
        4. Areas of concern
        
        Format your response as JSON with these keys: quality_score (1-10), strengths (list), weaknesses (list), recommendations (list).
        Be specific and technical in your analysis.
        
        Code to analyze:
        {code}
        """,
        
        "structure": """Analyze the following repository structure and provide insights about:
        1. Organization and clarity
        2. File/folder naming conventions
        3. Best practices followed or missing
        4. Suggested improvements
        
        Format your response as JSON with these keys: structure_score (1-10), strengths (list), weaknesses (list), recommendations (list).
        
        Structure to analyze:
        {content}
        """,
        
        "documentation": """Analyze the following documentation and provide insights about:
        1. Clarity and completeness
        2. Technical accuracy
        3. Examples and tutorials
        4. Areas needing improvement
        
        Format your response as JSON with these keys: doc_score (1-10), strengths (list), weaknesses (list), recommendations (list).
        
        Documentation to analyze:
        {content}
        """,
        
        "security": """Analyze the following code/configuration for security concerns:
        1. Potential vulnerabilities
        2. Security best practices
        3. Risk assessment
        4. Mitigation recommendations
        
        Format your response as JSON with these keys: security_score (1-10), risks (list), good_practices (list), recommendations (list).
        
        Content to analyze:
        {content}
        """,
        
        "dependencies": """Analyze the following dependency configuration and provide insights about:
        1. Dependency management practices
        2. Version control and specificity
        3. Potential conflicts or issues
        4. Improvement recommendations
        
        Format your response as JSON with these keys: dep_score (1-10), strengths (list), weaknesses (list), recommendations (list).
        
        Dependencies to analyze:
        {content}
        """
    }
    
    prompt = ChatPromptTemplate.from_template(prompts[analysis_type])
    output_parser = JsonOutputParser()
    
    try:
        prompt = ChatPromptTemplate.from_template(prompts[analysis_type])
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"code": content, "content": content})
        return result
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return {}

def create_task_ledger(repo_path: str) -> Dict[str, str]:
    """Create initial task ledger with repository information."""
    return {
        "repo_path": repo_path,
        "analysis_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "initialized"
    }

def orchestrator_agent(state: RepositoryAnalysisState) -> Dict:
    """Manages the analysis workflow."""
    try:
        print("\n🎮 Orchestrator planning next analysis step...")
        
        # Initialize if needed
        if not state.task_plan:
            state.task_plan = [
                ("StructureAnalyzer", "Analyze repository structure"),
                ("CodeAnalyzer", "Analyze code quality"),
                ("DependencyAnalyzer", "Analyze dependencies"),
                ("SecurityAnalyzer", "Check security"),
                ("DocumentationAnalyzer", "Assess documentation")
            ]
            state.messages.append("Analysis plan created.")
        
        # Determine next agent
        if state.task_complete:
            next_agent = "FinalReview"
        elif state.task_plan:
            next_task = state.task_plan.pop(0)
            next_agent = next_task[0]
            state.messages.append(f"Starting {next_task[1]}")
        else:
            next_agent = "FinalReview"
        
        state.current_agent = next_agent
        state.next_agent = next_agent
        state.messages.append(f"Orchestrator assigned task to {next_agent}")
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": next_agent
        }
    except Exception as e:
        print(f"Error in orchestrator_agent: {str(e)}")
        raise

def structure_analyzer_agent(state: RepositoryAnalysisState) -> Dict:
    """Analyzes repository structure and organization with LLM insights."""
    try:
        print("\n🔍 Structure Analyzer examining repository organization...")
        repo_path = state.task_ledger["repo_path"]
        
        # Static structure analysis
        file_types = {}
        total_files = 0
        max_depth = 0
        structure_content = []
        
        for root, dirs, files in os.walk(repo_path):
            depth = root[len(repo_path):].count(os.sep)
            max_depth = max(max_depth, depth)
            
            # Build directory structure for LLM analysis
            rel_path = os.path.relpath(root, repo_path)
            if rel_path != '.':
                structure_content.append(f"Directory: {rel_path}")
            
            for file in files:
                total_files += 1
                ext = os.path.splitext(file)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                if len(structure_content) < 50:  # Limit file listing for LLM
                    structure_content.append(f"File: {os.path.join(rel_path, file)}")

        # Get LLM insights on repository structure
        structure_text = "\n".join(structure_content)
        llm_insights = get_llm_analysis(structure_text, "structure")
        
        structure_insights = {
            "total_files": total_files,
            "max_depth": max_depth,
            "file_distribution": file_types,
            "structure_score": llm_insights.get('structure_score', 5),
            "llm_insights": llm_insights
        }
        
        state.analysis_results["structure"] = structure_insights
        
        message_parts = [
            f"📊 Repository Structure Analysis:",
            f"- Found {total_files} files across {len(file_types)} different types",
            f"- Maximum directory depth: {max_depth}"
        ]
        
        if llm_insights:
            message_parts.extend([
                "\nLLM Analysis:",
                f"- Structure Score: {llm_insights.get('structure_score', 5)}/10",
                "\nStrengths:",
                *[f"- {s}" for s in llm_insights.get('strengths', [])],
                "\nRecommendations:",
                *[f"- {r}" for r in llm_insights.get('recommendations', [])]
            ])
        
        state.messages.append("\n".join(message_parts))
        state.next_agent = "Orchestrator"
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in structure_analyzer_agent: {str(e)}")
        raise

def code_analyzer_agent(state: RepositoryAnalysisState) -> Dict:
    """Analyzes code quality with LLM insights."""
    try:
        print("\n🔍 Code Analyzer examining code quality...")
        repo_path = state.task_ledger["repo_path"]
        
        code_metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "complex_functions": 0
        }
        
        # Store samples of code for LLM analysis
        code_samples = []
        
        for ext in ['.py', '.js', '.java', '.cpp', '.cs']:
            for file_path in glob.glob(f"{repo_path}/**/*{ext}", recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        code_metrics["total_lines"] += len(lines)
                        
                        # Basic metrics
                        for line in lines:
                            line = line.strip()
                            if line.startswith(('#', '//', '/*', '*', '*/')):
                                code_metrics["comment_lines"] += 1
                            elif line:
                                code_metrics["code_lines"] += 1
                        
                        # Complexity check
                        code_metrics["complex_functions"] += len(re.findall(r'\b(if|for|while|switch)\b', content))
                        
                        # Collect representative code samples
                        if 100 < len(lines) < 300:  # Sample medium-sized files
                            code_samples.append({
                                'file': os.path.basename(file_path),
                                'content': content
                            })
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        # Get LLM insights for code samples
        llm_insights = []
        for sample in code_samples[:3]:  # Limit to 3 files for efficiency
            insight = get_llm_analysis(sample['content'], "code")
            if insight:
                insight['file'] = sample['file']
                llm_insights.append(insight)
        
        # Calculate metrics
        comment_ratio = (code_metrics["comment_lines"] / max(1, code_metrics["code_lines"])) * 100
        complexity_ratio = (code_metrics["complex_functions"] / max(1, code_metrics["code_lines"])) * 1000
        
        # Combine static and LLM analysis
        average_llm_score = sum(insight.get('quality_score', 5) for insight in llm_insights) / max(1, len(llm_insights))
        
        state.analysis_results["code_quality"] = {
            "metrics": code_metrics,
            "comment_ratio": comment_ratio,
            "complexity_ratio": complexity_ratio,
            "llm_insights": llm_insights,
            "quality_score": average_llm_score
        }
        
        # Generate detailed message
        message_parts = [
            f"📊 Code Quality Analysis:",
            f"- Total lines of code: {code_metrics['code_lines']}",
            f"- Comment ratio: {comment_ratio:.1f}%",
            f"- Complexity score: {complexity_ratio:.1f}"
        ]
        
        if llm_insights:
            message_parts.append("\nLLM Analysis:")
            for insight in llm_insights:
                message_parts.extend([
                    f"\nFile: {insight['file']}",
                    f"- Quality Score: {insight.get('quality_score', 5)}/10",
                    "Strengths:",
                    *[f"- {s}" for s in insight.get('strengths', [])],
                    "Recommendations:",
                    *[f"- {r}" for r in insight.get('recommendations', [])]
                ])
        
        state.messages.append("\n".join(message_parts))
        state.next_agent = "Orchestrator"
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in code_analyzer_agent: {str(e)}")
        raise

def dependency_analyzer_agent(state: RepositoryAnalysisState) -> Dict:
    """Analyzes project dependencies with LLM insights."""
    try:
        print("\n🔍 Dependency Analyzer checking project dependencies...")
        repo_path = state.task_ledger["repo_path"]
        
        dependency_files = {
            'python': ['requirements.txt', 'Pipfile', 'pyproject.toml'],
            'node': ['package.json', 'package-lock.json'],
            'dotnet': ['*.csproj', '*.fsproj'],
            'java': ['pom.xml', 'build.gradle']
        }
        
        found_dependencies = {}
        dep_contents = []
        
        for tech, files in dependency_files.items():
            for file_pattern in files:
                for file_path in glob.glob(f"{repo_path}/**/{file_pattern}", recursive=True):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Store content for LLM analysis
                            dep_contents.append({
                                'file': os.path.basename(file_path),
                                'content': content
                            })
                            
                            # Basic dependency extraction
                            if file_pattern == 'requirements.txt':
                                deps = [line.split('==')[0] for line in content.split('\n') if '==' in line]
                            elif file_pattern == 'package.json':
                                try:
                                    pkg_data = json.loads(content)
                                    deps = list(pkg_data.get('dependencies', {}).keys())
                                except:
                                    deps = []
                            else:
                                deps = []
                            
                            if deps:
                                found_dependencies[os.path.basename(file_path)] = deps
                    except Exception as e:
                        print(f"Error processing dependency file {file_path}: {str(e)}")
                        continue
        
        # Get LLM insights for dependency files
        llm_insights = []
        for dep_file in dep_contents[:3]:  # Limit to 3 files for efficiency
            insight = get_llm_analysis(dep_file['content'], "dependencies")
            if insight:
                insight['file'] = dep_file['file']
                llm_insights.append(insight)
        
        # Calculate dependency score
        average_llm_score = sum(insight.get('dep_score', 5) for insight in llm_insights) / max(1, len(llm_insights))
        
        state.analysis_results["dependencies"] = {
            "found_files": list(found_dependencies.keys()),
            "dependency_count": sum(len(deps) for deps in found_dependencies.values()),
            "llm_insights": llm_insights,
            "dependency_score": average_llm_score
        }
        
        # Generate detailed message
        message_parts = [
            f"📦 Dependency Analysis:",
            f"- Found {len(found_dependencies)} dependency files",
            f"- Total dependencies: {state.analysis_results['dependencies']['dependency_count']}"
        ]
        
        if llm_insights:
            message_parts.append("\nLLM Analysis:")
            for insight in llm_insights:
                message_parts.extend([
                    f"\nFile: {insight['file']}",
                    f"- Dependency Score: {insight.get('dep_score', 5)}/10",
                    "Strengths:",
                    *[f"- {s}" for s in insight.get('strengths', [])],
                    "Recommendations:",
                    *[f"- {r}" for r in insight.get('recommendations', [])]
                ])
        
        state.messages.append("\n".join(message_parts))
        state.next_agent = "Orchestrator"
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in dependency_analyzer_agent: {str(e)}")
        raise

def security_analyzer_agent(state: RepositoryAnalysisState) -> Dict:
    """Analyzes security concerns with LLM insights."""
    try:
        print("\n🔍 Security Analyzer checking for vulnerabilities...")
        repo_path = state.task_ledger["repo_path"]
        
        security_checks = {
            "secrets_found": 0,
            "sensitive_files": 0,
            "security_headers": 0
        }
        
        sensitive_patterns = [
            r'.*\.env$',
            r'.*\.pem$',
            r'.*\.key$',
            r'.*password.*',
            r'.*secret.*',
            r'.*credential.*'
        ]
        
        secret_patterns = [
            r'api[_-]?key',
            r'token',
            r'password',
            r'secret'
        ]
        
        # Collect security-relevant content for LLM analysis
        security_samples = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Check filename patterns
                if any(re.match(pattern, file_lower) for pattern in sensitive_patterns):
                    security_checks["sensitive_files"] += 1
                
                # Check content for sensitive patterns
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        content_lower = content.lower()
                        
                        # Check for secrets
                        if any(re.search(pattern, content_lower) for pattern in secret_patterns):
                            security_checks["secrets_found"] += 1
                        
                        # Collect samples for LLM analysis
                        if (any(pattern in file_lower for pattern in ['security', 'config', 'auth']) or
                            any(re.search(pattern, content_lower) for pattern in secret_patterns)):
                            security_samples.append({
                                'file': os.path.basename(file_path),
                                'content': content[:1500]  # Limit content size
                            })
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        # Get LLM insights for security-relevant files
        llm_insights = []
        for sample in security_samples[:3]:  # Limit to 3 files
            insight = get_llm_analysis(sample['content'], "security")
            if insight:
                insight['file'] = sample['file']
                llm_insights.append(insight)
        
        # Calculate security score
        average_llm_score = sum(insight.get('security_score', 5) for insight in llm_insights) / max(1, len(llm_insights))
        final_security_score = min(10, max(1,
            average_llm_score - 
            (2 if security_checks["secrets_found"] > 0 else 0) -
            (1 if security_checks["sensitive_files"] > 0 else 0)
        ))
        
        state.analysis_results["security"] = {
            "checks": security_checks,
            "llm_insights": llm_insights,
            "security_score": final_security_score
        }
        
        # Generate detailed message
        message_parts = [
            f"🔒 Security Analysis:",
            f"- Potential sensitive files: {security_checks['sensitive_files']}",
            f"- Possible exposed secrets: {security_checks['secrets_found']}",
            f"- Security score: {final_security_score}/10"
        ]
        
        if llm_insights:
            message_parts.append("\nLLM Security Analysis:")
            for insight in llm_insights:
                message_parts.extend([
                    f"\nFile: {insight['file']}",
                    "Identified Risks:",
                    *[f"- {r}" for r in insight.get('risks', [])],
                    "Security Recommendations:",
                    *[f"- {r}" for r in insight.get('recommendations', [])]
                ])
        
        state.messages.append("\n".join(message_parts))
        state.next_agent = "Orchestrator"
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in security_analyzer_agent: {str(e)}")
        raise

def documentation_analyzer_agent(state: RepositoryAnalysisState) -> Dict:
    """Analyzes documentation quality with LLM insights."""
    try:
        print("\n🔍 Documentation Analyzer assessing documentation quality...")
        repo_path = state.task_ledger["repo_path"]
        
        doc_metrics = {
            "readme_exists": False,
            "readme_size": 0,
            "doc_files": 0,
            "api_docs": 0,
            "examples": 0
        }
        
        # Collect documentation for LLM analysis
        doc_samples = []
        
        # Check README files
        readme_patterns = ['README.md', 'README.rst', 'README.txt']
        for pattern in readme_patterns:
            readme_path = os.path.join(repo_path, pattern)
            if os.path.exists(readme_path):
                doc_metrics["readme_exists"] = True
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    doc_metrics["readme_size"] = len(content)
                    doc_samples.append({
                        'file': pattern,
                        'content': content,
                        'type': 'readme'
                    })
                break
        
        # Analyze documentation files
        doc_extensions = ['.md', '.rst', '.txt', '.doc', '.docx']
        example_patterns = ['example', 'sample', 'tutorial', 'demo']
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_lower = file.lower()
                
                # Count and collect documentation files
                if any(file_lower.endswith(ext) for ext in doc_extensions):
                    doc_metrics["doc_files"] += 1
                    
                    # Collect content for specific doc types
                    if ('api' in file_lower or
                        any(pattern in file_lower for pattern in example_patterns)):
                        try:
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                                if 'api' in file_lower:
                                    doc_metrics["api_docs"] += 1
                                    doc_samples.append({
                                        'file': file,
                                        'content': content,
                                        'type': 'api'
                                    })
                                if any(pattern in file_lower for pattern in example_patterns):
                                    doc_metrics["examples"] += 1
                                    doc_samples.append({
                                        'file': file,
                                        'content': content,
                                        'type': 'example'
                                    })
                        except Exception as e:
                            print(f"Error processing doc file {file}: {str(e)}")
                            continue
        
        # Get LLM insights for documentation
        llm_insights = []
        for sample in doc_samples[:3]:  # Limit to 3 files
            insight = get_llm_analysis(sample['content'], "documentation")
            if insight:
                insight['file'] = sample['file']
                insight['type'] = sample['type']
                llm_insights.append(insight)
        
        # Calculate documentation score
        avg_llm_score = sum(insight.get('doc_score', 5) for insight in llm_insights) / max(1, len(llm_insights))
        final_doc_score = min(10, max(1, 
            avg_llm_score +
            (2 if doc_metrics["readme_exists"] else 0) +
            (1 if doc_metrics["api_docs"] > 0 else 0) +
            (1 if doc_metrics["examples"] > 0 else 0)
        ) / 2)  # Normalize to 1-10 scale
        
        state.analysis_results["documentation"] = {
            "metrics": doc_metrics,
            "llm_insights": llm_insights,
            "doc_score": final_doc_score
        }
        
        # Generate detailed message
        message_parts = [
            f"📚 Documentation Analysis:",
            f"- README {'exists' if doc_metrics['readme_exists'] else 'missing'}",
            f"- Documentation files: {doc_metrics['doc_files']}",
            f"- API documentation files: {doc_metrics['api_docs']}",
            f"- Example files: {doc_metrics['examples']}",
            f"- Documentation score: {final_doc_score}/10"
        ]
        
        if llm_insights:
            message_parts.append("\nLLM Documentation Analysis:")
            for insight in llm_insights:
                message_parts.extend([
                    f"\nFile: {insight['file']} (Type: {insight['type']})",
                    "Strengths:",
                    *[f"- {s}" for s in insight.get('strengths', [])],
                    "Recommendations:",
                    *[f"- {r}" for r in insight.get('recommendations', [])]
                ])
        
        state.messages.append("\n".join(message_parts))
        state.next_agent = "Orchestrator"
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": "Orchestrator"
        }
    except Exception as e:
        print(f"Error in documentation_analyzer_agent: {str(e)}")
        raise

def finalize_task(state: RepositoryAnalysisState) -> Dict:
    """Generate final analysis report with LLM-enhanced insights."""
    try:
        print("\n📊 Generating final analysis report...")
        
        # Collect scores
        scores = {
            "structure": state.analysis_results.get("structure", {}).get("structure_score", 0),
            "code_quality": state.analysis_results.get("code_quality", {}).get("quality_score", 0),
            "dependencies": state.analysis_results.get("dependencies", {}).get("dependency_score", 0),
            "security": state.analysis_results.get("security", {}).get("security_score", 0),
            "documentation": state.analysis_results.get("documentation", {}).get("doc_score", 0)
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        # Collect all LLM recommendations
        all_recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        # Structure recommendations
        structure_insights = state.analysis_results.get("structure", {}).get("llm_insights", {})
        if isinstance(structure_insights, dict) and structure_insights.get('recommendations'):
            all_recommendations["medium_priority"].extend(structure_insights['recommendations'])
        
        # Code quality recommendations
        code_insights = state.analysis_results.get("code_quality", {}).get("llm_insights", [])
        for insight in code_insights:
            if insight.get('recommendations'):
                all_recommendations["high_priority"].extend(insight['recommendations'])
        
        # Security recommendations
        security_insights = state.analysis_results.get("security", {}).get("llm_insights", [])
        for insight in security_insights:
            if insight.get('recommendations'):
                all_recommendations["high_priority"].extend(insight['recommendations'])
        
        # Documentation recommendations
        doc_insights = state.analysis_results.get("documentation", {}).get("llm_insights", [])
        for insight in doc_insights:
            if insight.get('recommendations'):
                all_recommendations["medium_priority"].extend(insight['recommendations'])
        
        # Dependency recommendations
        dep_insights = state.analysis_results.get("dependencies", {}).get("llm_insights", [])
        for insight in dep_insights:
            if insight.get('recommendations'):
                all_recommendations["low_priority"].extend(insight['recommendations'])
        
        # Generate final report
        state.final_report = (
            "📋 Repository Analysis Summary\n"
            "============================\n\n"
            f"Repository: {state.task_ledger['repo_path']}\n"
            f"Analysis Date: {state.task_ledger['analysis_started']}\n\n"
            "Scores by Category\n"
            "-----------------\n"
            f"Structure: {scores['structure']}/10\n"
            f"Code Quality: {scores['code_quality']}/10\n"
            f"Dependencies: {scores['dependencies']}/10\n"
            f"Security: {scores['security']}/10\n"
            f"Documentation: {scores['documentation']}/10\n"
            f"\nOverall Score: {overall_score:.1f}/10\n\n"
            "Key Recommendations\n"
            "------------------\n"
            "High Priority:\n" +
            "\n".join(f"- {r}" for r in all_recommendations["high_priority"][:5]) +
            "\n\nMedium Priority:\n" +
            "\n".join(f"- {r}" for r in all_recommendations["medium_priority"][:5]) +
            "\n\nLow Priority:\n" +
            "\n".join(f"- {r}" for r in all_recommendations["low_priority"][:5])
        )
        
        state.messages.append("Final analysis report generated.")
        state.task_complete = True
        state.next_agent = END
        
        return {
            "messages": state.messages,
            "task_ledger": state.task_ledger,
            "task_plan": state.task_plan,
            "counter": state.counter,
            "final_report": state.final_report,
            "task_complete": state.task_complete,
            "current_agent": state.current_agent,
            "next_agent": state.next_agent,
            "analysis_results": state.analysis_results,
            "next": END
        }
    except Exception as e:
        print(f"Error in finalize_task: {str(e)}")
        raise

def create_workflow_graph() -> StateGraph:
    """Create and configure the workflow graph."""
    try:
        print("DEBUG: Creating workflow graph")
        workflow = StateGraph(RepositoryAnalysisState)
        
        # Add nodes
        workflow.add_node("Orchestrator", orchestrator_agent)
        workflow.add_node("StructureAnalyzer", structure_analyzer_agent)
        workflow.add_node("CodeAnalyzer", code_analyzer_agent)
        workflow.add_node("DependencyAnalyzer", dependency_analyzer_agent)
        workflow.add_node("SecurityAnalyzer", security_analyzer_agent)
        workflow.add_node("DocumentationAnalyzer", documentation_analyzer_agent)
        workflow.add_node("FinalReview", finalize_task)
        
        # Add conditional edges from Orchestrator
        workflow.add_conditional_edges(
            "Orchestrator",
            lambda x: x.next_agent,
            {
                "StructureAnalyzer": "StructureAnalyzer",
                "CodeAnalyzer": "CodeAnalyzer",
                "DependencyAnalyzer": "DependencyAnalyzer",
                "SecurityAnalyzer": "SecurityAnalyzer",
                "DocumentationAnalyzer": "DocumentationAnalyzer",
                "FinalReview": "FinalReview"
            }
        )
        
        # Add edges back to Orchestrator
        workflow.add_edge("StructureAnalyzer", "Orchestrator")
        workflow.add_edge("CodeAnalyzer", "Orchestrator")
        workflow.add_edge("DependencyAnalyzer", "Orchestrator")
        workflow.add_edge("SecurityAnalyzer", "Orchestrator")
        workflow.add_edge("DocumentationAnalyzer", "Orchestrator")
        
        # Add start edge
        workflow.add_edge(START, "Orchestrator")
        
        return workflow.compile()
    except Exception as e:
        print(f"DEBUG: Error in create_workflow_graph: {str(e)}")
        raise

def run_repository_analysis(repo_path: str):
    """Run the repository analysis system."""
    try:
        print("\n🚀 Starting Repository Analysis System")
        print(f"Analyzing repository: {repo_path}")
        
        # Initialize state
        initial_state = RepositoryAnalysisState(
            messages=[f"Starting analysis of repository: {repo_path}"],
            task_ledger=create_task_ledger(repo_path)
        )
        
        # Create and run workflow
        workflow = create_workflow_graph()
        for step_dict in workflow.stream(initial_state):
            if "__end__" not in step_dict:
                messages = step_dict.get("messages", [])
                if messages:
                    print(messages[-1])
                
                final_report = step_dict.get("final_report")
                if final_report:
                    print("\n==== Final Report ====")
                    print(final_report)
                    break
                
    except Exception as e:
        print(f"Error in run_repository_analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Analyze the CTAGAPI repository
    repo_path = "https://github.com/AIDI-KHCC/CTAGAPI"  # Specify GitHub repo URL
    
    # Clone the repository first if needed
    try:
        import git
        repo_name = repo_path.split("/")[-1]
        local_path = os.path.join(os.getcwd(), repo_name)
        
        if not os.path.exists(local_path):
            print(f"Cloning repository to {local_path}...")
            git.Repo.clone_from(repo_path + ".git", local_path)
        
        # Run analysis on the cloned repo
        run_repository_analysis(local_path)
        
    except ImportError:
        print("Please install GitPython first: pip install GitPython")
    except Exception as e:
        print(f"Error analyzing repository: {str(e)}")