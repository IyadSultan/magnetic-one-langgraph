# magnetic_one_langgraph/utils/validation.py

import ast
import re
from typing import Optional, Dict, Any, List

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
                    return {"is_valid": False, "error": "Unsafe code detected."}
                    
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
