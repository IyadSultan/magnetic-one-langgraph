# magnetic_one_langgraph/tests/test_utilities.py

import pytest
from magnetic_one_langgraph.utils.metrics import MetricsCollector
from magnetic_one_langgraph.utils.validation import CodeValidator

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
