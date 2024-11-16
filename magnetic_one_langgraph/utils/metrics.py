# magnetic_one_langgraph/utils/metrics.py

from typing import Dict, Any, List
from collections import defaultdict

class MetricsCollector:
    """Collect and analyze metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def add_metric(self, name: str, data: Dict[str, Any], tags: Dict[str, Any] = None):
        """Add a metric with optional tags."""
        metric_data = {
            "data": data,
            "tags": tags or {}
        }
        self.metrics[name].append(metric_data)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        summary = {}
        for name, entries in self.metrics.items():
            count = len(entries)
            trends = self._analyze_trends(entries)
            summary[name] = {
                "count": count,
                "trends": trends
            }
        return summary

    def _analyze_trends(self, entries: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trends in metric data."""
        trends = {}
        if not entries:
            return trends

        keys = entries[0]['data'].keys()
        for key in keys:
            values = [entry['data'][key] for entry in entries if key in entry['data']]
            if all(x <= y for x, y in zip(values, values[1:])):
                trends[key] = "increasing"
            elif all(x >= y for x, y in zip(values, values[1:])):
                trends[key] = "decreasing"
            else:
                trends[key] = "stable"
        return trends
