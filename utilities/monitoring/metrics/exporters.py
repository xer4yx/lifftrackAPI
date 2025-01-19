from typing import Dict, List
import json
import os
from datetime import datetime, timezone
from core.entities import Metric
from core.interfaces import MetricsExporter

class JSONFileExporter(MetricsExporter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export(self, metrics: Dict[str, List[Metric]]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
        
        metrics_data = {
            name: [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "labels": m.labels
                }
                for m in metric_list
            ]
            for name, metric_list in metrics.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2) 