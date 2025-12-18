"""
Model Monitoring Service
Checks for data drift using Evidently AI.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

from src.config import MODELS_DIR

# Configuration
MONITORING_DIR = Path("monitoring")
MONITORING_DIR.mkdir(exist_ok=True)
REFERENCE_DATA_PATH = MONITORING_DIR / "reference_data.csv"
CURRENT_DATA_PATH = MONITORING_DIR / "current_data.csv"
REPORTS_DIR = MONITORING_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def load_data():
    """Load reference and current data."""
    if not REFERENCE_DATA_PATH.exists():
        print("Reference data not found. Run training first or extract from training set.")
        return None, None
        
    if not CURRENT_DATA_PATH.exists():
        print("No current production data to analyze.")
        return None, None
        
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)
    
    return reference, current

def run_drift_detection():
    """Run data drift detection."""
    print(f"Running drift detection at {datetime.now()}...")
    
    reference, current = load_data()
    if reference is None or current is None:
        return
    
    if len(current) < 10:
        print("Not enough production data to run drift detection (min 10 samples needed).")
        return

    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric()
    ])
    
    report.run(reference_data=reference, current_data=current)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.json"
    report.save_json(str(report_path))
    
    # Check result
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    dataset_drift = result['metrics'][1]['result']['dataset_drift']
    
    print(f"Report saved to {report_path}")
    print(f"Drift Share: {drift_share:.2%}")
    
    if dataset_drift:
        print("WARNING: Dataset Drift DETECTED!")
    else:
        print("Dataset is stable.")

if __name__ == "__main__":
    run_drift_detection()
