import json
from datetime import datetime
from pathlib import Path

def save_experiment_log(log_data, path="logs/experiment_log.json"):
    Path("logs").mkdir(exist_ok=True)

    log_data["timestamp"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(log_data, f, indent=4)