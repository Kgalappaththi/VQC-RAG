import json
from datetime import datetime
from pathlib import Path


def save_experiment_log(log_data, output_path="results/experiment_log.json"):
    Path("results").mkdir(parents=True, exist_ok=True)

    log_data["timestamp"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Experiment log saved to {output_path}")