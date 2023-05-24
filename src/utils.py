import logging
from datetime import datetime
from pathlib import Path
import random
import re
from typing import List, Tuple, Dict, Any

import spacy

logger = logging.getLogger(__name__)


def split_train_val(
        documents: List,
        split: float = 0.8,
        seed: int = 73
) -> Tuple[List, List]:
    """Splits a list of documents for training and validation by the ratio provided in split."""
    n_train = round(split * len(documents))
    
    n_val = len(documents) - n_train
    if n_val == 0:
        n_train -= 1
    random.seed(seed)
    random.shuffle(documents)
    return documents[:n_train], documents[n_train:]


def parse_logs_file(logs: str) -> List[Dict[str, Any]]:
    """Parse the log file that is created during training."""
    results = []
    experiments = logs.split("Start training for the following setup:")[1:]
    for experiment in experiments:
        results.append({
            "model": re.findall(r"model: (\w*)", experiment)[0],
            "language": re.findall(r"language: (\w*)", experiment)[0],
            "model_id": re.findall(r"Started train loop with id (\d*)", experiment)[0],
            "best_f1": re.findall(r"Best validation f1: (0.\d*)", experiment)[0],
        })

    return results


def setup_seed(seed: int) -> None:
    """Set the necessary seeds so that the code is reproducible."""
    random.seed(seed)
    spacy.util.fix_random_seed(seed)


def setup_directories(paths: List[Path]) -> None:
    """Ensure that the necessary directories for the program to run exist."""
    for path in paths:
        path.mkdir(exist_ok=True, parents=True)


def parse_sparknlp_log(path: str):
    """Parse the metrics in the log file produce by training a spark nlp model."""

    with open(path) as f_log:
        content = f_log.read()

    logger.info(content)

    micro_metrics_pattern = r"Micro-average\t prec: (\d*.\d*), rec: (\d*.\d*), f1: (\d*.\d*)\n"
    micro_metrics_re = re.compile(micro_metrics_pattern)
    micro_metrics = micro_metrics_re.findall(content)
    f1 = [float(f1) for p, r, f1 in micro_metrics]

    loss_re = re.compile(r"loss: (\d*.\d*)")
    losses = [float(loss) if loss.replace(".", "").isnumeric() else 0.0
              for loss in loss_re.findall(content)]

    metrics = {
        "train": {
            "loss": losses,
        },
        "validation": {
            "f1": f1
        }
    }

    return metrics


def generate_id() -> str:
    """Generates an id based on the current time."""
    return datetime.now().strftime("%Y%m%d%H%M%S")
