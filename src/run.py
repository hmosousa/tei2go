from argparse import Namespace
import logging
import logging.config
import os
from pathlib import Path
import warnings

import torch

from src.train import train_spacy, train_sparknlp
from src.cli import setup_parser
from src.utils import setup_seed, setup_directories
from src.data.preprocess import preprocess_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress tensorflow loggings in spacy
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore")  # ignore spacy warnings


def train(args: Namespace) -> None:
    """Train a model according to the arguments passed through the command line interface."""

    logger.info("-" * 100)
    logger.info("Start training for the following setup:")
    for arg, value in vars(args).items():
        logger.info(f"\t{arg}: {value}")

    preprocessed_data_path = args.data_path / "processed" / args.model / "_".join(args.data)
    if not preprocessed_data_path.exists():
        logger.info("Preprocessing data.")
        preprocess_data(args.data, args.data_path)
    else:
        logger.info("Found preprocessed data.")

    if args.model == "spacy":
        metrics = train_spacy(
            model_path=args.models_path,
            data_path=preprocessed_data_path,
            language=args.language,
            n_epochs=args.n_epochs,
            dropout=args.dropout,
            trans_hidden_width=args.trans_hidden_width,
            trans_maxout_pieces=args.trans_maxout_pieces,
            trans_use_upper=args.trans_use_upper,
            embed_width=args.embed_width,
            embed_depth=args.embed_depth,
            embed_size=args.embed_size,
            embed_window_size=args.embed_window_size,
            embed_maxout_pieces=args.embed_maxout_pieces,
            embed_subword_features=args.embed_subword_features,
        )

    elif args.model == "sparknlp":
        metrics = train_sparknlp(
            model_path=args.models_path,
            embeddings_path=args.resources_path / args.embeddings,
            data_path=preprocessed_data_path,
            language=args.language,
            n_epochs=args.n_epochs,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logs_path=args.logs_path,
            gpu=args.gpu
        )

    else:
        msg = f"Invalid model type {args.model}. Model type should be 'spacy' or 'sparknlp'."
        logger.warning(msg)
        return

    logger.info("Training done.")
    logger.info(f"Best validation f1: {max(metrics['validation']['f1']):.4f}")
    logger.info("-" * 100)


def setup_logger_file_handler(args: Namespace) -> None:
    """Adds a file handler to the module logger."""
    if len(args.data) == 1 and args.data[0].startswith("ph_"):  # only one model
        log_path = args.logs_path / f"professor_heideltime/{args.model}_{args.language}.log"
    elif len(args.data) == 1:  # base corpus
        log_path = args.logs_path / f"base/{args.model}_{args.language}.log"
    elif "ph_" in " ".join(args.data):  # use professor HeidelTime corpus
        log_path = args.logs_path / f"weak_label/{args.model}_{args.language}.log"
    else:
        log_path = args.logs_path / f"compilation/{args.model}_{args.language}.log"

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)


def main(root_path: Path) -> None:
    seed = 73  # ensure reproducibility
    setup_seed(seed)

    parser = setup_parser()
    args = parser.parse_args()
    args.models_path = root_path / "models"
    args.resources_path = root_path / "resources"
    args.logs_path = root_path / "logs"
    args.data_path = root_path / "data"
    paths = [
        args.models_path,
        args.resources_path,
        args.logs_path,
        args.data_path / "raw",
        args.data_path / "processed",
    ]
    setup_directories(paths)

    gpu = torch.cuda.is_available()
    args.gpu = gpu

    logging.config.fileConfig(root_path / "logging.conf", disable_existing_loggers=False)
    setup_logger_file_handler(args)

    train(args)


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    main(root_path)
