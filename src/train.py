import logging
from pathlib import Path
from typing import Dict

import sparknlp

from src.data import SparkNLPDataLoader, SpacyDataLoader
from src.models import (
    SparkNLPModel,
    SpacyModel,
    TransitionParserConfig,
    EmbedConfig
)

logger = logging.getLogger(__name__)


def train_spacy(
        model_path: Path,
        data_path: Path,
        language: str,
        n_epochs: int,
        dropout: float,
        trans_hidden_width,
        trans_maxout_pieces,
        trans_use_upper,
        embed_width,
        embed_depth,
        embed_size,
        embed_window_size,
        embed_maxout_pieces,
        embed_subword_features,

) -> Dict:
    """Train a spacy NER model to identify timexs."""
    train_dataloader = SpacyDataLoader(data_path / "train.json")
    val_dataloader = SpacyDataLoader(data_path / "validation.json")

    logger.info("Corpus statistics:")
    logger.info("\tTrain:")
    logger.info(f"\t\t# sentences: {len(train_dataloader.data)}")
    logger.info(f"\t\t# timexs: {len(train_dataloader)}")
    logger.info("\tValidation:")
    logger.info(f"\t\t# sentences: {len(val_dataloader.data)}")
    logger.info(f"\t\t# timexs: {len(val_dataloader)}")

    parser_config = TransitionParserConfig(
        hidden_width=trans_hidden_width,
        maxout_pieces=trans_maxout_pieces,
        use_upper=trans_use_upper,
    )
    embed_config = EmbedConfig(
        width=embed_width,
        depth=embed_depth,
        embed_size=embed_size,
        window_size=embed_window_size,
        maxout_pieces=embed_maxout_pieces,
        subword_features=embed_subword_features,
    )

    logger.info(f"Initializing model for language {language}.")
    model = SpacyModel(
        language=language,
        parser_config=parser_config,
        embed_config=embed_config
    )

    metrics = model.fit(
        train_data=train_dataloader,
        validation_data=val_dataloader,
        n_epochs=n_epochs,
        dropout=dropout,
        model_path=model_path
    )

    return metrics


def train_sparknlp(
        data_path: Path,
        embeddings_path: Path,
        language: str,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        dropout: float,
        model_path: Path,
        logs_path: Path,
        gpu: bool
) -> Dict:
    """Train a sparknlp NERDLApproach model to identify timexs."""
    logger.info(f"Starting spark session with GPU set to {gpu}.")
    spark = sparknlp.start(gpu)

    train_dataloader = SparkNLPDataLoader(data_path / "train", spark)

    logger.info(f"Initializing model for language {language}.")
    model = SparkNLPModel(
        spark=spark,
        embeddings_path=embeddings_path
    )

    (logs_path / "sparknlp").mkdir(exist_ok=True)

    metrics = model.fit(
        train_data=train_dataloader,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        model_path=model_path,
        logs_path=logs_path / "sparknlp",
    )

    return metrics
