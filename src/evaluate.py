from functools import partialmethod
import json
import os
from pathlib import Path
import time
from typing import List, Union, Tuple, Dict, Any

import sparknlp
from tieval.evaluate import span_identification
from tieval.models import HeidelTime
from tqdm import tqdm

from src.models import SpacyModel, SparkNLPModel
from src.data.utils import read_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress tensorflow loggings in sparknlp
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # disable progress bar

LANGUAGE_ACRONYM = {
    "english": "en",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
    "french": "fr",
    "german": "de",
}

Model = Union[SpacyModel, SparkNLPModel]  # defined for type hint
ModelName = str


def load_models(
        spark,
        language: str,
        model_path: Path,
        embeddings_path: Path,
) -> List[Tuple[ModelName, Model]]:
    """Load models for evaluation."""

    models = []

    heideltime = HeidelTime(language)
    models += [(f"HeidelTime", heideltime)]

    if (model_path / "base" / "spacy" / language).exists():
        model_spacy_base = SpacyModel(LANGUAGE_ACRONYM[language.lower()])
        model_spacy_base.load(model_path / "base" / "spacy" / language)
        models += [(f"Spacy NER Base", model_spacy_base)]

    if (model_path / "compilation" / "spacy" / language).exists():
        model_spacy_compilation = SpacyModel(LANGUAGE_ACRONYM[language.lower()])
        model_spacy_compilation.load(model_path / "compilation" / "spacy" / language)
        models += [(f"Spacy NER Compilation", model_spacy_compilation)]

    if (model_path / "weak_label" / "spacy" / language).exists():
        model_spacy_weak = SpacyModel(LANGUAGE_ACRONYM[language.lower()])
        model_spacy_weak.load(model_path / "weak_label" / "spacy" / language)
        models += [(f"Spacy NER Weak Label", model_spacy_weak)]

    if (model_path / "base" / "sparknlp" / language).exists():
        model_spark_base = SparkNLPModel(spark, embeddings_path)
        model_spark_base.load(model_path / "base" / "sparknlp" / language)
        models += [(f"Spark NER Base", model_spark_base)]

    return models


def complete_evaluation(
        data_path: Path,
        models: List[Tuple[ModelName, Model]],
        corpora: List[str]
) -> Dict[str, Any]:
    """Print the micro F1-score of models on the corpora."""
    result = {}
    for corpus in corpora:

        print(f"Corpus {corpus}.")

        _, _, test_documents = read_dataset(corpus, data_path)
        test_texts = [doc.text for doc in test_documents]
        test_annotations = [
            [tmx.endpoints for tmx in doc.timexs if not tmx.is_dct]
            for doc in test_documents
        ]

        model_result = {}
        for model_name, model in models:
            if model_name == "HeidelTime" and corpus.startswith("ph_"):
                continue

            ts = time.time()
            test_predictions = model.predict(test_texts)
            te = time.time()
            strict_metrics = span_identification(test_annotations, test_predictions)
            relaxed_metrics = span_identification(test_annotations, test_predictions, strict=False)
            model_result[model_name] = {
                "strict": strict_metrics,
                "relaxed": relaxed_metrics,
                "time": te - ts
            }
            print(f"\t{model_name:<30} F1:\t{strict_metrics['micro']['f1']:.4f}\t"
                  f"({relaxed_metrics['micro']['f1']:.4f})\t"
                  f"Time: {(te - ts):.4f}s")
        result[corpus] = model_result
    return result


def main(root_path: Path):
    embeddings_path = root_path / "resources" / "glove_840B_300_xx"
    model_path = root_path / "models"
    data_path = root_path / "data" / "raw"
    result_path = root_path / "result"
    result_path.mkdir(exist_ok=True)

    spark = sparknlp.start()

    corpora = {
        "english": [
            "tempeval_3",
            "meantime_english",
            "tcr",
            "ancient_time_english",
            "wikiwars",
            "ph_english"
        ],
        "portuguese": [
            "timebankpt",
            "ph_portuguese"
        ],
        "spanish": [
            "spanish_timebank",
            "tempeval_2_spanish",
            "meantime_spanish",
            "traint3",
            "ancient_time_spanish",
            "ph_spanish"
        ],
        "italian": [
            "tempeval_2_italian",
            "meantime_italian",
            "narrative_container",
            "ancient_time_italian",
            "ph_italian"
        ],
        "french": [
            "fr_timebank",
            "tempeval_2_french",
            "ancient_time_french",
            "ph_french"
        ],
        "german": [
            "krauts",
            "wikiwars_de",
            "ancient_time_german",
            "ph_german"
        ],
    }

    for language, corpus in corpora.items():
        models = load_models(spark, language, model_path, embeddings_path)
        metrics = complete_evaluation(data_path, models, corpus)
        with open(result_path / f"{language}.json", "w") as fout:
            json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    main(root_path)
