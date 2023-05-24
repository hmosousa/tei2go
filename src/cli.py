from argparse import ArgumentParser

from tieval.datasets import SUPPORTED_DATASETS


def setup_parser() -> ArgumentParser:
    """Setup command line interface parser."""

    description = "CLI to train the spacy and spark nlp models " \
                  "for temporal expression (timex) identification."
    parser = ArgumentParser(description=description)

    subparsers = parser.add_subparsers()

    spacy_parser = subparsers.add_parser(
        "spacy",
        help="Train spacy NER model."
    )
    _setup_spacy_parser(spacy_parser)

    sparknlp_parser = subparsers.add_parser(
        "sparknlp",
        help="Train spark NLP NER model."
    )
    _setup_sparknlp_parser(sparknlp_parser)

    return parser


def _setup_spacy_parser(parser: ArgumentParser):
    """Setup parser to train spacy model."""
    parser.add_argument("--model", type=str, default="spacy")
    parser.add_argument("--data", nargs="+", choices=SUPPORTED_DATASETS)
    parser.add_argument("--language", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning_rate", type=float)

    # transition base parser configuration
    parser.add_argument("--trans_hidden_width", type=int, default=64)
    parser.add_argument("--trans_maxout_pieces", type=int, default=2)
    parser.add_argument("--trans_use_upper", type=bool, default=True)

    # embeddings configuration
    parser.add_argument("--embed_width", type=int, default=96)
    parser.add_argument("--embed_depth", type=int, default=4)
    parser.add_argument("--embed_size", type=int, default=2000)
    parser.add_argument("--embed_window_size", type=int, default=1)
    parser.add_argument("--embed_maxout_pieces", type=int, default=3)
    parser.add_argument("--embed_subword_features", type=bool, default=True)


def _setup_sparknlp_parser(parser: ArgumentParser):
    """Setup parser to train spark nlp model."""
    parser.add_argument("--model", type=str, default="sparknlp")
    parser.add_argument("--data", nargs="+", choices=SUPPORTED_DATASETS)
    parser.add_argument("--language", type=str)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--embeddings", type=str)
