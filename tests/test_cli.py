from src.cli import setup_parser


def test_setup_parser_spacy():
    parser = setup_parser()

    cmd = "spacy  " \
          "--data tempeval_3 meantime_english " \
          "--language en " \
          "--n_epochs 30 " \
          "--dropout 0.0 " \
          "--trans_hidden_width 64 " \
          "--trans_maxout_pieces 2 " \
          "--trans_use_upper True " \
          "--embed_width 96 " \
          "--embed_depth 4 " \
          "--embed_size 2000 " \
          "--embed_window_size 1 " \
          "--embed_maxout_pieces 3 " \
          "--embed_subword_features True"

    args = parser.parse_args(cmd.split())
    assert args.data == ["tempeval_3", "meantime_english"]
    assert args.model == "spacy"
