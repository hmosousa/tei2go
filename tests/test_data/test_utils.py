from tieval.base import Document

from src.data.utils import (
    read_dataset,
    read_datasets,
    split_train_val
)


def test_read_dataset(tmp_path):
    """Assert that the read_dataset function always returns de same partition."""
    train0, val0, test0 = read_dataset("tempeval_3", tmp_path)
    train1, val1, test1 = read_dataset("tempeval_3", tmp_path)
    assert [d.name for d in train0] == [d.name for d in train1]
    assert [d.name for d in val0] == [d.name for d in val1]
    assert [d.name for d in test0] == [d.name for d in test1]


def test_read_datasets(tmp_path):
    train, val, test = read_datasets(["tempeval_3", "meantime_english"], tmp_path)

    assert isinstance(train, list)
    assert isinstance(train[0], Document)

    assert isinstance(val, list)
    assert isinstance(val[0], Document)

    assert isinstance(test, list)
    assert isinstance(test[0], Document)


def test_split_train_val_docs():
    """Ensure that the dataset spliter returns the same
    partition with when the seed is set (which is by default)."""
    for _ in range(10):
        s1 = split_train_val([1, 2, 3, 4, 5])
        s2 = split_train_val([1, 2, 3, 4, 5])
        assert s1 == s2
