from src.data.preprocess import (
    Span,
    _remove_overlapping_timexs
)


def test_span():
    spans = [Span(0, 10), Span(20, 30)]
    assert Span(13, 16) not in spans
    assert Span(3, 13) in spans
    assert Span(16, 23) in spans
    assert Span(0, 10) in spans
    assert Span(0, 30) in spans


def test_remove_overlapping_timexs():
    timexs = [
        (90, 'from 2009 to 2011'),
        (95, '2009'),
        (103, '2011'),
        (114, 'between 2014 and 2016'),
        (122, '2014'),
        (131, '2016')
    ]

    result = _remove_overlapping_timexs(timexs)
    expected_result = [(90, 'from 2009 to 2011'), (114, 'between 2014 and 2016')]
    assert result == expected_result
