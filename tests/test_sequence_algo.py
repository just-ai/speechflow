import pytest

from speechflow.data_pipeline.datasample_processors.algorithms.text_processing.sequence_algorithms import (
    expand_iterable,
    locate_replacements_from_alignment,
    needleman_wunsch,
)


@pytest.mark.parametrize(
    "array, lengths, expected",
    [
        ([1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 4]),
        [[3], [4], [3, 3, 3, 3]],
    ],
)
def test_expand_iterable(array, lengths, expected):
    expanded = expand_iterable(to_expand=array, expand_coefficients=lengths)
    assert expanded == expected


@pytest.mark.parametrize(
    "seq_a, seq_b, expected_spans",
    [
        ("Мама мыла раму".split(), "Мама красила раму".split(), {(1, 2): (1, 2)}),
        ("a b c d e".split(), "a b c1 c2 d e".split(), {(2, 3): (2, 4)}),
        ("a b c1 c2 d e".split(), "a b c d e".split(), {(2, 4): (2, 3)}),
        ("a1 a2 b c d e".split(), "a3 a4 b c d e".split(), {(0, 2): (0, 2)}),
        ("a1 a2 b c d e".split(), "a b c d e".split(), {(0, 2): (0, 1)}),
        ("a".split(), "a1 a2 a3".split(), {(0, 1): (0, 3)}),
        ("a1 a2 a3".split(), "a".split(), {(0, 3): (0, 1)}),
        ("a a c d".split(), "b b c d".split(), {(0, 2): (0, 2)}),
        ("a c d a".split(), "b c d b".split(), {(0, 1): (0, 1), (3, 4): (3, 4)}),
    ],
)
def test_locate_sub_spans(seq_a, seq_b, expected_spans):
    alignment = needleman_wunsch(seq_a, seq_b)
    replacments = locate_replacements_from_alignment(alignment, seq_a, seq_b)
    assert replacments == expected_spans
