from multilingual_text_parser.data_types import Doc

from annotator.utils import TextAlignment, TokenSeq


def test_text_alignment():
    target_text = "Германия сохраняет интерес к улучшению отношений между Россией"
    query_text = "интерес И отношений, Россия!"

    target_sent = Doc(target_text.lower(), True, True).sents[0]
    query_sent = Doc(query_text.lower(), True, True).sents[0]

    text_alignment = TextAlignment(TokenSeq(target_sent))
    is_match, t_idx, q_idx, q_seq = text_alignment(TokenSeq(query_sent, min_token_size=3))
    assert (t_idx, q_idx) == ([2, 5, 7], [0, 2, 4])

    text_alignment.set_roi(begin=2, end=7)
    is_match, t_idx, q_idx, q_seq = text_alignment(TokenSeq(query_sent, min_token_size=3))
    assert (t_idx, q_idx) == ([2, 5, 7], [0, 2, 4])
