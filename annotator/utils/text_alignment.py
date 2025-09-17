import typing as tp
import logging

from copy import deepcopy as copy
from dataclasses import dataclass

from multilingual_text_parser.data_types import Doc, Sentence, Token
from rapidfuzz import fuzz

from annotator.utils.fuzzy_sequence_matcher import best_matches

__all__ = ["TokenSeq", "TextAlignment"]

LOGGER = logging.getLogger("root")


@dataclass
class Chunk:
    tokens: tp.List[Token]
    keyword: str
    keyword_pos: int


class TokenSeq:
    def __init__(
        self,
        sent: Sentence,
        min_token_size: int = 1,
    ):
        self.sent = sent

        self.chunks: tp.List = []
        tokens_by_word = self.sent.get_words_with_punct()
        words = self.sent.get_words()
        for word, tokens in zip(words, tokens_by_word):
            is_bound_item = word == words[0] or word == words[-1]

            if (word.norm and len(word.norm) >= min_token_size) or is_bound_item:
                chunk = Chunk(tokens, word.norm, tokens.index(word))
                self.chunks.append(chunk)
            else:
                self.chunks[-1].tokens += tokens

    def __getitem__(self, keyword_index: int) -> str:
        return self.chunks[keyword_index].keyword

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def num_symbs(self) -> int:
        return len(self.sent.text)

    @property
    def num_words(self) -> int:
        return sum([len(chunk.tokens) for chunk in self.chunks])

    @property
    def num_keywords(self) -> int:
        return len(self.chunks)

    def get_keywords(self) -> tp.List[str]:
        return [chunk.keyword for chunk in self.chunks]

    def remove_keyword(self, keyword_index: int):
        if keyword_index < 0:
            keyword_index = len(self.chunks) + keyword_index
        assert 0 <= keyword_index < len(self.chunks), "invalid keyword index!"
        idx = keyword_index

        if len(self.chunks) == 1:
            del self.chunks[idx]
        elif idx + 1 < len(self.chunks):
            mod_chunk = self.chunks[idx + 1]
            mod_chunk.tokens = self.chunks[idx].tokens + mod_chunk.tokens
            mod_chunk.keyword_pos += len(self.chunks[idx].tokens)
            del self.chunks[idx]
        else:
            mod_chunk = self.chunks[idx - 1]
            mod_chunk.tokens = mod_chunk.tokens + self.chunks[idx].tokens
            del self.chunks[idx]

    def get_chunk(self, keyword_index: int) -> list:
        return self.chunks[keyword_index].tokens

    def get_keyword_idx_by_token(self, token_index: int) -> int:
        assert 0 <= token_index < self.num_words, "invalid word index!"
        for idx, chunk in enumerate(self.chunks):
            keyword_pos = self.get_keyword_pos(idx)
            num_next_words = len(chunk.tokens) - chunk.keyword_pos
            if keyword_pos + num_next_words > token_index:
                break
        return idx

    def get_keyword_pos(self, keyword_index: int) -> int:
        idx = keyword_index
        return (
            sum([len(chunk.tokens) for chunk in self.chunks[:idx]])
            + self.chunks[idx].keyword_pos
        )

    def startswith_keyword(self) -> bool:
        begin_chunk = self.get_chunk(0)
        return begin_chunk[0].norm == self[0]

    def endswith_keyword(self) -> bool:
        end_chunk = self.get_chunk(-1)
        return end_chunk[-1].norm == self[-1]


class TextAlignment:
    def __init__(
        self,
        target_seq: TokenSeq,
        sent_thr: float = 90.0,
        word_thr: float = 75.0,
        max_mismatched_keywords: float = 30.0,
        max_dist_between_tokens: tp.Optional[int] = None,
    ):
        self._target_seq = target_seq
        self._sent_thr = sent_thr
        self._word_thr = word_thr
        self._max_mismatched_keywords = max_mismatched_keywords / 100.0
        self._max_dist_between_tokens = max_dist_between_tokens
        self._roi: tp.Optional[tp.Tuple[int, int]] = None

    def set_roi(self, begin: tp.Optional[int] = None, end: tp.Optional[int] = None):
        begin = begin if begin else 0
        begin = self._target_seq.get_keyword_idx_by_token(begin)
        end = end if end else self._target_seq.num_words - 1
        end = self._target_seq.get_keyword_idx_by_token(end)
        self._roi = (begin, end)

    def _fuzzy_matching(
        self,
        targer_seq: TokenSeq,
        query_seq: TokenSeq,
    ) -> tp.Tuple:
        begin, end = 0, self._target_seq.num_keywords
        if self._roi:
            begin, end = self._roi

        targer = targer_seq.get_keywords()[begin : end + 1]
        query = query_seq.get_keywords()

        align = best_matches(
            targer,
            query,
            scorer=lambda u, v: fuzz.ratio(u, v),
            max_dist_between_tokens=self._max_dist_between_tokens,
        )

        t, q = zip(*align)
        t_tokens, t_idx = zip(*t)
        q_tokens, q_idx = zip(*q)

        t_idx = tuple(idx + begin for idx in t_idx)
        return t_tokens, t_idx, q_tokens, q_idx

    def _alignment(
        self, targer_seq: TokenSeq, query_seq: TokenSeq, num_reference_words: int = 2
    ) -> tp.Tuple[bool, tp.List[int], tp.List[int], tp.Optional[TokenSeq]]:
        assert not (targer_seq.is_empty and query_seq.is_empty), "empty sequence!"
        max_del_keywords = int(query_seq.num_keywords * self._max_mismatched_keywords) + 1

        is_match = False
        n = num_reference_words
        for _ in range(max_del_keywords):
            sent_score = 0
            try:
                ret = self._fuzzy_matching(targer_seq, query_seq)
                t_tokens, t_tokens_idx, q_tokens, q_tokens_idx = ret

                assert t_tokens_idx, "match list is empty!"

                sent_score = fuzz.ratio("".join(t_tokens), "".join(q_tokens))
                assert sent_score >= self._sent_thr, "sentences is not matches!"

                word_score = fuzz.ratio("".join(t_tokens[:n]), "".join(q_tokens[:n]))
                assert word_score >= self._word_thr, "begin words is not matches!"

                word_score = fuzz.ratio("".join(t_tokens[-n:]), "".join(q_tokens[-n:]))
                assert word_score >= self._word_thr, "end words is not matches!"

                assert query_seq.num_keywords == len(q_tokens_idx), "invalid match!"

                is_match = True
                break

            except Exception:
                if query_seq.num_keywords != len(q_tokens_idx):
                    for qt_id in range(query_seq.num_keywords):
                        if qt_id not in q_tokens_idx:
                            query_seq.remove_keyword(qt_id)
                            break

                elif sent_score >= self._sent_thr * 0.8:
                    for i, qt_id in enumerate(q_tokens_idx):
                        word_score = fuzz.ratio(t_tokens[i], q_tokens[i])
                        if word_score < self._word_thr:
                            query_seq.remove_keyword(qt_id)
                            break

                else:
                    break

        if not is_match:
            # logger.warning(f"query no match: {query_seq.sent.text}")
            return is_match, [], [], None

        t_idx = [targer_seq.get_keyword_pos(id) for id in t_tokens_idx]
        q_idx = [query_seq.get_keyword_pos(id) for id in q_tokens_idx]

        return is_match, t_idx, q_idx, query_seq

    def __call__(
        self, query_seq: TokenSeq
    ) -> tp.Tuple[bool, tp.List[int], tp.List[int], tp.Optional[TokenSeq]]:
        return self._alignment(self._target_seq, copy(query_seq))


if __name__ == "__main__":

    target_text = "Путин не исключил дополнительных мер поддержки россиян"
    query_text = "исключили поддержку"

    target_sent = Doc(target_text.lower(), sentenize=True, tokenize=True).sents[0]
    query_sent = Doc(query_text.lower(), sentenize=True, tokenize=True).sents[0]

    text_alignment = TextAlignment(TokenSeq(target_sent))
    print(text_alignment(TokenSeq(query_sent, min_token_size=3)))
