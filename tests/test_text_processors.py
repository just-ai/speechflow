import random

import torch
import pytest

from multilingual_text_parser.data_types import Doc
from multilingual_text_parser.parser import EmptyTextError, TextParser

from speechflow.data_pipeline.datasample_processors.data_types import TextDataSample
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.utils.fs import get_root_dir
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.profiler import Profiler

BOOK_NAME = {
    "RU": ["Tolstoy_War_and_Peace_ru.txt"],
    "EN": ["Tolstoy_War_and_Peace_en.txt", "Three_Men_in_a_Boat_en.txt"],
    "DE": ["remarque_im_westen_nichts_neues_de.txt"],
    "ES": ["Don_Quijote_de_la_Mancha_es.txt"],
    "IT": ["Tolstoy_War_and_Peace_it.txt"],
    "FR-FR": ["Le-Comte-de-Monte-Cristo_fr.txt"],
    "PT": ["O-evangelho-segundo-Jesus-Cristo_pt.txt"],
    "PT-BR": ["The_Alchemist_Portuguese_ptbr.txt"],
    "KK": ["Kochevniki_kk.txt"],
}


@pytest.mark.parametrize(
    "lang", ["RU", "EN", "DE", "ES", "IT", "FR-FR", "PT", "PT-BR", "KK"]
)
@pytest.mark.parametrize("chunk_size_min, chunk_size_max", [(1, 10000)])
def test_parse_book(
    lang: str,
    chunk_size_min: int,
    chunk_size_max: int,
    max_test_time: int = 60,
    with_profiler: bool = False,
):
    log_file = f"temp/log_file_{lang}.txt"
    with LoggingServer.ctx(log_file=log_file) as logger:

        if torch.cuda.is_available():
            device = f"cuda:{get_freer_gpu(strict=False)}"
        else:
            device = "cpu"

        text_parser = TextParser(lang=lang, device=device, with_profiler=with_profiler)
        text_proc = TTSTextProcessor(lang=lang)

        for book in BOOK_NAME[lang]:
            book_path = get_root_dir() / "tests/data/texts" / book
            all_text = book_path.read_text(encoding="utf-8")

            total_len = len(all_text)
            chunk_size = chunk_size_min
            timer = Profiler()
            while len(all_text) > chunk_size:
                if max_test_time > 0 and timer.get_time() > max_test_time:
                    break

                try:
                    doc = Doc(all_text[:chunk_size])
                    doc = text_parser.process(doc)
                    for sent in doc.sents:
                        ds = TextDataSample(sent=sent)
                        text_proc.process(ds)

                        if sent.exception_messages:
                            msg_all = f"{sent.text}:\n"
                            for msg in sent.exception_messages:
                                msg_all += f"\t{msg}\n"
                            logger.error(
                                trace(
                                    TextParser,
                                    exception=msg,
                                    message=f"lang: {lang},\n\n{sent.text}: {msg}",
                                )
                            )

                except EmptyTextError as empt_er:
                    logger.error(f"lang: {lang},\n\nEMPTY: {empt_er}")
                except Exception as e:
                    curr_len = len(all_text)
                    logger.info(f"language: {lang}, text: {doc.text_orig} \n\n")
                    logger.info(
                        f"progress: {round(100.0 - curr_len / total_len * 100.0, 3)}"
                    )
                    logger.error(trace(TextParser, f"lang: {lang}\n\n {e}"))
                    raise e

                all_text = all_text[chunk_size:]
                chunk_size = random.randint(chunk_size_min, chunk_size_max)


if __name__ == "__main__":
    test_parse_book(
        lang="EN",
        chunk_size_min=1,
        chunk_size_max=10000,
        with_profiler=False,
        max_test_time=-1,
    )
