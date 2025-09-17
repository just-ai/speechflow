import numpy as np
import multilingual_text_parser

from annotator.audiobook_spliter import AudiobookSpliter
from speechflow.io import AudioSeg
from speechflow.utils.fs import get_root_dir
from speechflow.utils.versioning import version_check

TEST_DATA_DIR = get_root_dir() / "tests/data/audiobook_spliter"


def test_audiobook_spliter():
    audio_spliter = AudiobookSpliter(lang="RU")
    data = audio_spliter.read_datasamples(
        file_list=sorted(list((TEST_DATA_DIR / "src").glob("*.wav")))
    )

    assert len(data) == 2
    assert len(data.item(0)["segmentation"]) == 4
    assert len(data.item(1)["segmentation"]) == 5

    for i, item in enumerate(data):
        for j, sega in enumerate(item["segmentation"]):
            gt_sega_path = TEST_DATA_DIR / "segs" / f"{i}_{j}.TextGrid"
            gt_sega = AudioSeg.load(gt_sega_path, audio_path=item["audio_path"])

            version_check(multilingual_text_parser, gt_sega.meta["text_parser_version"])
            assert sega.audio_chunk.duration == gt_sega.audio_chunk.duration
            assert sega.duration == gt_sega.duration
            assert sega.sent.text == gt_sega.sent.text
            assert sega.sent.stress == gt_sega.sent.stress
            assert len(sega.sent.syntagmas) == len(gt_sega.sent.syntagmas)
            assert len(sega.sent.tokens) == len(gt_sega.sent.tokens)
            assert sega.sent.get_attr("text") == gt_sega.sent.get_attr("text")
            assert sega.sent.get_attr("norm") == gt_sega.sent.get_attr("norm")
            assert sega.sent.get_attr("stress") == gt_sega.sent.get_attr("stress")
            assert sega.sent.get_attr("emphasis") == gt_sega.sent.get_attr("emphasis")
            assert sega.sent.get_attr("pos") == gt_sega.sent.get_attr("pos")
            assert sega.sent.get_attr("phonemes") == gt_sega.sent.get_attr("phonemes")
            assert sega.sent.get_attr("head_id") == gt_sega.sent.get_attr("head_id")
            assert sega.sent.get_attr("rel") == gt_sega.sent.get_attr("rel")

            np.testing.assert_allclose(
                gt_sega.get_timestamps(relative=True)[0],
                sega.get_timestamps(relative=True)[0],
                rtol=1e-5,
            )
            for gt_ph_ts, ph_ts in zip(
                gt_sega.get_timestamps(relative=True)[1],
                sega.get_timestamps(relative=True)[1],
            ):
                np.testing.assert_allclose(gt_ph_ts, ph_ts, rtol=1e-5)
