import pickle
import typing as tp

from pathlib import Path

import pytest

from numpy.linalg.linalg import norm

from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.io import AudioChunk
from speechflow.utils.fs import get_root_dir

tests_data_dir = get_root_dir() / "tests" / "data"
test_wav_path = tests_data_dir / "test_audio.wav"
resemblyzer_test_output = tests_data_dir / "biometric/resemblyzer_embedding_test.pkl"
speechbrain_test_output = tests_data_dir / "biometric/speechbrain_embedding_test.pkl"
wespeaker_test_output = tests_data_dir / "biometric/wespeaker_embedding_test.pkl"


@pytest.mark.parametrize(
    ("model_type", "path_to_wav", "precompute_path"),
    [
        ("resemblyzer", test_wav_path, resemblyzer_test_output),
        ("speechbrain", test_wav_path, speechbrain_test_output),
        ("wespeaker", test_wav_path, wespeaker_test_output),
    ],
)
def test_bio_processor(
    model_type: str, path_to_wav: tp.Union[Path, str], precompute_path: Path
):
    proc = VoiceBiometricProcessor(model_type, fast_resample=False)
    ds = AudioDataSample(audio_chunk=AudioChunk(file_path=path_to_wav))
    ds.audio_chunk.load()
    ds = proc.process(ds)

    assert len(ds.speaker_emb.shape) == 1
    assert ds.speaker_emb.shape[0] == proc.embedding_dim

    test_embedding = pickle.loads(precompute_path.read_bytes())
    assert norm(test_embedding - ds.speaker_emb) < 1.0e-3
