import torch

from multilingual_text_parser.data_types import Doc, Sentence

from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.tts_processors import (
    add_pauses_from_text,
)
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.utils.profiler import Profiler

__all__ = ["add_prosody_modifiers"]


def _get_prosody_modifier_vector(
    sent: Sentence, modifier_attribute: str, service_tokens_added: bool = False
) -> torch.Tensor:

    phonemes_by_token = sent.get_phonemes()
    tokens = [t for t in sent.tokens if not t.is_punctuation]

    modifier_vector = []
    for word, token in zip(phonemes_by_token, tokens):
        modifiers = token.modifiers if token.modifiers is not None else {}
        prosody_modifiers = modifiers.get("prosody", {})
        this_modifier = prosody_modifiers.get(modifier_attribute)
        if this_modifier is not None and this_modifier.isdigit():
            this_modifier = max(50, min(200, int(this_modifier)))
            modifier_vector.extend([this_modifier] * len(word))
        else:
            modifier_vector.extend([100] * len(word))

    if service_tokens_added:
        modifier_vector.insert(0, 100)
        modifier_vector.append(100)

    modifier_vector = [x / 100 for x in modifier_vector]
    return torch.tensor(modifier_vector)


@PipeRegistry.registry(
    inputs={"sent"}, outputs={"temp_modifier", "volume_modifier", "pitch_modifier"}
)
def add_prosody_modifiers(
    ds: TTSDataSample, max_value: float = 2.0, min_value: float = 0.1
):
    sent = ds.sent
    service_tokens_added = False

    temp = _get_prosody_modifier_vector(
        sent=sent, modifier_attribute="rate", service_tokens_added=service_tokens_added
    )
    temp = torch.clip(temp, min=min_value, max=max_value)
    ds.temp_modifier = 1 / temp

    burr = _get_prosody_modifier_vector(
        sent=sent, modifier_attribute="burr", service_tokens_added=service_tokens_added
    )
    burr = torch.clip(burr, min=min_value, max=max_value)

    ph_idx = 0
    for t in sent.tokens:
        if t.modifiers is None:
            prosody_tag = {}
        else:
            prosody_tag = t.modifiers.get("prosody", {})

        if "burr" in prosody_tag:
            for idx, ph in enumerate(t.phonemes):
                if "phoneme" not in prosody_tag:
                    if ph not in ["R", "R0", "P", "P0", "K", "K0"]:
                        burr[ph_idx + idx] = 1.0
                else:
                    if ph not in prosody_tag["phoneme"]:
                        burr[ph_idx + idx] = 1.0

        ph_idx += t.num_phonemes

    ds.temp_modifier *= 1 / burr

    volume = _get_prosody_modifier_vector(
        sent=sent,
        modifier_attribute="volume",
        service_tokens_added=service_tokens_added,
    )
    ds.volume_modifier = torch.clip(volume, min=min_value, max=max_value)

    pitch = _get_prosody_modifier_vector(
        sent=sent, modifier_attribute="pitch", service_tokens_added=service_tokens_added
    )
    ds.pitch_modifier = torch.clip(pitch, min=min_value, max=max_value)

    return ds


if __name__ == "__main__":
    from multilingual_text_parser.parser import TextParser

    text = """
    <prosody pitch="120">Заброшенный Невский</prosody> рынок <prosody volume="150">могут,
    снести</prosody> ради строительства моста через Неву.
    """

    text_parser = TextParser(lang="RU")
    text_processor = TTSTextProcessor(lang="RU", add_service_tokens=True)

    sentence = text_parser.process(Doc(text)).sents[0]

    _ds = TTSDataSample(sent=sentence)
    _ds = add_pauses_from_text(_ds)
    _ds = text_processor.process(_ds)

    with Profiler():
        _ds = add_prosody_modifiers(_ds)

    for x, p, v, t in zip(
        _ds.transcription_text, _ds.pitch_modifier, _ds.volume_modifier, _ds.temp_modifier
    ):
        print(x, f"pitch: {p}", f"volume: {v}", f"temp: {t}")
