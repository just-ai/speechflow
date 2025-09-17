import typing as tp

from pathlib import Path

from tqdm import tqdm

from speechflow.utils.plotting import plot_durations_and_signals, plot_tensor
from speechflow.utils.profiler import Profiler
from speechflow.utils.seed import get_seed
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
    TTSOptions,
)
from tts.acoustic_models.interface.prosody_reference import REFERENECE_TYPE
from tts.acoustic_models.modules.common.length_regulators import SoftLengthRegulator
from tts.vocoders.eval_interface import VocoderEvaluationInterface, VocoderOptions


def plotting(tts_in: TTSForwardInput, tts_out: TTSForwardOutput, doc, signals=("pitch",)):
    try:
        dura = tts_out.variance_predictions["durations"]
        max_len = tts_out.spectrogram[0].shape[0]

        if tts_out.spectrogram.shape[0] == 1:
            lr = SoftLengthRegulator(sigma=999999)

            signal = {}
            for name in signals:
                if name not in tts_out.variance_predictions:
                    continue

                signal[name] = tts_out.variance_predictions[name][0]

                name = f"aggregate_{name}"
                if name in tts_out.variance_predictions:
                    val = tts_out.variance_predictions[name]
                    val, _ = lr(val.unsqueeze(-1), dura, max_len)
                    signal[name] = val[0, :, 0]

            val = tts_in.ling_feat.breath_mask * (-1)
            val, _ = lr(val.unsqueeze(-1), dura, max_len)
            signal["breath_mask"] = val[0, :, 0]

            plot_durations_and_signals(
                tts_out.spectrogram[0],
                dura[0],
                doc.sents[0].get_phonemes(as_tuple=True),
                signal,
            )
        else:
            plot_tensor(tts_out.spectrogram)
    except Exception as e:
        print(e)
    finally:
        Profiler.sleep(1)


def synthesize(
    tts_interface: TTSEvaluationInterface,
    voc_interface: tp.Optional[VocoderEvaluationInterface],
    text: tp.Union[str, Path],
    lang: str,
    speaker_name: tp.Optional[tp.Union[str, tp.Dict[str, str]]] = None,
    speaker_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    style_reference: tp.Optional[
        tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
    ] = None,
    tts_opt: TTSOptions = TTSOptions(),
    voc_opt: VocoderOptions = VocoderOptions(),
    seed: int = 0,
    use_profiler: bool = False,
):
    tts_ctx = TTSContext.create(
        lang, speaker_name, speaker_reference, style_reference, seed
    )
    tts_ctx = tts_interface.prepare_embeddings(tts_ctx, opt=tts_opt)

    doc = tts_interface.prepare_text(text, lang, opt=tts_opt)
    doc = tts_interface.predict_prosody_by_text(doc, tts_ctx, opt=tts_opt)

    # for sent in doc.sents:
    #     for t in sent.tokens:
    #         print(f"{t.text}: {t.prosody}")

    tts_in = tts_interface.prepare_batch(
        tts_interface.split_sentences(doc)[0],
        tts_ctx,
        opt=tts_opt,
    )

    with Profiler(enable=use_profiler):
        tts_out = tts_interface.evaluate(tts_in, tts_ctx, opt=tts_opt)

        if voc_interface is not None:
            voc_out = voc_interface.synthesize(
                tts_in,
                tts_out,
                lang=tts_ctx.prosody_reference.default.lang,
                speaker_name=tts_ctx.prosody_reference.default.speaker_name,
                opt=voc_opt,
            )
        else:
            voc_out = None

    # plotting(tts_in, tts_out, doc)
    return voc_out.audio_chunk


if __name__ == "__main__":
    tts_model_path = "/path/to/checkpoint"
    voc_model_path = "/path/to/checkpoint"  # for E2E TTS the same as tts_model_path
    prosody_model_path = (
        "/path/to/checkpoint"  # text-based prosody prediction model [optional]
    )
    device = "cpu"

    tts = TTSEvaluationInterface(
        tts_ckpt_path=tts_model_path,
        prosody_ckpt_path=prosody_model_path,
        device=device,
    )
    voc = VocoderEvaluationInterface(
        ckpt_path=voc_model_path,
        device=device,
    )

    print(tts.get_languages())
    print(tts.get_speakers())

    opt = TTSOptions()
    # if prosody_model_path is not None:
    #     opt.down_contur_ids = ["2", "3", "5"]
    #     opt.up_contur_ids = ["4"]

    tests = [
        {
            "lang": "RU",
            "speaker_name": "Natasha",
            "style_reference": "/path/to/reference_audio",
            "utterances": """
                В городе было пасмурно и серо. Снег начался после полудня и шел в течение нескольких часов.
                На дорогах и тротуарах образовался плотный снежный покров.
                Некоторые горожане напоследок решили не лишать себя зимних забав и слепили снеговика.
            """,
        },
    ]

    for i in tqdm(range(10), desc="Generating TTS samples"):
        for idx, test in enumerate(tests):
            audio_chunk = synthesize(
                tts,
                voc,
                test["utterances"],
                test["lang"],
                speaker_name=test["speaker_name"],
                speaker_reference=test["style_reference"],
                style_reference=test["style_reference"],
                tts_opt=opt,
                seed=get_seed(),
            )
            audio_chunk.save(
                f"tts_result_{test['speaker_name']}_test{idx}_v{i}.wav",
                overwrite=True,
            )
