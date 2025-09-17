"""
example:
    streamlit run tts_demo.py -- -tts=/path/to/checkpoint -voc=/path/to/checkpoint -ref=/path/to/audio_reference -d=cuda:0

"""

import sys
import typing as tp
import argparse

from pathlib import Path

import torch
import streamlit as st

from annotated_text import annotated_text

try:
    THIS_PATH = Path(__file__).absolute()
    ROOT = THIS_PATH.parents[2]
    sys.path.append(ROOT.as_posix())
except Exception as e:
    print(e)

from dataclasses import dataclass

from multilingual_text_parser.data_types import Doc

from speechflow.io import AudioChunk, AudioFormat
from speechflow.utils.gpu_info import get_freer_gpu
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
    TTSOptions,
)
from tts.vocoders.eval_interface import VocoderEvaluationInterface, VocoderOptions


@dataclass
class EvaluationInterface:
    tts_interface: TTSEvaluationInterface
    tts_opt: TTSOptions
    voc_interface: VocoderEvaluationInterface
    voc_opt: VocoderOptions

    def synthesize(
        self,
        text: str,
        lang: str,
        speaker_name: str,
        style_reference: Path,
    ) -> tp.Tuple[AudioChunk, tp.List]:
        tts_ctx = TTSContext.create(lang, speaker_name, style_reference=style_reference)
        tts_ctx = self.tts_interface.prepare_embeddings(tts_ctx, opt=self.tts_opt)

        doc = self.tts_interface.prepare_text(text, lang, opt=self.tts_opt)
        doc = self.tts_interface.predict_prosody_by_text(doc, tts_ctx, opt=self.tts_opt)

        tts_in = self.tts_interface.prepare_batch(
            self.tts_interface.split_sentences(doc)[0],
            tts_ctx,
            opt=self.tts_opt,
        )

        tts_out = self.tts_interface.evaluate(tts_in, tts_ctx, opt=self.tts_opt)
        voc_out = self.voc_interface.synthesize(
            tts_in,
            tts_out,
            lang=tts_ctx.prosody_reference.default.lang,
            speaker_name=tts_ctx.prosody_reference.default.speaker_name,
            opt=self.voc_opt,
        )

        colors = ["#8ea", "#faa", "#afa", "#fea", "#8ef", "#afe", "#faf", "#eaf"]

        utterances = []
        for sent in doc.sents:
            for t in sent.tokens:
                if t.is_service:
                    continue
                if t.prosody and t.prosody == "-1":
                    utterances.append(f"{t.text} ")
                else:
                    utterances.append((f"{t.text} ", t.prosody, colors[int(t.prosody)]))

        return voc_out.audio_chunk, utterances


def parse_args(sys_args):
    parser = argparse.ArgumentParser("Experiment Viewer")
    parser.add_argument(
        "-tts",
        "--tts_path",
        help="Path to tts checkpoint or experiment folder",
        type=Path,
    )
    parser.add_argument(
        "-voc",
        "--vocoder_path",
        help="Path to vocoder checkpoint",
        type=Path,
    )
    parser.add_argument(
        "-prosody",
        "--prosody_path",
        help="Path to prosody predictor checkpoint or experiment folder",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-ref",
        "--style_reference",
        help="Path to audio for style transfer",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to process on",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=5000,
        help="Maximum text length for synthesis",
    )
    parser.add_argument(
        "--demo_mode",
        type=bool,
        default=False,
        help="Enable demo mode",
    )

    try:
        return parser.parse_known_args(sys_args)[0]
    except SystemExit:
        st.error("This page is unavailable, check your configuration!")
        st.stop()


def prepare_input_args() -> tp.Tuple[Path, Path, Path, Path, str, int]:
    args = parse_args(sys.argv[1:])

    if args.tts_path.is_dir():
        experiment_path = Path(
            st.sidebar.text_input(
                "Path to the experiment folder", value=args.tts_path.as_posix()
            )
        )

        files = []
        for path in experiment_path.rglob("*.ckpt"):
            files.append(path.relative_to(experiment_path))

        if not files:
            st.sidebar.markdown("TTS models not found!")
            st.stop()
        else:
            files = sorted(files, key=lambda f: (experiment_path / f).stat().st_mtime)

        tts_path = Path(
            st.sidebar.selectbox(
                "List of TTS checkpoints", options=files, index=len(files) - 1
            )
        )
        tts_path = Path(
            st.sidebar.text_input(
                "Path to acoustic model",
                value=(experiment_path / tts_path).as_posix(),
                disabled=True,
            )
        )
    else:
        if not args.demo_mode:
            tts_path = Path(
                st.sidebar.text_input(
                    "Path to acoustic model",
                    value=args.tts_path.as_posix(),
                    disabled=True,
                )
            )
        else:
            tts_path = args.tts_path

    if not args.demo_mode:
        voc_path = Path(
            st.sidebar.text_input(
                "Path to vocoder model",
                value=args.vocoder_path.as_posix(),
                disabled=True,
            )
        )
    else:
        voc_path = args.vocoder_path

    if args.prosody_path:
        if not args.demo_mode:
            prosody_path = Path(
                st.sidebar.text_input(
                    "Path to prosody model",
                    value=args.prosody_path.as_posix(),
                    disabled=True,
                )
            )
        else:
            prosody_path = args.prosody_path
    else:
        prosody_path = None

    if args.style_reference:
        if not args.demo_mode:
            style_reference = Path(
                st.sidebar.text_input(
                    "Path to style reference",
                    value=args.style_reference.as_posix(),
                    disabled=True,
                )
            )
        else:
            style_reference = args.style_reference
    else:
        style_reference = None

    if not tts_path.exists() or not voc_path.exists():
        st.sidebar.markdown("Models not found!")
        st.stop()

    if not torch.cuda.is_available():
        device = "cpu"
    elif args.device == "auto":
        device = f"cuda:{get_freer_gpu()}"
    else:
        device = args.device

    return tts_path, voc_path, prosody_path, style_reference, device, args.max_chars


@st.cache_resource
def load_synthesis_interface(
    tts_ckpt_path: Path,
    vocoder_ckpt_path: Path,
    prosody_ckpt_path: tp.Optional[Path] = None,
    device: str = "cpu",
) -> EvaluationInterface:
    tts = TTSEvaluationInterface(
        tts_ckpt_path=tts_ckpt_path,
        prosody_ckpt_path=prosody_ckpt_path,
        device=device,
    )
    tts_opt = TTSOptions()

    if prosody_ckpt_path is not None:
        tts_opt.down_contur_ids = ["2", "3", "5"]
        tts_opt.up_contur_ids = ["4"]

    voc = VocoderEvaluationInterface(
        ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    voc_opt = VocoderOptions()

    return EvaluationInterface(tts, tts_opt, voc, voc_opt)


def prepare_sidebar():
    langs = synt_interface.tts_interface.get_languages()
    lang = st.sidebar.selectbox(f"Languages ({len(langs)})", langs, index=0)

    speaker_names = synt_interface.tts_interface.get_speakers()
    speaker_name = st.sidebar.selectbox(f"Voices ({len(speaker_names)})", speaker_names)

    audio_format = AudioFormat[
        st.sidebar.selectbox("Audio format", options=AudioFormat.formats(), index=0)
    ]
    return lang, speaker_name, audio_format


def view_help():
    st.markdown(
        """
# Speech Synthesis

This is a demo of Text-to-Speech models built on [SpeechFlow](https://github.com/just-ai/speechflow).

You can control intonation using tags within the text.
The <break time="1s"/> tag allows you to insert a pause in the text.
Wrapping a word with the <intonation label="1">text</intonation> tag assigns it a corresponding prosodic markup class.
There are 8 markup classes in total, numbered from 0 to 7. Using the -1 index removes automatically assigned prosodic markup.

Use the "+" symbol to place stress in any word.

**For example**:
```
<intonation label="-1">С помощью просодической разметки я могу делать</intonation>
<intonation label="7">акценты</intonation> <intonation label="-1">на отдельных</intonation>
<intonation label="4">словах</intonation>. <intonation label="-1">Моя интонация может</intonation>
<intonation label="4">подниматься</intonation>, или <intonation label="5">опускаться</intonation>.
```
```
Ночью - 23 <intonation label="5">июня</intonation> - начал извергаться самый высокий действующий вулкан Евразии - Ключевской.
Об этом сообщила руководитель Камчатской группы реагирования на вулканические извержения,
ведущий научный сотрудник Института вулканологии и сейсмологии ДВО Ран Ольга <intonation label="5">Гирина</intonation>.
Зафиксировано ночью не просто свечение, а вершинное эксплозивное извержение стромболианского типа.
Пока такое извержение никому не опасно. <intonation label="3">
Ни населению, ни авиации, </intonation>  пояснила ТАСС -  <intonation label="1">госпожа Гирина</intonation>.
```
        """
    )


def prepare_sents(doc: Doc):
    text_porosdy = []
    for i, token in enumerate(doc.tokens):
        if token.text != "<SIL>":
            if i != 0 and not token.is_punctuation:
                text_porosdy.append(" ")
            if token.prosody is not None and token.prosody != "-1":
                text_porosdy.append((token.text, token.prosody))
            else:
                text_porosdy.append(token.text)
    return text_porosdy


st.set_page_config(
    page_title="SpeechFlow TTS",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

(
    tts_path,
    voc_path,
    prosody_path,
    style_reference,
    device,
    max_chars,
) = prepare_input_args()

synt_interface = load_synthesis_interface(tts_path, voc_path, prosody_path, device)

# draw sidebar #
lang, speaker_name, audio_format = prepare_sidebar()

# draw help
view_help()

text = st.text_area(
    "", "Введите текст" if lang == "RU" else "Input text", height=300, max_chars=max_chars
)

if st.button("Generate", type="primary"):
    audio_chunk, utterances = synt_interface.synthesize(
        text, lang, speaker_name, style_reference
    )
    st.audio(
        audio_chunk.to_bytes(audio_format.name),
        format=f"audio/{'mpeg' if audio_format.name == 'mp3' else audio_format.name}",
        loop=True,
    )
    if prosody_path:
        annotated_text(*utterances)
