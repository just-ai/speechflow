import json
import pickle
import typing as tp
import logging

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from os import environ as env
from pathlib import Path

import numpy as np
import torch
import multilingual_text_parser

from multilingual_text_parser.data_types import Doc, Sentence, Syntagma, Token
from multilingual_text_parser.parser import TextParser

import speechflow

from nlp.prosody_prediction.eval_interface import ProsodyPredictionInterface
from speechflow.data_pipeline.collate_functions.tts_collate import TTSCollateOutput
from speechflow.data_pipeline.core.components import PipelineComponents
from speechflow.data_pipeline.datasample_processors import add_pauses_from_text
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.io import AudioChunk, check_path, tp_PATH
from speechflow.logging import trace
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.init import init_class_from_config, init_method_from_config
from speechflow.utils.seed import set_all_seed
from speechflow.utils.versioning import version_check
from tts import acoustic_models
from tts.acoustic_models.data_types import (
    TTSForwardInput,
    TTSForwardInputWithSSML,
    TTSForwardOutput,
)
from tts.acoustic_models.interface.prosody_reference import (
    REFERENECE_TYPE,
    ComplexProsodyReference,
)

__all__ = [
    "TTSEvaluationInterface",
    "TTSContext",
    "TTSOptions",
]

LOGGER = logging.getLogger("root")

DEFAULT_SIL_TOKENS_NUM = 2


@dataclass
class TTSContext:
    prosody_reference: ComplexProsodyReference
    default_ds: tp.Optional[TTSDataSample] = None
    embeddings: tp.Optional[tp.Dict] = None
    additional_inputs: tp.Optional[tp.Dict] = None
    seed: int = 0
    sdk_version: str = speechflow.__version__

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = {}
        if self.additional_inputs is None:
            self.additional_inputs = {}

    @staticmethod
    def create(
        lang: str,
        speaker_name: tp.Optional[tp.Union[str, tp.Dict[str, str]]] = None,
        speaker_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
        style_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
        seed: int = 0,
    ) -> "TTSContext":
        prosody_reference = ComplexProsodyReference.create(
            lang,
            speaker_name,
            speaker_reference,
            style_reference,
        )
        return TTSContext(prosody_reference=prosody_reference, seed=seed)

    def copy(self) -> "TTSContext":
        return deepcopy(self)


@dataclass
class TTSOptions:
    gate_threshold: float = 0.2
    sigma_multiplier: float = 0.0
    average_val: tp.Optional[tp.Dict[str, float]] = None
    begin_pause: tp.Optional[float] = None
    end_pause: tp.Optional[float] = None
    forward_pass: bool = False
    speech_quality: float = 5.0
    tres_bin: float = 0.0
    predict_proba: bool = True
    down_contur_ids: tp.Optional[tp.Sequence[int]] = None
    up_contur_ids: tp.Optional[tp.Sequence[int]] = None

    def __post_init__(self):
        if self.average_val is None:
            self.average_val = {}

        self.average_val.setdefault("duration", 0.0)
        self.average_val.setdefault("energy", 0.0)
        self.average_val.setdefault("pitch", 0.0)
        self.average_val.setdefault("rate", 0.0)
        self.average_val.setdefault("duration_scale", 1.0)
        self.average_val.setdefault("energy_scale", 1.0)
        self.average_val.setdefault("pitch_scale", 1.0)
        self.average_val.setdefault("rate_scale", 1.0)

    def copy(self) -> "TTSOptions":
        return deepcopy(self)


class TTSEvaluationInterface:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        tts_ckpt_path: tp_PATH,
        prosody_ckpt_path: tp.Optional[tp_PATH] = None,
        pauses_ckpt_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
        device_pipe: tp.Optional[str] = None,
        **kwargs,
    ):
        env["DEVICE"] = device_pipe if device_pipe is not None else device

        tts_ckpt = ExperimentSaver.load_checkpoint(tts_ckpt_path)
        cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(tts_ckpt)

        if "feature_extractor" in cfg_model["model"]:
            cfg_model["model"] = {
                "type": "ParallelTTSModel",
                "params": cfg_model["model"]["feature_extractor"]["init_args"],
            }
            tts_ckpt["params"] = cfg_model["model"]["params"]
            tts_ckpt["params"]["n_symbols_per_token"] = 3

            sd = tts_ckpt["state_dict"]
            for k in list(sd.keys()):
                if any(
                    x in k
                    for x in [
                        "feature_extractor.proj",
                        "backbones",
                        "head",
                        "sm_loss",
                        "wavlm_loss",
                        "melspec_loss",
                        "multiperiod_disc",
                        "multiresd_disc",
                    ]
                ):
                    sd.pop(k)
                elif "feature_extractor" in k:
                    sd[k.replace("feature_extractor.tts_model", "model")] = sd.pop(k)

        version_check(
            multilingual_text_parser, tts_ckpt["versions"]["libs"]["text_parser"]
        )
        version_check(speechflow, tts_ckpt["versions"]["speechflow"])

        self.info = tts_ckpt.get("info", {})
        if not self.info:
            self.info = self._load_info(tts_ckpt_path)

        self.lang_id_map = tts_ckpt.get("lang_id_map", {})
        self.speaker_id_map = tts_ckpt.get("speaker_id_map", {})
        self.device = torch.device(device)

        # update data config
        cfg_data["processor"].pop("dump", None)

        pauses_from_ts_cfg = cfg_data["preproc"]["pipe_cfg"].get(
            "add_pauses_from_timestamps", {}
        )
        pause_step = pauses_from_ts_cfg.get("step", 0.05)

        cfg_data["preproc"]["pipe"].insert(0, "add_pauses_from_text")
        cfg_data["preproc"]["pipe_cfg"]["add_pauses_from_text"] = {
            "level": "syntagmas",
            "num_symbols": 1 if pauses_ckpt_path else DEFAULT_SIL_TOKENS_NUM,
            "pause_from_punct_map": {
                ",": "normal",
                "-": "weak",
                "—": "normal",
                ".": "strong",
            },
            "step": pause_step,
        }
        cfg_data["preproc"]["pipe"].append("add_prosody_modifiers")

        text_cfg = cfg_data["preproc"]["pipe_cfg"].get("text", {})
        text_cfg["add_service_tokens"] = True

        self.lang = text_cfg.get("lang", "RU")
        self.num_prosodic_classes = text_cfg.get("num_prosodic_classes")
        text_proc = TTSTextProcessor(
            lang=self.lang, num_prosodic_classes=self.num_prosodic_classes
        )
        if tts_ckpt["alphabet"] != text_proc.alphabet:
            raise ValueError(
                f"The current TTS model is trained with a different alphabet! "
                f"Unknown symbols: {set(tts_ckpt['alphabet']) ^ set(text_proc.alphabet)}"
            )

        cfg_data["collate"]["type"] = (
            "TTSCollateWithSSML"
            if "TTSCollate" in cfg_data["collate"]["type"]
            else cfg_data["collate"]["type"]
        )

        # init singleton handlers
        singleton_handlers = cfg_data.section("singleton_handlers", mutable=True)

        self.speaker_id_setter = singleton_handlers.get("SpeakerIDSetter")
        self.speaker_id_setter["resume_from_checkpoint"] = None
        if self.speaker_id_setter and self.lang_id_map:
            self.speaker_id_setter["lang_id_map"] = [
                f"{key}:{value}" for key, value in self.lang_id_map.items()
            ]
            self._dump_to_file("temp/lang_id_map.json", self.lang_id_map)
        if self.speaker_id_setter and self.speaker_id_map:
            self.speaker_id_setter["speaker_id_map"] = [
                f"{key}:{value}" for key, value in self.speaker_id_map.items()
            ]
            self._dump_to_file("temp/speaker_id_map.json", self.speaker_id_map)

        self.stat_ranges = self.info["singleton_handlers"].get("StatisticsRange")
        if self.stat_ranges is not None:
            handler = singleton_handlers.get("StatisticsRange", {})
            handler["statistics_file"] = "temp/StatisticsRange_data.json"
            self._dump_to_file(handler["statistics_file"], self.stat_ranges.statistics)

        self.mean_bio_embs = self.info["singleton_handlers"].get("MeanBioEmbeddings")
        if self.mean_bio_embs is not None:
            self.mean_bio_embs = self.mean_bio_embs.data

        if self.mean_bio_embs is not None:
            embs = self.mean_bio_embs
            embs = {s: emb.tolist() for s, emb in embs.items()}
            handler = singleton_handlers.get("MeanBioEmbeddings", {})
            handler["mean_embeddings_file"] = "temp/MeanBioEmbeddings_data.json"
            self._dump_to_file(handler["mean_embeddings_file"], embs)

        self.dataset_stat = self.info["singleton_handlers"].get("DatasetStatistics")
        if self.dataset_stat is not None:
            self.dataset_stat.unpickle()
            self.bio_embs = self.dataset_stat.speaker_embedding
            self.audio_hours_per_speaker = {
                name: value.sum() / 3600
                for name, value in self.dataset_stat.wave_duration.items()
            }
        else:
            self.bio_embs = None
            self.audio_hours_per_speaker = None  # type: ignore

        # init data pipeline
        self.pipeline = PipelineComponents(cfg_data, "test")
        self.sample_rate = cfg_data.section("preproc").find_field(
            "sample_rate", default_value=24000
        )
        self.hop_len = cfg_data.section("preproc").find_field("hop_len")

        ignored_fields = {
            "word_timestamps",
            "phoneme_timestamps",
            "speaker_emb_mean",
        }
        pipeline = self.pipeline.with_ignored_fields(
            ignored_metadata_fields={"sega"}, ignored_data_fields=ignored_fields
        )

        self.text_pipe = pipeline.with_ignored_fields(
            ignored_data_fields={"audio_chunk"},
        ).with_ignored_handlers(
            ignored_data_handlers={
                "add_pauses_from_text",
            }
        )
        self.biometric_pipe = pipeline.with_ignored_fields(
            ignored_data_fields={
                "sent",
                "pitch",
                "gate",
            },
        )
        self.audio_pipe = self.biometric_pipe.with_ignored_handlers(
            ignored_data_handlers={
                "VoiceBiometricProcessor",
                "WaveAugProcessor",
                "DenoisingProcessor",
            }
        )
        self.biometric_pipe.data_preprocessing = [
            item
            for item in self.biometric_pipe.data_preprocessing
            if item not in self.audio_pipe.data_preprocessing
            or "store_field" in str(item)
        ]

        self.add_pauses_from_text = init_method_from_config(
            add_pauses_from_text, cfg_data["preproc"]["pipe_cfg"]["add_pauses_from_text"]
        )
        self.text_parser = {}

        # init batch processor
        cfg_model["batch"]["type"] = (
            "TTSBatchProcessorWithSSML"
            if cfg_model["batch"]["type"]
            in ["TTSBatchProcessor", "VocoderBatchProcessor"]
            else cfg_model["batch"]["type"]
        )

        batch_processor_cls = getattr(acoustic_models, cfg_model["batch"]["type"])
        self.batch_processor = init_class_from_config(
            batch_processor_cls, cfg_model["batch"]
        )()
        self.batch_processor.set_device(self.device)

        if prosody_ckpt_path is not None:
            if "_prosody" not in cfg_data["file_search"]["ext"]:
                LOGGER.warning("Current TTS model not support of prosody model!")
                self.prosody_ckpt_path = self.prosody_interface = None
            else:
                self.prosody_ckpt_path = Path(prosody_ckpt_path)
                self.prosody_interface = ProsodyPredictionInterface(
                    ckpt_path=self.prosody_ckpt_path,
                    lang=self.lang,
                    num_prosodic_classes=self.num_prosodic_classes,
                    device=device,
                    text_parser=self.text_parser,
                )
        else:
            self.prosody_ckpt_path = self.prosody_interface = None

        if pauses_ckpt_path is not None:
            self.pauses_interface = None
        else:
            self.pauses_interface = None

        # init model
        tts_ckpt["params"]["n_langs"] = tts_ckpt.get("n_langs", len(self.lang_id_map))
        tts_ckpt["params"]["n_speakers"] = tts_ckpt.get(
            "n_speakers", len(self.speaker_id_map)
        )
        tts_ckpt["params"]["alphabet_size"] = len(tts_ckpt["alphabet"])

        model_cls = getattr(acoustic_models, cfg_model["model"]["type"])
        self.model = model_cls(tts_ckpt["params"])
        self.model.eval()
        try:
            self.model.load_state_dict(tts_ckpt["state_dict"], strict=True)
        except Exception as e:
            LOGGER.error(trace(self, e))
            self.model.load_state_dict(tts_ckpt["state_dict"], strict=False)
        self.model.to(self.device)

        # init averages
        if self.stat_ranges is not None:
            model_averages = self.model.get_params().get("averages", {})
            self.averages = self._get_averages_by_speaker(
                self.stat_ranges.statistics, model_averages
            )
        else:
            self.averages = None

    @staticmethod
    def _load_info(ckpt_path: Path) -> tp.Dict[str, tp.Any]:
        for i in range(2):
            info_path = list(ckpt_path.parents[i].rglob("*info.pkl"))
            if info_path:
                print(f"Load info data from path {info_path[0].as_posix()}")
                with ExperimentSaver.portable_pathlib():
                    return pickle.loads(info_path[0].read_bytes())

        raise FileNotFoundError("*info.pkl file not found!")

    @staticmethod
    def _dump_to_file(file_name: str, data: tp.Any):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        if Path(file_name).suffix == ".txt":
            Path(file_name).write_text(data, encoding="utf-8")
        elif Path(file_name).suffix == ".json":
            Path(file_name).write_text(json.dumps(data, indent=4), encoding="utf-8")
        elif Path(file_name).suffix == ".pkl":
            Path(file_name).write_bytes(pickle.dumps(data))
        else:
            raise NotImplementedError

    @staticmethod
    def _get_averages_by_speaker(
        stat_ranges: tp.Dict[str, tp.Any], model_averages: tp.Dict[str, tp.Any]
    ):
        averages: tp.Dict[str, tp.Dict[str, float]] = defaultdict(dict)

        for var in stat_ranges.keys():
            if var in model_averages:
                interval = model_averages[var]["interval"]
                averages[var]["default"] = np.array(interval).mean()
            else:
                averages[var]["default"] = np.array(0.5)

        if stat_ranges is not None:
            for var in stat_ranges.keys():
                for name, field in stat_ranges[var].items():
                    if field["max"] > 0.0:
                        if var != "rate":
                            averages[var][name] = field["mean"]  # / field["max"]
                        else:
                            averages[var][name] = field["mean"]
                    else:
                        averages[var][name] = averages[var]["default"]

        return averages

    def get_languages(self):
        return sorted(list(self.lang_id_map.keys()))

    def get_speakers(
        self,
        hours_per_speaker: tp.Optional[tp.Union[float, tp.Tuple[float, float]]] = None,
    ) -> tp.List[str]:
        if hours_per_speaker and self.dataset_stat:
            if isinstance(hours_per_speaker, float):
                names = [
                    name
                    for name, value in self.audio_hours_per_speaker.items()
                    if value > hours_per_speaker
                ]
            else:
                names = [
                    name
                    for name, value in self.audio_hours_per_speaker.items()
                    if hours_per_speaker[0] < value < hours_per_speaker[1]
                ]
        else:
            if self.audio_hours_per_speaker:
                names = list(self.audio_hours_per_speaker.keys())
            else:
                names = list(self.speaker_id_map.keys())

        return sorted(names)

    def predict_pauses(
        self,
        doc: Doc,
        begin_pause: tp.Optional[float],
        end_pause: tp.Optional[float],
        speaker_id: tp.Optional[int] = None,
    ):
        if self.pauses_interface is not None:
            pauses_output = self.pauses_interface.predict(
                doc,
                begin_pause=begin_pause,
                end_pause=end_pause,
                speaker_id=speaker_id,
            )
            pauses_durations = [
                pauses_output.durations[i][pauses_output.sil_mask[i] > 0]
                for i in range(pauses_output.durations.shape[0])
            ]
        else:
            pauses_durations = None

        doc.pauses_durations = (
            [None] * len(doc.sents) if pauses_durations is None else pauses_durations
        )
        return doc

    def prepare_text(
        self,
        text: str,
        lang: str,
        opt: TTSOptions = TTSOptions(),
    ) -> Doc:
        if self.lang_id_map and lang not in self.lang_id_map:
            raise ValueError(f"Language {lang} not support in current TTS model!")

        if lang not in self.text_parser:
            LOGGER.info(f"Initial TextParser for {lang} language")
            self.text_parser[lang] = TextParser(lang, device=str(self.device))

        doc = self.text_parser[lang].process(Doc(text))

        doc = self.predict_pauses(doc, opt.begin_pause, opt.end_pause)
        return doc

    def predict_prosody_by_text(
        self, doc: Doc, ctx: TTSContext, opt: TTSOptions = TTSOptions()
    ) -> Doc:
        if self.prosody_interface is not None:
            doc = self.prosody_interface.predict(
                doc,
                tres_bin=opt.tres_bin,
                predict_proba=opt.predict_proba,
                seed=ctx.seed,
                down_contur_ids=opt.down_contur_ids,
                up_contur_ids=opt.up_contur_ids,
            )
        return doc

    def prepare_embeddings(
        self,
        ctx: TTSContext,
        opt: TTSOptions = TTSOptions(),
    ) -> TTSContext:
        ctx.prosody_reference.initialize(
            self.speaker_id_map,
            self.bio_embs,
            self.mean_bio_embs,
            self.biometric_pipe,
            self.audio_pipe,
            seed=ctx.seed,
        )

        set_all_seed(ctx.seed)

        ctx.prosody_reference.set_feats_from_model(self.model)

        ds = TTSDataSample(
            lang_id=self.lang_id_map.get(ctx.prosody_reference.default.lang, 0),
            mel=ctx.prosody_reference.default.style_spectrogram,
            speaker_name=ctx.prosody_reference.default.speaker_name,
            speaker_id=ctx.prosody_reference.default.speaker_id,
            speaker_emb=ctx.prosody_reference.default.speaker_emb,
            speaker_emb_mean=ctx.prosody_reference.default.speaker_emb_mean,
            speech_quality_emb=torch.FloatTensor([[opt.speech_quality] * 4]),
        )

        ds.averages = {}
        if self.averages is not None:
            for key in self.averages.keys():
                value = self.averages[key].get(ds.speaker_name)  # type: ignore
                if value is None or value == 0.0:
                    value = 20 if key == "rate" else 0.0
                if opt.average_val.get(key, 0.0) == 0.0:
                    ds.averages[key] = value
                else:
                    ds.averages[key] = deepcopy(opt.average_val[key])

                scale = opt.average_val.get(f"{key}_scale", 1.0)
                ds.averages[key] *= 4 * scale
        else:
            ds.averages["energy"] = 150
            ds.averages["pitch"] = 220
            ds.averages["rate"] = 12

        ds.ranges = {}
        if self.stat_ranges is not None:
            for attr in self.stat_ranges.get_keys():
                f0_min, f0_max = self.stat_ranges.get_range(attr, ds.speaker_name)  # type: ignore
                ds.ranges[attr] = np.asarray(
                    [f0_min, f0_max, f0_max - f0_min], dtype=np.float32
                )

        batch = self.audio_pipe.to_batch([ds])
        model_inputs, _, _ = self.batch_processor(batch)

        with torch.no_grad():
            output = self.model.embedding_component(model_inputs)

        embeddings = {
            k: v.cpu().numpy() for k, v in output.embeddings.items() if v is not None
        }

        ctx.default_ds = ds
        ctx.embeddings = embeddings
        return ctx

    def split_sentences(
        self,
        doc: Doc,
        max_sentence_length: tp.Optional[int] = None,
        max_text_length_in_batch: tp.Optional[int] = None,
        one_sentence_per_batch: bool = False,
    ) -> tp.List[tp.List[Sentence]]:
        sents = []
        for sent in doc.sents:
            sent = self.add_pauses_from_text(TTSDataSample(sent=sent)).sent

            if max_sentence_length and sent.num_phonemes > max_sentence_length:
                pause = Token(TTSTextProcessor.sil)
                pause.phonemes = (TTSTextProcessor.sil,)
                new_tokens: tp.List[Token] = []
                total_sent_length = 0
                for token in sent.tokens + [None]:
                    if token and token.num_phonemes > max_sentence_length:
                        raise RuntimeError("Invalid text!")

                    if (
                        token is None
                        or total_sent_length + token.num_phonemes > max_sentence_length
                    ):
                        new_tokens = [pause] + new_tokens + [pause]
                        new_sent = deepcopy(sent)
                        new_sent.tokens = new_tokens
                        new_sent.syntagmas = [Syntagma(new_tokens)]
                        sents.append(new_sent)
                        new_tokens = [token]
                        total_sent_length = token.num_phonemes if token else 0
                    else:
                        new_tokens.append(token)
                        total_sent_length += token.num_phonemes
            else:
                sents.append(sent)

        sents_by_batch = [[sents[0]]]
        total_text_length = sents[0].num_phonemes
        for sent in sents[1:]:
            if one_sentence_per_batch or (
                max_text_length_in_batch
                and total_text_length + sent.num_phonemes > max_text_length_in_batch
            ):
                sents_by_batch.append([])
                total_text_length = 0

            sents_by_batch[-1].append(sent)
            total_text_length += sent.num_phonemes

        return sents_by_batch

    def prepare_batch(
        self,
        sents: tp.List[Sentence],
        ctx: TTSContext,
        opt: TTSOptions = TTSOptions(),
    ) -> TTSForwardInputWithSSML:
        samples = []
        for sent in sents:
            new_ds = ctx.default_ds.copy()
            new_ds.sent = sent
            samples.append(new_ds)

        batch = self.text_pipe.datasample_to_batch(samples, skip_corrupted_samples=False)

        additional_inputs = {}

        for name, field in ctx.additional_inputs.items():
            if field is not None and isinstance(field, np.ndarray):
                field = torch.as_tensor(field)
                field = field.expand((batch.size, field.shape[1], field.shape[2]))
                additional_inputs[name] = field.to(self.device)

        for k, v in ctx.embeddings.items():
            if isinstance(v, np.ndarray):
                additional_inputs[k] = torch.cat(
                    [torch.from_numpy(v).to(self.device)] * batch.size
                )
            elif isinstance(v, torch.Tensor):
                additional_inputs[k] = torch.cat([v] * batch.size)
            elif v is not None:
                additional_inputs[k] = v

        if opt.sigma_multiplier is not None:
            additional_inputs["sigma_multiplier"] = opt.sigma_multiplier

        model_inputs, _, _ = self.batch_processor(batch)

        model_inputs.additional_inputs.update(additional_inputs)
        model_inputs.prosody_reference = ctx.prosody_reference.copy()
        model_inputs.output_lengths = None
        return model_inputs

    def predict_variance(self, inputs, ignored_variance: tp.Set = None):
        (
            variance_embeddings,
            variance_predictions,
            additional_content,
        ) = self.model.get_variance(inputs, ignored_variance)
        return variance_embeddings, variance_predictions, additional_content

    @torch.inference_mode()
    def evaluate(
        self,
        inputs: TTSForwardInputWithSSML,
        ctx: TTSContext,
        opt: TTSOptions = TTSOptions(),
    ) -> TTSForwardOutput:
        set_all_seed(ctx.seed)

        if opt.forward_pass:
            outputs = self.model(inputs)
        else:
            outputs = self.model.inference(inputs)
            if outputs.gate is not None and opt.gate_threshold is not None:
                outputs.output_mask = (
                    (outputs.gate.sigmoid() > opt.gate_threshold).cumsum(1) > 0
                ).squeeze(-1)

        setattr(outputs, "additional_inputs", inputs.additional_inputs)
        return outputs

    def synthesize(
        self,
        text: str,
        lang: str,
        speaker_name: str,
        opt: TTSOptions = TTSOptions(),
    ) -> TTSForwardOutput:
        ctx = TTSContext.create(lang, speaker_name)
        text_by_sentence = self.prepare_text(text, lang, opt)
        text_by_sentence = self.predict_prosody_by_text(text_by_sentence, ctx, opt)
        ctx = self.prepare_embeddings(ctx, opt)
        inputs = self.prepare_batch(self.split_sentences(text_by_sentence)[0], ctx, opt)
        outputs = self.evaluate(inputs, ctx, opt)
        return outputs

    @check_path(assert_file_exists=True)
    def resynthesize(
        self,
        wav_path: tp_PATH,
        ref_wav_path: tp.Optional[tp_PATH] = None,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: TTSOptions = TTSOptions(),
    ) -> TTSForwardOutput:
        audio_chunk = (
            AudioChunk(file_path=wav_path).load(sr=self.sample_rate).volume(1.25)
        )
        ds = TTSDataSample(audio_chunk=audio_chunk)
        batch = self.audio_pipe.datasample_to_batch([ds])
        collated: TTSCollateOutput = batch.collated_samples  # type: ignore

        if ref_wav_path is not None:
            ref_audio_chunk = AudioChunk(file_path=ref_wav_path).load(sr=self.sample_rate)
            ref_ds = TTSDataSample(audio_chunk=ref_audio_chunk)
            ref_batch = self.audio_pipe.datasample_to_batch([ref_ds])
            ref_collated: TTSCollateOutput = ref_batch.collated_samples  # type: ignore
            collated.speaker_emb = ref_collated.speaker_emb
            collated.speaker_emb_mean = ref_collated.speaker_emb_mean
            collated.spectrogram = ref_collated.spectrogram
            collated.spectrogram_lengths = ref_collated.spectrogram_lengths
            collated.averages = ref_collated.averages
            collated.speech_quality_emb = ref_collated.speech_quality_emb
            collated.additional_fields = ref_collated.additional_fields

        _input = TTSForwardInputWithSSML(
            spectrogram=collated.spectrogram,
            spectrogram_lengths=collated.spectrogram_lengths,
            ssl_feat=collated.ssl_feat,
            ssl_feat_lengths=collated.ssl_feat_lengths,
            xpbert_feat=collated.xpbert_feat,
            xpbert_feat_lengths=collated.xpbert_feat_lengths,
            speaker_emb=collated.speaker_emb,
            speaker_emb_mean=collated.speaker_emb,
            speech_quality_emb=collated.speech_quality_emb,
            averages=collated.averages,
            additional_inputs=collated.additional_fields,
            # energy=collated.energy,
            # pitch=collated.pitch,
        )

        if lang is not None:
            _input.lang_id = torch.LongTensor([self.lang_id_map[lang]])
        if speaker_name is not None:
            _input.speaker_id = torch.LongTensor([self.speaker_id_map[speaker_name]])

        ctx = TTSContext.create(lang, speaker_name)

        _input.to(self.device)
        outputs = self.evaluate(_input, ctx, opt)
        return outputs

    def inference(self, batch_input: TTSForwardInput, **kwargs):
        return self.model.inference(batch_input, **kwargs)
