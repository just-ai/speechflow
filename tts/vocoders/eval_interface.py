import typing as tp

from copy import deepcopy
from dataclasses import dataclass
from os import environ as env

import torch
import numpy.typing as npt

from speechflow.data_pipeline.collate_functions.tts_collate import TTSCollateOutput
from speechflow.data_pipeline.core import PipelineComponents
from speechflow.data_pipeline.datasample_processors import SignalProcessor
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    TTSDataSample,
)
from speechflow.io import AudioChunk, Config, check_path, tp_PATH
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.vocoders.data_types import VocoderForwardInput, VocoderForwardOutput
from tts.vocoders.denoiser import Denoiser
from tts.vocoders.vocos.modules.feature_extractors.tts import TTSFeatures
from tts.vocoders.vocos.pretrained import Vocos

__all__ = ["VocoderEvaluationInterface", "VocoderOptions"]


@dataclass
class VocoderOptions:
    denoiser_strength: float = 0.005
    denoiser_use_energies: bool = True

    def copy(self) -> "VocoderOptions":
        return deepcopy(self)


class VocoderLoader:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        ckpt_path: tp_PATH,
        device: str = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
        **kwargs,
    ):
        env["DEVICE"] = device

        self.ckpt_path = ckpt_path

        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(self.ckpt_path)
        else:
            checkpoint = ckpt_preload

        self.cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(
            checkpoint
        )

        if "hubert_model_path" in kwargs:
            self.cfg_data.preproc.pipe_cfg.ssl.ssl_params.pretrain_path = kwargs[
                "hubert_model_path"
            ]
        if "hubert_vocab_path" in kwargs:
            self.cfg_data.preproc.pipe_cfg.ssl.ssl_params.vocab_path = kwargs[
                "hubert_vocab_path"
            ]

        self.pipe = self._load_data_pipeline(self.cfg_data)
        self.pipe_for_reference = self.pipe.with_ignored_handlers(
            ignored_data_handlers={"SSLProcessor"}
        )

        self.lang_id_map = checkpoint.get("lang_id_map", {})
        if self.lang_id_map is None:
            self.lang_id_map = {}

        self.speaker_id_map = checkpoint.get("speaker_id_map", {})
        if self.speaker_id_map is None:
            self.speaker_id_map = {}

        cfg_model["model"]["feature_extractor"]["init_args"].n_langs = len(
            self.lang_id_map
        )
        cfg_model["model"]["feature_extractor"]["init_args"].n_speakers = len(
            self.speaker_id_map
        )

        self.sample_rate = find_field(self.cfg_data, "sample_rate")
        self.hop_len = find_field(self.cfg_data, "hop_len")
        self.n_mels = find_field(self.cfg_data, "n_mels")
        self.preemphasis_coef = self.find_preemphasis_coef(self.cfg_data)

        # Load model
        if self._check_vocos_signature(checkpoint):
            self.model = self._load_vocos_model(cfg_model, checkpoint)
        else:
            raise NotImplementedError

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        if not isinstance(self.model.feature_extractor, TTSFeatures):
            self.denoiser = Denoiser(
                self._get_bias_audio(),
                fft_size=find_field(self.cfg_data, "n_fft"),
                win_size=find_field(self.cfg_data, "win_len"),
                hop_size=find_field(self.cfg_data, "hop_len"),
            ).to(self.device)
        else:
            self.denoiser = None

    @staticmethod
    def _check_vocos_signature(checkpoint: tp.Dict) -> bool:
        return "Vocos" in checkpoint["files"]["model.yml"]

    @staticmethod
    def _load_data_pipeline(cfg_data: Config) -> PipelineComponents:
        cfg_data["processor"].pop("dump", None)
        if "singleton_handlers" in cfg_data:
            cfg_data.pop("singleton_handlers")
        pipe = PipelineComponents(Config(cfg_data).trim("ml"), data_subset_name="test")
        return pipe.with_ignored_fields(
            ignored_data_fields={"sent", "phoneme_timestamps"}
        ).with_ignored_handlers(
            ignored_data_handlers={"SignalProcessor", "WaveAugProcessor", "normalize"}
        )

    @staticmethod
    def _load_vocos_model(cfg_model: Config, checkpoint: tp.Dict[str, tp.Any]) -> Vocos:
        model = Vocos.init_from_config(cfg_model["model"])
        model.eval()

        state_dict: tp.Dict[str, tp.Any] = checkpoint["state_dict"]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "discriminators" not in k and "loss" not in k
        }

        state_dict = {
            k.replace("head.dac_", "head.dac_model."): v for k, v in state_dict.items()
        }

        if isinstance(model.feature_extractor, TTSFeatures):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("feature_extractor")
            }
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

        if hasattr(model.head, "remove_weight_norm"):
            model.head.remove_weight_norm()

        return model

    @staticmethod
    def find_preemphasis_coef(cfg_data: Config):
        beta = None
        for item in cfg_data["preproc"]["pipe_cfg"].values():
            if isinstance(item, dict) and item.get("type", None) == "SignalProcessor":
                if "preemphasis" in item.get("pipe", []):
                    if item["pipe_cfg"].get("preemphasis"):
                        beta = item["pipe_cfg"]["preemphasis"].get("beta", 0.97)
                    else:
                        beta = 0.97
        return beta

    @torch.no_grad()
    def _get_bias_audio(self, num_frames: int = 80):
        zero_input = VocoderForwardInput(
            spectrogram=torch.zeros((1, num_frames, self.n_mels)),
            spectrogram_lengths=torch.LongTensor([num_frames]),
        ).to(self.device)
        return self.model.inference(zero_input).waveform


class VocoderEvaluationInterface(VocoderLoader):
    @torch.inference_mode()
    def evaluate(
        self,
        inputs: VocoderForwardInput,
        opt: VocoderOptions,
    ) -> VocoderForwardOutput:
        outputs = self.model.inference(inputs.to(self.device))

        waveforms = []
        for signal, spec_len in zip(outputs.waveform, inputs.spectrogram_lengths):
            signal_len = spec_len * self.hop_len
            waveforms.append(signal[:signal_len])

        waveform = torch.cat(waveforms).unsqueeze(0)

        if self.denoiser is not None and opt.denoiser_strength > 0:
            waveform = self.denoiser(
                waveform,
                strength=opt.denoiser_strength,
                use_energies=opt.denoiser_use_energies,
            )

        waveform = waveform.cpu().numpy()[0]

        if self.preemphasis_coef is not None:
            waveform = self.inv_preemphasis(waveform, self.preemphasis_coef)

        outputs.audio_chunk = AudioChunk(data=waveform, sr=self.sample_rate)
        return outputs

    @staticmethod
    def inv_preemphasis(waveform: npt.NDArray, beta: float = 0.97) -> npt.NDArray:
        audio_chunk = AudioChunk(data=waveform, sr=1)
        inv_wave = SignalProcessor.inv_preemphasis(
            AudioDataSample(audio_chunk=audio_chunk), beta=beta
        ).audio_chunk.waveform
        return inv_wave

    def synthesize(
        self,
        tts_input: TTSForwardInput,
        tts_output: TTSForwardOutput,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: VocoderOptions = VocoderOptions(),
    ) -> VocoderForwardOutput:
        voc_in = VocoderForwardInput.init_from_tts(tts_input, tts_output)
        if lang is not None:
            voc_in.lang_id = torch.LongTensor([self.lang_id_map.get(lang, 0)])
        if speaker_name is not None:
            voc_in.speaker_id = torch.LongTensor(
                [self.speaker_id_map.get(speaker_name, 0)]
            )
        return self.evaluate(voc_in, opt)

    @check_path(assert_file_exists=True)
    def resynthesize(
        self,
        wav_path: tp_PATH,
        ref_wav_path: tp.Optional[tp_PATH] = None,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: VocoderOptions = VocoderOptions(),
    ) -> VocoderForwardOutput:
        audio_chunk = (
            AudioChunk(file_path=wav_path).load(sr=self.sample_rate).volume(1.25)
        )
        ds = TTSDataSample(audio_chunk=audio_chunk)
        batch = self.pipe.datasample_to_batch([ds])
        collated: TTSCollateOutput = batch.collated_samples  # type: ignore

        if ref_wav_path is not None:
            ref_audio_chunk = AudioChunk(file_path=ref_wav_path).load(sr=self.sample_rate)
            ref_ds = TTSDataSample(audio_chunk=ref_audio_chunk)
            ref_batch = self.pipe_for_reference.datasample_to_batch([ref_ds])
            ref_collated: TTSCollateOutput = ref_batch.collated_samples  # type: ignore
            collated.speaker_emb = ref_collated.speaker_emb
            collated.speaker_emb_mean = ref_collated.speaker_emb_mean
            collated.spectrogram = ref_collated.spectrogram
            collated.spectrogram_lengths = ref_collated.spectrogram_lengths
            collated.averages = ref_collated.averages
            collated.speech_quality_emb = ref_collated.speech_quality_emb
            collated.additional_fields = ref_collated.additional_fields

        if collated.speech_quality_emb is not None:
            collated.speech_quality_emb = collated.speech_quality_emb * 0 + 5

        if self.model.__class__.__name__ == "Vocos":
            _input = VocoderForwardInput(
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

            if self.lang_id_map and lang is not None:
                _input.lang_id = torch.LongTensor([self.lang_id_map[lang]])
            if self.speaker_id_map and speaker_name is not None:
                _input.speaker_id = torch.LongTensor([self.speaker_id_map[speaker_name]])

            _output = self.evaluate(_input, opt)
        else:
            raise NotImplementedError

        return _output


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    voc = VocoderEvaluationInterface(ckpt_path="/path/to/checkpoint")

    test_file_path = get_root_dir() / "tests/data/test_audio.wav"

    voc_out = voc.resynthesize(test_file_path, lang="RU")
    voc_out.audio_chunk.save("resynt.wav", overwrite=True)
