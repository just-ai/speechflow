import enum
import typing as tp
import logging

from pathlib import Path

import numpy as np
import torch

from speechflow.data_pipeline.datasample_processors.data_types import AudioCodecFeatures
from speechflow.io import AudioChunk, check_path, tp_PATH
from speechflow.utils.fs import get_root_dir
from speechflow.utils.profiler import Profiler

__all__ = [
    "DescriptAC",
    "StableAC",
    "VocosAC",
]

LOGGER = logging.getLogger("root")


class ACFeatureType(enum.Enum):
    """
    "latent" : Tensor[B x N*D x T]
        Projected latents (continuous representation of input before quantization)
    "quantized" : Tensor[B x N x T]
        Codebook indices for each codebook (quantized discrete representation of input)
    "continuous" : Tensor[B x D x T]
        Quantized continuous representation of input
    """

    latent = 0
    quantized = 1
    continuous = 2


class BaseAudioCodecModel(torch.nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        feat_type: ACFeatureType = ACFeatureType.continuous,
    ):
        super().__init__()

        self.device = device
        self.sample_rate = 24000
        self.embedding_dim = 0

        self._feat_type = (
            ACFeatureType[feat_type] if isinstance(feat_type, str) else feat_type
        )

    def preprocess(self, audio_chunk: AudioChunk) -> torch.Tensor:
        assert np.issubdtype(
            audio_chunk.dtype, np.floating
        ), "Audio data must be floating-point!"

        audio_chunk = audio_chunk.resample(sr=self.sample_rate, fast=True)
        data = torch.tensor(audio_chunk.waveform, device=self.device)
        return data.unsqueeze(0)

    @staticmethod
    def postprocessing(feat: AudioCodecFeatures) -> AudioCodecFeatures:
        return feat


class DescriptAC(BaseAudioCodecModel):
    @check_path
    def __init__(
        self,
        model_name: tp.Literal["44khz", "24khz", "16khz"] = "24khz",
        model_bitrate: tp.Literal["8kbps", "16kbps"] = "8kbps",
        feat_type: ACFeatureType = ACFeatureType.continuous,
        pretrain_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
    ):
        import dac

        super().__init__(device, feat_type)

        if model_name != "24khz":
            raise NotImplementedError(
                f"model name {model_name} is not supported in current wrapper"
            )

        if model_bitrate != "8kbps":
            raise NotImplementedError(
                f"model bitrate {model_bitrate} is not supported in current wrapper"
            )

        self.sample_rate = 24000
        self.n_classes = 1024

        if self._feat_type == ACFeatureType.latent:
            self.embedding_dim = 256
        elif self._feat_type == ACFeatureType.quantized:
            self.embedding_dim = 32
        elif self._feat_type == ACFeatureType.continuous:
            self.embedding_dim = 1024
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        if pretrain_path is None or not pretrain_path.exists():
            model_path = dac.utils.download(
                model_type=model_name, model_bitrate=model_bitrate
            )
            self.model = dac.DAC.load(model_path.as_posix())
        else:
            LOGGER.info(
                f"Load Descript audio codec model from {pretrain_path.as_posix()}"
            )
            self.model = dac.DAC.load(pretrain_path.as_posix())

        self.model.to(device)

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk, **kwargs) -> AudioCodecFeatures:
        ac_feat = AudioCodecFeatures()
        data = self.preprocess(audio_chunk).unsqueeze(1)

        x = self.model.preprocess(data, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)

        if self._feat_type == ACFeatureType.latent:
            ac_feat.encoder_feat = latents.squeeze(0).t().cpu()
        elif self._feat_type == ACFeatureType.quantized:
            ac_feat.encoder_feat = codes.squeeze(0).long().t().cpu()
        elif self._feat_type == ACFeatureType.continuous:
            ac_feat.encoder_feat = z.squeeze(0).t().cpu()
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        return self.postprocessing(ac_feat)

    @torch.no_grad()
    def encode(self, data: torch.Tensor):
        x = self.model.preprocess(data, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        return self.model.decode(z)

    @torch.no_grad()
    def decode_from_codes(self, codes: torch.Tensor):
        z, _, _ = self.model.quantizer.from_codes(codes.t().unsqueeze(0))
        return self.decode(z).squeeze(1).t()


class StableAC(BaseAudioCodecModel):
    @check_path
    def __init__(
        self,
        model_name: str = "stabilityai/stable-codec-speech-16k",
        model_bitrate: tp.Literal[
            "1x46656_400bps", "2x15625_700bps", "4x729_1000bps"
        ] = "1x46656_400bps",
        feat_type: ACFeatureType = ACFeatureType.continuous,
        pretrain_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
    ):
        from stable_codec import StableCodec

        super().__init__(device, feat_type)

        self.sample_rate = 16000
        self.n_classes = int(model_bitrate.split("_")[0][2:])

        if self._feat_type == ACFeatureType.latent:
            self.embedding_dim = 6
        elif self._feat_type == ACFeatureType.quantized:
            self.embedding_dim = int(model_bitrate[0])
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        if pretrain_path is not None:
            self.model = StableCodec(
                model_config_path=(pretrain_path / "model_config.json").as_posix(),
                ckpt_path=(pretrain_path / "model.ckpt").as_posix(),
                device=device,
            )
        else:
            self.model = StableCodec(pretrained_model=model_name, device=device)

        self.model.set_posthoc_bottleneck(model_bitrate)

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk, **kwargs) -> AudioCodecFeatures:
        ac_feat = AudioCodecFeatures()
        data = self.preprocess(audio_chunk).unsqueeze(1)

        latents, codes = self.model.encode(data, posthoc_bottleneck=True)

        if self._feat_type == ACFeatureType.latent:
            ac_feat.encoder_feat = latents.squeeze(0).t().cpu()
        elif self._feat_type == ACFeatureType.quantized:
            ac_feat.encoder_feat = torch.cat(codes, dim=-1).squeeze(0).long().cpu()
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        return self.postprocessing(ac_feat)

    @torch.no_grad()
    def encode(self, data: torch.Tensor):
        latents, codes = self.model.encode(data, posthoc_bottleneck=True)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def decode_from_codes(self, codes: torch.Tensor):
        codes = [codes[:, i].unsqueeze(0).unsqueeze(-1) for i in range(codes.shape[-1])]
        return self.model.decode(codes, posthoc_bottleneck=True)[0].t()


class VocosAC(BaseAudioCodecModel):
    def __init__(
        self,
        ckpt_path: Path,
        device: str = "cpu",
        feat_type: ACFeatureType = ACFeatureType.continuous,
    ):
        from tts.vocoders.eval_interface import VocoderEvaluationInterface

        super().__init__(device, feat_type)

        self.sample_rate = 24000

        self.voc = VocoderEvaluationInterface(ckpt_path=ckpt_path, device=device)
        dim = self.voc.model.feature_extractor.vq_enc.params.encoder_output_dim

        if self._feat_type == ACFeatureType.latent:
            self.embedding_dim = dim
        elif self._feat_type == ACFeatureType.quantized:
            self.embedding_dim = (
                self.voc.model.feature_extractor.vq_codes.params.vq_num_quantizers
            )
        elif self._feat_type == ACFeatureType.continuous:
            self.embedding_dim = dim
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk, **kwargs) -> AudioCodecFeatures:
        from tts.vocoders.data_types import VocoderForwardInput

        ac_feat = AudioCodecFeatures()
        vq_only = kwargs.get("vq_only", False)

        ds = kwargs.get("ds").copy()
        ds.to_tensor()

        lens = torch.LongTensor([ds.magnitude.shape[0]]).to(self.device)

        _input = VocoderForwardInput(
            spectrogram=ds.mel.unsqueeze(0).to(self.device),
            spectrogram_lengths=lens,
            linear_spectrogram=ds.magnitude.unsqueeze(0).to(self.device),
            ssl_feat=ds.ssl_feat.encoder_feat.unsqueeze(0).to(self.device),
            ssl_feat_lengths=lens,
            energy=ds.energy.unsqueeze(0).to(self.device),
            pitch=ds.pitch.unsqueeze(0).to(self.device),
            speaker_emb=ds.speaker_emb.to(self.device),
            lang_id=torch.LongTensor([self.voc.lang_id_map[ds.lang]] * len(lens)).to(
                self.device
            ),
            additional_inputs={
                "style_embedding": ds.additional_fields["style_embedding"]
                .unsqueeze(0)
                .to(self.device)
            },
        )
        _output = self.voc.model.inference(_input, vq_only=vq_only)
        _addc = _output.additional_content

        if self._feat_type == ACFeatureType.latent:
            ac_feat.encoder_feat = _addc["vq_latent"].queeze(0).cpu()
        elif self._feat_type == ACFeatureType.quantized:
            ac_feat.encoder_feat = _addc["vq_codes"].squeeze(0).cpu()
        elif self._feat_type == ACFeatureType.continuous:
            ac_feat.encoder_feat = _addc["vq_z"].squeeze(0).cpu()
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        if not vq_only:
            ac_feat.waveform = _output.waveform[0].squeeze(0).cpu()

        return self.postprocessing(ac_feat)


if __name__ == "__main__":
    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(_wav_path, end=3.9).load()

    for _feat_type in [
        ACFeatureType.continuous,
        ACFeatureType.quantized,
        ACFeatureType.latent,
    ]:
        for _ac_cls in [DescriptAC, StableAC]:
            try:
                _ac_model = _ac_cls(feat_type=_feat_type)
            except Exception as e:
                print(e)
                continue

            with Profiler(_ac_cls.__name__) as prof:
                try:
                    _ac_feat = _ac_model(_audio_chunk)
                except Exception as e:
                    print(e)
                    continue

            print(f"{_ac_cls.__name__}: {_ac_feat.encoder_feat.shape}")
            assert _ac_feat.encoder_feat.shape[-1] == _ac_model.embedding_dim

            if _feat_type == ACFeatureType.quantized:
                waveform = _ac_model.decode_from_codes(_ac_feat.encoder_feat)
                temp = AudioChunk(data=waveform.cpu().numpy(), sr=_ac_model.sample_rate)
                temp.save(f"{_ac_cls.__name__}_reconstruction.wav", overwrite=True)
