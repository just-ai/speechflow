import math
import typing as tp
import logging

from pathlib import Path

import numpy as np
import torch
import joblib
import transformers

from speechflow.data_pipeline.datasample_processors.data_types import SSLFeatures
from speechflow.io import AudioChunk, check_path, tp_PATH
from speechflow.utils.fs import get_root_dir
from speechflow.utils.tensor_utils import fold, unfold

__all__ = [
    "SSLFeatures",
    "Whisper",
    "Wav2Vec",
    "WavLM",
    "ECAPABiometric",
]

LOGGER = logging.getLogger("root")


class BaseSSLModel(torch.nn.Module):
    def __init__(
        self,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.sample_rate = 16000
        self.embedding_dim = 0

    def preprocessing(self, audio_chunk: AudioChunk) -> torch.Tensor:
        assert np.issubdtype(
            audio_chunk.dtype, np.floating
        ), "Audio data must be floating-point!"

        # if audio_chunk.sr != self.sample_rate:
        #     LOGGER.warning(
        #         trace(
        #             self,
        #             message=f"Only {self.sample_rate} sample rate is available "
        #                     f"but got sample rate={audio_chunk.sr}!",
        #             full=False,
        #         ),
        #     )

        audio_chunk = audio_chunk.resample(sr=self.sample_rate, fast=True)
        data = torch.from_numpy(audio_chunk.waveform).to(self.device)
        return data.unsqueeze(0)

    def postprocessing(self, ssl_feat: SSLFeatures) -> SSLFeatures:
        ssl_feat.encoder_feat = ssl_feat.encoder_feat.cpu()

        if ssl_feat.logits is not None:
            ssl_feat.logits = ssl_feat.logits.cpu()

        if ssl_feat.centroids is not None:
            ssl_feat.centroids = ssl_feat.centroids.cpu()

        return ssl_feat


class Whisper(BaseSSLModel):
    def __init__(
        self,
        model_name: tp.Literal[
            "tiny", "base", "small", "medium", "large-v2"
        ] = "large-v2",
        device: str = "cpu",
    ):
        import whisper

        super().__init__(device)

        self.model = whisper.load_model(model_name, device)
        self.options = whisper.DecodingOptions(fp16=False)
        self.dec_task = whisper.decoding.DecodingTask(self.model, self.options)
        self.pos_emb = self.model.encoder.positional_embedding.clone()
        self.log_mel_spectrogram = whisper.log_mel_spectrogram

        self.model.eval()
        self.embedding_dim = self.model.dims.n_audio_state

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        mel = self.log_mel_spectrogram(data.squeeze(0)).unsqueeze(0)

        assert mel.shape[-1] <= self.pos_emb.shape[0]
        self.model.encoder.positional_embedding = self.pos_emb[
            : math.ceil(mel.shape[-1] / 2)
        ]
        emb = self.dec_task._get_audio_features(mel)

        ssl_feat.encoder_feat = emb.squeeze(0)
        return self.postprocessing(ssl_feat)


class Wav2Vec(BaseSSLModel):
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        model_name: str = "anton-l/wav2vec2-large-xlsr-53-russian",
        pretrain_path: tp.Optional[tp_PATH] = None,
        vocab_path: tp.Optional[tp_PATH] = None,
        kmeans_path: tp.Optional[tp_PATH] = None,
        feature_type: tp.Literal[
            "logits", "centroids", "last_hidden_state", "partial"
        ] = "last_hidden_state",
        level: int = 4,
        stream_mod: tp.Optional[dict] = None,
        device: str = "cpu",
    ):
        super().__init__(device)

        self._feature_type = feature_type
        self._level = level
        self._stream_mod = stream_mod
        self._vocab_path = vocab_path
        self._kmeans_path = kmeans_path

        self._init_model(model_name, feature_type, pretrain_path, vocab_path, kmeans_path)

        if feature_type == "logits" and hasattr(self.model, "lm_head"):
            self.embedding_dim = self.model.lm_head.out_features
        else:
            if hasattr(self.model, "lm_head"):
                self.embedding_dim = self.model.lm_head.in_features
            else:
                self.embedding_dim = self.model.config.output_hidden_size

        if hasattr(self.model, "config"):
            self._pad = self.model.config.conv_kernel[0] // 2
        else:
            self._pad = 0

    def _init_model(
        self,
        model_name: str,
        feature_type: str,
        pretrain_path: tp.Optional[tp_PATH],
        vocab_path: tp.Optional[tp_PATH],
        kmeans_path: tp.Optional[tp_PATH],
    ):
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

        if vocab_path is not None:
            raise ValueError("Vocabulary is not support for Wav2Vec!")

        if kmeans_path is not None:
            raise ValueError("KMeans is not support for Wav2Vec!")
        else:
            self.kmeans = None

    def encode(
        self,
        input_values: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        model = self.model.wav2vec2
        extract_features = model.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = model.feature_projection(extract_features)

        position_embeddings = model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = model.encoder.dropout(hidden_states)

        for layer in model.encoder.layers[:level]:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]

        return hidden_states

    def get_tokens(self, ssl_feat: SSLFeatures) -> SSLFeatures:
        if ssl_feat.logits is None:
            return ssl_feat

        logits = ssl_feat.logits
        text = self.processor.batch_decode(logits.argmax(dim=-1))[0]
        ssl_feat.text = text

        if hasattr(self.processor.tokenizer, "backend"):
            if self.processor.tokenizer.backend.name() == "espeak":
                return ssl_feat

        tokens_id = self.processor.tokenizer(text).input_ids

        dictionary = self.processor.tokenizer.get_vocab()
        dictionary = {v: k for k, v in dictionary.items()}
        tokens_text = [dictionary[t] for t in tokens_id]
        tokens_text = [" " if s == "|" else s for s in tokens_text]

        ssl_feat.tokens_id = tokens_id
        ssl_feat.tokens_text = tokens_text
        return ssl_feat

    def get_discrete_units(self, ssl_feat: SSLFeatures) -> SSLFeatures:
        if self.kmeans is None:
            return ssl_feat

        ssl_feat.centroids = self.kmeans(ssl_feat.encoder_feat.squeeze(0))
        ssl_feat.centroids = ssl_feat.centroids + 1
        return ssl_feat

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        processed = self.processor(
            data.squeeze(0),
            padding=transformers.tokenization_utils_base.PaddingStrategy.DO_NOT_PAD,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        processed = {k: v.to(self.device) for k, v in processed.data.items()}

        if self._stream_mod is not None:
            waveform = processed["input_values"].squeeze(0)
            s_ = self._stream_mod["chunk_size"]
            l_, r_ = self._stream_mod["context_size"]
            processed["input_values"] = unfold(waveform, s_, l_, r_, pad_size=self._pad)

        if self._feature_type == "logits":
            outputs = self.model(**processed)
            ssl_feat.encoder_feat = outputs.logits
            ssl_feat.logits = outputs.logits
        elif self._feature_type in ["centroids", "last_hidden_state"]:
            outputs = self.model(**processed, output_hidden_states=True)
            ssl_feat.encoder_feat = outputs.hidden_states[-1]
            if hasattr(outputs, "logits"):
                ssl_feat.logits = outputs.logits
        elif self._feature_type == "partial":
            if self._level > 0:
                ssl_feat.encoder_feat = self.encode(
                    input_values=processed["input_values"],
                    level=self._level,
                )
            else:
                outputs = self.model(
                    input_values=processed["input_values"],
                    attention_mask=processed["attention_mask"],
                    output_hidden_states=True,
                )
                ssl_feat.encoder_feat = outputs.hidden_states[self._level]
                if hasattr(outputs, "logits"):
                    ssl_feat.logits = outputs.logits

        if self._stream_mod is not None:
            for feat_name in ["encoder_feat", "logits"]:
                feat = getattr(ssl_feat, feat_name)
                if feat is None:
                    continue
                s_ = self._stream_mod["chunk_size"]
                l_, r_ = self._stream_mod["context_size"]
                setattr(ssl_feat, feat_name, fold(feat, s_, l_, r_))

        ssl_feat = self.get_tokens(ssl_feat)
        ssl_feat = self.get_discrete_units(ssl_feat)

        if self._feature_type == "centroids":
            ssl_feat.encoder_feat = ssl_feat.centroids

        ssl_feat.encoder_feat = ssl_feat.encoder_feat.squeeze(0)
        if ssl_feat.logits is not None:
            ssl_feat.logits = ssl_feat.logits.squeeze(0)

        return self.postprocessing(ssl_feat)


class WavLM(BaseSSLModel):
    def __init__(
        self,
        model_name: tp.Literal["WAVLM_BASE_PLUS", "WAVLM_LARGE"] = "WAVLM_LARGE",
        model_dir: tp.Optional[tp_PATH] = None,
        num_layer: int = 9,
        device: str = "cpu",
    ):
        import torchaudio

        super().__init__(device)
        """
        num_layer: 9 - asr task, base+; -1 asr task large
        more details: https://arxiv.org/pdf/2110.13900.pdf (see Fig. 2)
        """
        self._num_layer = num_layer

        pipe = getattr(torchaudio.pipelines, model_name)

        if model_dir is not None and model_dir.exists():
            self.model = pipe.get_model(dl_kwargs={"model_dir": model_dir}).to(
                self.device
            )
        else:
            self.model = pipe.get_model().to(self.device)

        self.model.eval()
        self.sample_rate = pipe.sample_rate
        self.embedding_dim = pipe._params["encoder_embed_dim"]

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        feat = self.model.extract_features(data)[0]

        if self._num_layer:
            emb = feat[self._num_layer]
        else:
            emb = feat

        ssl_feat.encoder_feat = emb.squeeze(0)
        return self.postprocessing(ssl_feat)


class ECAPABiometric(BaseSSLModel):
    def __init__(
        self,
        model_name: tp.Union[str, Path] = "spkrec-ecapa-voxceleb",
        device: str = "cpu",
    ):
        from speechbrain.pretrained import EncoderClassifier

        super().__init__(device)

        self.sample_rate = 16000
        self.embedding_dim = 192

        if not Path(model_name).exists():
            model_name = (
                get_root_dir()
                / f"speechflow/data/temp/biometric/speechbrain/{model_name}"
            )
            model_name.mkdir(parents=True, exist_ok=True)

        self.model = EncoderClassifier.from_hparams(
            source=f"speechbrain/{Path(model_name).name}",
            savedir=Path(model_name).absolute().as_posix(),
            run_opts={"device": self.device},
        )

        _model = self.model.mods.embedding_model
        _state_dict = _model.fc.state_dict()
        _state_dict_mean = {
            "weight": _state_dict["conv.weight"][:, :3072, :],
            "bias": _state_dict["conv.bias"],
        }
        _state_dict_bn = {
            "weight": _model.asp_bn.state_dict()["norm.weight"][:3072],
            "bias": _model.asp_bn.state_dict()["norm.bias"][:3072],
            "running_mean": _model.asp_bn.state_dict()["norm.running_mean"][:3072],
            "running_var": _model.asp_bn.state_dict()["norm.running_var"][:3072],
        }

        self.proj = torch.nn.Conv1d(3072, 192, 1, device=device)
        self.bn = torch.nn.BatchNorm1d(3072, device=device)
        self.proj.load_state_dict(_state_dict_mean)
        self.bn.load_state_dict(_state_dict_bn)

    @torch.inference_mode()
    def get_feats(self, wavs, wav_lens=None):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        feats = self.model.mods.compute_features(wavs)
        feats = self.model.mods.mean_var_norm(feats, wav_lens)

        return feats

    @torch.inference_mode()
    def get_embeddings(self, features):
        _model = self.model.mods.embedding_model
        x = features.transpose(1, 2)

        xl = []
        for layer in _model.blocks:
            try:
                x = layer(x, lengths=None)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = _model.mfa(x)
        x = self.bn(x)
        x = self.proj(x).transpose(1, 2)

        return x

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        features = self.get_feats(data).to(self.device)
        emb = self.get_embeddings(features=features)

        ssl_feat.encoder_feat = emb.squeeze(0)
        return self.postprocessing(ssl_feat)


class KMeans:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.km_model = joblib.load(model_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)
        self.C_torch = torch.from_numpy(self.C_np).to(device)
        self.Cnorm_torch = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C_torch)
                + self.Cnorm_torch
            )
            return dist.argmin(dim=1)
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


if __name__ == "__main__":
    from speechflow.utils.profiler import Profiler

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(_wav_path, end=3.9).load()

    for _ssl_cls in [Whisper, Wav2Vec, WavLM, ECAPABiometric]:
        try:
            _ssl_model = _ssl_cls()
        except Exception:
            continue

        with Profiler(_ssl_cls.__name__) as prof:
            _ssl_feat = _ssl_model(_audio_chunk)

        print(f"{_ssl_cls.__name__}: {_ssl_feat.encoder_feat.shape}")
        assert _ssl_feat.encoder_feat.shape[-1] == _ssl_model.embedding_dim
