import typing as tp

import torch

from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from tts.acoustic_models.modules.common.blocks import ConvPrenet
from tts.acoustic_models.modules.common.gpts.gpt_acoustic import GPTA
from tts.acoustic_models.modules.common.gpts.layers.modules import make_pad_mask
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "XTTSDecoder",
    "XTTSDecoderParams",
]


class XTTSDecoderParams(DecoderParams):
    target_audio_feat: tp.Literal["spectrogram", "ac_feat"] = "spectrogram"
    prompt_audio_feat: tp.Literal["spectrogram"] = "spectrogram"
    n_heads: int = 8
    n_layers: int = 12
    n_tokens: tp.Optional[int] = None
    n_levels: tp.Optional[int] = None
    p_dropout: float = 0.1
    use_prenet: bool = False
    decoder_name: tp.Literal["gpt"] = "gpt"


class XTTSDecoder(Component):
    params: XTTSDecoderParams

    def __init__(self, params: XTTSDecoderParams, input_dim):
        super().__init__(params, input_dim)

        prompt_audio_feat_dim = params.get_feat_dim(params.prompt_audio_feat)

        if params.target_audio_feat == "ac_feat":
            target_audio_feat_dim = None
        else:
            target_audio_feat_dim = params.get_feat_dim(params.target_audio_feat)

        self.prenet_layer = ConvPrenet(
            in_channels=input_dim,
            out_channels=params.decoder_inner_dim,
        )
        self.gpt = GPTA(
            dim_hidden=params.decoder_inner_dim,
            n_heads=params.n_heads,
            n_layers=params.n_layers,
            dim_prompt_text=params.decoder_inner_dim,
            dim_prompt_audio=prompt_audio_feat_dim,
            dim_response=target_audio_feat_dim,
            use_prenet=params.use_prenet,
            num_tokens_audio=params.n_tokens,
            decoder_name=params.decoder_name,
        )

        if params.target_audio_feat == "ac_feat":
            self.linear = nn.Linear(params.decoder_inner_dim, params.n_tokens + 1)
        else:
            self.linear = nn.Linear(params.decoder_inner_dim, target_audio_feat_dim)

        self._ignore_index = -100

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def _get_response(self, inputs):
        _response = getattr(inputs.model_inputs, self.params.target_audio_feat)
        _response_lens = getattr(
            inputs.model_inputs, f"{self.params.target_audio_feat}_lengths"
        )

        if self.params.target_audio_feat == "ac_feat":
            if self.params.n_levels:
                _response = _response[:, :, : self.params.n_levels]

            _shape = _response.shape
            _response = _response.reshape(_shape[0], -1).unsqueeze(-1)
            _response_lens = _response_lens * _shape[-1]

        return _response, _response_lens

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()
        x = self.prenet_layer(x.transpose(1, -1)).transpose(1, -1)

        _prompt = getattr(inputs.model_inputs, "prompt", None)
        if _prompt is None:
            _prompt = inputs.model_inputs

        _prompt_audio = getattr(_prompt, self.params.prompt_audio_feat)
        _prompt_audio_lens = getattr(_prompt, f"{self.params.prompt_audio_feat}_lengths")

        _response, _response_lens = self._get_response(inputs)

        output = self.gpt(
            prompt_text=x,
            prompt_text_lens=x_lens,
            prompt_audio=_prompt_audio,
            prompt_audio_lens=_prompt_audio_lens,
            response=_response,
            response_lens=_response_lens,
        )

        start_resp = output.prompt_lens.max()
        pred = output.emb[:, start_resp:-1]
        logits = self.linear(pred)

        targets = output.target.squeeze(1)
        targets_mask = make_pad_mask(output.target_lens).unsqueeze(-1)

        if self.gpt.is_use_continuous_resp:
            logits = logits.masked_fill(targets_mask, 0)
            targets = targets.masked_fill(targets_mask, 0)
            loss = F.l1_loss(logits, targets)
        else:
            targets = targets.masked_fill(targets_mask, self._ignore_index)
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets.squeeze(-1),
                ignore_index=self._ignore_index,  # internal cuda error, it's pytorch bug
                reduction="sum",
            ) / torch.sum(_response_lens)

        inputs.additional_losses.update({"loss_gpt": loss})

        return DecoderOutput.copy_from(inputs).set_content(logits, _response_lens + 1)

    def inference_step(self, inputs: VarianceAdaptorOutput, max_steps: int = 1000, **kwargs) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()
        x = self.prenet_layer(x.transpose(1, -1)).transpose(1, -1)

        _prompt_audio = getattr(inputs.model_inputs, self.params.prompt_audio_feat)
        _prompt_audio_lens = getattr(
            inputs.model_inputs, f"{self.params.prompt_audio_feat}_lengths"
        )

        # _response = -4 * torch.ones((1, 1, _prompt_audio.shape[-1])).to(self.linear.weight.device)
        # _response_lens = torch.LongTensor([1]).to(self.linear.weight.device)
        #
        # _response = getattr(inputs.model_inputs, self.params.target_audio_feat)
        # _response_lens = getattr(
        #     inputs.model_inputs, f"{self.params.target_audio_feat}_lengths"
        # )
        #
        # _response = _response[:, :150, :]
        # _response_lens = 0 * _response_lens + 150

        # if self.params.target_audio_feat == "ac_feat":
        #     _response = _response[:, :, :1]

        _response = None
        _response_lens = None
        _result = []
        for _ in tqdm(range(max_steps * self.params.n_levels), desc="GPT generation"):
            output = self.gpt(
                prompt_text=x,
                prompt_text_lens=x_lens,
                prompt_audio=_prompt_audio,
                prompt_audio_lens=_prompt_audio_lens,
                response=_response,
                response_lens=_response_lens,
            )

            _start_resp = output.prompt_lens.max()
            _pred = output.emb[:, _start_resp:]
            _logits = self.linear(_pred)

            if self.gpt.is_use_continuous_resp:
                _result.append(_logits[:, -1, :].unsqueeze(1))
            else:
                ids = _logits.argmax(dim=-1)
                _result.append(ids[:, -1].unsqueeze(-1).unsqueeze(-1))

            if int(_result[-1]) == 1024:
                break

            if _response is None:
                _response = _result[-1]
                _response_lens = torch.ones((_logits.shape[0],)).long().to(_logits.device)
            else:
                _response = torch.cat([_response, _result[-1]], dim=1)
                _response_lens += 1

        _result = _result[: self.params.n_levels * (len(_result) // self.params.n_levels)]
        content = torch.cat(_result, dim=1)
        content_lengths = _response_lens

        outputs = DecoderOutput.copy_from(inputs).set_content(content, content_lengths)
        return outputs
