import random
import typing as tp

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from tts.acoustic_models.modules.common.gpts import modules
from tts.acoustic_models.modules.common.gpts.decoders.gpt import GPTDecoder
from tts.acoustic_models.modules.common.gpts.layers.layers import MultiEmbedding
from tts.acoustic_models.modules.common.gpts.layers.misc import ConditioningEncoder
from tts.acoustic_models.modules.common.gpts.layers.modules import (
    SinePositionalEmbedding,
    make_pad_mask,
)


@dataclass
class GPTAOutput:
    emb: Tensor
    prompt: tp.Optional[Tensor] = None
    prompt_lens: tp.Optional[Tensor] = None
    response: tp.Optional[Tensor] = None
    response_lens: tp.Optional[Tensor] = None
    target: tp.Optional[Tensor] = None
    target_lens: tp.Optional[Tensor] = None


class GPTA(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        n_heads: int,
        n_layers: int,
        dim_prompt_text: tp.Optional[int] = None,
        dim_prompt_audio: tp.Optional[int] = None,
        dim_response: tp.Optional[int] = None,
        num_tokens_text: tp.Optional[int] = None,
        num_tokens_audio: tp.Optional[int] = None,
        num_levels_text: int = 1,
        num_levels_audio: int = 1,
        num_langs: tp.Optional[int] = None,
        is_norm_first: bool = True,
        use_prenet: bool = True,
        len_audio_prompt_max: int = -1,
        decoder_name: tp.Literal["gpt"] = "gpt",
        **kwargs,
    ):
        super().__init__()

        assert bool(dim_prompt_text) != bool(num_tokens_text), "\n".join(
            [
                "one and only one of variable 'dim_prompt_text' or 'num_tokens_text'"
                "should be defined, but received",
                f"dim_prompt_text: {dim_prompt_text}; num_tokens_text: {num_tokens_text}",
            ]
        )
        assert bool(dim_response) != bool(num_tokens_audio), "\n".join(
            [
                "one and only one of variable 'dim_response' or 'num_tokens_audio'"
                "should be defined, but received",
                f"dim_response: {dim_response}; \
                    num_tokens_audio: {num_tokens_audio}",
            ]
        )

        if kwargs:
            print(f"unused parameters for GPTA: {kwargs}")

        self._dim_hidden = dim_hidden
        self._dim_prompt_text = dim_prompt_text
        self._dim_prompt_audio = dim_prompt_audio
        self._dim_response = dim_response
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._is_norm_first = is_norm_first
        self._use_prenet = use_prenet
        self._len_audio_prompt_max = len_audio_prompt_max

        self._num_tokens_text = num_tokens_text
        self._num_tokens_audio = num_tokens_audio

        self._decoder_name = decoder_name

        self.positional_encoding = self._get_pos_embs()
        self.service_tokens = self._get_service_tokens()
        self.is_use_continuous_resp = dim_response is not None

        if num_langs:
            self.emb_lang = torch.nn.Embedding(num_langs, self._dim_hidden)

        self.emb_text = self.get_emb_proj(
            dim_emb=dim_prompt_text,
            num_tokens=num_tokens_text,
            num_levels=num_levels_text,
        )
        self.emb_audio = self.get_emb_proj(
            dim_emb=dim_prompt_audio,
            num_tokens=None,
            num_levels=num_levels_audio,
        )

        if self.is_use_continuous_resp:
            self.emb_response = self.get_emb_proj(
                dim_emb=dim_response,
                num_tokens=None,
                num_levels=num_levels_audio,
            )
        else:
            self.emb_response = self.get_emb_proj(
                dim_emb=None,
                num_tokens=num_tokens_audio + 1,  # plus stop token
                num_levels=1,
            )

        self.prenet_audio = modules.PrenetAudio(
            dim_model=self._dim_hidden,
            dim_internal=self._dim_hidden,
            is_enable=self._use_prenet,
            is_channel_first=False,
        )
        self.prenet_text = modules.PrenetText(
            dim_model=self._dim_hidden, is_enable=self._use_prenet
        )
        if self.is_use_continuous_resp:
            self.prenet_response = modules.PrenetAudio(
                dim_model=self._dim_hidden,
                dim_internal=self._dim_hidden,
                is_enable=self._use_prenet,
                is_channel_first=False,
            )
        else:
            self.prenet_response = nn.Identity()

        if self._decoder_name == "gpt":
            decoder = GPTDecoder
        else:
            raise NotImplementedError

        self.decoder = decoder(
            dim_hidden=self._dim_hidden,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            is_norm_first=self._is_norm_first,
            d_state=16,
            expand=2,
        )
        self.proj = nn.Linear(self._dim_hidden, self._dim_hidden, bias=True)

        self.eos_value = 10

    def get_emb_proj(
        self,
        dim_emb: tp.Optional[int] = None,
        num_tokens: tp.Optional[int] = None,
        num_levels: int = 1,
    ):
        assert num_levels > 0

        if num_tokens is not None:
            assert dim_emb is None
            emb_proj = MultiEmbedding(
                max_n_levels=num_levels, n_tokens=num_tokens, token_dim=self._dim_hidden
            )
        elif dim_emb:
            emb_proj = ConditioningEncoder(
                dim_emb, embedding_dim=self._dim_hidden, attn_blocks=3, num_attn_heads=2
            )
        else:
            assert False, "at leas one element should be defined"

        return emb_proj

    def _get_service_tokens(self):
        """begin of text begin of audio begin of response."""

        tokens = nn.ParameterDict()
        tokens["bot"] = nn.Parameter(torch.randn(1, 1, self._dim_hidden))
        tokens["boa"] = nn.Parameter(torch.randn(1, 1, self._dim_hidden))
        tokens["bor"] = nn.Parameter(torch.randn(1, 1, self._dim_hidden))

        if self._dim_response is not None:
            tokens["eos"] = nn.Parameter(
                torch.zeros(1, 1, self._dim_response), requires_grad=False
            )

        return tokens

    def _get_pos_embs(self):
        return nn.ModuleDict(
            {
                "prompt": SinePositionalEmbedding(
                    dim_model=self._dim_hidden, dropout=0.1, scale=False, alpha=True
                ),
                "response": SinePositionalEmbedding(
                    dim_model=self._dim_hidden, dropout=0.1, scale=False, alpha=True
                ),
            }
        )

    def prepare_prompt_text(self, text, text_lens, lang_emb=None):
        """
        text: [n_batch, length, dim]
        """
        batch_size = text.shape[0]

        text_prompt_emb = self.emb_text(text)
        text_prompt_emb = self.prenet_text(text_prompt_emb)

        bot = torch.repeat_interleave(self.service_tokens["bot"], batch_size, 0)
        text_prompt_emb = torch.cat([bot, text_prompt_emb], dim=1)
        text_lens = text_lens + 1

        if lang_emb is not None:
            text_prompt_emb = torch.cat([lang_emb, text_prompt_emb], dim=1)
            text_lens = text_lens + 1

        return text_prompt_emb, text_lens

    def prepare_prompt_audio(
        self,
        audio: tp.Optional[torch.Tensor] = None,
        audio_lens: tp.Optional[torch.Tensor] = None,
    ):
        if audio is None:
            return None, None
        assert audio_lens is not None

        batch_size = audio.shape[0]

        audio_prompt_emb = self.emb_audio(audio)
        audio_prompt_emb = self.prenet_audio(audio_prompt_emb)

        boa = torch.repeat_interleave(self.service_tokens["boa"], batch_size, 0)
        audio_prompt_emb = torch.cat([boa, audio_prompt_emb], dim=1)
        audio_lens = audio_lens + 1

        return audio_prompt_emb, audio_lens

    def prepare_response(
        self,
        resp: tp.Optional[torch.Tensor] = None,
        resp_lens: tp.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if resp is None:
            batch_size = kwargs.get("batch_size")
            resp_emb = torch.repeat_interleave(self.service_tokens["bor"], batch_size, 0)
            resp_lens = torch.ones((batch_size,)).long().to(resp_emb.device)
            return resp_emb, resp_lens, None

        batch_size = resp.shape[0]

        if self.training:
            resp_mask = make_pad_mask(resp_lens).unsqueeze(-1)
            resp_lens = resp_lens + 1  # plus end of response

            if self.is_use_continuous_resp:
                eos_token = self.eos_value + self.service_tokens["eos"]
                resp = resp.masked_fill(resp_mask, value=self.eos_value)
                eos = torch.repeat_interleave(eos_token, batch_size, 0)
                resp = torch.cat([resp, eos], dim=1)
            else:
                resp = resp.masked_fill(resp_mask, value=self._num_tokens_audio)
                resp = F.pad(resp, (0, 0, 0, 1), value=self._num_tokens_audio)

        resp_emb = self.emb_response(resp)
        resp_emb = self.prenet_response(resp_emb)

        bor = torch.repeat_interleave(self.service_tokens["bor"], batch_size, 0)
        resp_emb = torch.cat([bor, resp_emb], dim=1)
        resp_lens = resp_lens + 1  # plus stop token

        return resp_emb, resp_lens, resp

    @staticmethod
    def cat_lists(tensors: tp.List[torch.Tensor], lens: tp.List[torch.Tensor]):
        return torch.cat([x[:l] for x, l in zip(tensors, lens)])

    @staticmethod
    def cat_on_edge(
        list_tensors: tp.List[torch.Tensor], list_lens: tp.List[torch.Tensor]
    ):
        assert len(list_tensors) == len(list_lens)

        iter_tensors = zip(*list_tensors)
        iter_lens = zip(*list_lens)
        concated = list()
        for tensor_item, len_item in zip(iter_tensors, iter_lens):
            concated.append(GPTA.cat_lists(tensor_item, len_item))
        concated = pad_sequence(concated, batch_first=True)
        lens = sum(list_lens)

        return concated, lens

    def padding_mask(self, lens: tp.List[torch.Tensor], device):
        masks = []
        for _len in lens:
            masks.append(make_pad_mask(_len).to(device))

        masks = torch.concat(masks, dim=1)
        masks = torch.repeat_interleave(masks, self._n_heads, dim=0).unsqueeze(1)
        return masks

    def attn_mask(self, device, len_cross: int = 0, len_causal: int = 0):
        assert bool(len_cross > 0) or bool(len_causal > 0)

        mask_cross = torch.zeros(len_cross, len_cross, dtype=torch.bool, device=device)

        mask_causal = torch.triu(
            torch.ones(len_causal, len_causal, dtype=torch.bool, device=device),
            diagonal=1,
        )

        mask_cross = F.pad(mask_cross, (0, len_causal), value=True)
        mask_causal = F.pad(mask_causal, (len_cross, 0), value=False)

        mask = torch.cat([mask_cross, mask_causal], dim=0)
        return mask

    def forward(
        self,
        prompt_text: torch.Tensor,
        prompt_text_lens: torch.Tensor,
        prompt_audio: tp.Optional[torch.Tensor] = None,
        prompt_audio_lens: tp.Optional[torch.Tensor] = None,
        response: tp.Optional[torch.Tensor] = None,
        response_lens: tp.Optional[torch.Tensor] = None,
        lang_id: tp.Optional[torch.Tensor] = None,
    ) -> GPTAOutput:
        """"""
        _device = prompt_text.device

        lang_emb = None
        if lang_id is not None:
            assert self.emb_lang, "Num langs wasn't been defined"
            lang_emb = self._lang_emb(lang_id)

        prompt_text, prompt_text_lens = self.prepare_prompt_text(
            text=prompt_text, text_lens=prompt_text_lens, lang_emb=lang_emb
        )
        prompt_audio, prompt_audio_lens = self.prepare_prompt_audio(
            audio=prompt_audio, audio_lens=prompt_audio_lens
        )

        prompt, prompt_lens = self.cat_on_edge(
            list_tensors=[prompt_text, prompt_audio],
            list_lens=[prompt_text_lens, prompt_audio_lens],
        )
        prompt, _ = self.positional_encoding["prompt"](prompt)

        response, response_lens, target = self.prepare_response(
            resp=response, resp_lens=response_lens, batch_size=prompt.shape[0]
        )

        padding_lens = [prompt_lens]
        if response_lens is not None:
            padding_lens.append(response_lens)

        x = prompt
        if response is not None:
            x = torch.cat([x, response], dim=1)

        mask_pad = self.padding_mask(padding_lens, device=_device)
        mask_attn = self.attn_mask(
            len_cross=prompt_lens.max(),
            len_causal=response_lens.max() if response_lens is not None else 0,
            device=_device,
        )

        # TODO: check mask calculation
        if mask_pad.shape[1] == 1:
            mask_pad = mask_pad.expand(-1, mask_attn.shape[1], -1)

        mask = mask_pad.logical_or(mask_attn.unsqueeze(0))
        mask_float = torch.zeros_like(mask, dtype=prompt.dtype)
        mask_float = mask_float.masked_fill(mask, float("-inf"))

        emb, _ = self.decoder((x, None), mask=mask_float)
        emb = self.proj(emb)

        output = GPTAOutput(
            emb=emb,
            prompt=prompt,
            prompt_lens=prompt_lens,
            response=response,
            response_lens=response_lens,
            target=target,
            target_lens=response_lens - 1,
        )
        return output
