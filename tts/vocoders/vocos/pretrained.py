import typing as tp

import torch

from torch import nn

from speechflow.io import Config
from speechflow.utils.init import init_class_from_config
from tts.vocoders.data_types import VocoderForwardInput, VocoderForwardOutput
from tts.vocoders.vocos.modules import VOCOS_BACKBONES, VOCOS_FEATURES, VOCOS_HEADS
from tts.vocoders.vocos.modules.backbones import Backbone
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor
from tts.vocoders.vocos.modules.heads import WaveformGenerator


def instantiate_class(
    args: tp.Union[tp.Any, tp.Tuple[tp.Any, ...]], init: tp.Dict[str, tp.Any]
) -> tp.Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.

    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)

    class_path = f"tts.vocoders.vocos.{init.get('class_name', {})}"
    class_module, class_name = class_path.rsplit(".", 1)

    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)

    if "tts_cfg" in kwargs:
        kwargs["tts_cfg"]["n_langs"] = kwargs.pop("n_langs")
        kwargs["tts_cfg"]["n_speakers"] = kwargs.pop("n_speakers")
        kwargs["tts_cfg"]["alphabet_size"] = 246

    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    """The Vocos class represents a Fourier-based neural vocoder for audio synthesis.

    This class is primarily designed for inference, with support for loading from
    pretrained model checkpoints. It consists of three main components: a feature
    extractor, a backbone, and a head.

    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: WaveformGenerator,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def init_from_config(cls, cfg: Config) -> "Vocos":
        feat_cls, feat_params_cls = VOCOS_FEATURES[cfg["feature_extractor"]["class_name"]]
        feat_params = init_class_from_config(
            feat_params_cls, cfg["feature_extractor"]["init_args"]
        )()
        feat = feat_cls(feat_params)

        backbone_cls, backbone_params_cls = VOCOS_BACKBONES[cfg["backbone"]["class_name"]]
        backbone_params = init_class_from_config(
            backbone_params_cls, cfg["backbone"]["init_args"]
        )()
        backbone = backbone_cls(backbone_params)

        if "pretrain_path" in cfg.head.init_args:
            cfg.head.init_args.pretrain_path = None

        head_cls, head_params_cls = VOCOS_HEADS[cfg["head"]["class_name"]]
        head_params = init_class_from_config(head_params_cls, cfg["head"]["init_args"])()
        head = head_cls(head_params)

        return cls(feature_extractor=feat, backbone=backbone, head=head)

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        """Method to run a copy-synthesis from audio waveform. The feature extractor first
        processes the audio input, which is then passed through the backbone and the head
        to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).

        """
        features, _, _ = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        """Method to decode audio waveform from already calculated features. The features
        input is passed through the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).

        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x, **kwargs)
        return audio_output

    @torch.no_grad()
    def inference(self, inputs: VocoderForwardInput, **kwargs) -> VocoderForwardOutput:
        feat, losses, ft_additional = self.feature_extractor(inputs, **kwargs)
        kwargs.update(ft_additional)
        waveform, _, _ = self.decode(feat, **kwargs)
        return VocoderForwardOutput(waveform=waveform, additional_content=ft_additional)
