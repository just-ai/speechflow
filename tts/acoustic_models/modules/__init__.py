from speechflow.training.utils.collection import ComponentCollection
from tts.acoustic_models.modules.components import (
    decoders,
    duration_predictors,
    encoders,
    forced_alignment,
    postnet,
    style_encoders,
    variance_adaptors,
    variance_predictors,
)
from tts.acoustic_models.modules.params import *

from . import common, forward_tacotron, prosody, tacotron2
from .components import variance_adaptors as va

TTS_ENCODERS = ComponentCollection()
TTS_ENCODERS.registry_module(encoders, lambda x: "Encoder" in x)
TTS_ENCODERS.registry_module(tacotron2, lambda x: "Encoder" in x)
TTS_ENCODERS.registry_module(forward_tacotron, lambda x: "Encoder" in x)
TTS_ENCODERS.registry_module(prosody, lambda x: "Encoder" in x)

TTS_DECODERS = ComponentCollection()
TTS_DECODERS.registry_module(decoders, lambda x: "Decoder" in x)
TTS_DECODERS.registry_module(tacotron2, lambda x: "Decoder" in x)
TTS_DECODERS.registry_module(forward_tacotron, lambda x: "Decoder" in x)

TTS_POSTNETS = ComponentCollection()
TTS_POSTNETS.registry_module(postnet, lambda x: "Postnet" in x)
TTS_POSTNETS.registry_module(tacotron2, lambda x: "Postnet" in x)
TTS_POSTNETS.registry_module(forward_tacotron, lambda x: "Postnet" in x)

TTS_VARIANCE_ADAPTORS = ComponentCollection()
TTS_VARIANCE_ADAPTORS.registry_module(variance_adaptors)

TTS_VARIANCE_PREDICTORS = ComponentCollection()
TTS_VARIANCE_PREDICTORS.registry_module(variance_predictors)
TTS_VARIANCE_PREDICTORS.registry_module(duration_predictors)
TTS_VARIANCE_PREDICTORS.registry_module(style_encoders)
TTS_VARIANCE_PREDICTORS.registry_component(
    forced_alignment.GradTTSFA, forced_alignment.GradTTSFAParams
)

TTS_LENGTH_REGULATORS = ComponentCollection()
TTS_LENGTH_REGULATORS.registry_component(common.LengthRegulator)
TTS_LENGTH_REGULATORS.registry_component(common.SoftLengthRegulator)
