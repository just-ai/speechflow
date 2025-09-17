from speechflow.training.utils.collection import ComponentCollection

from . import backbones, feature_extractors, heads

VOCOS_FEATURES = ComponentCollection()
VOCOS_FEATURES.registry_module(feature_extractors, lambda x: "Feature" in x)

VOCOS_BACKBONES = ComponentCollection()
VOCOS_BACKBONES.registry_module(backbones, lambda x: "Backbone" in x)

VOCOS_HEADS = ComponentCollection()
VOCOS_HEADS.registry_module(heads, lambda x: "Head" in x)
