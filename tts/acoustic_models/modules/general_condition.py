import enum

from torch import nn

from tts.acoustic_models.modules.common.conditional_layers import ConditionalLayer
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import GeneralConditionParams

__all__ = ["GeneralCondition", "ModelLevel"]


class ModelLevel(enum.Enum):
    level_0 = 0
    level_1 = 1
    level_2 = 2
    level_3 = 3

    @property
    def value(self) -> int:
        return self._value_


class GeneralCondition(Component):
    params: GeneralConditionParams

    def __init__(self, params: GeneralConditionParams, input_dim, level: ModelLevel):
        super().__init__(params, input_dim)

        cond_params = params.general_condition.get(level.name)
        if cond_params is not None:
            self.content_condition_list = []
            self.content_condition_modules = nn.ModuleList()
            self._output_dim = []
        else:
            self.content_condition_list = None
            self.content_condition_modules = None
            self._output_dim = input_dim
            return

        for idx, in_dim in enumerate(
            input_dim if isinstance(input_dim, list) else [input_dim]
        ):
            conditions = []
            cond_layers = nn.ModuleList()
            for item in cond_params:
                if "content" in item and idx not in item["content"]:
                    continue

                if "condition_dim" in item:
                    condition_dim = item["condition_dim"]
                else:
                    condition_dim = 0
                    for emb_type in item["condition"]:
                        if emb_type.startswith("average") and emb_type != "average_emb":
                            avg_params = self.params.averages[emb_type.split("_", 1)[1]]
                            condition_dim += avg_params["emb_dim"]
                        else:
                            condition_dim += getattr(params, f"{emb_type}_dim")

                layer = ConditionalLayer(item["condition_type"], in_dim, condition_dim)
                in_dim = layer.output_dim

                conditions.append(item["condition"])
                cond_layers.append(layer)

            self.content_condition_list.append(conditions)
            self.content_condition_modules.append(cond_layers)
            self._output_dim.append(in_dim)

    @property
    def output_dim(self):
        return self._output_dim

    def forward_step(self, inputs: ComponentInput) -> ComponentOutput:  # type: ignore
        if self.content_condition_modules is None:
            return inputs

        for idx, (cond_layers, condition_list) in enumerate(
            zip(self.content_condition_modules, self.content_condition_list)
        ):
            x, x_lens, x_mask = inputs.get_content_and_mask(idx)

            for layer, condition in zip(cond_layers, condition_list):
                c = self.get_condition(inputs, condition)
                x = layer(x, c, x_mask)

            inputs.set_content(x, idx=idx)

        return inputs
