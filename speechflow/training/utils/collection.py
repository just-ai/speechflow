import typing as tp

from speechflow.training.base_model import BaseTorchModel, BaseTorchModelParams

__all__ = ["ComponentCollection", "ModelCollection"]


class ComponentCollection:
    components: tp.Dict[str, tp.Union[object, tp.Tuple[object, object]]]

    def __init__(self):
        self.components = {}

    def _check(self, component_name: str):
        if component_name in self.components:
            raise KeyError(f"Component '{component_name}' already registered")

    def registry_module(self, module, filter_names=None):
        keys = module.__dict__.keys()
        if filter_names is not None:
            keys = [k for k in keys if filter_names(k)]

        for key in keys:
            if f"{key}Params" in module.__dict__.keys():
                self._check(key)
                self.components[key] = (
                    module.__dict__[key],
                    module.__dict__[f"{key}Params"],
                )

    def registry_component(
        self, component: object, component_params: tp.Optional[object] = None
    ):
        component_name = component.__name__
        self._check(component_name)
        if component_params is not None:
            self.components[component_name] = (component, component_params)
        else:
            self.components[component_name] = component

    def __contains__(self, key):
        return key in self.components

    def __getitem__(self, item: str) -> tp.Union[object, tp.Tuple[object, object]]:
        if item not in self.components:
            raise KeyError(f"Component '{item}' not found")

        return self.components[item]


class ModelCollection:
    models: tp.Dict[str, tp.Tuple[BaseTorchModel, BaseTorchModelParams]]

    def __init__(self):
        self.models = {}

    def __contains__(self, key):
        return key in self.models

    def __getitem__(self, item: str) -> tp.Tuple[BaseTorchModel, BaseTorchModelParams]:
        if item not in self.models:
            raise KeyError(f"Model '{item}' not found")

        return self.models[item]

    def _check(self, model_name: str):
        if model_name in self.models:
            raise KeyError(f"Model '{model_name}' already registered")

    def registry_model(
        self,
        model: BaseTorchModel,
        model_params: BaseTorchModelParams,
        tag: tp.Optional[str] = None,
    ):
        model_name = model.name if tag is None else tag
        self._check(model_name)
        self.models[model_name] = (model, model_params)

    def find_model(self, model_name):
        for model, model_params in self.models.values():
            if model.__name__ == model_name:
                return model, model_params
        else:
            raise KeyError(f"Model '{model_name}' not found")
