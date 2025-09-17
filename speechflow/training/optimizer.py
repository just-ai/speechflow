import typing as tp
import logging

from collections.abc import Iterable

import torch
import torch_optimizer

from torch import optim

from speechflow.logging import trace
from speechflow.training import lr_schedulers
from speechflow.utils.init import init_class_from_config

__all__ = ["Optimizer"]

LOGGER = logging.getLogger("root")


class Optimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        method: dict,
        lr_scheduler: dict,
        criterion: tp.Optional[torch.nn.Module] = None,
        parameter_groups: tp.Optional[tp.Dict] = None,
    ):
        super().__init__()
        self.model = model
        self.parameter_groups = parameter_groups if parameter_groups else {}
        self.net_params: tp.List = []
        self.criterion = criterion
        self.optimizer = self._get_optimizer(method)
        self.lr_scheduler = self._get_lr_scheduler(lr_scheduler)
        self.optimizer.load_state_dict = self.load_state_dict

    def _is_named_group(self, module) -> tp.Tuple[bool, str]:
        groups = list(self.parameter_groups.keys())
        if isinstance(module, Iterable):
            for name in groups:
                if name in module:
                    return True, name
        else:
            for name in groups:
                if name == module.__class__.__name__:
                    return True, name

        return False, ""

    def _set_param_group(self, module, parent_module_name, net_params: list):
        module_name = module.__class__.__name__

        if module is None:
            return

        is_named, group_name = self._is_named_group(module)

        if not is_named:
            if "_params" in module.__dict__:
                for m in module.__dict__["_modules"].values():
                    self._set_param_group(m, module_name, net_params)
                return

            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                first_item = module[0]
                if first_item is not None and "_params" in first_item.__dict__:
                    for m in module.__dict__["_modules"].values():
                        self._set_param_group(m, module_name, net_params)
                    return

            if isinstance(module, torch.nn.ModuleDict) and len(module) > 0:
                first_item = list(module.values())[0]
                if first_item is not None and "_params" in first_item.__dict__:
                    for m in module.__dict__["_modules"].values():
                        self._set_param_group(m, module_name, net_params)
                    return
        else:
            LOGGER.info(trace(self, f"find group '{group_name}'"))

        module_params = [p for p in module.parameters() if p.requires_grad]

        if len(module_params) > 0:
            net_params.append({"params": module_params})
            net_params[-1].update(
                {
                    "module_name": module_name,
                    "parent_module_name": parent_module_name,
                    "group_name": group_name,
                }
            )

    def _get_optimizer(self, cfg: dict) -> torch.optim:
        self._set_param_group(
            self.model,
            self.model.__class__.__name__,
            self.net_params,
        )

        if self.criterion is not None:
            self._set_param_group(
                self.criterion, self.criterion.__class__.__name__, self.net_params
            )

        if hasattr(optim, cfg["type"]):
            optim_cls = getattr(optim, cfg["type"])
        elif hasattr(torch_optimizer, cfg["type"]):
            optim_cls = getattr(torch_optimizer, cfg["type"])
        else:
            LOGGER.error(trace(self, f"optimizer {cfg['type']} not implemented!"))
            raise NotImplementedError

        return init_class_from_config(optim_cls, cfg)(params=self.net_params)

    def _get_lr_scheduler(self, cfg: dict) -> optim.lr_scheduler._LRScheduler:
        if hasattr(optim.lr_scheduler, cfg["type"]):
            scheduler_cls = getattr(optim.lr_scheduler, cfg["type"])
        elif hasattr(lr_schedulers, cfg["type"]):
            scheduler_cls = getattr(lr_schedulers, cfg["type"])
        else:
            LOGGER.error(trace(self, f"scheduler {cfg['type']} not implemented!"))
            raise NotImplementedError

        return init_class_from_config(scheduler_cls, cfg)(optimizer=self.optimizer)

    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_param_groups(self, current_iter: int):
        add_groups = []
        remove_groups = []
        for name, val in self.parameter_groups.items():
            for group in self.net_params:
                if group["group_name"] == name:
                    if (
                        val.get("begin_iter", 0)
                        <= current_iter
                        < val.get("end_iter", 1_000_000)
                    ):
                        add_groups += [group]
                    else:
                        remove_groups += [group]

        other_groups = []
        for group in self.net_params:
            if group["group_name"] not in self.parameter_groups:
                other_groups += [group]

        if "remove_other" in self.parameter_groups:
            val = self.parameter_groups["remove_other"]
            if val.get("begin_iter", 0) <= current_iter < val.get("end_iter", 1_000_000):
                remove_groups += other_groups
                other_groups = []

        new_param_groups = add_groups + other_groups

        if self.optimizer.param_groups != new_param_groups:
            for group in remove_groups:
                for p in group["params"]:
                    self.optimizer.state.pop(p, None)

            self.optimizer.param_groups = new_param_groups

    @property
    def current_lr(self):
        return self.lr_scheduler.get_lr()[0]

    def state_dict(self) -> dict:
        ret = {}
        ret["lr_scheduler_type"] = self.lr_scheduler.__class__.__name__
        ret["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict: dict):
        try:
            if state_dict["lr_scheduler_type"] == self.lr_scheduler.__class__.__name__:
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
        except Exception:
            pass
            # self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
