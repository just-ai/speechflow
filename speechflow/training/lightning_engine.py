import typing as tp
import logging

import torch
import pytorch_lightning as pl

from speechflow.data_pipeline.core import BaseBatchProcessor
from speechflow.data_pipeline.core.batch import Batch
from speechflow.logging import log_to_file, trace
from speechflow.training import BaseCriterion, BaseTorchModel, ExperimentSaver, Optimizer

__all__ = [
    "LightningEngine",
    "GANLightningEngine",
    "GANLightningEngineWithManualOptimization",
]

LOGGER = logging.getLogger("root")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LightningEngine(pl.LightningModule):
    def __init__(
        self,
        model: BaseTorchModel,
        criterion: BaseCriterion,
        batch_processor: BaseBatchProcessor,
        optimizer: Optimizer,
        saver: ExperimentSaver,
        detect_grad_nan: bool = False,
        use_clearml_logger: bool = False,
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.saver = saver
        self.detect_grad_nan = detect_grad_nan

        self.saver.to_save["params"] = self.model.get_params()
        self.saver.to_save["params_after_init"] = self.model.get_params(after_init=True)

        self.validation_losses = []

        if use_clearml_logger:
            from clearml import Task

            LOGGER.info(trace(self, message="Init ClearML task"))
            self.clearml_task = Task.init(
                task_name=saver.expr_path.name, project_name=saver.expr_path.parent.name
            )
            LOGGER.info(trace(self, message="ClearML task has been initialized"))
        else:
            self.clearml_task = None

    def on_fit_start(self):
        self.batch_processor.set_device(self.device)

    def on_load_checkpoint(self, checkpoint):
        self.optimizer.update_param_groups(checkpoint["global_step"])

    def forward(self, inputs) -> dict:
        outputs = self.model(inputs)
        return outputs

    def calculate_loss(
        self, step_name: str, batch_idx: int, outputs, targets
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        losses = self.criterion(outputs, targets, batch_idx, self.global_step)

        total_loss = torch.tensor(0.0).to(self.device)
        losses_to_log = {}
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float).to(self.device)
            if step_name != "train" or loss.requires_grad:
                if "constant" not in name:
                    total_loss += loss
                losses_to_log[f"{name}/{step_name}"] = loss.detach()
            else:
                losses_to_log[f"{name}/{step_name}"] = loss

        losses_to_log.update({f"TotalLoss/{step_name}": total_loss.detach()})

        if self.training and torch.isnan(total_loss).any():
            raise RuntimeError("Unexpected NaN values in loss computation")

        return total_loss, losses_to_log

    @staticmethod
    def aggregate_losses_to_log(losses_to_log: tp.List[tp.Dict[str, torch.Tensor]]):
        keys = sorted(losses_to_log, key=lambda x: len(x))[0].keys()
        aggregated = {k: [dic[k] for dic in losses_to_log] for k in keys}
        aggregated = {k: torch.stack(v).mean() for k, v in aggregated.items()}
        return aggregated

    def training_step(self, batch: Batch, batch_idx: int):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )
        outputs = self(inputs)
        total_loss, losses_to_log = self.calculate_loss(
            "train", batch_idx, outputs, targets
        )
        self.optimizer.update_param_groups(self.global_step)
        self.log_dict(losses_to_log, add_dataloader_idx=False)

        if total_loss.requires_grad:
            self.log(
                "total_loss",
                total_loss,
                prog_bar=True,
                on_step=True,
                add_dataloader_idx=False,
            )

        return total_loss

    def on_train_epoch_end(self):
        log_to_file(trace(self, message=f"Epoch {self.trainer.current_epoch} completed!"))

    def validation_step(self, batch, batch_idx):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )
        outputs = self(inputs)
        _, losses_to_log = self.calculate_loss("valid", batch_idx, outputs, targets)
        self.validation_losses.append(losses_to_log)

    def on_validation_epoch_end(self):
        losses_to_log = self.validation_losses
        validation_losses = self.aggregate_losses_to_log(losses_to_log)
        self.log_dict(validation_losses, add_dataloader_idx=False)
        self.log("Epoch", float(self.trainer.current_epoch), add_dataloader_idx=False)
        self.validation_losses = []

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs, targets, metadata = self.batch_processor(
                batch, batch_idx, self.global_step
            )
            outputs = self(inputs)
        return inputs, outputs, targets, metadata

    def on_after_backward(self):
        if self.detect_grad_nan:
            for name, param in self.named_parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    LOGGER.warning(
                        trace(
                            self,
                            "Detected inf or NaN values in gradients."
                            "Not updating model parameters!",
                        )
                    )
                    self.zero_grad()

    def configure_optimizers(self):
        optimizer = self.optimizer.optimizer
        scheduler = {
            "scheduler": self.optimizer.lr_scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self.saver.to_save)


class GANLightningEngine(pl.LightningModule):
    def __init__(
        self,
        generator_model: BaseTorchModel,
        discriminator_model: BaseTorchModel,
        criterion: BaseCriterion,
        batch_processor: BaseBatchProcessor,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        saver: ExperimentSaver,
    ):
        super().__init__()

        assert pl.__version__ == "1.5.9", RuntimeError(
            "pytorch_lightning==1.5.9 required"
        )

        self.model = generator_model
        self.discriminator_model = discriminator_model
        self.criterion = criterion
        self.batch_processor = batch_processor
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.saver = saver

        self.saver.to_save["params"] = {
            "generator_model": self.model.get_params(),
            "discriminator_params": self.discriminator_model.get_params(),
        }
        self.saver.to_save["params_after_init"] = {
            "generator_model": self.model.get_params(after_init=True),
            "discriminator_params": self.discriminator_model.get_params(after_init=True),
        }

    def on_pretrain_routine_start(self):
        self.batch_processor.set_device(self.device)

    def forward(self, inputs) -> dict:
        outputs = self.model(inputs)
        return outputs

    def calculate_loss(
        self,
        step_name: str,
        batch_idx: int,
        outputs,
        targets,
        is_generator_training: bool,
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        losses = self.criterion(
            outputs, targets, batch_idx, self.global_step, is_generator_training
        )

        total_loss = torch.tensor(0.0).to(self.device)
        losses_to_log = {}
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float).to(self.device)
            total_loss += loss
            losses_to_log[f"{name}/{step_name}"] = loss.detach()

        losses_to_log.update({f"TotalLoss/{step_name}": total_loss.detach()})
        return total_loss, losses_to_log

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )

        # Train Generator
        if optimizer_idx == 0:
            # Generator takes only inputs:
            outputs = self.model(inputs)
            # Discriminator takes generator outputs and inputs.
            outputs = self.discriminator_model(outputs, inputs, self.global_step)

        # Train Discriminator
        else:
            # Generator takes only inputs:
            with torch.no_grad():
                outputs = self.model(inputs)
            # Discriminator takes generator outputs and inputs.
            outputs = self.discriminator_model(outputs, inputs, self.global_step)

        is_generator_training = True if optimizer_idx == 0 else False
        total_loss, losses_to_log = self.calculate_loss(
            "train", batch_idx, outputs, targets, is_generator_training
        )

        self.log_dict(losses_to_log)

        return total_loss

    def training_epoch_end(self, outputs: tp.List[tp.Any]):
        log_to_file(trace(self, message=f"Epoch {self.trainer.current_epoch} completed!"))

    def validation_step(self, batch, batch_idx):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )
        outputs = self.model(inputs)
        outputs = self.discriminator_model(outputs, inputs, self.global_step)
        _, losses_to_log = self.calculate_loss(
            "valid", batch_idx, outputs, targets, False
        )
        return losses_to_log

    def validation_epoch_end(self, outputs: tp.List[tp.Any]):
        losses_to_log = outputs
        validation_losses = LightningEngine.aggregate_losses_to_log(losses_to_log)
        self.log_dict(validation_losses)
        self.log("Epoch", self.trainer.current_epoch)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs, targets, metadata = self.batch_processor(
                batch, batch_idx, self.global_step
            )
            outputs = self.model(inputs)
            outputs = self.discriminator_model(outputs, inputs, self.global_step)
        return inputs, outputs, targets, metadata

    def configure_optimizers(self):
        generator_optimizer = self.g_optimizer.optimizer
        generator_scheduler = {
            "scheduler": self.g_optimizer.lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        discriminator_optimizer = self.d_optimizer.optimizer
        discriminator_scheduler = {
            "scheduler": self.d_optimizer.lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [generator_optimizer, discriminator_optimizer], [
            generator_scheduler,
            discriminator_scheduler,
        ]

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self.saver.to_save)


class GANLightningEngineWithManualOptimization(GANLightningEngine):
    def __init__(
        self,
        generator_model: BaseTorchModel,
        discriminator_model: BaseTorchModel,
        criterion: BaseCriterion,
        batch_processor: BaseBatchProcessor,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        saver: ExperimentSaver,
    ):
        try:
            super().__init__(
                generator_model,
                discriminator_model,
                criterion,
                batch_processor,
                g_optimizer,
                d_optimizer,
                saver,
            )
        except Exception as e:
            LOGGER.error(trace(self, e))

        self.validation_losses = []
        self.automatic_optimization = False

        self.on_pretrain_routine_start = None
        self.training_epoch_end = None
        self.validation_epoch_end = None

        assert int(pl.__version__[0]) >= 2, RuntimeError(
            "pytorch_lightning==2.0.7 or higher required"
        )

    def on_fit_start(self):
        self.batch_processor.set_device(self.device)

    def _evaluate_model(self, inputs, targets, batch_idx: int, optimizer_idx: int):
        # Train Generator
        if optimizer_idx == 0:
            # Generator takes only inputs:
            outputs = self.model(inputs)
            # Discriminator takes generator outputs and inputs.
            outputs = self.discriminator_model(outputs, inputs, self.global_step)

        # Train Discriminator
        else:
            # Generator takes only inputs:
            with torch.no_grad():
                outputs = self.model(inputs)
            # Discriminator takes generator outputs and inputs.
            outputs = self.discriminator_model(outputs, inputs, self.global_step)

        is_generator_training = True if optimizer_idx == 0 else False
        total_loss, losses_to_log = self.calculate_loss(
            "train", batch_idx, outputs, targets, is_generator_training
        )
        self.log_dict(losses_to_log, add_dataloader_idx=False)

        if total_loss.requires_grad:
            self.log(
                "total_loss",
                total_loss,
                prog_bar=True,
                on_step=True,
                add_dataloader_idx=False,
            )

        return total_loss

    def _manual_optimization_step(
        self, loss, opt, sch, freq, batch_idx, gradient_clip_val=4.0
    ):
        opt.zero_grad()
        if loss.requires_grad:
            self.manual_backward(loss)
            self.clip_gradients(
                opt, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm="norm"
            )
            if freq > 0 and (batch_idx + 1) % freq == 0:
                opt.step()
                sch.step()

    def training_step(self, batch: Batch, batch_idx: int):
        inputs, targets, _ = self.batch_processor(batch, batch_idx, self.global_step)

        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        if hasattr(self.trainer, "optimizer_frequencies"):
            g_freq, d_freq = self.trainer.optimizer_frequencies
        else:
            g_freq = d_freq = 1

        loss = self._evaluate_model(inputs, targets, batch_idx, optimizer_idx=0)
        self._manual_optimization_step(loss, g_opt, g_sch, g_freq, batch_idx)

        loss = self._evaluate_model(inputs, targets, batch_idx, optimizer_idx=1)
        self._manual_optimization_step(loss, d_opt, d_sch, d_freq, batch_idx)

    def on_train_epoch_end(self):
        log_to_file(trace(self, message=f"Epoch {self.trainer.current_epoch} completed!"))

    def validation_step(self, batch, batch_idx):
        losses_to_log = super().validation_step(batch, batch_idx)
        self.validation_losses.append(losses_to_log)

    def on_validation_epoch_end(self):
        losses_to_log = self.validation_losses
        validation_losses = LightningEngine.aggregate_losses_to_log(losses_to_log)
        self.log_dict(validation_losses, add_dataloader_idx=False)
        self.log("Epoch", float(self.trainer.current_epoch), add_dataloader_idx=False)
        self.validation_losses = []
