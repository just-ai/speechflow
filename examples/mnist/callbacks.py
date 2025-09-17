import torch

from pytorch_lightning.callbacks import Callback

from speechflow.data_server.loader import DataLoader

__all__ = ["AccuracyCallback"]


class AccuracyCallback(Callback):
    def __init__(self, data_loader: DataLoader):
        super().__init__()
        self._data_loader = data_loader

    def on_train_epoch_end(self, trainer, pl_module):
        self._data_loader.reset()

        _all, _correct = 0, 0
        for batch_idx, batch in enumerate(self._data_loader.get_epoch_iterator()):
            outputs = pl_module.test_step(batch, batch_idx)
            inputs, outputs, targets, metadata = outputs

            for idx, out in enumerate(outputs.logits[:]):
                pred_label = torch.argmax(out).item()
                true_label = targets.label[idx].item()
                _correct += pred_label == true_label
                _all += 1

        metrics = {
            "test/Accuracy": _correct / _all,
            "test/False": _all - _correct,
        }
        pl_module.log_dict(metrics, prog_bar=True, logger=True)
