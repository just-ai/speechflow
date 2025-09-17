import typing as tp
import logging

from itertools import combinations

import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning import Callback
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoTokenizer

from nlp.prosody_prediction.data_types import (
    ProsodyPredictionOutput,
    ProsodyPredictionTarget,
)
from speechflow.data_server.loader import DataLoader
from speechflow.utils.pad_utils import pad_1d, pad_2d

LOGGER = logging.getLogger("root")

__all__ = ["ProsodyCallback"]


class ProsodyCallback(Callback):
    def __init__(
        self,
        data_loader: DataLoader,
        names: tp.List[tp.Literal["binary", "category"]],
        tokenizer_name: str,
        n_classes: int,
    ):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax_2 = torch.nn.Softmax(dim=2)
        self.names = names
        self._data_loader = data_loader
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=True, use_fast=True
        )
        self.n_classes = n_classes

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._data_loader.reset()
        outputs_all = ProsodyPredictionOutput(
            binary=[],  # type: ignore
            category=[],  # type: ignore
        )
        targets_all = ProsodyPredictionTarget(
            binary=[],  # type: ignore
            category=[],  # type: ignore
        )

        for batch_idx in range(len(self._data_loader)):
            batch = self._data_loader.next_batch()
            _, outputs, targets, _ = pl_module.test_step(batch, batch_idx)

            if outputs.binary is not None:
                outputs_all.binary.extend(outputs.binary.cpu())  # type: ignore
                targets_all.binary.extend(targets.binary.cpu())  # type: ignore

            if outputs.category is not None:
                outputs_all.category.extend(outputs.category.cpu())  # type: ignore
                targets_all.category.extend(targets.category.cpu())  # type: ignore

            # if batch_idx == 0:
            #     output_text = self._log_text(batch, outputs)
            #     pl_module.logger.experiment.add_text("text_with_labels", output_text, global_step=pl_module.global_step)

        if outputs_all.binary:
            outputs_all.binary, _ = pad_2d(outputs_all.binary, pad_val=0, n_channel=2)
            targets_all.binary, _ = pad_1d(targets_all.binary, pad_val=-100)

        if outputs_all.category:
            outputs_all.category, _ = pad_2d(
                outputs_all.category,
                pad_val=0,
                n_channel=outputs_all.category[0].shape[1],
            )
            targets_all.category, _ = pad_1d(targets_all.category, pad_val=-100)

        metrics, reports = self.compute_metrics(outputs_all, targets_all)
        pl_module.log_dict(metrics, prog_bar=True, logger=True)

        if pl_module.logger is not None:
            for name in self.names:
                pl_module.logger.experiment.add_text(
                    f"report_{name}", reports[name], global_step=pl_module.global_step
                )

    def compute_metrics(
        self,
        outputs: ProsodyPredictionOutput,
        targets: ProsodyPredictionTarget,
    ):
        metrics = {}
        reports = {}
        for name in self.names:
            predicted = getattr(outputs, name).squeeze(-1)
            target = getattr(targets, name).float()
            orig_shape = predicted.shape

            _is_prosody = target != -100
            predicted = predicted.masked_select(_is_prosody.unsqueeze(-1)).reshape(
                (-1, orig_shape[-1])
            )
            target = target.masked_select(_is_prosody)
            probabilities = (
                self.softmax(predicted)
                if name == "category"
                else self.softmax(predicted)[:, 1]
            )

            if name == "binary":
                metrics[f"{name}/EER"], threshold = self.compute_eer(
                    target, probabilities
                )
                predictions = (probabilities >= threshold).type(torch.uint8)
            else:
                metrics[f"{name}/EER"] = self.compute_multiclass_eer(
                    target, probabilities
                )
                predictions = predicted.argmax(-1)

            metrics[f"{name}_EER"] = metrics[f"{name}/EER"]

            roc_auc = roc_auc_score(
                target,
                probabilities,
                multi_class="ovo",
                average="macro",
                labels=torch.arange(self.n_classes)
                if name == "category"
                else torch.arange(2),
            )
            acc = accuracy_score(target, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                target, predictions, average="macro"
            )

            metrics[f"{name}/Accuracy"] = acc
            metrics[f"{name}/F1"] = f1
            metrics[f"{name}/Precision"] = precision
            metrics[f"{name}/Recall"] = recall
            metrics[f"{name}/Roc_auc"] = roc_auc

            reports[name] = classification_report(target, predictions, output_dict=False)

        return metrics, reports

    def compute_multiclass_eer(self, y_true, y_score):
        y_true_unique = np.unique(y_true)
        n_classes = y_true_unique.shape[0]
        n_pairs = n_classes * (n_classes - 1) // 2
        pair_scores = np.empty(n_pairs)

        for ix, (a, b) in enumerate(combinations(y_true_unique, 2)):
            a_mask = y_true == a
            b_mask = y_true == b
            ab_mask = np.logical_or(a_mask, b_mask) == 1

            a_true = a_mask[ab_mask]
            b_true = b_mask[ab_mask]
            idx = torch.arange(ab_mask.shape[0])[ab_mask]
            probs = torch.index_select(y_score, 0, idx)

            a_true_score, _ = self.compute_eer(a_true.type(torch.uint8), probs[:, int(a)])
            b_true_score, _ = self.compute_eer(b_true.type(torch.uint8), probs[:, int(b)])
            pair_scores[ix] = (a_true_score + b_true_score) / 2

        return np.average(pair_scores)

    @staticmethod
    def compute_eer(label, pred):
        fpr, tpr, threshold = roc_curve(label, pred)
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
        eer_1 = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        eer_2 = fnr[np.nanargmin(np.absolute(fnr - fpr))]

        eer = (eer_1 + eer_2) / 2
        return eer, eer_threshold

    def _log_text(self, batch, outputs):
        output_text = ""
        result = {
            "tokens": "",
            "sep": "",
            "pred_binary": "",
            "pred_category_prob": "",
            "pred_category_label": "",
            "targets_category": "",
        }

        inputs = batch.collated_samples.input_ids
        targets = {
            "binary": batch.collated_samples.binary,
            "category": batch.collated_samples.category,
        }
        predictions = {
            "binary": self.softmax_2(getattr(outputs, "binary").squeeze(-1))[:, :, 1]
            if "binary" in self.names
            else None,
            "category": self.softmax_2(getattr(outputs, "category").squeeze(-1))
            if "category" in self.names
            else None,
        }

        for sample_idx in range(4):
            for token_idx in range(len(inputs[sample_idx])):
                if targets["binary"][sample_idx][token_idx] != -100:
                    result[
                        "tokens"
                    ] += f"| {self._tokenizer.decode(inputs[sample_idx][token_idx])} "
                    result["sep"] += "| ---- "
                    for name in self.names:
                        label = predictions[name][sample_idx][token_idx]
                        if name == "binary":
                            result[f"pred_{name}"] += f"| {round(label.item(), 2)} "
                        else:
                            label_idx = label.argmax(-1)
                            result[
                                "pred_category_prob"
                            ] += f"| {round(label[label_idx].item(), 2)} "
                            result["pred_category_label"] += f"| {label_idx} "
                            result[
                                "targets_category"
                            ] += f"| {targets[name][sample_idx][token_idx]} "

            result["tokens"] += "|"
            result["sep"] += "|"
            result["pred_binary"] += "|"
            result["pred_category_prob"] += "|"
            result["pred_category_label"] += "|"
            result["targets_category"] += "|"

            output_text += "\n".join(
                [result[part] for part in result if len(result[part]) != 0]
            )
            output_text += "\n\n"

            result = {
                "tokens": "",
                "sep": "",
                "pred_binary": "",
                "pred_category_prob": "",
                "pred_category_label": "",
                "targets_category": "",
            }

        return output_text
