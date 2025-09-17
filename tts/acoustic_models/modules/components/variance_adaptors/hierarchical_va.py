import typing as tp
import logging

import torch

from torch.nn import functional as F

from speechflow.logging import trace
from speechflow.utils.tensor_utils import (
    apply_mask,
    get_lengths_from_durations,
    get_mask_from_lengths,
    merge_additional_outputs,
)
from tts.acoustic_models.modules.common import VarianceEmbedding
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import EncoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import VarianceAdaptorParams

__all__ = ["HierarchicalVarianceAdaptor", "HierarchicalVarianceAdaptorParams"]

LOGGER = logging.getLogger("root")

DP_NAME = "durations"


class HierarchicalVarianceAdaptorParams(VarianceAdaptorParams):
    pass


class HierarchicalVarianceAdaptor(Component):
    params: HierarchicalVarianceAdaptorParams

    def __init__(
        self,
        params: HierarchicalVarianceAdaptorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        from tts.acoustic_models.modules import (
            TTS_LENGTH_REGULATORS,
            TTS_VARIANCE_PREDICTORS,
        )

        super().__init__(params, input_dim)

        if isinstance(input_dim, tp.Sequence):
            self.input_dim = input_dim
        else:
            self.input_dim = (input_dim,)

        self.va_variances: tp.Tuple[str, ...] = params.va_variances  # type: ignore
        self.va_variance_params = params.va_variance_params

        assert isinstance(self.va_variances, tp.Sequence)
        assert isinstance(self.va_variance_params, tp.Dict)

        self.predictors = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            predictor_cls = TTS_VARIANCE_PREDICTORS[variance_params.predictor_type]
            predictor_params = variance_params.predictor_params

            if isinstance(predictor_cls, tp.Sequence):
                predictor_cls, predictor_param_cls = predictor_cls
                predictor_params = predictor_param_cls.check_deprecated_params(
                    predictor_params
                )
            else:
                predictor_param_cls = None

            if variance_params.input_content_dim is None:
                input_content_dim = [None] * len(variance_params.input_content)
            else:
                input_content_dim = variance_params.input_content_dim

            content_dim = []
            for i, index in enumerate(variance_params.input_content):
                if str(index).isdigit():
                    content_dim.append(self.input_dim[index])
                elif input_content_dim[i] not in [None, "None"]:
                    content_dim.append(input_content_dim[i])
                else:
                    if hasattr(params, f"{index}_proj_dim"):
                        content_dim.append(getattr(params, f"{index}_proj_dim"))
                    else:
                        for vp_name, vp in self.va_variance_params.items():
                            if vp.tag == index:
                                content_dim.append(vp.predictor_params.vp_latent_dim)
                            elif vp_name == index:
                                content_dim.append(vp.dim)

            content_dim = sum(d for d in content_dim)

            vp_params = predictor_param_cls.init_from_parent_params(
                params, predictor_params.vp_params
            )
            for key in vp_params.to_dict():
                if key in predictor_params.to_dict():
                    setattr(vp_params, key, getattr(predictor_params, key))

            if hasattr(vp_params, "variance_params"):
                vp_params.variance_params = variance_params

            self.predictors[name] = predictor_cls(vp_params, content_dim)

            predictor_params.vp_output_dim = self.predictors[name].output_dim

        self.embeddings = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            if variance_params.as_embedding:
                self.embeddings[name] = VarianceEmbedding(
                    interval=variance_params.interval,
                    n_bins=variance_params.n_bins,
                    emb_dim=variance_params.emb_dim,
                    log_scale=variance_params.log_scale,
                )

        self.length_regulators = torch.nn.ModuleDict()
        for name in self.va_variances:
            variance_params = self.va_variance_params[name]
            if variance_params.upsample or name == DP_NAME:
                length_regulator_cls = TTS_LENGTH_REGULATORS[
                    params.va_length_regulator_type
                ]
                self.length_regulators[name] = length_regulator_cls()

        self._ssml_length_regulator = None

    @property
    def output_dim(self):
        out_dim = list(self.input_dim)
        for name in self.va_variances:
            if name == DP_NAME:
                continue

            var_params = self.va_variance_params[name]
            dim = self.predictors[name].output_dim

            if var_params.as_embedding:
                dim = var_params.emb_dim * dim

            if var_params.cat_to_content is not None:
                cnt_ids = (
                    var_params.cat_to_content
                    if var_params.cat_to_content
                    else range(len(out_dim))
                )
                for i in cnt_ids:
                    out_dim[i] += dim

            if var_params.overwrite_content is not None:
                cnt_ids = (
                    var_params.overwrite_content
                    if var_params.overwrite_content
                    else range(len(out_dim))
                )
                for i in cnt_ids:
                    out_dim[i] = dim

        return tuple(out_dim)

    @property
    def _ssml_lr(self):
        from tts.acoustic_models.modules import TTS_LENGTH_REGULATORS

        if self._ssml_length_regulator is None:
            length_regulator_cls = TTS_LENGTH_REGULATORS[
                self.params.va_length_regulator_type
            ]
            self._ssml_length_regulator = length_regulator_cls()
            self._ssml_length_regulator.sigma = torch.nn.Parameter(
                torch.tensor(999999.0), requires_grad=False
            )
        return self._ssml_length_regulator

    @staticmethod
    def _get_targets(
        inputs: EncoderOutput, variance_names, variance_params
    ) -> tp.Dict[str, torch.Tensor]:
        targets = {}
        content = inputs.get_content()
        for var_name in variance_names:
            target_name = var_name
            if variance_params[var_name].target is not None:
                target_name = variance_params[var_name].target

            if "content" in target_name:
                targets[var_name] = content[int(target_name.replace("content_", ""))]
            else:
                target = getattr(
                    inputs.model_inputs, target_name.replace("_encoder", ""), None
                )
                if target is None:
                    if "dummy" not in var_name:
                        target = inputs.model_inputs.additional_inputs.get(target_name)

                targets[var_name] = target

        return targets

    @staticmethod
    def _get_modifier_name(var_name: str) -> str:
        if var_name == "energy":
            return "volume_modifier"
        elif var_name == "aggregate_pitch":
            return "pitch_modifier"
        else:
            return ""

    @staticmethod
    def _get_input_content(
        content,
        content_length,
        variance_params,
        model_inputs,
    ):
        cat_content = []
        for name, is_detach in zip(
            variance_params.input_content, variance_params.detach_input
        ):
            if str(name).isdigit():
                cat_content.append(content[name])
            else:
                feat = model_inputs.additional_inputs.get(name)
                if feat is None:
                    feat = getattr(model_inputs, name, None)
                if feat is not None:
                    if feat.ndim == 2:
                        feat = feat.unsqueeze(1)
                    if feat.shape[1] == 1:
                        if len(cat_content) > 0:
                            t_dim = cat_content[-1].shape[1]
                        else:
                            t_dim = content[0].shape[1]
                        feat = feat.expand(-1, t_dim, -1)

                cat_content.append(feat)

            if is_detach:
                cat_content[-1] = cat_content[-1].detach()

        full_content = torch.cat(cat_content, dim=-1)

        for cnt, lens in zip(content, content_length):
            if full_content.shape[1] == cnt.shape[1]:
                full_content_length = lens
                break
        else:
            raise RuntimeError

        return full_content, full_content_length

    def _predict_durations(
        self,
        content,
        content_length,
        targets,
        model_inputs,
        **kwargs,
    ):
        var_params = self.va_variance_params[DP_NAME]

        content, content_length = self._get_input_content(
            content,
            content_length,
            var_params,
            model_inputs,
        )

        prediction, content, loss = self.predictors[DP_NAME](
            content,
            content_length,
            model_inputs,
            target=targets.get(DP_NAME),
            name=DP_NAME,
        )

        if (
            (self.training and var_params.use_target)
            or kwargs.get("use_target_variances", False)
        ) and targets.get(DP_NAME) is not None:
            durations = targets[DP_NAME]
        else:
            durations = prediction.detach() if var_params.detach_output else prediction

        return durations, prediction, content, loss

    def _process_durations(
        self,
        content,
        content_length,
        targets,
        model_inputs,
        **kwargs,
    ):
        variance_params = self.va_variance_params.get(DP_NAME)
        ignored_variance = kwargs.get("ignored_variance")
        current_iter = model_inputs.batch_idx

        if (
            DP_NAME in self.predictors
            and ignored_variance
            and DP_NAME in ignored_variance
        ):
            return model_inputs.durations, model_inputs.durations, {}, {}

        if DP_NAME not in targets and DP_NAME not in self.predictors:
            return None, None, {}, {}

        if "fa_postprocessed" in model_inputs.additional_inputs:
            durations = model_inputs.additional_inputs["fa_postprocessed"]
            durations = durations.squeeze(1).squeeze(-1)
            return durations, durations, {f"{DP_NAME}_postprocessed": durations}, {}

        if DP_NAME in self.predictors:
            (
                durations,
                durations_prediction,
                durations_content,
                durations_loss,
            ) = self._predict_durations(
                content,
                content_length,
                targets,
                model_inputs,
                **kwargs,
            )
        else:
            durations = durations_prediction = targets.get(DP_NAME)
            durations_content = {}
            durations_loss = {}

        if self.training and not (
            variance_params.begin_iter <= current_iter < variance_params.end_iter
            and current_iter % variance_params.every_iter == 0
        ):
            durations = durations_prediction = targets.get(DP_NAME)
            durations_content = {}
            durations_loss = {}

        if variance_params.denormalize:
            _range = model_inputs.ranges[DP_NAME]
            durations = durations * _range[:, 2:3] + _range[:, 0:1]
            if model_inputs.durations is not None:
                model_inputs.durations = (
                    model_inputs.durations * _range[:, 2:3] + _range[:, 0:1]
                )
                _, _length = self._get_input_content(
                    content,
                    content_length,
                    self.va_variance_params[DP_NAME],
                    model_inputs,
                )
                model_inputs.durations = apply_mask(
                    model_inputs.durations, get_mask_from_lengths(_length)
                )

        durations_content.update({f"{DP_NAME}_postprocessed": durations})
        return durations, durations_prediction, durations_content, durations_loss

    def _predict_variance(
        self,
        name,
        content,
        content_length,
        target,
        model_inputs,
        modifier=None,
        **kwargs,
    ):
        variance_params = self.va_variance_params[name]

        content, content_length = self._get_input_content(
            content,
            content_length,
            variance_params,
            model_inputs,
        )

        predictor = self.predictors[name]
        try:
            prediction, additional_content, additional_losses = predictor(
                content,
                content_length,
                model_inputs,
                target=target,
                name=name,
                **kwargs,
            )
        except Exception as e:
            LOGGER.error(trace(self, e, message=f"{name} prediction failed"))
            raise e

        if modifier is not None and prediction.shape[1] == modifier.shape[1]:
            prediction *= modifier.unsqueeze(-1) if prediction.ndim > 2 else modifier

        return prediction, additional_content, additional_losses

    def _process_variance(
        self,
        content,
        content_length,
        targets,
        model_inputs,
        **kwargs,
    ) -> tp.Tuple[tp.Dict[str, torch.Tensor], ...]:
        variance_embeddings = {}
        variance_predictions = {}
        variance_content = {}
        variance_losses = {}

        for name in self.va_variances:
            variance_params = self.va_variance_params.get(name)
            current_iter = model_inputs.batch_idx

            if name == DP_NAME:
                continue

            if kwargs.get("ignored_variance") and name in kwargs.get("ignored_variance"):
                continue

            if (
                self.training
                and variance_params.skip
                and not (
                    variance_params.begin_iter <= current_iter < variance_params.end_iter
                )
            ):
                continue

            (
                variance_predictions[name],
                variance_content[name],
                variance_losses[name],
            ) = self._predict_variance(
                name,
                content,
                content_length,
                targets.get(name),
                model_inputs,
                modifier=getattr(model_inputs, self._get_modifier_name(name), None),
                **kwargs,
            )

            if self.training:
                if not (
                    variance_params.begin_iter <= current_iter < variance_params.end_iter
                    and current_iter % variance_params.every_iter == 0
                ):
                    variance_losses.pop(name)
                    variance_predictions[name] = variance_predictions[name].detach()
                elif variance_params.use_loss:
                    loss_type = variance_params.loss_type
                    loss = getattr(F, loss_type)(
                        variance_predictions[name], targets[name].detach()
                    )
                    variance_losses[name].update({f"{name}_{loss_type}": loss})

        for name, prediction in variance_predictions.items():
            embeddings, variance = self._postprocessing_variance(
                name, prediction, targets.get(name), model_inputs, **kwargs
            )
            variance_embeddings[name] = embeddings
            variance_content[name].update({f"{name}_postprocessed": variance})

            variance_params = self.va_variance_params.get(name)
            if variance_params.tag != "default":
                variance_content[name].update({f"{variance_params.tag}": embeddings})

        return (
            variance_embeddings,
            variance_predictions,
            variance_content,
            variance_losses,
        )

    def _postprocessing_variance(
        self,
        name,
        prediction,
        target,
        model_inputs,
        **kwargs,
    ):
        variance_params = self.va_variance_params[name]
        ranges = model_inputs.ranges

        if (
            (self.training and variance_params.use_target)
            or kwargs.get("use_target_variances", False)
        ) and target is not None:
            variance = target
        else:
            variance = (
                prediction.detach() if variance_params.detach_output else prediction
            )

        if variance_params.denormalize:
            range_name = name
            if variance_params.target is not None:
                range_name = variance_params.target

            range_name = range_name.replace("aggregate_", "")
            variance = variance * ranges[range_name][:, 2:3] + ranges[range_name][:, 0:1]

        if variance_params.as_embedding:
            embedding_builder = self.embeddings[name]
            variance_emb = embedding_builder(variance)
        else:
            variance_emb = variance

        if variance_emb.ndim == 2:
            variance_emb = variance_emb.unsqueeze(-1)
            variance = variance.unsqueeze(-1)

        return variance_emb, variance

    def _process_content(
        self,
        content,
        content_lengths,
        variance_embeddings,
        durations,
        model_inputs,
    ):
        def _tensor_upsampling(content_, length_regulator_):
            if model_inputs.output_lengths is None:
                model_inputs.output_lengths = get_lengths_from_durations(durations)

            return length_regulator_(
                content_,
                durations,
                upsample_x2=True,
                max_length=model_inputs.output_lengths.max(),
            )

        def _cat_tensors(content_, embedding_):
            if embedding_.shape[1] == 1:
                embedding_ = embedding_.expand(content_.shape[0], content_.shape[1], -1)
            delta = content_.shape[1] - embedding_.shape[1]
            if abs(delta) > 1:
                message = f"delta: {delta}, content.shape: {content_.shape}, embedding.shape: {embedding_.shape}"
                LOGGER.warning(trace(self, message=message))
            if delta > 0:
                shape = list(embedding_.shape)
                shape[1] = delta
                embedding_ = torch.cat(
                    [embedding_, torch.zeros(shape).to(embedding_.device)], dim=1
                )
            if delta < 0:
                embedding_ = embedding_[:, :delta, :]
            return torch.cat([content_, embedding_], dim=-1)

        var_params = self.va_variance_params
        attention_weights = None

        for name, embedding in variance_embeddings.items():
            if name not in var_params:
                continue

            if name == DP_NAME and durations is not None:
                lr = self.length_regulators[DP_NAME]
                cnt_ids = (
                    var_params[DP_NAME].cat_to_content
                    if var_params[DP_NAME].cat_to_content
                    else range(len(content))
                )
                for i in cnt_ids:
                    content[i], attention_weights = _tensor_upsampling(content[i], lr)
                    content_lengths[i] = model_inputs.output_lengths
                continue

            if var_params[name].upsample:
                lr = self.length_regulators[name]
                embedding, attention_weights = _tensor_upsampling(embedding, lr)

            if var_params[name].cat_to_content is not None:
                cnt_ids = (
                    var_params[name].cat_to_content
                    if var_params[name].cat_to_content
                    else range(len(content))
                )
                for i in cnt_ids:
                    content[i] = _cat_tensors(content[i], embedding)

            if var_params[name].overwrite_content is not None:
                cnt_ids = (
                    var_params[name].overwrite_content
                    if var_params[name].overwrite_content
                    else range(len(content))
                )
                for i in cnt_ids:
                    content[i] = embedding
                    if embedding.shape[1] == model_inputs.input_lengths.max():
                        content_lengths[i] = model_inputs.input_lengths
                    elif embedding.shape[1] == model_inputs.output_lengths.max():
                        content_lengths[i] = model_inputs.output_lengths

        return content, content_lengths, attention_weights

    def forward_step(self, inputs: EncoderOutput, **kwargs) -> VarianceAdaptorOutput:  # type: ignore
        content = inputs.get_content()
        content_length = inputs.get_content_lengths()
        targets = self._get_targets(inputs, self.va_variances, self.va_variance_params)

        (durations, dura_prediction, dura_content, dura_loss,) = self._process_durations(
            content,
            content_length,
            targets,
            inputs.model_inputs,
            **kwargs,
        )

        (
            variance_embeddings,
            variance_predictions,
            variance_content,
            variance_losses,
        ) = self._process_variance(
            content,
            content_length,
            targets,
            inputs.model_inputs,
            **kwargs,
        )

        if dura_prediction is not None:
            variance_embeddings[DP_NAME] = durations
            variance_predictions[DP_NAME] = durations
            variance_content[DP_NAME] = dura_content
            variance_losses[DP_NAME] = dura_loss

        content, content_lengths, attention_weights = self._process_content(
            content,
            content_length,
            variance_embeddings,
            durations,
            inputs.model_inputs,
        )

        additional_content = merge_additional_outputs(
            inputs.additional_content, variance_embeddings
        )
        additional_content = merge_additional_outputs(
            additional_content, tuple(variance_content.values())
        )
        additional_losses = merge_additional_outputs(
            inputs.additional_losses, tuple(variance_losses.values())
        )

        return VarianceAdaptorOutput(
            content=content,
            content_lengths=content_lengths,
            attention_weights=attention_weights,
            variance_predictions=variance_predictions,
            embeddings=inputs.embeddings,
            model_inputs=inputs.model_inputs,
            additional_content=additional_content,
            additional_losses=additional_losses,
        )
