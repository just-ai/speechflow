from pathlib import Path

import torch

from examples import mnist
from speechflow.data_pipeline.core import PipelineComponents
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.init import init_class_from_config


def evaluation(ckpt_path: Path, image_path: Path):
    checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
    cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

    cfg_model["trainer"]["accelerator"] = "cpu"
    cfg_model["trainer"]["devices"] = 1

    model_cls = getattr(mnist, cfg_model["model"]["type"])
    mnist_net = model_cls(checkpoint["params"])
    mnist_net.eval()

    mnist_net.load_state_dict(checkpoint["state_dict"])

    pipeline = PipelineComponents(cfg_data, "test")

    batch_processor_cls = getattr(mnist, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    metadata = {"file_path": image_path, "label": image_path.parents[0].name}
    batch = pipeline.metadata_to_batch([metadata])
    inputs, targets, metadata = batch_processor(batch)

    result = mnist_net(inputs)
    print("predicted class:", torch.argmax(result.logits).item())


if __name__ == "__main__":
    _ckpt_path = ExperimentSaver.get_last_checkpoint(
        "_logs/16_May_2024_14_31_31_mnist_expr"
    )
    _image_path = next(Path("temp/test/5").glob("*.png"))
    evaluation(_ckpt_path, _image_path)
