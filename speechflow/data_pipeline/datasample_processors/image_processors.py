from PIL import Image

from speechflow.data_pipeline.core import BaseDSProcessor
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import ImageDataSample

__all__ = ["ImageProcessor"]


class ImageProcessor(BaseDSProcessor):
    """Image processors."""

    def __init__(self):
        """Everything that we need to init."""
        super().__init__()

    @PipeRegistry.registry(inputs={"file_path"}, outputs={"image"})
    def process(self, ds: ImageDataSample) -> ImageDataSample:
        """Returns processed data."""
        from torchvision import transforms

        ds = super().process(ds)

        pil_img = Image.open(ds.file_path)
        pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)
        ds.image = pil_to_tensor
        return ds.to_numpy()
