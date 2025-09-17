import uuid

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.utils import save_image
from tqdm import tqdm

from speechflow.utils.fs import get_root_dir


def prepare_dataset(data_root: Path):
    data_root.mkdir(parents=True, exist_ok=True)
    flist_path = data_root / "filelist.txt"
    if flist_path.exists():
        flist_path.unlink()

    transform = transforms.Compose([transforms.ToTensor()])

    def save(batch_loader, subset: str):
        with open(flist_path, "a") as f:
            f.write(f"[{subset.upper()}]\n")
            for data, target in tqdm(batch_loader, desc=f"Extract {subset} data"):
                label = str(target[0].numpy())
                data_path = data_root / subset / label
                data_path.mkdir(parents=True, exist_ok=True)
                file_path = data_path / (uuid.uuid4().hex + ".png")

                save_image(data, file_path)
                f.write(f"{file_path.relative_to(data_root)}|{label}\n")

    root = get_root_dir() / "examples/simple_datasets/mnist"
    train_dataset = torchvision.datasets.MNIST(
        root=root.as_posix(), train=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=1, shuffle=False
    )
    save(train_loader, "train")

    test_dataset = torchvision.datasets.MNIST(
        root=root.as_posix(), download=True, train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=1, shuffle=False
    )
    save(test_loader, "test")


def prepare_data():
    data_root = Path("temp")
    if not data_root.exists():
        prepare_dataset(data_root)


if __name__ == "__main__":
    prepare_data()
    print("Prepare MNIST dataset is done!")
