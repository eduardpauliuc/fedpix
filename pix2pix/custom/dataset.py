import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms


def load_img(path):
    return Image.open(path)


class MapDataset(Dataset):
    def __init__(self, root_dir, jitter=True, AtoB=False):
        self.root_dir = root_dir
        self.list_files = sorted(os.listdir(self.root_dir), key=lambda f: int(f[:-4]))
        self.jitter = jitter
        self.AtoB = AtoB

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = load_img(img_path)
        image_array = np.array(image)

        input_image = image_array[:, :600, :] if self.AtoB else image_array[:, 600:, :]
        target_image = image_array[:, 600:, :] if self.AtoB else image_array[:, :600, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        if self.jitter:
            input_image = config.transform_only_input(image=input_image)["image"]
        else:
            input_image = config.transform_only_input_no_jitter(image=input_image)["image"]

        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        transforms.ToPILImage()(make_grid(x)).save("x.png")

        save_image(make_grid(x), "x.jpg")
        save_image(y, "y.png")
        import sys

        sys.exit()
