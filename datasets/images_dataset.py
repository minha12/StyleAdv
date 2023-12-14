import os
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from utils.align import process_input_image


class ImageDataset(Dataset):
    def __init__(self, image_folder, return_relative_paths=False, run_align=True):
        self.root_dir = image_folder
        self.return_relative_paths = return_relative_paths
        self.run_align = run_align
        self.image_paths = sorted(
            [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(image_folder)
                for f in fn
                if f.endswith(("jpg", "png", "jpeg"))
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        processed_image = process_input_image(image_path, self.run_align)
        if self.return_relative_paths:
            relative_path = os.path.relpath(image_path, self.root_dir)
            return processed_image, relative_path
        else:
            filename = os.path.basename(image_path)
            return processed_image, filename
