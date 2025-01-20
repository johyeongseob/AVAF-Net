import os, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3665, 0.3665, 0.3665], std=[0.1911, 0.1911, 0.1911])
])


class SingleViewDataset(Dataset):
    def __init__(self, root_dir=None, view=None):
        self.root_dir = root_dir
        self.data = []
        self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']
        self.view = view

        print(f"SingleViewDataset. classes: {self.classes}, View: {self.view}.")

        # 이미지 경로와 레이블을 한 번에 생성
        for label, class_name in enumerate(self.classes):
            dir_path = os.path.join(self.root_dir, class_name, self.view)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = transform(image)  # [C, H, W]
        return image, label


class MultiViewDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']
        self.views = ['Down', 'Upper', 'Left', 'Right']

        print(f"MultiViewDataset. classes: {self.classes}, View: {self.views}.")

        for label, class_name in enumerate(self.classes):
            single_view_dir = os.path.join(self.root_dir, class_name, self.views[0])
            file_names = os.listdir(single_view_dir)
            for file_name in file_names:
                views_paths = []
                for view in self.views:
                    modified_file_name = file_name.replace("Down", view)
                    file_path = os.path.join(self.root_dir, class_name, view, modified_file_name)
                    views_paths.append(file_path)
                if len(views_paths) == len(self.views):
                    self.data.append((views_paths, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        views_paths, label = self.data[idx]
        images_tensor = torch.stack([
            transform(Image.open(view_path).convert("RGB"))
            for view_path in views_paths
        ])  # [4, C, H, W]

        return images_tensor, label


class MultiViewWithNormalDataset(Dataset):
    def __init__(self, root_dir, augmentation=None):
        self.root_dir = root_dir
        self.augmentation = augmentation
        self.data = []
        self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']
        self.views = ['Down', 'Upper', 'Left', 'Right']

        print(f"MultiViewWithNormalDataset. augmentation: {augmentation}, classes: {self.classes}, View: {self.views}.")

        for label, class_name in enumerate(self.classes):
            single_view_dir = os.path.join(self.root_dir, class_name, self.views[0])
            file_names = os.listdir(single_view_dir)

            for file_name in file_names:
                views_paths = []
                for view in self.views:
                    modified_file_name = file_name.replace("Down", view)
                    file_path = os.path.join(self.root_dir, class_name, view, modified_file_name)
                    views_paths.append(file_path)

                inverse_mapping = {
                    "BrightLine": random.choice(["Scratch", "Normal"]),
                    "Deformation": "Normal",
                    "Dent": "Normal",
                    "Scratch": random.choice(["BrightLine", "Normal"]),
                    "Normal": random.choice(["BrightLine", "Deformation", "Dent", "Scratch"]),
                }
                inverse_class_name = inverse_mapping.get(class_name, "")
                inverse_view_dir = os.path.join(self.root_dir, inverse_class_name, self.views[0])
                inverse_names = os.listdir(inverse_view_dir)
                inverse_name, inverse_view = random.choice(inverse_names), random.choice(self.views)
                modified_normal_name = inverse_name.replace("Down", inverse_view)
                inverse_path = os.path.join(self.root_dir, inverse_class_name, inverse_view, modified_normal_name)
                views_paths.append(inverse_path)

                if len(views_paths) == 5:
                    self.data.append((views_paths, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        views_paths, label = self.data[idx]

        # 랜덤 시드 고정
        seed = idx
        random.seed(seed)
        torch.manual_seed(seed)

        # Random augmentation parameters
        apply_aug = (random.random() <= 0.3)
        random_int = random.choice([90, 180, 270])

        augmentation_list = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(degrees=(random_int, random_int))
        ]

        if apply_aug and self.augmentation:
            aug = random.choice(augmentation_list)  # Choose a random augmentation
        else:
            aug = None  # No augmentation

        images_tensor = torch.stack([
            transform(aug(Image.open(view_path).convert("RGB")) if aug else Image.open(view_path).convert("RGB"))
            for view_path in views_paths
        ])  # [5, C, H, W]

        return images_tensor, label

if __name__ == '__main__':
    train_dir = 'dataset/train'
    train_dataset = MultiViewWithNormalDataset(root_dir=train_dir, augmentation=True)
