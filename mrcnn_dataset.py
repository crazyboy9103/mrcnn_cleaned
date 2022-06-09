import os
from tqdm import tqdm
import numpy as np
labels = {
    1: "container_truck", 
    2: "forklift", 
    3: "reach_stacker", 
    4: "ship"
}
def process_dataset(data_dir):
    # Process data and return 
    #    num_classes: num objs + 1 (background)  
    #    images: dict of img paths 
    #    labels: dict of labels (dict)
    # e.g.
    # images = {i: img for i, img in enumerate(images)}
    # labels = {i: label for i, label in enumerate(labels)}

    # num_classes = len(label_map)
    # num_classes += 1 # background class
    return num_classes, images, labels

import cv2
import torch
from torch.utils.data import Dataset

W_new, H_new = 448, 336

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_name, transform=None):
        assert data_name.endswith(".pt"), "data_name must end with .pt"
        try:
            self.load(data_name)
            self.dir = data_dir
            print(f"Data loaded from {data_name} from directory {data_dir}")
            self.transform = transform # override transform to apply augmentation
        except:
            print(f"Error loading {data_name}, processing data from {data_dir}")
            self.dir = data_dir
            self.transform = transform
            self.num_classes, self.images, self.labels = process_dataset(data_dir)
            self.save(data_name)

        print("Dataset Initialized")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        img = cv2.imread(os.path.join(self.dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W_new, H_new), interpolation=cv2.INTER_AREA)

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        img = img.transpose((2,0,1)).astype(np.float32)
        img = img / 255.0
        img = torch.as_tensor(img, dtype=torch.float32)
    
        boxes = torch.as_tensor(label['boxes'], dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(label['labels'], dtype=torch.int64)
        target["masks"] = torch.as_tensor(label['masks'], dtype=torch.bool)
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes), ), dtype=torch.int64)

        if self.transform:
            img, target = self.transform(img, target)

        return img, target
    
    def save(self, data_name):
        temp = {
            "transform":self.transform,
            "images":self.images,
            "labels":self.labels,
            "num_classes":self.num_classes
        }
        
        torch.save(temp, data_name)
        print(f"dataset saved to {data_name}")

    def load(self, data_name):
        temp = torch.load(data_name)
        self.transform = temp["transform"]
        self.images = temp["images"]
        self.labels = temp["labels"]
        self.num_classes = temp["num_classes"]