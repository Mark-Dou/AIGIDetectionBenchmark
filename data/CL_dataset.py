import os
import torch
from typing import List, Dict
from pathlib import Path
from PIL import Image

from .process import get_processing_model, processing_LGrad, processing_LNP, processing, processing_DCT

def dataset_folder(opt, root):
    return AIGCImageDataset(opt, root)

class AIGCImageDataset(torch.utils.data.Dataset):
    def __init__(self, opt, root_dir):
        super(AIGCImageDataset, self).__init__()
        
        self.opt = get_processing_model(opt, opt.device)

        if self.opt.detection_model == 'SAFE':
            TRANSFORM = Get_Transforms(self.opt)
            self.transform = TRANSFORM[0] if self.opt.isTrain else TRANSFORM[1]

        self.root_dir = Path(root_dir)
        self.samples = []

        for class_id, class_name in enumerate(sorted(os.listdir(root_dir))):

            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.suffix in ['.jpg', '.jpeg', '.png', '.JPEG']:
                    self.samples.append((image_path, class_id))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_id = self.samples[idx]
        img = Image.open(str(image_path)).convert('RGB')

        if self.opt.detection_model == 'LGrad':
            self.opt.cropSize = 256
            img = processing_LGrad(img, self.opt.gen_model, self.opt, self.opt.device)
        elif self.opt.detection_model == 'LNP':
            img = processing_LNP(img, self.opt.model_restoration, self.opt)
        elif self.opt.detection_model in ['GramNet', 'CNNSpot', 'NPR']:
            img = processing(img, self.opt, 'imagenet')
        elif self.opt.detection_model == 'FreDect':
            img = processing_DCT(img, self.opt)
        elif self.opt.detection_model in ['UnivFD', 'SimE', 'FatFormer']:
            img = processing(img, self.opt, 'clip')
        elif self.opt.detection_model == 'SAFE':
            img = self.transform(img)
        else:
            raise ValueError(f"Unsupported model_type: {self.opt.detection_model}")

        return {"image": img, "label": class_id}

def get_dataset(opt, name, id, dataroot, multiclass):
    dset_lst = []

    dataroot = os.path.join(dataroot, name)
    classes = os.listdir(dataroot) if multiclass[id] else ['']
    for cls in classes:
        root = dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)

    return torch.utils.data.ConcatDataset(dset_lst)

def create_sequential_datasets(opt,
                               dataset_names: List[str],
                               multiclass: List[str],
                               dataroot: str = "./CL_datasets",):
    data = {}
    for id, name in enumerate(dataset_names):
        dataset = get_dataset(opt, name, id, dataroot, multiclass)
        data[name] = dataset

    return data