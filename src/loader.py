from os import path, listdir as ls
from collections.abc import Iterable
from typing import Tuple
import json
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import torch
from pathlib import Path
from albumentations import Compose
import albumentations.augmentations.transforms as tf

affine_transformations = Compose(
    [
        tf.ShiftScaleRotate(scale_limit=0.3),
    ], p = 1
)

class Loader(Dataset):
    def __init__(
        self, 
        projection:str, 
        train:str, 
        metadata_prefix:str,
        augment:bool=False, 
        experts:str ='123',
        ):
        super().__init__()
        assert train in ['train', 'val', 'test']
        assert projection in ['PA', 'LAT', 'PA_LAT']
        assert isinstance(augment, bool)
        assert experts in ['1', '2', '3', '12', '13', '23', '123']
        self.train = train
        self.projection = projection
        self.augment = augment
        self.experts = ['expert'+expert for expert in experts]
        self.metadata_prefix = metadata_prefix
        self.indices_path = path.join(path.dirname(__file__), '../indices.json')
        self.metadata = self._load_metadata()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path_im, path_mask = self.metadata[idx]
        image = cv2.imread(path_im, 0)
        image = cv2.resize(image, (512,512))
        image = self.normalize(image)
        if path_mask is not None:
            mask = np.load(path_mask)['mask'].astype('uint8')
            mask = cv2.resize(mask, (512,512))
        else:
            mask = np.zeros((512,512)).astype('uint8')
        image = image.astype('float32')
        mask = mask.astype('uint8')
        if self.train == 'val' or self.train == 'test':
            return image, mask
        if self.augment:
            image, mask = self.augment_image(image, mask)
        else:
            image, mask = self.flip_augment_image(image, mask)
        return image, mask

    def _load_proj_metadata(self, proj):
        with open(self.indices_path, 'r') as file:
            indices = json.load(file)[self.train]
        pneu_inds = indices['pneumonia']
        norma_inds = indices['norma']
        all_pneu = ls(self.metadata_prefix+'images/pneumonia/')
        all_pneu_proj = [elem for elem in all_pneu if proj.lower() in elem]
        all_norma = ls(self.metadata_prefix+'images/norma/')
        all_norma_proj = [elem for elem in all_norma if proj.lower() in elem]
        metadata = list()
        pneu_names = [
            elem for elem in all_pneu_proj if int(elem.split('_')[0]) in pneu_inds
            ]
        norma_names = [
            elem for elem in all_norma_proj if int(elem.split('_')[0]) in norma_inds
            ]
        for exp in self.experts:
            exp_path = self.metadata_prefix+'masks/'+exp+'/'
            exp_list = [elem for elem in ls(exp_path) if proj.lower() in elem]
            exp_list = [elem for elem in exp_list if int(elem.split('_')[0]) in pneu_inds]
            for pneu in pneu_names:
                case_prefix = pneu.split('.')[0]
                mask_path = [elem for elem in exp_list if elem.startswith(case_prefix)]
                if mask_path:
                    case = (self.metadata_prefix+'images/pneumonia/'+pneu, exp_path+mask_path[0])
                    metadata.append(case)
        norma_cases = [(self.metadata_prefix+'images/norma/'+elem, None) for elem in norma_names]
        return metadata + norma_cases

    def _load_metadata(self):
        if self.projection in ['PA', 'LAT']:
            return self._load_proj_metadata(self.projection)
        else:
            pa_metadata = self._load_proj_metadata('PA')
            lat_metadata = self._load_proj_metadata('LAT')
            return pa_metadata+lat_metadata

    def _apply_albumentation_augment(self, image, mask) -> Tuple:
        augmented = affine_transformations(
            image=image, mask=mask
            )
        return augmented['image'], augmented['mask']

    def augment_image(self, image, mask) -> Tuple:
        coin_flip = np.random.random()
        if coin_flip < 0.5:
            image, mask = self._apply_albumentation_augment(image, mask)
        return image, mask

    def flip_augment_image(self, image, mask):
        if np.random.random() < 0.5:
            return image, mask
        if np.random.random() < 0.5:
            return np.fliplr(image), np.fliplr(mask)
        return image, mask
        
    def normalize(self, images):
        mean = 138.5
        return (images-mean)/256.

    def collate(self, batch):
        images = torch.stack([torch.from_numpy(elem[0].copy()) for elem in batch], dim=0).float()
        masks = torch.stack([torch.from_numpy(elem[1].copy()) for elem in batch], dim=0)
        return {'images':images.unsqueeze(1), 'masks':masks}

    
class JointLoader(Dataset):
    def __init__(
        self, 
        train, 
        metadata_prefix:str,
        augment=True, 
        experts='123',
        ):
        assert train in ['train', 'val', 'test']
        assert isinstance(augment, bool)
        self.train = train
        self.augment = augment
        self.experts = ['expert'+expert for expert in experts]
        self.metadata_prefix = metadata_prefix
        self.indices_path = self.indices_path = path.join(path.dirname(__file__), '../indices.json')
        self.metadata = self._load_metadata()
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        pa, lat = self.metadata[idx]
        im_pa, mask_pa = self._get_item(pa)
        im_lat, mask_lat = self._get_item(lat)
        return (im_pa, im_lat), (mask_pa, mask_lat)

    def _get_item(self, case):
        path_im, path_mask = case
        image = cv2.imread(path_im, 0)
        image = cv2.resize(image, (512,512))
        if path_mask is not None:
            mask = np.load(path_mask)['mask']
            mask = cv2.resize(mask, (512,512))
        else:
            mask = np.zeros((512,512))
        image = image.astype('float32')
        mask = mask.astype('uint8')
        if self.train == 'val' or self.train == 'test':
            return image, mask
        if self.augment:
            image, mask = self.augment_image(image, mask)
        else:
            image, mask = self.flip_augment_image(image, mask)
        return image, mask
        
    def _load_proj_metadata(self, proj):
        with open(self.indices_path, 'r') as file:
            indices = json.load(file)[self.train]
        pneu_inds = indices['pneumonia']
        norma_inds = indices['norma']
        all_pneu = ls(self.metadata_prefix+'images/pneumonia/')
        all_pneu_proj = [elem for elem in all_pneu if proj.lower() in elem]
        all_norma = ls(self.metadata_prefix+'images/norma/')
        all_norma_proj = [elem for elem in all_norma if proj.lower() in elem]
        metadata = list()
        pneu_names = [
            elem for elem in all_pneu_proj if int(elem.split('_')[0]) in pneu_inds
            ]
        norma_names = [
            elem for elem in all_norma_proj if int(elem.split('_')[0]) in norma_inds
            ]
        for exp in self.experts:
            exp_path = self.metadata_prefix+'masks/'+exp+'/'
            exp_list = [elem for elem in ls(exp_path) if proj.lower() in elem]
            exp_list = [elem for elem in exp_list if int(elem.split('_')[0]) in pneu_inds]
            for pneu in pneu_names:
                case_prefix = pneu.split('.')[0]
                mask_path = [elem for elem in exp_list if elem.startswith(case_prefix)]
                if mask_path:
                    case = (self.metadata_prefix+'images/pneumonia/'+pneu, exp_path+mask_path[0])
                    metadata.append(case)
        norma_cases = [(self.metadata_prefix+'images/norma/'+elem, None) for elem in norma_names]
        return metadata + norma_cases

    def _load_metadata(self):
        metadata_pa = self._load_proj_metadata('PA')
        metadata_lat = self._load_proj_metadata('LAT')
        pair_elems = list()
        for case_pa in metadata_pa:
            name_pa = case_pa[0].split('/')[-1]
            name_lat = name_pa.replace('pa', 'lat')
            status = case_pa[0].split('/')[-2]
            if status == 'pneumonia':
                expert = case_pa[1].split('/')[-2]
                case_lat = [elem for elem in metadata_lat if elem[0].endswith(name_lat) and status in elem[0] and expert in elem[1]]
            else:
                case_lat = [elem for elem in metadata_lat if elem[0].endswith(name_lat) and status in elem[0]]
            if case_lat:
                pair_elems.append((case_pa, case_lat[0]))
        return pair_elems

    def _apply_albumentation_augment(self, image, mask) -> Tuple:
        augmented = affine_transformations(
            image=image, mask=mask
            )
        return augmented['image'], augmented['mask']

    def augment_image(self, image, mask) -> Tuple:
        coin_flip = np.random.random()
        if coin_flip < 0.5:
            image, mask = self._apply_albumentation_augment(image, mask)
        return image, mask

    def flip_augment_image(self, image, mask):
        if np.random.random() < 0.5:
            return image, mask
        if np.random.random() < 0.5:
            return np.fliplr(image), np.fliplr(mask)
        return np.flipud(image), np.flipud(mask)  

    def normalize(self, images, proj):
        mean = 138.5
        return (images-mean)/256.

    def collate(self, batch):
        images_pa = torch.stack([torch.from_numpy(elem[0][0].copy()) for elem in batch], dim=0)
        images_lat = torch.stack([torch.from_numpy(elem[0][1].copy()) for elem in batch], dim=0)
        masks_pa = torch.stack([torch.from_numpy(elem[1][0].copy()) for elem in batch], dim=0)
        masks_lat = torch.stack([torch.from_numpy(elem[1][1].copy()) for elem in batch], dim=0)
        return {
                'images_pa':images_pa.unsqueeze(1), 
                'images_lat':images_lat.unsqueeze(1),
                'masks_pa': masks_pa,
                'masks_lat': masks_lat,
            }
