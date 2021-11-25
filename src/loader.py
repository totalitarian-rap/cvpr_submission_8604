from os import path
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

metadata_prefix = '/ayb/vol1/datasets/chest/'
indices_path = '/ayb/vol1/datasets/chest/indices.json'

affine_transformations = Compose(
    [
        tf.ShiftScaleRotate(scale_limit=0.3),
    ], p = 1
)

class Loader(Dataset):
    def __init__(self, projection, train, augment=False, indices=None, experts='123'):
        super().__init__()
        assert train in ['train', 'val', 'test']
        assert projection in ['PA', 'LAT', 'PA_LAT']
        assert isinstance(augment, bool)
        assert indices is None or isinstance(indices, Iterable)
        assert experts in ['1', '2', '3', '12', '13', '23', '123']
        self.train = train
        self.projection = projection
        self.augment = augment
        self.indices = indices
        self.experts = ['expert'+expert for expert in experts]
        self.experts.append('norma')
        self.metadata = self._load_metadata()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplemetedError("We cannot handle slices yet!")
        image_path = self.metadata[idx]['image_path']
        mask_path = self.metadata[idx]['mask_path']
        image = cv2.imread(image_path, 0)
        image = self.normalize(image)
        mask = np.load(mask_path)['mask'].astype('uint8')
        if self.train == 'val' or self.train == 'test':
            return image, mask
        if self.augment:
            image, mask = self.augment_image(image, mask)
        else:
            image, mask = self.flip_augment_image(image, mask)
        return image, mask

    def _load_proj_metadata(self, proj):
        metadata_json = metadata_prefix+'Chest_'+proj+'/metadata_Chest_'+proj+'.json'
        with open(metadata_json, 'r') as file:
            metadata = json.load(file)
        with open(indices_path, 'r') as file:
            all_indices = json.load(file)
        all_indices = [int(elem) for elem in all_indices[self.train]]
        if self.indices is not None:
            all_indices = [elem for elem in all_indices if elem in self.indices]
        metadata = [
            elem for elem in metadata.values() if elem['case_uid'] in all_indices
            ]
        metadata = [elem for elem in metadata if elem['expert'] in self.experts]
        return metadata

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
        
    def normalize(self, images):
        mean = 138.5
        return (images-mean)/256.

    def collate(self, batch):
        images = torch.stack([torch.from_numpy(elem[0].copy()) for elem in batch], dim=0).float()
        masks = torch.stack([torch.from_numpy(elem[1].copy()) for elem in batch], dim=0)
        return {'images':images.unsqueeze(1), 'masks':masks}

    
class JointLoader(Dataset):
    def __init__(self, train, augment=True, indices=None, experts='123'):
        assert train in ['train', 'val', 'test']
        assert isinstance(augment, bool)
        assert indices is None or isinstance(indices, Iterable)
        self.train = train
        self.augment = augment
        self.indices = indices
        self.experts = ['expert'+expert for expert in experts]
        self.experts.append('norma')
        self.metadata = self._load_metadata()
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplemetedError("We cannot handle slices yet!")
        image_pa, mask_pa = self._get_item(self.metadata[idx][0], 'PA')
        image_lat, mask_lat = self._get_item(self.metadata[idx][1], 'LAT')
        return (image_pa, image_lat), (mask_pa, mask_lat)

    def _get_item(self, metadata_elem, proj):
        image_path = metadata_elem['image_path']
        mask_path = metadata_elem['mask_path']
        image = cv2.imread(image_path, 0)
        image = self.normalize(image, proj).astype('float32')
        mask = np.load(mask_path)['mask'].astype('uint8')
        if self.train == 'val' or self.train == 'test':
            return image, mask
        if self.augment:
            image, mask = self.augment_image(image, mask)
        return image, mask
        
    def _load_proj_metadata(self, proj):
        metadata_json = metadata_prefix+'Chest_'+proj+'/metadata_Chest_'+proj+'.json'
        with open(metadata_json, 'r') as file:
            metadata = json.load(file)
        with open(indices_path, 'r') as file:
            all_indices = json.load(file)
        all_indices = [int(elem) for elem in all_indices[self.train]]
        if self.indices is not None:
            all_indices = [elem for elem in all_indices if elem in self.indices]
        metadata = [
            elem for elem in metadata.values() if elem['case_uid'] in all_indices
            ]
        metadata = [elem for elem in metadata if elem['expert'] in self.experts]
        return metadata

    def _load_metadata(self):
        metadata_pa = self._load_proj_metadata('PA')
        metadata_lat = self._load_proj_metadata('LAT')
        pair_elems = list()
        for elem_lat in metadata_lat:
            expert, uid = elem_lat['expert'], elem_lat['study_uid']
            elem_pa = [
                    elem for elem in metadata_pa 
                    if elem['expert'] == expert and elem['study_uid'] == uid
                ]
            if elem_pa:
                pair_elems.append((elem_pa[0], elem_lat))
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