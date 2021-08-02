from pathlib import Path

import cv2
import numpy as np
import glob
import os
import json

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class AimmoDataset(ISDataset):
    def __init__(self, dataset_path, aimmo_cfg=None, split='valid',
                 images_dir_name='images', masks_dir_name='labels', ann_type='poly_points',
                 **kwargs):
        super(AimmoDataset, self).__init__(**kwargs)

        self.class_map = {label:idx for idx, label in enumerate(aimmo_cfg.CLASS_LIST)}
        self.ignore_class = aimmo_cfg.IGNORE_CLASS
        self.dataset_path = os.path.join(dataset_path, split)
        self._images_path = os.path.join(self.dataset_path, images_dir_name)
        self._labels_path = os.path.join(self.dataset_path, masks_dir_name)
        ## TODO: remove
        sample_idx = ['/Disk5/Front/20200618_094543_Front_002940.png',
        '/Disk5/Left/20200604_111620_Left_001260.png',
        '/Disk6/Front/20200507_135706_Front_000660.png',
        '/Disk5/Front/20200611_104618_Front_000900.png',
        '/Disk5/Left/20200618_094543_Left_003480.png',
        '/Disk4/Right/20200623_144835_Right_000120.png']
        # annotation
        self._ann_datas = list()
        for path in glob.glob(self._labels_path+"/**/*.json", recursive=True):
            annotations = json.load(open(path))
            ## TODO: remove
            sample_path = annotations['parent_path']+"/"+annotations['filename']

            if sample_path not in sample_idx:
                continue
            self._ann_datas.append(annotations)
        # img
        self.dataset_samples = list()
        path_list = glob.glob(self._images_path+"/**/*.*", recursive=True)
        img_extends = ['jpg', 'jpeg', 'JPG', 'bmp', 'png']
        for path in path_list:
            if path.split(".")[-1] in img_extends:
                json_path = Path(path.replace(images_dir_name, masks_dir_name))
                json_path = json_path.with_suffix('.json')
                if os.path.isfile(json_path):
                    if path.replace(self.dataset_path+"/images", "") not in sample_idx:
                        continue
                    self.dataset_samples.append(path)
                else:
                    raise FileNotFoundError("No annotation file at image: "+path)
        
    def make_mask(self, anns, img_size):
        background = np.zeros((img_size[0], img_size[1], 1))
        mask_color = 0
        for ann in anns:
            if ann['label'] in self.ignore_class:
                continue
            mask_color +=1
            points = np.array(ann['points'])
            if len(points.shape) !=2:
                for seg in points:
                    seg = np.array(seg, dtype=np.int32)
                    mask_ = cv2.fillPoly(background, [seg], mask_color)
            else:
                seg = np.array(points, dtype=np.int32)
                mask_ = cv2.fillPoly(background, [seg], mask_color)
        return mask_.astype(np.int32)

    def get_sample(self, index) -> DSample:
        annotations = self._ann_datas[index]

        image_name = os.path.join(annotations['parent_path'], annotations['filename'])
        image_path = self._images_path + image_name
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = self.make_mask(annotations['annotations'], image.shape)
        instances_ids = np.unique(instances_mask).astype('int32').tolist()
        instances_ids.remove(0)
        if len(instances_ids) ==0:
            err_msg = 'No Instanse on image'+self._ann_datas[index]['parent_path'] +"/"+self._ann_datas[index]['filename']
            raise ValueError(err_msg)
        return DSample(image, instances_mask, objects_ids=instances_ids)