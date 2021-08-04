
import sys
import numpy as np
import collections
import torch

sys.path.insert(0, '..')
from isegm.inference import utils
from isegm.inference import clicker
from isegm.inference.predictors.base import BasePredictor

class ISController(object):
    """Interface for interactive segmentation.
       The model input is a combination of the mask inferred from the previous click and the current click information.
    
    Attributes:
        device (torch.device): set gpu device.
        net (isegm.model.is_hrnet_model.HRNetModel): pre-trained Interactive Segmentation model
        score_threshold (float): using for mask score threshold. this value is decided during training and set on checkpoint file.
        predictor (isegm.inference.predictors.base.BasePredictor): this instance have method for process that predict mask
        image (np.array:np.uint8): image for segmentation labeling.
        states (list): for recoding interaction result.
        init_mask (np.array:np.float32): a segmentation mask created before work.(AI-assist). By setting this value, the mask can be modified.
        clicker (isegm.inference.clicker.Clicker): management of click and make click coordinate to disk map.
        iteration (int): number of click
        
    Args:
        ckpt_path (str): checkpoint file path.
    """

    def __init__(self, ckpt_path):
        self._device = torch.device('cuda:0')
        self._net, self._score_threshold = utils.load_is_model(ckpt_path, self._device, is_api=True)
        self._predictor = BasePredictor(self._net, self._device, prob_thresh=self._score_threshold)
        self._image = None
        self._states = list()
        self._init_mask = None
        self._clicker = clicker.Clicker()
        self.iteration = 0
        
    def set_image(self, img):
        assert img.dtype==np.uint8, "Image dtype must be uint8"
        self._image = img
        self._predictor.set_input_image(img)
        
    def set_init_mask(self, mask):
        mask = np.array(mask, dtype=np.float32)
        assert self._image.shape[:2]==mask.shape, "Init Mask shape must be same with image"
        self._init_mask = torch.tensor(mask, device=self._device)
    
    def predict_mask(self, click_coord, is_positive):
        if self._image is None:
            raise ValueError('No image')
        self._states.append({
            'clicker': self._clicker.get_state(),
            'predictor': self._predictor.get_states()
        })
        click = clicker.Click(is_positive=is_positive, coords=(click_coord[1], click_coord[0]))
        self._clicker.add_click(click)
        
        if self.iteration != 0:
            pred = self._predictor.get_prediction(self._clicker)
        else:
            pred = self._predictor.get_prediction(self._clicker, init_mask=self._init_mask)
            
        torch.cuda.empty_cache()
        self.iteration += 1
        
        pred_mask = np.where(pred>self._score_threshold, 1, 0).astype(np.uint8)
        return pred_mask
        
    def undo_click(self):
        if len(self._states)==0:
            raise IndexError('Last mask state')
        prev_state = self._states.pop()
        self._clicker.set_state(prev_state['clicker'])
        self._predictor.set_states(prev_state['predictor'])
        mask = self._predictor.get_states().cpu().numpy()
        mask = np.squeeze(np.where(mask>self._score_threshold, 1, 0)).astype(np.uint8)
        
        self.iteration -= 1
        return mask
    
    def finish_object(self):
        self._clicker.reset_clicks()
        self._predictor.reset_states()
        self._states = list()
        self.iteration = 0
        
    def finish_image(self):
        self._clicker.reset_clicks()
        self._predictor.reset_states()
        self._image = None
        self._init_mask = None
        self._states = list()
        self.iteration = 0
        
        