from .base import BasePredictor
from isegm.inference.transforms import ZoomIn
from isegm.model.is_hrnet_model import HRNetModel


def get_predictor(net, device,
                  prob_thresh=0.49,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  lbfgs_params=None):
    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    
    predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip)

    return predictor
