from .base import BasePredictor
from isegm.inference.transforms import ZoomIn
from isegm.model.is_hrnet_model import HRNetModel


def get_predictor(net, device,
                  prob_thresh=0.49,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  lbfgs_params=None):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']


    if predictor_params is not None:
        predictor_params_.update(predictor_params)
    predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)

    return predictor
