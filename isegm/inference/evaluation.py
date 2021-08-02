from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []
    all_ras = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        for inst_id in range(len(sample)):
            _, sample_ious, sample_ras, _ = evaluate_sample(sample.image,
                                                sample.gt_mask_per_instance(inst_id),
                                                predictor, sample_id=index, **kwargs)
        all_ious.append(sample_ious)
        all_ras.append(sample_ras)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, all_ras, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr=0.9, max_ra_thr=0.05,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    ra_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            ra = utils.get_RA(gt_mask, pred_mask)
            ra_list.append(ra)

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), np.array(ra_list, dtype=np.float32), pred_probs
