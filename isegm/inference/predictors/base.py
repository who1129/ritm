import torch
import torch.nn.functional as F
from torchvision import transforms


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 max_size=None,
                 **kwargs):
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, init_mask=None):
        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if init_mask is None:
            prev_mask = self.prev_prediction
        else:
            prev_mask = torch.tensor(init_mask).unsqueeze(0).unsqueeze(0)
        input_image = torch.cat((input_image, prev_mask), dim=1)
        clicks_lists = [clicks_list]

        pred_logits = self._get_prediction(input_image, clicks_lists)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=input_image.size()[2:])

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, input_image, clicks_lists):
        points_nd = self._get_points_nd(clicks_lists)
        return self.net(input_image, points_nd)['instances']

    def _get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return self.prev_prediction

    def set_states(self, states):
        self.prev_prediction = states
    
    def reset_states(self):
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
