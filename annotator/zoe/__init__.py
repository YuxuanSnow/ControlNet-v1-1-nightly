# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
from annotator.util import annotator_ckpts_path


class ZoeDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath)['model'])
        model = model.cuda()
        model.device = 'cuda'
        model.eval()
        self.model = model

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)  # smallest 2%
            vmax = np.percentile(depth, 85) # largest 85%

            depth -= vmin
            depth /= vmax - vmin   # normalize to [0, 1]
            depth = 1.0 - depth # inverse depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8) # limit the depth value into range [0, 255]

            return depth_image
    
    def estimate(self, input_image):

        assert input_image.ndim == 3
        image_depth = input_image

        image_depth = torch.from_numpy(image_depth).float().cuda()
        image_depth = image_depth / 255.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model.infer(image_depth)

        depth = depth[0, 0].detach().cpu().numpy()

        return depth

    def estimate_normalize(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image

        image_depth = torch.from_numpy(image_depth).float().cuda()
        image_depth = image_depth / 255.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model.infer(image_depth)

        depth = depth[0, 0].detach().cpu().numpy()

        vmin = 0.1 # np.percentile(depth, 2)  # smallest 2%
        vmax = 10 # np.percentile(depth, 85) # largest 85%

        depth -= vmin
        depth /= vmax - vmin   # normalize to [0, 1]
        depth = 1.0 - depth # inverse depth
        depth = depth.clip(0, 1)   
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8) # limit the depth value into range [0, 255]

        return depth, depth_image

