from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from annotator.zoe import ZoeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

preprocessor = None

model_name = 'control_v11f1p_sd15_depth'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned-emaonly.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
for param in model.parameters():
    param.requires_grad = False
ddim_sampler = DDIMSampler(model)

preprocessor = ZoeDetector()

from PIL import Image
input_image = Image.open('/home/yuxuan/project/Data/official_splits/test/living_room/rgb_01204.jpg')
input_image = np.array(input_image) 
# input_image = np.load('/home/yuxuan/Downloads/img_folders/exp100_omnidirectdepth_nottrain/res_number/NYU_v2/1269_rgb.png.npy') * 255
input_image = HWC3(input_image)
detected_map = preprocessor(resize_image(input_image, 512)) # apply depth map
detected_map = HWC3(detected_map) # bring 1 channel to 3 channel by copying
detected_map_org = preprocessor.estimate(resize_image(input_image, 512))
detected_map_num, detected_map_img = preprocessor.estimate_normalize(resize_image(input_image, 512))
detected_map = HWC3(detected_map_num) # bring 1 channel to 3 channel by copying

img = resize_image(input_image, 512)
H, W, C = img.shape
num_samples = 1

detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) # interpolate to the original resolution

control_org = torch.from_numpy(detected_map.copy()).float().cuda().requires_grad_() # / 255.0 # normalize to [0, 1] again? (why? maybe because of the interpolation)
control = torch.stack([control_org for _ in range(num_samples)], dim=0) # multiple samples
control = einops.rearrange(control, 'b h w c -> b c h w').clone().requires_grad_()

seed = random.randint(0, 65535)
seed_everything(seed)

if config.save_memory:
    model.low_vram_shift(is_diffusing=False)

cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(["a room" + ', ' + 'DSLR, photorealistic'] * num_samples)]}
un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(['watermark, cartoon'] * num_samples)]}
shape = (4, H // 8, W // 8)

model.control_scales = ([1.0] * 13)
# Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

eta = 1.0
scale = 9.0

samples, intermediates = ddim_sampler.sample(10, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)]

from matplotlib import pyplot as plt
plt.imshow(results[3])