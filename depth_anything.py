import cv2
import torch
import yaml
from depth_anything_v2.dpt import DepthAnythingV2
from functools import lru_cache


@lru_cache(maxsize=3)
def load_model(encoder_config, model_path_config, max_depth_config):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    depth_anything = DepthAnythingV2(**{**model_configs[encoder_config], 'max_depth': max_depth_config})
    depth_anything.load_state_dict(torch.load(model_path_config, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    return depth_anything


def depth_calculation(filename: str):
    '''Calcola la distanza dei punti nell'immagine'''

    with open('config/settings.yaml', 'r') as settings_file:
        env_settings = yaml.safe_load(settings_file)
    encoder_config      = env_settings['encoder']       # 'vitb'
    input_size_config   = env_settings['input_size']    # 518
    max_depth_config    = env_settings['max_depth']     # 80
    model_path_config   = env_settings['model_path']    # checkpoints/depth_anything_v2_metric_vkitti_vitb.pth

    depth_anything = load_model(encoder_config, model_path_config, max_depth_config)

    raw_image = cv2.imread(filename)
    depth_map = depth_anything.infer_image(raw_image, input_size_config)

    return depth_map