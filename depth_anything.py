import cv2
import torch
import yaml
from depth_anything_v2.dpt import DepthAnythingV2

def depth_calculation(filename: str):
    '''NOTA: I valori delle distanze NON sono in metri'''

    # Leggo alcune cose da settings.yaml
    with open('settings.yaml', 'r') as settings_file:
        env_settings = yaml.safe_load(settings_file)
    encoder_config = env_settings['encoder'] # 'vitb'
    input_size_config = env_settings['input_size'] # 518

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    depth_anything = DepthAnythingV2(**model_configs[encoder_config])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder_config}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    raw_image = cv2.imread(filename)
    depth_map = depth_anything.infer_image(raw_image, input_size_config)

    # Ritorna una cosa simile a questa
    # [[4.7106595 4.7855444 4.860429  ... 4.098736  4.00116   3.9036021]
    #  [4.793133  4.885724  4.978315  ... 4.109568  4.055221  4.000884 ]
    #  [4.8756065 4.9859037 5.0962014 ... 4.1204004 4.1092825 4.0981655]
    #  ...
    #  [8.532101  8.525809  8.519517  ... 8.92571   8.923376  8.921043 ]
    #  [8.551974  8.5424    8.532825  ... 8.934305  8.93417   8.934035 ]
    #  [8.571844  8.558989  8.546132  ... 8.942899  8.944961  8.947023 ]]
    return depth_map