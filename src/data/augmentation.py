from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def transform(hr_height):
    lr_transform = transforms.Compose(
        [
            transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = np.array([0.485, 0.456, 0.406]),
                                 std = np.array([0.229, 0.224, 0.225])),
        ]
    )
    hr_transform = transforms.Compose(
        [
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = np.array([0.485, 0.456, 0.406]),
                                 std = np.array([0.229, 0.224, 0.225])),
        ]
    )
    return lr_transform, hr_transform