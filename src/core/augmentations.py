from typing import Tuple

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image


class SquarePad:
    """
    Pad the image to a square.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, 0, "constant")

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class BaseAlbumentationConverter:
    transform: A.Compose

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        result = self.transform(image=img_np)["image"]
        return Image.fromarray(result)


class ChannelShuffle(BaseAlbumentationConverter):
    def __init__(self, p: float = 0.5) -> None:
        self.transform = A.Compose([A.ChannelShuffle(p=p)])

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(p={self.transform.transforms[0].p})"


class Downscale(BaseAlbumentationConverter):
    def __init__(self, scale_range: Tuple[float, float], p: float = 0.5) -> None:
        self.transform = A.Compose([A.Downscale(scale_range=scale_range, p=p)])

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(scale_range={self.transform.transforms[0].scale_range}, p={self.transform.transforms[0].p})"
        )


class GaussNoise(BaseAlbumentationConverter):
    def __init__(
        self, std_range: Tuple[float, float] = (0.1, 0.1), p: float = 0.5
    ) -> None:
        self.transform = A.Compose([A.GaussNoise(std_range=std_range, p=p)])

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(std_range={self.transform.transforms[0].std_range}, p={self.transform.transforms[0].p})"
        )
