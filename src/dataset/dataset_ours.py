import glob
import os

import cv2
import numpy as np

from utils import io_utils


class DatasetOurs(object):
    def __init__(self, root_dir, eps=1e-30, initial_resize=1.0, lights_to_load=None):
        self.root = root_dir
        self.eps = eps
        self.initial_resize = initial_resize
        self.lights_to_load = lights_to_load

    def get(self, color=False, raw=False):
        data = self.get_data()
        images = self.get_images(color, raw)
        return data, images

    def get_data(self):
        mask = self.get_mask(mask_path=os.path.join(self.root, "mask.png"))
        assert mask.dtype == bool
        cameraMatrix = self.get_cameraMatrix()
        light_params = self.get_light_params()
        if self.lights_to_load is not None:
            light_params = light_params[np.array(self.lights_to_load)]
        data = {
            'mask': mask,
            'cameraMatrix': cameraMatrix,
            'n_lights': len(light_params),
        }
        return data

    def get_cameraMatrix(self):
        path = os.path.join(self.root, "camera_params.txt")
        if not os.path.exists(path):
            return None
        cameraMatrix = np.loadtxt(path)
        assert cameraMatrix.shape == (3, 3)
        if self.initial_resize != 1:
            cameraMatrix = io_utils.scale_cameraMatrix(cameraMatrix, self.initial_resize)
        return cameraMatrix

    def get_mask(self, mask_path=None):
        if mask_path is None:
            mask_path = os.path.join(self.root, "mask.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.initial_resize != 1:
            mask = cv2.resize(mask, None, fx=self.initial_resize, fy=self.initial_resize,
                              interpolation=cv2.INTER_NEAREST)
        mask[mask != 255] = 0
        return mask == 255

    def get_light_params(self):
        path = os.path.join(self.root, "light_params.txt")
        assert os.path.exists(path), path
        with open(path) as f:
            n_lights = int(f.readline().rstrip())
            lparams = np.loadtxt(f)
        assert lparams.shape == (n_lights, 2), f"lparams.shape == {lparams.shape}"
        return lparams

    def read_image(self, image_path):
        img = np.load(image_path)
        if self.initial_resize != 1:
            img = cv2.resize(img, None, fx=self.initial_resize, fy=self.initial_resize,
                             interpolation=cv2.INTER_NEAREST)
        return img

    def get_images(self, color=False, raw=False):
        from utils import general_utils
        data = self.get_data()
        n_lights = data["n_lights"]
        if self.lights_to_load is not None:
            light_idx_to_load = np.array(self.lights_to_load)
            n_lights = len(light_idx_to_load)
        else:
            light_idx_to_load = np.arange(n_lights)

        image_paths = sorted(glob.glob(os.path.join(self.root, "image", "*.npy")))
        images = [self.read_image(p) for p in image_paths]

        if len(images) == n_lights + 1:
            amb = images[n_lights]
            images = [
                (images[i].astype(float) - amb.astype(float))
                .clip(0, general_utils.get_dtype_max(amb.dtype))
                .astype(amb.dtype)
                for i in range(n_lights)
            ]
        elif len(images) != n_lights:
            raise ValueError(f"Input image has {len(images)}, expected {n_lights}")

        images = np.array(images)
        if color and images.ndim == 3:
            images = self.cvt_gray2psuedo_color(images)
        elif not color and images.ndim == 4:
            images = self.cvt_color2gray(images)
        if not raw:
            images = images.astype(float) / np.max(images)
            images[:, ~data["mask"]] = 0
        self._check_images_shape(color, images)
        images = images[light_idx_to_load]
        return images

    def cvt_color2gray(self, images):
        dtype = images.dtype
        images = np.array([io_utils.rgb2gray(img) for img in images])
        if dtype in [np.uint8, np.uint16]:
            images = np.clip(images, 0, np.iinfo(dtype).max).astype(dtype)
        elif dtype in [np.float32, np.float64]:
            images = np.clip(images, 0, np.finfo(dtype).max).astype(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return images

    def cvt_gray2psuedo_color(self, images):
        return np.tile(images, (3, 1, 1, 1)).transpose(1, 2, 3, 0)

    @staticmethod
    def _check_images_shape(color, images):
        if color:
            assert images.ndim == 4, f"images shape: {images.shape}"
        else:
            assert images.ndim == 3
