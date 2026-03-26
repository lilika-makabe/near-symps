import os

import cv2
import numpy as np


def light_params_to_lpos(r, angles, degree=True):
    if degree:
        angles = np.deg2rad(angles)
    res = np.vstack((r * np.sin(angles), r * np.cos(angles), 0 * r)).T
    assert res.shape[1] == 3
    return res


def scale_cameraMatrix(cameraMatrix, scale):
    new_cameraMatrix = np.copy(cameraMatrix)
    new_cameraMatrix[:-1, :] *= scale
    return new_cameraMatrix


def rgb2gray(img):
    assert img.ndim >= 3
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def add_alpha_channel(image, mask):
    if image.ndim == 2:
        image = np.array([image, image, image]).transpose(1, 2, 0)
    alpha = np.where(mask == 0, 0, 1).astype(image.dtype)
    if image.dtype in [np.uint8, np.uint16]:
        alpha *= np.iinfo(image.dtype).max
    elif image.dtype in [np.float32, np.float64]:
        alpha *= np.finfo(image.dtype).max
    else:
        raise ValueError('Unsupported dtype: {}'.format(image.dtype))
    image = np.concatenate([image, alpha[..., None]], axis=-1)
    return image


def normal2color(normal, mask=None, dtype=np.uint8):
    n = normal.copy()
    n[..., 2] = -n[..., 2]
    temp0 = n[..., 0].copy()
    temp1 = n[..., 1].copy()
    n[..., 1] = -temp1
    n[..., 0] = temp0
    res = (np.iinfo(dtype).max * (n + 1) / 2)
    res = np.clip(res, 0, np.iinfo(dtype).max).astype(dtype)
    return res[..., ::-1]


def color2normal(color, mask=None, dtype=np.uint8):
    if dtype is None:
        dtype = color.dtype
    c = color.astype(float)
    n = (2 * c / np.iinfo(dtype).max) - 1
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    n = n[..., ::-1]
    n[..., 2] = -n[..., 2]
    temp0 = n[..., 0].copy()
    temp1 = n[..., 1].copy()
    n[..., 1] = -temp1
    n[..., 0] = temp0
    return n


def save_args_to_yaml(file, arg):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    import yaml
    with open(file, 'wt') as f:
        yaml.safe_dump(vars(arg), f)
    return os.path.exists(file)


def crop_images(mask, samples, square=False):
    indices = np.argwhere(mask.squeeze())
    left = indices[:, 1].min()
    right = indices[:, 1].max() + 1
    top = indices[:, 0].min()
    bottom = indices[:, 0].max() + 1

    if square:
        width = right - left
        height = bottom - top
        if height > width:
            pad = int((height - width) / 2)
            left = left - pad
            right = right + (height - width) - pad
        else:
            pad = int((width - height) / 2)
            top = top - pad
            bottom = bottom + (width - height) - pad

    for i in range(len(samples)):
        cropped = _crop_image((top, bottom), (left, right), samples[i])
        samples[i] = cropped
    return samples


def _crop_image(top_bottom, left_right, image):
    image_shape = image.shape[0:2]
    pad_needed, top_bottom_pad, left_right_pad = _calculate_need_padding(top_bottom, left_right, image_shape)
    if pad_needed:
        image = np.pad(image, (left_right_pad[0], left_right_pad[1], top_bottom_pad[0], top_bottom_pad[1]))
        left_right = (left_right[0] + left_right_pad[0], left_right[1] + left_right_pad[0])
        top_bottom = (top_bottom[0] + top_bottom_pad[0], top_bottom[1] + top_bottom_pad[0])
    image_out = image[top_bottom[0]:top_bottom[1], left_right[0]:left_right[1], ...]
    return image_out


def _calculate_need_padding(top_bottom, left_right, image_shape):
    top_pad = max(0, -top_bottom[0])
    left_pad = max(0, -left_right[0])
    bottom_pad = max(0, top_bottom[1] - image_shape[0])
    right_pad = max(0, left_right[1] - image_shape[1])
    top_bottom_pad = (top_pad, bottom_pad)
    left_right_pad = (left_pad, right_pad)
    pad_needed = (top_pad != 0) or (bottom_pad != 0) or (left_pad != 0) or (right_pad != 0)
    return pad_needed, top_bottom_pad, left_right_pad
