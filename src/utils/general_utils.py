import os

import cv2
import joblib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
from tqdm import tqdm


class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def get_dtype_max(dtype):
    if dtype in [np.uint8, np.uint16]:
        return np.iinfo(dtype).max
    elif dtype in [np.float32, np.float64, float]:
        return np.finfo(dtype).max
    else:
        raise ValueError("Unexpected dtype: {}".format(dtype))


def get_uvcoord(shape):
    h, w = shape
    uvcoord = np.meshgrid(np.arange(w), np.arange(h))
    uvcoord = np.stack(uvcoord, axis=0)
    return uvcoord


def get_sensor_coord(shape, cameraMatrix):
    uvcoord = get_uvcoord(shape)
    cxcy = cameraMatrix[:2, 2]
    fxfy = cameraMatrix[(0, 1), (0, 1)]
    XYdivbyZ = (uvcoord.astype(float) - cxcy[:, None, None]) / fxfy[:, None, None]
    return XYdivbyZ


def move_left(mask):
    return np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_bottom_right(mask):
    return np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def plot_results_in_subplots(data, output_path, figshape=None, subtitle=[], show_bar=[], minmax_list=None,
                             axis_off=True, cmap='jet'):
    plt.rcParams["figure.subplot.right"] = 1.0
    plt.rcParams["figure.subplot.wspace"] = 0.05
    if figshape is None:
        figshape = (1, len(data))
    nrows, ncols = figshape
    if axis_off:
        plt.axis("off")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True, figsize=(12 * ncols, 6 * nrows))
    if type(axes) is not np.ndarray:
        axes = [axes]
    assert nrows * ncols == len(data)
    for i in range(nrows):
        for j in range(ncols):
            data_id = i * ncols + j
            if minmax_list is None:
                minmax = [np.nanmin(data[data_id]), np.nanmax(data[data_id])]
            elif len(minmax_list) == len(data):
                minmax = minmax_list[data_id]
            elif len(minmax_list) == 1:
                minmax = minmax_list[0]
            else:
                raise ValueError
            pos = j if figshape[0] == 1 else (i, j)
            im1 = axes[pos].matshow(data[data_id], vmin=minmax[0], vmax=minmax[1], cmap=cmap)
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes[pos])
            if len(subtitle) == len(data):
                axes[pos].set_title(subtitle[data_id])
            axes[pos].set_axis_off()
            axes[pos].set_aspect('equal')
            if len(show_bar) == 0 or (len(show_bar) == len(data) and show_bar[data_id]):
                cax = divider.append_axes('right', '5%', pad='3%')
                fig.colorbar(im1, cax=cax, format='%.2e')
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close()


def batch_data(M, batch_size=None, batch_num=None):
    assert ~(batch_size is None and batch_num is None)
    if batch_num is None:
        batch_num = np.ceil(len(M) / batch_size)
    Mbs = np.array_split(M, batch_num, 0)
    return Mbs


def solve_nearlight_PS_given_lightdirection_multicore(image, scaled_led_dir, mask, n_jobs=8, avoid_normalize=False,
                                                      gt=None):
    n_light, H, W = image.shape
    assert scaled_led_dir.shape == (n_light, H, W, 3)
    assert mask.shape == (H, W)

    if not avoid_normalize:
        light_direction = scaled_led_dir / (1e-30 + np.linalg.norm(scaled_led_dir, axis=-1, keepdims=True))
    else:
        import warnings
        warnings.warn("Not normalize the light direction")
        light_direction = scaled_led_dir.copy()
    light_direction[:, ~mask, :] = 0
    image[:, ~mask] = 0

    n_light, h, w = image.shape
    light_direction = light_direction.reshape(n_light, -1, 3).transpose(1, 0, 2)
    Lb = batch_data(light_direction, batch_num=n_jobs)
    Mb = batch_data(image.reshape(n_light, -1).T, batch_num=n_jobs)
    gtb = batch_data(gt, batch_num=n_jobs) if gt is not None else [None] * len(Lb)
    Maskb = batch_data(mask.reshape(-1), batch_num=n_jobs)
    from copy import deepcopy
    func = deepcopy(_estimate_nearlight_PS_on_a_batch)
    result = ProgressParallel(n_jobs=n_jobs, temp_folder="/dev/shm", max_nbytes='3000G',
                              backend='multiprocessing', verbose=10)(
        joblib.delayed(func)(Lb[i], Mb[i], Maskb[i], gtb[i]) for i in range(len(Lb)))
    rho = np.concatenate([x[0] for x in result]).reshape(H, W)
    normal = np.concatenate([x[1] for x in result]).reshape(H, W, 3)
    del result
    return rho, normal


def _estimate_nearlight_PS_on_a_batch(light_direction, m, mask, gt=None):
    assert m.ndim == 2
    n_points, n_light = m.shape
    assert light_direction.shape == (n_points, n_light, 3)
    assert light_direction.dtype == m.dtype == float
    assert mask.dtype == bool
    assert mask.shape == (n_points,)

    mask_expanded = np.tile(mask[:, None], (1, m.shape[-1]))
    normal = np.zeros((m.shape[0], 3), dtype=float)
    rho = np.zeros(m.shape[0], dtype=float)

    if np.all(mask == False):
        return rho, normal
    masked_m = m.flatten()[mask_expanded.flatten()]
    masked_light_direction = light_direction[mask, ...]
    masked_sparse_L = construct_sparse_L(masked_light_direction)
    res, istop, itn, r1norm = scipy.sparse.linalg.lsqr(masked_sparse_L, masked_m, atol=0, btol=0, conlim=0)[:4]
    rhon_near = res.reshape(-1, 3)
    normal[mask, :] = rhon_near / (np.linalg.norm(rhon_near, axis=-1, keepdims=True) + 1e-30)
    rho[mask] = np.linalg.norm(rhon_near, axis=-1)
    return rho, normal


def construct_sparse_L(L):
    Np, Nl = L.shape[:2]
    data_term = L.flatten()
    tmp = np.arange(3 * Np).reshape(-1, 3)
    row_idx = np.repeat(tmp, Nl, axis=0).flatten()
    tmp = np.arange(Np * Nl).reshape(Np, Nl)
    col_idx = np.repeat(tmp, 3)
    A = coo_matrix((data_term, (row_idx, col_idx))).T
    return A


def map_depth_map_to_point_clouds(depth_map, mask, K):
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)
    u = np.zeros((H, W, 3))
    u[..., 0] = xx
    u[..., 1] = yy
    u[..., 2] = 1
    u = u[mask].T
    p_tilde = (np.linalg.inv(K) @ u).T
    return p_tilde * depth_map[mask, np.newaxis]


def construct_facets_from_depth_map_mask(mask):
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce(
        (facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.hstack((4 * np.ones((np.sum(facet_top_left_mask), 1)),
                      idx[facet_top_left_mask][:, None],
                      idx[facet_bottom_left_mask][:, None],
                      idx[facet_bottom_right_mask][:, None],
                      idx[facet_top_right_mask][:, None])).astype(int)


def depth_to_mesh(depth_map, mask, cameraMatrix, output_path=None, ksize=3):
    import pyvista as pv
    mask_ = cv2.boxFilter(mask.astype(int), -1, ksize=(ksize, ksize), normalize=False) == ksize ** 2
    facets = construct_facets_from_depth_map_mask(mask_)
    vertices = map_depth_map_to_point_clouds(depth_map, mask_, cameraMatrix)
    surface = pv.PolyData(vertices, facets)
    surface.save(output_path)
    return surface
