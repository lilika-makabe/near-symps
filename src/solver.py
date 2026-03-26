import copy
import os.path
import sys

import cv2
import numpy as np

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from constraints import ConstraintMixin
from dataset.dataset_ours import DatasetOurs
from distance_estimation import DistanceEstimationMixin
from point_estimation import PointEstimationMixin
from utils import general_utils, io_utils
from utils.vis_utils import plot_scaled_distances, discrete_matshow

np.random.seed(0)
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))


class Solver(ConstraintMixin, DistanceEstimationMixin, PointEstimationMixin):
    def __init__(self, use_all_combinations=False):
        self.eps = 1e-30
        self.use_all_combinations = use_all_combinations

    def precompute_params(self, lparams, num_thresh=3):
        radii, angle = lparams
        self.smallest_radius = np.abs(radii).min()
        self.radii = radii / self.smallest_radius
        self.angles = angle
        self.pol2ind = {(self.radii[i], angle[i]): i for i in range(len(self.radii))}
        self.ind2pol = {i: (self.radii[i], angle[i]) for i in range(len(self.radii))}
        self.unique_radii = np.unique(np.abs(self.radii))
        self.unique_angles = np.unique(angle)
        mask_pradii = self.radii > 0
        self.pinds = np.where(mask_pradii)[0]
        self.pind2mind = {i: self.pol2ind[(-self.radii[i], angle[i])] for i in self.pinds}
        self.n_lights = len(self.radii)
        self.all_lights_on_line = True if self.unique_angles.size == 1 else False

        self.A_eq3_wo_cameraMatrix = self._build_eq3_wo_cameraMatrix(num_thresh)
        self.A_eq4 = self._build_eq4(num_thresh)
        self.A_eq1 = self._build_eq1(num_thresh)
        self.A_eq2 = self._build_eq2(num_thresh)

    def solve(self, args):
        np.random.seed(0)
        from utils import log_util
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        io_utils.save_args_to_yaml(os.path.join(output_dir, "solver_args.yaml"), args)
        logger = log_util.parse_logger(output_logdir=output_dir)
        dataset = DatasetOurs(root_dir=args.root_dir, initial_resize=args.initial_resize)
        lparams = dataset.get_light_params()
        self.precompute_params(lparams.T)

        self.dataset = dataset
        data, images = dataset.get(raw=False, color=False)
        general_utils.plot_results_in_subplots(data=[img for img in images],
                                               output_path=os.path.join(output_dir, "images_input.png"),
                                               figshape=(1, len(images)),
                                               cmap='gray')
        cameraMatrix = data["cameraMatrix"]
        mask = data["mask"]
        images = images.astype(float)
        assert images.ndim == 3

        if not args.w_cameraMatrix:
            cameraMatrix = None
        logger.info("start scaled distance estimation")

        estimation_flag_mask, scaled_distances = self.estimate_scaled_distances(args, mask, images, cameraMatrix,
                                                                                num_thresh=args.num_thresh)
        if not self.use_all_combinations:
            # Paper mode: mask out pixels where SVD rank is insufficient
            scaled_distances[:, estimation_flag_mask < len(scaled_distances) - 1] = np.nan
            estimation_flag_mask[estimation_flag_mask < len(scaled_distances) - 1] = np.nan
        plot_scaled_distances(scaled_distances.copy(), os.path.join(output_dir, "rho.png"),
                              mask=mask.astype(bool))

        discrete_matshow(estimation_flag_mask, os.path.join(output_dir, "estimation_flag_mask.png"))
        logger.info("start 3d point estimation")
        xyz_dash, xyz_dash_raw = self.estimate_3d_points(scaled_distances, mask,
                                                         n_jobs=args.n_jobs,
                                                         num_thresh=args.num_thresh, logger=logger, return_raw=True,
                                                         cameraMatrix=cameraMatrix)
        general_utils.plot_results_in_subplots(data=[xyz_dash[..., i] for i in range(3)],
                                               output_path=os.path.join(output_dir, "xyz.png"),
                                               minmax_list=[[-10, 10]])
        general_utils.plot_results_in_subplots(data=[xyz_dash_raw[..., i] for i in range(3)],
                                               output_path=os.path.join(output_dir, "xyz_raw.png"),
                                               minmax_list=[[-10, 10]])
        depth = xyz_dash[..., -1]
        res = {"scaled_distances": scaled_distances,
               "shifted_xyz": xyz_dash,
               "depth": depth,
               "ours_mask": np.logical_and(mask, estimation_flag_mask),
               "mask": mask,
               }
        if not self.all_lights_on_line:
            logger.info("start normal estimation")
            albedo_est, normal_est = self.estimate_PS_normal(images, xyz_dash, mask, args.n_jobs)
            rgb_images = dataset.get_images(raw=True, color=True)
            albedo_color = self.estimate_colored_albedo(rgb_images, xyz_dash, normal_est)
            res["albedo_gray"] = albedo_est
            res["albedo"] = albedo_color
            res["normal"] = normal_est
        xyz_dash *= self.smallest_radius
        np.savez(os.path.join(output_dir, "res.npz"), **res)

        cv2.imwrite(os.path.join(output_dir, "input_mask.png"), (mask.astype(float) * 255).astype(int))

    def inpaint_nan(self, shifted_xyz, mask=None, zero_thresh=1e-2):
        from scipy import interpolate
        zmask = np.logical_and(mask, (shifted_xyz[..., 2] < zero_thresh))
        nanmask = np.logical_and(np.any(np.isnan(shifted_xyz), axis=-1), mask)
        invalid_mask = np.logical_or(nanmask, zmask)
        cv2.imwrite("nanmask.png", invalid_mask.astype(np.uint8) * 255)
        for i in range(3):
            image = shifted_xyz[..., i]
            valid_mask = np.logical_and(~invalid_mask, mask)
            coords = np.array(np.nonzero(valid_mask)).T
            values = image[valid_mask]
            it = interpolate.LinearNDInterpolator(coords, values, fill_value=np.nanmean(values))
            inpainted = it(list(np.ndindex(image.shape))).reshape(image.shape)
            shifted_xyz[..., i] = inpainted
        shifted_xyz[~mask, :] = np.nan
        return shifted_xyz

    def estimate_colored_albedo(self, rgb_images, shifted_xyz, est_normal):
        shifted_light_pos = io_utils.light_params_to_lpos(self.radii, self.angles, degree=True)
        light_direction = np.array([ith_lpos - shifted_xyz for ith_lpos in shifted_light_pos])
        light_direction = light_direction / np.linalg.norm(light_direction, axis=-1, keepdims=True) ** 2
        n_light, h, w, ch = light_direction.shape
        assert est_normal.shape == (h, w, ch)
        shading = np.abs(np.sum(light_direction * est_normal[None, ...], axis=-1))
        albedo = []
        for i in range(3):
            albedo_ch = rgb_images[..., i] / shading
            albedo_ch[shading < 0] = np.nan
            albedo_ch = np.nanmean(albedo_ch, axis=0)
            albedo_ch[np.isnan(albedo_ch)] = 0
            albedo.append(albedo_ch)
        albedo = np.dstack(albedo)
        assert albedo.shape == (h, w, 3)
        return albedo

    def scaled_led_pos(self):
        return io_utils.light_params_to_lpos(self.radii, self.angles, degree=True)

    def estimate_PS_normal(self, images, shifted_xyz, mask, n_jobs):
        scaled_led_pos = io_utils.light_params_to_lpos(self.radii, self.angles, degree=True)
        point2light = np.array([ith_lpos - shifted_xyz for ith_lpos in scaled_led_pos])
        scaled_light_direction = point2light / np.linalg.norm(point2light, axis=-1, keepdims=True) ** 2
        Albedo, N_est = general_utils.solve_nearlight_PS_given_lightdirection_multicore(images.astype(float),
                                                                                        mask=mask,
                                                                                        scaled_led_dir=scaled_light_direction,
                                                                                        n_jobs=n_jobs,
                                                                                        avoid_normalize=True)
        return Albedo, N_est


class SolvePS:
    def run(self, args):
        args_tmp = copy.deepcopy(args)
        solver = Solver(use_all_combinations=args_tmp.all_combinations)
        solver.solve(args_tmp)
        import evaluate
        evaluate.main(args_tmp)


def main(args):
    SolvePS().run(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help="dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output directory (default: output/<basename of root_dir>)")
    parser.add_argument("--n_jobs", default=3, type=int)
    parser.add_argument("--num_thresh", default=10, type=int)
    parser.add_argument("--initial_resize", type=float, default=1.)
    parser.add_argument("--w_cameraMatrix", action="store_true",
                        help="use camera intrinsic matrix (requires camera_params.txt)")
    parser.add_argument("--all_combinations", action="store_true",
                        help="use all pairwise combinations for constraints instead of consecutive pairs")
    parser.add_argument("--lights_to_load", nargs="+", default=list(range(50)), type=int)

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join("output", os.path.basename(args.root_dir))
    main(args)
