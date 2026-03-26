import os
import sys

import cv2
import numpy as np

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils import io_utils, general_utils


class Evaluator(object):
    def __init__(self, args):
        pass

    def output(self, output_dir, dataset, logger=None):
        data = dataset.get_data()
        self.mask = data["mask"]
        self.cameraMatrix = data.get("cameraMatrix")
        res = dict(np.load(os.path.join(output_dir, "res.npz")))
        depth_est = res.get("depth")
        normal_est = res.get("normal")
        from utils import log_util
        if logger is None:
            logger = log_util.parse_logger(name="__main__", output_logdir=os.path.join(output_dir, "log"))

        if normal_est is not None:
            self.save_normal(os.path.join(output_dir, "normal_fromours.png"), normal_est)
        else:
            logger.warning("Normal map not found in res.npz, skipping.")

        if depth_est is not None:
            if self.cameraMatrix is not None:
                general_utils.depth_to_mesh(depth_est.copy(), self.mask,
                                            cameraMatrix=self.cameraMatrix,
                                            output_path=os.path.join(output_dir, "depth_fromours.ply"),
                                            ksize=3)
        else:
            logger.warning("Depth map not found in res.npz, skipping.")

    def save_normal(self, output, normal):
        mask = self.mask.copy()
        normal[~mask] = np.nan
        visnormal = io_utils.normal2color(normal, dtype=np.uint8)
        visnormal = io_utils.add_alpha_channel(visnormal, mask)
        visnormal = io_utils.crop_images(mask, [visnormal])[0]
        cv2.imwrite(output, visnormal)


def main(args):
    from dataset.dataset_ours import DatasetOurs
    dataset = DatasetOurs(root_dir=args.root_dir, initial_resize=args.initial_resize)
    evaluator = Evaluator(args)
    output_dir = args.output_dir
    from utils import log_util
    logger = log_util.parse_logger(os.path.join(output_dir, "log"))
    logger.info("evaluating...")
    evaluator.output(output_dir, dataset, logger=logger)
