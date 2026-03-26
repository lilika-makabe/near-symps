import copy

import joblib
import numpy as np
import scipy.sparse as sp

from utils import general_utils


class DistanceEstimationMixin:
    """Scaled distance estimation via per-pixel SVD."""

    def estimate_scaled_distances(self, args, mask, images, cameraMatrix, num_thresh=3):
        from utils.general_utils import ProgressParallel

        n_light, h, w = images.shape
        images = images.reshape(n_light, -1)
        _estimate_scaled_distances_batchwise = copy.deepcopy(self.estimate_scaled_distances_batchwise)
        uvcoord = np.meshgrid(np.arange(w), np.arange(h))
        uvcoord = np.stack(uvcoord, axis=0)
        uvcoord = uvcoord.reshape(2, -1).T
        uvb = general_utils.batch_data(uvcoord, batch_num=args.n_jobs)
        maskb = general_utils.batch_data(mask.reshape(-1), batch_num=args.n_jobs)
        Ib = general_utils.batch_data(images.T, batch_num=args.n_jobs)
        cameraMatricies = [cameraMatrix] * len(Ib)
        res = ProgressParallel(n_jobs=args.n_jobs, batch_size=1, temp_folder="/dev/shm", max_nbytes='3000G',
                               backend='multiprocessing', verbose=10)(
            joblib.delayed(_estimate_scaled_distances_batchwise)(maskb[i], Ib[i], uvb[i], cameraMatricies[i],
                                                                 num_thresh)
            for i in range(args.n_jobs))

        estimation_flag_mask = np.concatenate([item[0] for item in res]).reshape(h, w).astype(float)
        estimation_flag_mask[~mask] = 0
        scaled_distances = np.concatenate([item[1] for item in res])
        scaled_distances = scaled_distances.reshape(h, w, -1).transpose(2, 0, 1)
        return estimation_flag_mask, scaled_distances

    def estimate_scaled_distances_batchwise(self, maskb, Ib, uvb, cameraMatrix, num_thresh=3):
        results = [[], []]
        for i in range(uvb.shape[0]):
            flag, dist = self.estimate_scaled_distances_pixelwise(maskb[i], Ib[i], uvb[i], cameraMatrix, num_thresh)
            results[0].append(flag)
            results[1].append(dist)
        return results

    def estimate_scaled_distances_pixelwise(self, mask, image, uv, cameraMatrix, num_thresh=3):
        if not mask:
            return 0, np.zeros(len(image))
        A_eq1 = self.build_eq1(image)
        A_eq2 = self.build_eq2(image)
        A_eq3 = self.build_eq3(uv, cameraMatrix)
        A_eq4 = self.build_eq4()
        A = sp.vstack((A_eq1, A_eq2))
        A_ = sp.vstack((A_eq3, A_eq4))
        concatA = sp.vstack((A, A_))
        flag, distances = self._solve_homogeneous(concatA)
        return flag, distances

    def _solve_homogeneous(self, concatA):
        u, s, vt = np.linalg.svd(concatA.toarray())
        res = np.squeeze(np.asarray(vt[-1]))[:self.n_lights]
        assert res.shape == (self.n_lights,), f"res shape : {res.shape}"
        assert np.any(res != 0)
        res = np.abs(res)
        res /= np.linalg.norm(res)
        flag = np.linalg.matrix_rank(concatA.toarray())
        return flag, res
