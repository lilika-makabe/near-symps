import itertools

import joblib
import numpy as np

from utils import general_utils


class PointEstimationMixin:
    """3D point estimation from scaled distances via sphere intersection."""

    def estimate_3d_points(self, e, mask, cameraMatrix=None, n_jobs=30, logger=None,
                           num_thresh=3, return_raw=False):
        if self.all_lights_on_line:
            xyz_dash_raw = self._estimate_3d_points_on_line(e, cameraMatrix, n_jobs, num_thresh=num_thresh)
        else:
            xyz_dash_raw = self._estimate_3d_points_general(e, n_jobs, logger=logger, num_thresh=num_thresh)

        xyz_dash_raw[~mask, :] = np.nan
        xyz_dash = self.inpaint_nan(xyz_dash_raw.copy(), mask.astype(bool))
        if return_raw:
            return xyz_dash, xyz_dash_raw
        return xyz_dash

    def get_inds_in_axis(self, axis_ind):
        pp_ind = self.lpos_inds_to_vector_ind(pos_in_axis="++", axis_ind=axis_ind)
        p_ind = self.lpos_inds_to_vector_ind(pos_in_axis="+", axis_ind=axis_ind)
        mm_ind = self.lpos_inds_to_vector_ind(pos_in_axis="--", axis_ind=axis_ind)
        m_ind = self.lpos_inds_to_vector_ind(pos_in_axis="-", axis_ind=axis_ind)
        return pp_ind, p_ind, mm_ind, m_ind

    def get_xdash_func_candidates(self):
        combinations = [[pind1, pind2] for pind1, pind2 in itertools.combinations(self.pinds, 2)
                        if self.angles[pind1] != self.angles[pind2]]
        func_list = []
        for inputs in combinations:
            func_list.append(self._get_xdash_func_candidates(inputs))
        return func_list

    def _get_xdash_func_candidates(self, inputs):
        nom_f = lambda radii, thetas, es: ((es[0] - es[1]) * radii[1] * np.cos(thetas[1]) -
                                           (es[2] - es[3]) * radii[0] * np.cos(thetas[0]))
        denom_f = lambda radii, thetas: -4 * radii[0] * radii[1] * np.sin(thetas[0] - thetas[1])
        p_ind1, p_ind2 = inputs
        m_ind1 = self.pind2mind[p_ind1]
        m_ind2 = self.pind2mind[p_ind2]
        angles = np.deg2rad(self.angles[[p_ind1, p_ind2]])
        radii = self.radii[[p_ind1, p_ind2]]

        nom = lambda es: nom_f(radii=radii, thetas=angles, es=es[[p_ind1, m_ind1, p_ind2, m_ind2]])
        denom = denom_f(radii=radii, thetas=angles)
        return lambda es, rhoi_s_sq: nom(es) / (self.eps + rhoi_s_sq * denom)

    def get_ydash_func_candidates(self):
        combinations = [[pind1, pind2] for pind1, pind2 in itertools.combinations(self.pinds, 2)
                        if self.angles[pind1] != self.angles[pind2]]
        func_list = []
        for inputs in combinations:
            func_list.append(self._get_ydash_func_candidates(inputs))
        return func_list

    def _get_ydash_func_candidates(self, inputs):
        nom_f = lambda radii, thetas, es: ((es[0] - es[1]) * radii[1] * np.sin(thetas[1]) -
                                           (es[2] - es[3]) * radii[0] * np.sin(thetas[0]))
        denom_f = lambda radii, thetas: -4 * radii[0] * radii[1] * np.sin(thetas[1] - thetas[0])
        p_ind1, p_ind2 = inputs
        m_ind1 = self.pind2mind[p_ind1]
        m_ind2 = self.pind2mind[p_ind2]
        angles = np.deg2rad(self.angles[[p_ind1, p_ind2]])
        radii = self.radii[[p_ind1, p_ind2]]

        nom = lambda es: nom_f(radii=radii, thetas=angles, es=es[[p_ind1, m_ind1, p_ind2, m_ind2]])
        denom = denom_f(radii=radii, thetas=angles)
        return lambda es, rhoi_s_sq: nom(es) / (self.eps + rhoi_s_sq * denom)

    def get_xydash_func_candidates(self, image_shape, cameraMatrix):
        func_list = []
        for p_ind in self.pinds:
            func_list.append(self._get_xydash_func_candidates(image_shape, cameraMatrix, p_ind))
        return func_list

    def _get_xydash_func_candidates(self, image_shape, cameraMatrix, p_ind):
        xy = general_utils.get_sensor_coord(image_shape, cameraMatrix)
        y_over_x = xy[1] / xy[0]
        x_over_y = xy[0] / xy[1]
        denom_x_f = lambda radius, theta: -4 * radius * (np.sin(theta) + y_over_x * np.cos(theta))
        denom_y_f = lambda radius, theta: -4 * radius * (x_over_y * np.sin(theta) + np.cos(theta))
        m_ind = self.pind2mind[p_ind]
        angle = np.deg2rad(self.angles[p_ind])
        radius = self.radii[p_ind]

        denom_y = denom_y_f(radius, angle)
        denom_x = denom_x_f(radius, angle)

        def func(es, rhoi_s_sq):
            x_dash = (es[p_ind] - es[m_ind]) / (self.eps + rhoi_s_sq * denom_x)
            y_dash = (es[p_ind] - es[m_ind]) / (self.eps + rhoi_s_sq * denom_y)
            y_dash_from_x = x_dash * y_over_x
            x_dash_from_y = y_dash * x_over_y
            xy_dash_1 = np.dstack((x_dash, y_dash_from_x))
            xy_dash_2 = np.dstack((x_dash_from_y, y_dash))
            xy_dash = np.nanmean(np.array([xy_dash_1, xy_dash_2]), axis=0)
            return xy_dash

        return func

    def get_zdash_func_candidates(self):
        func_list = []
        for pind in self.pinds:
            func_list.append(self._get_zdash_func_candidates(pind))
        return func_list

    def _get_zdash_func_candidates(self, pind):
        mind = self.pind2mind[pind]
        cur_radius = self.radii[pind]
        cur_angle = np.deg2rad(self.angles[pind])

        def func(es, x, y, rhoi_s_sq):
            lx = cur_radius * np.sin(cur_angle)
            ly = cur_radius * np.cos(cur_angle)
            z_dash_sq = (es[pind] / (self.eps + rhoi_s_sq)) - (x ** 2 + y ** 2 + cur_radius ** 2) \
                        + 2 * x * lx + 2 * y * ly
            z_dash_sq[z_dash_sq <= 0] = 0
            return z_dash_sq ** 0.5

        return func

    def get_rhoi_s_sq_candidates(self):
        if len(self.unique_radii) < 2:
            raise ValueError(
                "There are not enough unique radii to estimate depth. At least 2 different radii is required.")
        radii_combination = list(itertools.combinations(self.unique_radii, 2))
        combinations = []
        for r1, r2 in radii_combination:
            inds_on_the_ring1 = [i for i in self.pinds if self.radii[i] == r1]
            inds_on_the_ring2 = [i for i in self.pinds if self.radii[i] == r2]
            light_combinations = list(itertools.product(inds_on_the_ring1, inds_on_the_ring2))
            for pind1, pind2 in light_combinations:
                combinations.append([pind1, pind2])
        res = []
        for inputs in combinations:
            res.append(self._get_rhoi_s_sq_candidates(inputs))
        return res

    def _get_rhoi_s_sq_candidates(self, inputs):
        p_ind1, p_ind2 = inputs
        m_ind1 = self.pind2mind[p_ind1]
        m_ind2 = self.pind2mind[p_ind2]
        radii = self.radii[[p_ind1, p_ind2]]
        return lambda es: (es[p_ind1] + es[m_ind1] - es[p_ind2] - es[m_ind2]) \
                          / (self.eps + 2 * (radii[0] ** 2 - radii[1] ** 2))

    def get_sz(self, xyz_dash, cameraMatrix, mask):
        xy = general_utils.get_sensor_coord(xyz_dash.shape[:2], cameraMatrix)
        sz_candidates = np.array([
            xyz_dash[..., i] / (xy[i] + self.eps) - xyz_dash[..., -1] for i in range(2)])
        mask_zero_divide = np.all(xy != 0, axis=0)
        sz_candidates[:, ~np.logical_and(mask_zero_divide, mask)] = np.nan
        sz_dash = np.nanmedian(sz_candidates)
        return sz_dash

    def get_xyz_dash_candidates_general(self, e, n_jobs=30, num_thresh=5, logger=None):
        x_dash_fs = self.get_xdash_func_candidates()
        if logger is not None:
            logger.info("x candidates are collected.")
        y_dash_fs = self.get_ydash_func_candidates()
        if logger is not None:
            logger.info("y candidates are collected.")
        z_dash_fs = self.get_zdash_func_candidates()
        if logger is not None:
            logger.info("z candidates are collected.")
        rhoi_s_sq_candidates = self.get_rhoi_s_sq_candidates()
        if logger is not None:
            logger.info("rhoi candidates are collected.")
        _get_xyz_dash_candidates = self._get_xyz_dash_candidates_general
        if logger is not None:
            logger.info("copied function")

        x, y, z, w = (np.array(rhoi_s_sq_candidates, dtype=object),
                       np.array(x_dash_fs, dtype=object),
                       np.array(y_dash_fs, dtype=object),
                       np.array(z_dash_fs, dtype=object))
        if len(x) > num_thresh: x = x[np.random.choice(np.arange(len(x)), num_thresh)]
        if len(y) > num_thresh: y = y[np.random.choice(np.arange(len(y)), num_thresh)]
        if len(z) > num_thresh: z = z[np.random.choice(np.arange(len(z)), num_thresh)]
        if len(w) > num_thresh: w = w[np.random.choice(np.arange(len(w)), num_thresh)]

        combinations = np.stack(np.meshgrid(x, y, z, w, indexing='ij'), axis=-1).reshape(-1, 4)
        if logger is not None:
            logger.info(f"{len(combinations)} combinations are found.")
            batch_size = max(1, int(len(combinations) / n_jobs))
            logger.info("Batchsize is {}".format(batch_size))

        res = general_utils.ProgressParallel(n_jobs=n_jobs, batch_size=batch_size)(
            joblib.delayed(_get_xyz_dash_candidates)(e, combinations[i].tolist())
            for i in range(len(combinations)))

        data_x = np.array([item[0] for item in res])
        data_y = np.array([item[1] for item in res])
        data_z = np.array([item[2] for item in res])
        return data_x, data_y, data_z

    def get_xyz_dash_candidates_on_line(self, e, cameraMatrix, n_jobs=30, num_thresh=5, logger=None):
        assert cameraMatrix is not None, "cameraMatrix is needed when all the lights are on the same line."
        xy_dash_fs = self.get_xydash_func_candidates(e.shape[1:], cameraMatrix)
        z_dash_fs = self.get_zdash_func_candidates()
        rhoi_s_sq_candidates = self.get_rhoi_s_sq_candidates()
        _get_xyz_dash_candidates = self._get_xyz_dash_candidates_on_line

        x, y, z = (np.array(rhoi_s_sq_candidates, dtype=object),
                    np.array(xy_dash_fs, dtype=object),
                    np.array(z_dash_fs, dtype=object))
        if len(x) > num_thresh: x = x[np.random.choice(np.arange(len(x)), num_thresh)]
        if len(y) > num_thresh: y = y[np.random.choice(np.arange(len(y)), num_thresh)]
        if len(z) > num_thresh: z = z[np.random.choice(np.arange(len(z)), num_thresh)]

        combinations = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        batch_size = max(1, int(len(combinations) / n_jobs))
        if logger is not None:
            logger.info(f"{len(combinations)} combinations are found.")
            logger.info("Batchsize is {}".format(batch_size))

        res = general_utils.ProgressParallel(n_jobs=n_jobs, batch_size=batch_size)(
            joblib.delayed(_get_xyz_dash_candidates)(e, combinations[i].tolist())
            for i in range(len(combinations)))

        data_x = np.array([item[0] for item in res])
        data_y = np.array([item[1] for item in res])
        data_z = np.array([item[2] for item in res])
        return data_x, data_y, data_z

    def _get_xyz_dash_candidates_general(self, e, inputs):
        rhoi_s_sq_func, xfunc, yfunc, zfunc = inputs
        rhoi_s_sq = rhoi_s_sq_func(e)
        x_dash = xfunc(e, rhoi_s_sq)
        y_dash = yfunc(e, rhoi_s_sq)
        z_dash = zfunc(x=x_dash, y=y_dash, rhoi_s_sq=rhoi_s_sq, es=e)
        mask = np.logical_or(np.isnan(z_dash), z_dash == 0)
        x_dash[mask] = np.nan
        y_dash[mask] = np.nan
        z_dash[mask] = np.nan
        return x_dash, y_dash, z_dash, rhoi_s_sq

    def _get_xyz_dash_candidates_on_line(self, es, inputs):
        rhoi_s_sq_func, xyfunc, zfunc = inputs
        rhoi_s_sq = rhoi_s_sq_func(es)
        xy_dash = xyfunc(es, rhoi_s_sq)
        x_dash, y_dash = xy_dash[..., 0], xy_dash[..., 1]
        assert x_dash.shape == y_dash.shape == rhoi_s_sq.shape, f"{x_dash.shape}, {y_dash.shape},{rhoi_s_sq.shape}"
        z_dash = zfunc(es=es, x=x_dash, y=y_dash, rhoi_s_sq=rhoi_s_sq)
        return x_dash, y_dash, z_dash, rhoi_s_sq

    def _estimate_3d_points_general(self, e, n_jobs, logger=None, num_thresh=10):
        assert e.shape[0] < e.shape[1], f"e.shape should be [#lights,H,W] where the input is {e.shape}"
        xyz_dash = self.get_xyz_dash_candidates_general(e, n_jobs=n_jobs, logger=logger, num_thresh=num_thresh)
        if logger is not None:
            logger.info("xyz candidates are collected.")
        x_dash, y_dash, z_dash = map(lambda item: np.nanmedian(item, axis=0), xyz_dash)
        xyz_dash = np.dstack((x_dash, y_dash, z_dash))
        H, W = e.shape[1:]
        assert xyz_dash.shape == (H, W, 3)
        return xyz_dash

    def _estimate_3d_points_on_line(self, e, cameraMatrix, n_jobs, num_thresh=10):
        assert e.shape[0] < e.shape[1], f"e.shape should be [#lights,H,W] where the input is {e.shape}"
        xyz_dash = self.get_xyz_dash_candidates_on_line(e, cameraMatrix, n_jobs=n_jobs, num_thresh=num_thresh)
        x_dash, y_dash, z_dash = map(lambda item: np.nanmedian(item, axis=0), xyz_dash)
        xyz_dash = np.dstack((x_dash, y_dash, z_dash))
        H, W = e.shape[1:]
        assert xyz_dash.shape == (H, W, 3)
        return xyz_dash
