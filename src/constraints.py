import itertools

import numpy as np
import scipy.sparse as sp

from utils import io_utils


def _pair_iter(indices, use_all_combinations=False):
    """Yield pairs of indices: consecutive pairs (default) or all combinations."""
    if use_all_combinations:
        yield from itertools.combinations(indices, 2)
    else:
        for i in range(len(indices) - 1):
            yield indices[i], indices[i + 1]


class ConstraintMixin:
    """Constraint matrix builders for symmetric near-light photometric stereo."""

    def _build_eq1(self, num_thresh):
        base_light_pos = io_utils.light_params_to_lpos(self.radii, self.angles)[:, :2]
        if self.all_lights_on_line:
            basis_inds_list = [[pind] for pind in self.pinds]
            basis_inds_list = basis_inds_list[:num_thresh]
            basis_r = lambda basis_inds: self.radii[basis_inds]
            get_coeffs = lambda target_ind, basis_inds: self.radii[target_ind] / basis_r(basis_inds)
        else:
            basis_inds_list = [[pind1, pind2] for pind1, pind2 in list(itertools.combinations(self.pinds, 2)) if
                               self.angles[pind1] != self.angles[pind2]]
            basis_inds_list = basis_inds_list[:num_thresh]
            A = lambda basis_inds: np.vstack(base_light_pos[basis_inds]).T
            get_coeffs = lambda target_ind, basis_inds: \
                np.linalg.lstsq(A(basis_inds=basis_inds), base_light_pos[target_ind], rcond=None)[0]

        data = []
        row = []
        col = []
        row_cnt = 0
        for basis_inds in basis_inds_list:
            for target_ind in self.pinds:
                if target_ind in basis_inds: continue
                if self.radii[target_ind] < 0: raise ValueError("Negative radius encountered")
                coeffs = get_coeffs(target_ind, basis_inds)
                if isinstance(coeffs, int): coeffs = [coeffs]
                data += [coeffs[i] for i in range(len(basis_inds))]
                data += [-coeffs[i] for i in range(len(basis_inds))]
                data += [-1., 1.]
                col += [basis_inds[i] for i in range(len(basis_inds))]
                col += [self.pind2mind[basis_inds[i]] for i in range(len(basis_inds))]
                col += [target_ind, self.pind2mind[target_ind]]
                row += [row_cnt] * (len(basis_inds) + 1) * 2
                row_cnt += 1

        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A

    def build_eq1(self, image):
        return self.A_eq1 @ sp.diags(image)

    def _build_eq2(self, num_thresh):
        data = []
        row = []
        col = []
        row_cnt = 0
        for pind1, pind2 in _pair_iter(self.pinds, self.use_all_combinations):
            mind1 = self.pind2mind[pind1]
            mind2 = self.pind2mind[pind2]
            data += [1, 1, -1, -1]
            col += [pind1, mind1, pind2, mind2]
            row += [row_cnt] * 4
            row_cnt += 1

        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A

    def build_eq2(self, image):
        return self.A_eq2 @ sp.diags(image)

    def build_eq3(self, uv=None, cameraMatrix=None):
        if cameraMatrix is None:
            return self.A_eq3_wo_cameraMatrix
        else:
            return self._build_eq3_w_cameraMatrix(uv, cameraMatrix)

    def _build_eq3_wo_cameraMatrix(self, num_thresh=3):
        unique_angles = np.unique(self.angles)
        base_light_pos = io_utils.light_params_to_lpos(self.radii, self.angles)[:, :2]
        if unique_angles.size == 1:
            basis_inds_list = [[pind] for pind in self.pinds]
            basis_r = lambda basis_inds: self.radii[basis_inds]
            get_coeffs = lambda target_ind, basis_inds: self.radii[target_ind] / basis_r(basis_inds)
        else:
            basis_inds_list = [[pind1, pind2] for pind1, pind2 in list(itertools.combinations(self.pinds, 2)) if
                               self.angles[pind1] != self.angles[pind2]]
            basis_inds_list = basis_inds_list[:num_thresh]
            A = lambda basis_inds: np.vstack(base_light_pos[basis_inds]).T
            get_coeffs = lambda target_ind, basis_inds: \
                np.linalg.lstsq(A(basis_inds=basis_inds), base_light_pos[target_ind], rcond=None)[0]

        data = []
        row = []
        col = []
        row_cnt = 0
        for basis_inds in basis_inds_list:
            for target_ind in self.pinds:
                if target_ind in basis_inds: continue
                if self.radii[target_ind] < 0: raise ValueError("Negative radius encountered")
                coeffs = get_coeffs(target_ind, basis_inds)
                if isinstance(coeffs, int): coeffs = [coeffs]
                data += [coeffs[i] for i in range(len(basis_inds))]
                data += [-coeffs[i] for i in range(len(basis_inds))]
                data += [-1, 1]
                col += [basis_inds[i] for i in range(len(basis_inds))]
                col += [self.pind2mind[basis_inds[i]] for i in range(len(basis_inds))]
                col += [target_ind, self.pind2mind[target_ind]]
                row += [row_cnt] * (len(basis_inds) + 1) * 2
                row_cnt += 1

        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A

    def _build_eq3_w_cameraMatrix(self, uv, cameraMatrix):
        cxcy = cameraMatrix[:2, 2]
        fxfy = cameraMatrix[(0, 1), (0, 1)]
        x, y = (uv.astype(float) - cxcy) / fxfy

        data = []
        row = []
        col = []
        row_cnt = 0
        for pind1, pind2 in _pair_iter(self.pinds, self.use_all_combinations):
            mind1 = self.pind2mind[pind1]
            mind2 = self.pind2mind[pind2]
            radii_ratio = self.radii[pind2] / self.radii[pind1]
            angle_ind1 = np.deg2rad(self.angles[pind1])
            angle_ind2 = np.deg2rad(self.angles[pind2])
            scale_ind1 = radii_ratio * (x * np.sin(angle_ind2) + y * np.cos(angle_ind2))
            scale_ind2 = 1. * (x * np.sin(angle_ind1) + y * np.cos(angle_ind1))
            data += [scale_ind1, -scale_ind1, -scale_ind2, scale_ind2]
            col += [pind1, mind1, pind2, mind2]
            row += [row_cnt] * 4
            row_cnt += 1
        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A

    def _build_eq4(self, num_thresh):
        A = self.build_eq4_with_same_ring(num_thresh)
        A_ = self.build_eq4_with_3different_radii()
        A = sp.vstack([A, A_])
        return A

    def build_eq4(self):
        return self.A_eq4

    def build_eq4_with_same_ring(self, num_thresh):
        data = []
        row = []
        col = []
        row_cnt = 0
        for r in self.unique_radii:
            if r < 0: continue
            inds_on_the_same_ring = [i for i in self.pinds if self.radii[i] == r]
            if len(inds_on_the_same_ring) < 2: continue
            for pind1, pind2 in _pair_iter(inds_on_the_same_ring, self.use_all_combinations):
                mind1 = self.pind2mind[pind1]
                mind2 = self.pind2mind[pind2]
                data += [1, 1, -1, -1]
                col += [pind1, mind1, pind2, mind2]
                row += [row_cnt] * 4
                row_cnt += 1
        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A

    def build_eq4_with_3different_radii(self):
        data = []
        row = []
        col = []
        row_cnt = 0
        if len(self.unique_radii) < 3: return sp.coo_matrix(([], ([], [])), shape=(0, self.n_lights))
        radii_combination = list(itertools.combinations(self.unique_radii, 3))
        for r1, r2, r3 in radii_combination:
            inds_on_the_ring1 = [i for i in self.pinds if self.radii[i] == r1]
            inds_on_the_ring2 = [i for i in self.pinds if self.radii[i] == r2]
            inds_on_the_ring3 = [i for i in self.pinds if self.radii[i] == r3]
            light_combinations = list(itertools.product(inds_on_the_ring1, inds_on_the_ring2, inds_on_the_ring3))
            light_combinations = [light_combinations[0]]
            for pind1, pind2, pind3 in light_combinations:
                mind1 = self.pind2mind[pind1]
                mind2 = self.pind2mind[pind2]
                mind3 = self.pind2mind[pind3]
                scale2 = 1 / ((self.radii[pind2] / self.radii[pind1]) ** 2 - 1)
                scale3 = 1 / ((self.radii[pind3] / self.radii[pind1]) ** 2 - 1)
                scale1 = -scale2 + scale3
                data += [scale2, scale2, -scale3, -scale3, scale1, scale1]
                col += [pind2, mind2, pind3, mind3, pind1, mind1]
                row += [row_cnt] * 6
                row_cnt += 1
        A = sp.coo_matrix((data, (row, col)), shape=(row_cnt, self.n_lights))
        return A
