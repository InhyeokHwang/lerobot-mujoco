import numpy as np
from pytransform3d import rotations

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

class Quest3Preprocessor:
    def __init__(self):
        self._T = grd_yup2grd_zup
        self._Tinv = fast_mat_inv(self._T)

        self._left = np.eye(4)
        self._right = np.eye(4)

    def process(self, left_raw, right_raw):
        self._left  = mat_update(self._left, left_raw)
        self._right = mat_update(self._right, right_raw)

        left  = self._T @ self._left  @ self._Tinv
        right = self._T @ self._right @ self._Tinv

        return left, right

    @staticmethod
    def mat_to_pose(T):
        q = rotations.quaternion_from_matrix(T[:3, :3])
        return T[:3, 3], q[[1, 2, 3, 0]]  # xyzw