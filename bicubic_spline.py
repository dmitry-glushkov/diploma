from spline import *

class Bispline:
    N: int = 0
    M: int = 0
    matrix_f: np.array([])
    vector_u: list = []
    vector_v: list = []
    matrix_M_20: np.array([])
    matrix_M_02: np.array([])
    matrix_M_22: np.array([])
    spl_u: list = []
    spl_v: list = []


    def __init__(self, vector_u, vector_v, matrix_f):
        self.vector_u = vector_u
        self.vector_v = vector_v
        self.matrix_f = np.array(matrix_f)
        self.calculate_spline()
        
    def calculate_spline(self):
        matrix_f_rot90 = np.rot90(self.matrix_f)
        N = len(self.matrix_f)
        M = len(self.matrix_f[0])

        matrix_M_20 = []
        for i in range(M):
            spl = Spline(self.vector_u, matrix_f_rot90[i, :])
            spl.calculate_spline() # 1
            self.spl_u.append(spl)
            matrix_M_20.append(spl.calculate_spline())
        self.matrix_M_20 = np.array(matrix_M_20)

        matrix_M_02 = []
        for j in range(N):
            spl = Spline(self.vector_v, matrix_f_rot90[:, j])
            spl.calculate_spline() # 1
            self.spl_v.append(spl)
            matrix_M_02.append(spl.calculate_spline())
        self.matrix_M_02 = np.array(matrix_M_02)

        matrix_M_22 = []
        for j in range(N):
            spl = Spline(self.vector_v, self.matrix_M_20[:, j])
            spl.calculate_spline() # 1
            matrix_M_22.append(spl.calculate_spline())
        self.matrix_M_22 = np.array(matrix_M_22)
        self.matrix_M_20 = np.transpose(self.matrix_M_20) # ???

    # ПРАВИТЬ
    def derivative(self, u, v):
        # print ('len spl_u:', len(self.spl_u))
        # print ('len spl_v:', len(self.spl_v))
        # print ('len vector_u:', len(self.vector_u))
        # print ('len vector_v:', len(self.vector_v))
        idx_u = 0
        idx_v = 0
        for i in range(len(self.vector_u) - 1):
            if (u < self.vector_u[i+1] and u >= self.vector_u[i]):
                idx_u = i
        for j in range(len(self.vector_v) - 1):
            if (v < self.vector_v[j+1] and v >= self.vector_v[j]):
                idx_v = j

        deriv_u = self.spl_u[idx_v].test_derivative(u)
        deriv_v = self.spl_v[idx_u].test_derivative(v)

        return deriv_u, deriv_v

    def get_point(self, u, v):
        idx_i = 0
        for i in range(len(self.vector_u)-1):
            if (u >= self.vector_u[i] and u < self.vector_u[i+1]):
                idx_i = i
        
        idx_j = 0
        for j in range(len(self.vector_v)-1):
            if (v >= self.vector_v[j] and v < self.vector_v[j+1]):
                idx_j = j

        hi = self.vector_u[idx_i + 1] - self.vector_u[idx_i]
        w = (u-self.vector_u[idx_i]) / hi
        psi_w = np.array([1-w, w, hi**2 * w*(w-1)*(w-2), hi**2 * w*(w**2 - 1)])

        dj = self.vector_v[idx_j + 1] - self.vector_v[idx_j]
        ksi = (v - self.vector_v[idx_j]) / dj
        psi_ksi = np.array([1-ksi, ksi, dj**2 * ksi*(ksi-1)*(ksi-2), dj**2 * ksi*(ksi**2 - 1)])

        matrix_F = np.array([
            [self.matrix_f[idx_i, idx_j], self.matrix_f[idx_i, idx_j+1], \
            1/6*self.matrix_M_02[idx_i, idx_j], 1/6*self.matrix_M_02[idx_i, idx_j]],
            [self.matrix_f[idx_i+1, idx_j], self.matrix_f[idx_i+1, idx_j+1], \
            1/6*self.matrix_M_02[idx_i+1, idx_j], 1/6*self.matrix_M_02[idx_i+1, idx_j]],
            [1/6*self.matrix_M_20[idx_i, idx_j], 1/6*self.matrix_M_20[idx_i, idx_j+1], \
            1/36*self.matrix_M_22[idx_i, idx_j], 1/36*self.matrix_M_22[idx_i, idx_j+1]],
            [1/6*self.matrix_M_20[idx_i+1, idx_j], 1/6*self.matrix_M_20[idx_i+1, idx_j+1], \
            1/36*self.matrix_M_22[idx_i+1, idx_j], 1/36*self.matrix_M_22[idx_i+1, idx_j+1]]
        ])

        return np.dot(psi_w, np.dot(matrix_F, psi_ksi))