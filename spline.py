from vector4f import *
from calculations import *

class Spline:
    n: int = 0
    vector_m: list = []
    vector_u: list = []
    vector_f: list = []

    def __init__(self, vector_x, vector_f):
        self.vector_u = vector_x
        self.vector_f = vector_f

    def calculate_spline(self):
        N = len(self.vector_f)  #(N = книжная N+1 - количество промежутков)
        vector_h = np.zeros(N-1)
        vector_b = np.zeros(N)
        vector_avg_f = np.zeros(N-1)
        above_diagonal = np.zeros(N-1)
        main_diagonal = np.zeros(N)
        below_diagonal = np.zeros(N-1)
        
        # Вспомогательные значения.
        for i in range(N-1):
            vector_h[i] = self.vector_u[i+1] - self.vector_u[i]

        for i in range(N-1):
            vector_avg_f[i] = (self.vector_f[i+1]-self.vector_f[i]) / vector_h[i]

        # Вектор b
        for i in range(N):
            if (i == 0 or i == N-1): # т.к. f0'' = fN'' = 0
                vector_b[i] = 0
            else:
                vector_b[i] = 6*(vector_avg_f[i] - vector_avg_f[i-1])

        # Диагонали матрицы A
        for i in range(N-1):
            if (i == 0):
                above_diagonal[i] = 0
            else:
                above_diagonal[i] = vector_h[i]

        for i in range(N):
            if (i == 0 or i == N-1):
                main_diagonal[i] = 1
            else:
                main_diagonal[i] = 2*(vector_h[i-1] + vector_h[i])

        for i in range(N-1):
            if (i == N-2):
                below_diagonal[i] = 0
            else:
                below_diagonal[i] = vector_h[i]


        self.vector_m = TDMA_solver(above_diagonal, main_diagonal, below_diagonal, vector_b)
        return self.vector_m

    def get_point(self, x):
        idx = 0

        for i in range(len(self.vector_u) - 1):
            if (x < self.vector_u[i+1] and x >= self.vector_u[i]):
                idx = i
        
        if (idx == len(self.vector_u)):
            idx = 0
        hi = self.vector_u[idx+1] - self.vector_u[idx]
        value = self.vector_m[idx] * (self.vector_u[idx+1] - x)**3 / (6*hi) \
                + self.vector_m[idx+1] * (x - self.vector_u[idx])**3 / (6*hi) \
                + (self.vector_f[idx] - self.vector_m[idx] * hi**2 / 6) * (self.vector_u[idx+1] - x) / hi \
                + (self.vector_f[idx+1]- self.vector_m[idx+1] * hi**2 / 6) * (x - self.vector_u[idx])/ hi
        return value

    def derivative(self, x):
        idx = 0

        for i in range(len(self.vector_u) - 1):
            if (x < self.vector_u[i+1] and x >= self.vector_u[i]):
                idx = i
        
        hi = self.vector_u[idx+1] - self.vector_u[idx]
        value = -self.vector_m[idx] * (self.vector_u[idx+1] - x)**2 / (2*hi) \
                + self.vector_m[idx+1] * (x - self.vector_u[idx])**2 / (2*hi) \
                + (self.vector_f[idx] / hi - self.vector_m[idx] * hi/6) \
                + (self.vector_f[idx+1] / hi- self.vector_m[idx+1] * hi/6)
        return value

    def test_derivative(self, u):
        eps = 0.000002
        av = self.get_point(u - eps) - self.get_point(u + eps)
        o = self.get_point(u) + av
        return o


def get_normal(u, spline_x: Spline, spline_y: Spline):
    return Vector4f(spline_x.derivative(u), spline_y.derivative(u), 0, 1)




# # тесты
# vec_u = np.arange(0, 2*np.pi, np.pi/4)
# x = []
# y = []
# for u in vec_u:
#     x.append(np.cos(u))
#     y.append(np.sin(u))

# Sx = Spline(vec_u, x)
# Sy = Spline(vec_u, y)
# Sx.calculate_spline()
# Sy.calculate_spline()

# for u in vec_u:
#     print([Sx.derivative(u) / np.sqrt(Sx.derivative(u)**2 + Sy.derivative(u)**2), 
#            Sy.derivative(u) / np.sqrt(Sx.derivative(u)**2 + Sy.derivative(u)**2)])

# print('------')
# for u in vec_u:
#     print([Sx.test_derivative(u) / np.sqrt(Sx.test_derivative(u)**2 + Sy.test_derivative(u)**2),
#         Sy.test_derivative(u) / np.sqrt(Sx.test_derivative(u)**2 + Sy.test_derivative(u)**2)])
