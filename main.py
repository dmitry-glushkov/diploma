from bicubic_spline import *
from renderer import *
from calculations import *

U_STEPS = 2
V_STEPS = 256
IT_STEP = 0.15

def create_object(app, spl_x, spl_y, spl_z, vector_u, vector_v):
    # При некоторых значениях возникает ошибка
    vertexes, faces = [], []
    idx = 0
    u_steps = len(vector_u)
    v_steps = len(vector_v)
    for u in vector_u:
        for v in vector_v:
            vertexes.append([spl_x.get_point(u, v), spl_y.get_point(u, v), spl_z.get_point(u, v), 1])
            if v == 2*np.pi - 2*np.pi/v_steps and u != 1 - 1/u_steps:
                faces.append([idx, idx-v_steps+1])
                faces.append([idx, (idx+v_steps) % (u_steps*v_steps)])
            elif u == 1 - 1/u_steps and v != 2*np.pi - 2*np.pi/v_steps:
                faces.append([idx, idx+1])
            elif u == 1 - 1/u_steps and v == 2*np.pi - 2*np.pi/v_steps:
                faces.append([idx, idx - v_steps + 1])
            else:
                faces.append([idx, idx + 1])
                faces.append([idx, idx + v_steps])

            idx += 1

    return Object(app, vertexes, faces), vertexes

def init_charge(a, b, r, R, h):
    points, shell_points, v_len, u_len = init_points(a, b, r, R, h)
    points = np.array(points)
    return points, u_len, v_len

def points_to_matrix(points, len_u, len_v, fl=True):
    if fl:
        points = np.array(points)
    matrix_f = points.reshape((len_u, len_v))
    matrix_f_x = np.empty([len_u, len_v])
    matrix_f_y = np.empty([len_u, len_v])
    matrix_f_z = np.empty([len_u, len_v])
    for i in range(len_u):
        for j in range(len_v):
            #if (i != len_u and j != len_v):
            matrix_f_x[i, j] = matrix_f[i, j].x
            matrix_f_y[i, j] = matrix_f[i, j].y
            #
            matrix_f_z[i, j] = matrix_f[i, j].z

    matrix_f_x = np.r_[matrix_f_x, [matrix_f_x[0]]] # зацикливаем по строкам
    matrix_f_x = np.c_[matrix_f_x, matrix_f_x[:, 0]] # зацикливаем по столбцам
    matrix_f_y = np.r_[matrix_f_y, [matrix_f_y[0]]]
    matrix_f_y = np.c_[matrix_f_y, matrix_f_y[:, 0]]

    return matrix_f_x, matrix_f_y, matrix_f_z

def points_shift(points, vector_u, vector_v, spl_x, spl_y, spl_z):
    new_points = []
    # new_vector_u = []
    # new_vector_v = []
    idx = 0
    for u in vector_u: 
        for v in vector_v:
            tmp = get_num_normal(u, v, spl_x, spl_y, spl_z)
            new_x = points[idx][0] - tmp[0]*IT_STEP
            new_y = points[idx][1] - tmp[1]*IT_STEP
            new_z = points[idx][2] - tmp[2]*IT_STEP
            new_points.append(Vector4f(new_x, new_y, new_z, idx))
            #np.append(new_points, [points[idx] + [get_num_normal(u, v, spl_x, spl_y, spl_z), 0]])
            idx += 1

    return new_points

def get_normal(u, v, spl_x, spl_y, spl_z):
    x_u, x_v = spl_x.derivative(u, v)
    y_u, y_v = spl_y.derivative(u, v)
    z_u, z_v = spl_z.derivative(u, v)

    norm = np.cross([x_u, y_u, z_u], [x_v, y_v, z_v])

    print(norm)

    return norm

def get_num_deriv(u, v, spl_x, spl_y, spl_z):
    eps = 0.02
    x_u = (spl_x.get_point(u+eps, v) - spl_x.get_point(u-eps, v)) / 2
    x_v = (spl_x.get_point(u, v+eps) - spl_x.get_point(u, v-eps)) / 2
    y_u = (spl_y.get_point(u+eps, v) - spl_y.get_point(u-eps, v)) / 2
    y_v = (spl_y.get_point(u, v+eps) - spl_y.get_point(u, v-eps)) / 2
    z_u = (spl_z.get_point(u+eps, v) - spl_z.get_point(u-eps, v)) / 2
    z_v = (spl_z.get_point(u, v+eps) - spl_z.get_point(u, v-eps)) / 2
    r_u = [x_u, y_u, z_u]
    r_v = [x_v, y_v, z_v]

    return r_u, r_v

def get_num_normal(u, v, spl_x, spl_y, spl_z):

    r_u, r_v = get_num_deriv(u, v, spl_x, spl_y, spl_z)
    normal = np.cross(r_u, r_v)
    #print(":::", normal)
    normal = normal / np.linalg.norm(normal)
    #print(normal)
    return normal


def test_sphere():
    # тест сфера 
    vec_teta = np.arange(0, np.pi + np.pi/30, np.pi/30)
    vec_fi = np.arange(0, 2*np.pi + np.pi/30, 2*np.pi/60)

    matrix_f_x = np.empty([len(vec_teta)-1, len(vec_fi)-1])
    for i in range(len(matrix_f_x)):
        for j in range(len(matrix_f_x[0])):
            matrix_f_x[i, j] = np.sin(vec_teta[i]) * np.cos(vec_fi[j])
    matrix_f_x = np.r_[matrix_f_x, [matrix_f_x[0]]] # зацикливаем по строкам
    matrix_f_x = np.c_[matrix_f_x, matrix_f_x[:, 0]] # зацикливаем по столбцам

    matrix_f_y = np.empty([len(vec_teta)-1, len(vec_fi)-1])
    for i in range(len(matrix_f_y)):
        for j in range(len(matrix_f_y[0])):
            matrix_f_y[i, j] = np.sin(vec_teta[i]) * np.sin(vec_fi[j])
    matrix_f_y = np.r_[matrix_f_y, [matrix_f_y[0]]] # зацикливаем по строкам
    matrix_f_y = np.c_[matrix_f_y, matrix_f_y[:, 0]] # зацикливаем по столбцам

    matrix_f_z = np.empty([len(vec_teta), len(vec_fi)])
    for i in range(len(matrix_f_z)):
        for j in range(len(matrix_f_z[0])):
            matrix_f_z[i, j] = np.cos(vec_teta[i])

    spl_x = Bispline(vec_teta, vec_fi, matrix_f_x)
    spl_x.calculate_spline()
    spl_y = Bispline(vec_teta, vec_fi, matrix_f_y)
    spl_y.calculate_spline()
    spl_z = Bispline(vec_teta, vec_fi, matrix_f_z)
    spl_z.calculate_spline()

    u_steps = U_STEPS
    v_steps = V_STEPS
    vertexes, faces = [], []
    idx = 0
    for u in np.arange(0, np.pi, 1/5):
        for v in np.arange(0, 2*np.pi, 1/30):
            vertexes.append([spl_x.get_point(u, v), spl_y.get_point(u, v), spl_z.get_point(u, v), 1])


    return spl_x, spl_y, spl_z, vec_teta, vec_fi
    # inner_test = Object(app, vertexes, faces)
    # app.init_object(inner_test)

    # app.run()



if __name__ == '__main__':
    app = SoftwareRenderer()
    app.bind_camera(0, 0, -12)

    a = 1
    b = 1
    r = 1
    R = 6
    h = 4.5
    points, len_u, len_v = init_charge(a, b, r, R, h)
    vector_u = np.arange(0, 1 + 1/len_u, 1/len_u)
    vector_v = np.arange(0, 2*np.pi + 2*np.pi/len_v, 2*np.pi/len_v)
    matrix_f_x, matrix_f_y, matrix_f_z = points_to_matrix(points, len_u, len_v)

    spl_x = Bispline(vector_u, vector_v, matrix_f_x)
    spl_y = Bispline(vector_u, vector_v, matrix_f_y)
    spl_z = Bispline(vector_u[:-1], vector_v[:-1], matrix_f_z)

    vector_u = np.arange(0, 1, 1 / U_STEPS)
    vector_v = np.arange(0, 2*np.pi, 2*np.pi / V_STEPS)
    len_u = len(vector_u)
    len_v = len(vector_v)

    charge, points = create_object(app, spl_x, spl_y, spl_z, vector_u, vector_v)

    app.init_object(charge)
    run = True
    it = 0
    while run:
        app.draw()

        
        if it < 20:
            new_points = points_shift(points, vector_u, vector_v, spl_x, spl_y, spl_z)
            vector_u = np.arange(0, 1 + 1/len_u, 1/len_u)
            vector_v = np.arange(0, 2*np.pi + 2*np.pi/len_v, 2*np.pi/len_v)
            new_matrix_f_x, new_matrix_f_y, new_matrix_f_z = points_to_matrix(new_points, len_u, len_v)
            spl_x = Bispline(vector_u, vector_v, new_matrix_f_x)
            spl_y = Bispline(vector_u, vector_v, new_matrix_f_y)
            spl_z = Bispline(vector_u[:-1], vector_v[:-1], new_matrix_f_z)
            vector_u = np.arange(0, 1, 1 / U_STEPS)
            vector_v = np.arange(0, 2*np.pi, 2*np.pi / V_STEPS)
            len_u = len(vector_u)
            len_v = len(vector_v)
            charge, points = create_object(app, spl_x, spl_y, spl_z, vector_u, vector_v)

            app.delete_object()
            app.init_object(charge)
            it += 1

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        pg.display.set_caption(str(app.clock.get_fps()))
        pg.display.flip()
        app.clock.tick(app.FPS)