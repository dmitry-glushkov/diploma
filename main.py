from numpy.core.fromnumeric import reshape
from pygame import constants
from bicubic_spline import *
from renderer import *
from calculations import *

U_STEPS = 2
V_STEPS = 160
IT_STEP = 0.1
DELTA = 0.5
ITERATIONS = 2

def create_object(app, spl_x, spl_y, spl_z, vector_u, vector_v):
    vertexes, faces = [], []
    idx = 0
    u_steps = len(vector_u)
    v_steps = len(vector_v)
    for u in vector_u:
        for v in vector_v:
            x = spl_x.get_point(u, v)
            y = spl_y.get_point(u, v)
            try:
                z = spl_z.get_point(u, v)
                tmp_z = z
            except:
                z = tmp_z
            vertexes.append([x, y, z, 1])
            # #if v == 2*np.pi - 2*np.pi/v_steps and u != 1 - 1/u_steps:
            # if v == 1 - 1/v_steps and u != 1- 1/u_steps:
            #     faces.append([idx, idx-v_steps+1])
            #     faces.append([idx, (idx+v_steps) % (u_steps*v_steps)])
            # #elif u == 1 - 1/u_steps and v != 2*np.pi - 2*np.pi/v_steps:
            # elif u == 1 - 1/u_steps and v != 1 - 1/v_steps:
            #     faces.append([idx, idx+1])
            # #elif u == 1 - 1/u_steps and v == 2*np.pi - 2*np.pi/v_steps:
            # elif u == 1 - 1/u_steps and v == 1 - 1/v_steps:
            #     faces.append([idx, idx - v_steps + 1])
            # else:
            #     faces.append([idx, idx + 1])
            #     faces.append([idx, idx + v_steps])

            # if v == 1 - 1/v_steps and u != 1 - 1/u_steps:
            #     faces.append([idx, (idx+1-v_steps)])
            # else:
            #     faces.append([idx, (idx+1) % (u_steps*v_steps)])

            faces.append([idx, (idx+1) % (u_steps*v_steps)])
            idx += 1



    return Object(app, vertexes, faces), vertexes

def init_charge(a, b, r, R, h):
    H = 9
    points, shell_points, v_len, u_len = init_points(a, b, r, R, h)
    points = np.array(points)
    return points, u_len, v_len

def points_to_matrix(points, len_u, len_v, fl=True):
    # if fl:
    #     points = np.array(points)
    matrix_f = np.reshape(points, (len_u, len_v, 3))
    matrix_f_x = np.empty([len_u, len_v])
    matrix_f_y = np.empty([len_u, len_v])
    matrix_f_z = np.empty([len_u, len_v])
    for i in range(len_u):
        for j in range(len_v):
            #if (i != len_u and j != len_v):
            matrix_f_x[i, j] = matrix_f[i, j][0]
            matrix_f_y[i, j] = matrix_f[i, j][1]
            matrix_f_z[i, j] = matrix_f[i, j][2]

    #matrix_f_x = np.r_[matrix_f_x, [matrix_f_x[0]]] # зацикливаем по строкам
    matrix_f_x = np.c_[matrix_f_x, matrix_f_x[:, 0]] # зацикливаем по столбцам
    matrix_f_x = np.c_[matrix_f_x, matrix_f_x[:, 1]]
    #matrix_f_x = np.c_[matrix_f_x, matrix_f_x[:, 2]]
    matrix_f_x = np.c_[matrix_f_x[:, -1], matrix_f_x]
    matrix_f_x = np.c_[matrix_f_x[:, -2], matrix_f_x]
    #matrix_f_y = np.r_[matrix_f_y, [matrix_f_y[0]]]
    matrix_f_y = np.c_[matrix_f_y, matrix_f_y[:, 0]]
    matrix_f_y = np.c_[matrix_f_y, matrix_f_y[:, 1]]
    #matrix_f_y = np.c_[matrix_f_y, matrix_f_y[:, 2]]
    matrix_f_y = np.c_[matrix_f_y[:, -1], matrix_f_y]
    matrix_f_y = np.c_[matrix_f_y[:, -2], matrix_f_y]


    #matrix_f_z = np.r_[matrix_f_z, [matrix_f_z[0]]]
    matrix_f_z = np.c_[matrix_f_z, matrix_f_z[:, 0]]
    matrix_f_z = np.c_[matrix_f_z, matrix_f_z[:, 1]]
    matrix_f_z = np.c_[matrix_f_z[:, -1], matrix_f_z]
    matrix_f_z = np.c_[matrix_f_z[:, -2], matrix_f_z]

    return matrix_f_x, matrix_f_y, matrix_f_z

def points_shift(points, vector_u, vector_v, spl_x, spl_y, spl_z, R):
    new_points = []
    idx = 0
    for u in vector_u: 
        for v in vector_v[2:-2]:
            if idx < len(points):
                tmp = get_num_normal(u, v, spl_x, spl_y, spl_z, vector_u, vector_v)
                new_x = points[idx][0] - tmp[0]*IT_STEP
                new_y = points[idx][1] - tmp[1]*IT_STEP
                new_z = points[idx][2] - tmp[2]*IT_STEP
                if (np.linalg.norm([new_x, new_y]) < R):
                    new_points.append([new_x, new_y, new_z])
                else:
                    new_points.append([points[idx][0], points[idx][1], points[idx][2]])
                idx += 1

    new_points = np.array(new_points)
    return new_points

def get_num_deriv(u, v, spl_x, spl_y, spl_z, vector_u, vector_v):
    eps = 0.01

    # tmp_idx_v = np.where(vector_v == v)
    # tmp_idx_u = np.where(vector_u == u)
    # idx_v = tmp_idx_v[0][0]
    # idx_u = tmp_idx_u[0][0]

    if u == vector_u[0]:
        x_u = 2*spl_x.get_point(eps, v) - 2*spl_x.get_point(0, v) \
            - (spl_x.get_point(2*eps, v) - spl_x.get_point(0, v))/2 
        y_u = 2*spl_y.get_point(eps, v) - 2*spl_y.get_point(0, v) \
            - (spl_y.get_point(2*eps, v) - spl_y.get_point(0, v))/2
        z_u = 2*spl_z.get_point(eps, v) - 2*spl_z.get_point(0, v) \
            - (spl_z.get_point(2*eps, v) - spl_z.get_point(0, v))/2
    elif u == vector_u[-1]:
        x_u = 2*spl_x.get_point(u, v) - 2*spl_x.get_point(u-eps, v) \
            - (spl_x.get_point(u, v) - spl_x.get_point(u - 2*eps, v))/2
        y_u = 2*spl_y.get_point(u, v) - 2*spl_y.get_point(u-eps, v) \
            - (spl_y.get_point(u, v) - spl_y.get_point(u - 2*eps, v))/2
        z_u = 2*spl_z.get_point(u, v) - 2*spl_z.get_point(u-eps, v) \
            - (spl_z.get_point(u, v) - spl_z.get_point(u - 2*eps, v))/2
    else:
        # x_u = (spl_x.get_point(u+vector_u[idx_u+1], v) - spl_x.get_point(u-vector_u[idx_u-1], v)) / 2
        # y_u = (spl_y.get_point(u+vector_u[idx_u+1], v) - spl_y.get_point(u-vector_u[idx_u-1], v)) / 2
        # z_u = (spl_z.get_point(u+vector_u[idx_u+1], v) - spl_z.get_point(u-vector_u[idx_u-1], v)) / 2
        x_u = (spl_x.get_point(u+eps, v) - spl_x.get_point(u-eps, v)) / 2
        y_u = (spl_y.get_point(u+eps, v) - spl_y.get_point(u-eps, v)) / 2
        z_u = (spl_z.get_point(u+eps, v) - spl_z.get_point(u-eps, v)) / 2

    if v == vector_v[0]:
        x_v = spl_x.get_point(u, v + (vector_v[1] - vector_v[0])) - spl_x.get_point(u, vector_v[-1]) / 2
        y_v = spl_y.get_point(u, v + (vector_v[1] - vector_v[0])) - spl_y.get_point(u, vector_v[-1]) / 2
        z_v = spl_z.get_point(u, v + (vector_v[1] - vector_v[0])) - spl_z.get_point(u, vector_v[-1]) / 2
    elif v == vector_v[-1]:
        x_v = spl_x.get_point(u, vector_v[0]) - spl_x.get_point(u, v - (vector_v[-1] - vector_v[-2])) / 2
        y_v = spl_y.get_point(u, vector_v[0]) - spl_y.get_point(u, v - (vector_v[-1] - vector_v[-2])) / 2
        z_v = spl_z.get_point(u, vector_v[0]) - spl_z.get_point(u, v - (vector_v[-1] - vector_v[-2])) / 2
    else:
        # x_v = (spl_x.get_point(u, v+vector_v[idx_v+1]) - spl_x.get_point(u, v-vector_v[idx_v-1])) / 2
        # y_v = (spl_y.get_point(u, v+vector_v[idx_v+1]) - spl_y.get_point(u, v-vector_v[idx_v-1])) / 2
        # z_v = (spl_z.get_point(u, v+vector_v[idx_v+1]) - spl_z.get_point(u, v-vector_v[idx_v-1])) / 2
        x_v = (spl_x.get_point(u, v+eps) - spl_x.get_point(u, v-eps)) / 2
        y_v = (spl_y.get_point(u, v+eps) - spl_y.get_point(u, v-eps)) / 2
        z_v = (spl_z.get_point(u, v+eps) - spl_z.get_point(u, v-eps)) / 2

    r_u = [x_u, y_u, z_u]
    r_v = [x_v, y_v, z_v]
    return r_u, r_v

def get_num_normal(u, v, spl_x, spl_y, spl_z, vector_u, vector_v):
    r_u, r_v = get_num_deriv(u, v, spl_x, spl_y, spl_z, vector_u, vector_v)
    normal = np.cross(r_u, r_v)
    normal = normal / np.linalg.norm(normal)
    return normal

def insert_new_points(points, vector_u, vector_v, spl_x, spl_y, spl_z):
    new_points = []
    idx_u = 0
    idx_v = 0
    len_v = len(vector_v) 
    len_u = len(vector_u)
    new_len_v = len_v
    new_len_u = len_u

    for point_idx in range(len(points)):
        new_points.append(points[point_idx])
        idx_v_row = point_idx % len_v
        if idx_v_row == 0 and point_idx != 0:
            idx_u += 1

        if idx_v_row + 1 != len_v and point_idx + 1 != len(points):
            points_subt = points[point_idx] - points[point_idx + 1]
        else:
            points_subt = points[point_idx] - points[point_idx - (len_v-1)]
        distance = np.linalg.norm(np.array(points_subt))

        if distance > DELTA:
            first_term_new_x = spl_x.get_point(vector_u[idx_u], vector_v[idx_v_row])
            first_term_new_y = spl_y.get_point(vector_u[idx_u], vector_v[idx_v_row])
            if (idx_v_row + 1) != len_v:
                second_term_new_x = spl_x.get_point(vector_u[idx_u], vector_v[idx_v_row + 1])
                second_term_new_y = spl_y.get_point(vector_u[idx_u], vector_v[idx_v_row + 1])
            else:
                second_term_new_x = spl_x.get_point(vector_u[idx_u], vector_v[idx_v_row - (len_v-1)])
                second_term_new_y = spl_y.get_point(vector_u[idx_u], vector_v[idx_v_row - (len_v-1)])
            new_x = (first_term_new_x + second_term_new_x) / 2
            new_y = (first_term_new_y + second_term_new_y) / 2

            new_points.append([new_x, new_y, points[point_idx][2]])
            new_len_v += 1

    new_vector_u = np.arange(0, 1, 1/new_len_u)
    new_vector_v = np.arange(0, 1, 1/new_len_v)

    return new_points, new_vector_u, new_vector_v 

def distance_between_points(first_point, second_point): 
    return np.sqrt((first_point[0] - second_point[0])**2 + (first_point[1] - second_point[1])**2)

def delete_inner_points(points, len_u, len_v):
    new_points = []
    matrix_p = np.reshape(points, (len_u, len_v, 3))
    depth = 9
    for layer_idx in range(len(matrix_p)):
        layer_points_tmp = matrix_p[layer_idx]
        idx = 0
        while idx < len(layer_points_tmp) - depth - 2:
            len_to_delete = 0
            distance_to_next = distance_between_points(matrix_p[layer_idx][idx], matrix_p[layer_idx][idx+1])
            fl = False
            for j in range(idx + 2, idx+depth-1, 1):
                distance_to_j = distance_between_points(matrix_p[layer_idx][idx], matrix_p[layer_idx][j])
                distance_to_j_plus = distance_between_points(matrix_p[layer_idx][idx], matrix_p[layer_idx][j+1])
                if distance_to_next > distance_to_j and distance_to_next < distance_to_j_plus:
                    fl = True
                    len_to_delete = j - idx

            new_x = matrix_p[layer_idx][idx][0]
            new_y = matrix_p[layer_idx][idx][1]
            new_z = matrix_p[layer_idx][idx][2]
            new_points.append([new_x, new_y, new_z])

            if fl:
                idx = idx + len_to_delete + 1
            else:
                idx += 1

        for i in range(len(layer_points_tmp) - depth - 3, len(layer_points_tmp)):
            new_x = matrix_p[layer_idx][i][0]
            new_y = matrix_p[layer_idx][i][1]
            new_z = matrix_p[layer_idx][i][2]
            new_points.append([new_x, new_y, new_z])

    new_points = np.array(new_points)
    new_len_v = int((len(new_points) / len_u))
    return new_points, new_len_v

if __name__ == '__main__':
    app = SoftwareRenderer()
    app.bind_camera(0, 0, -8)
    a = 0.5
    b = 2
    r = 0.5
    R = 3
    h = 4.5
    H = 6

    control_points, len_u, len_v = init_charge(a, b, r, R, h)
    matrix_f_x, matrix_f_y, matrix_f_z = points_to_matrix(control_points, len_u, len_v)

    control_vector_u = np.arange(0, 1, 1/len_u)
    control_vector_v = np.arange(-2/len_v, 1 + 2/len_v, 1/len_v)
    draw_vector_u = np.arange(0, 1, 1/(U_STEPS))
    draw_vector_v = np.arange(0, 1, 1/(V_STEPS))

    spl_x = Bispline(control_vector_u, control_vector_v, matrix_f_x)
    spl_y = Bispline(control_vector_u, control_vector_v, matrix_f_y)
    spl_z = Bispline(control_vector_u, control_vector_v, matrix_f_z)

    charge, control_points = create_object(app, spl_x, spl_y, spl_z, draw_vector_u, draw_vector_v)
    len_control_u = len(draw_vector_u)
    len_control_v = len(draw_vector_v)
    control_vector_u = np.arange(0, 1, 1/len_control_u)
    control_vector_v = np.arange(-2/len_control_v, 1 + 2/len_control_v, 1/len_control_v)
    app.init_object(charge)
    run = True
    it = 0
    while run:
        app.draw()
        key = pg.key.get_pressed()
        #if it < ITERATIONS:
        if key[pg.K_SPACE]:
            it += 1

            if it == 8:
                print(it)

            control_points = points_shift(control_points, control_vector_u, control_vector_v, spl_x, spl_y, spl_z, R)
            if it <= 40:
                control_points, len_control_v = delete_inner_points(control_points, len_control_u, len_control_v)
            matrix_f_x, matrix_f_y, matrix_f_z = points_to_matrix(control_points, len_control_u, len_control_v)

            control_vector_u = np.arange(0, 1, 1/len_control_u)
            control_vector_v = np.arange(-2/len_control_v, 1 + 2/len_control_v, 1/len_control_v)

            spl_x = Bispline(control_vector_u, control_vector_v, matrix_f_x)
            spl_y = Bispline(control_vector_u, control_vector_v, matrix_f_y)
            spl_z = Bispline(control_vector_u, control_vector_v, matrix_f_z)

            # draw_vector_u = np.arange(0, 1, 1/4)
            # draw_vector_v = np.arange(0, 1, 1/512)

            charge, _ = create_object(app, spl_x, spl_y, spl_z, draw_vector_u, draw_vector_v)

            app.delete_object()
            app.init_object(charge)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        pg.display.set_caption(str(str(app.clock.get_fps()) + "   it: " + str(it)))
        pg.display.flip()
        app.clock.tick(app.FPS)