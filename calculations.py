import numpy as np
from vector4f import *

def TDMA_solver(above_diagonal, main_diagonal, below_diagonal, vector_f):  
    alpha = [0]
    beta = [0]
    n = len(vector_f)
    x = [0] * n

    for i in range(n-1):
        alpha.append(-below_diagonal[i]/(above_diagonal[i]*alpha[i] + main_diagonal[i]))
        beta.append((vector_f[i] - above_diagonal[i]*beta[i])/(above_diagonal[i]*alpha[i] + main_diagonal[i]))
            
    x[n-1] = (vector_f[n-1] - above_diagonal[n-2]*beta[n-1])/(main_diagonal[n-1] + above_diagonal[n-2]*alpha[n-1])

    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
    
    return x

def init_points(a, b, r, R, height):
    points = []
    shell_points = []

    type_count = 16 # всего различных сторон фигуры
    points_per_border = 5 # точек на одно ребро
    points_per_layer = points_per_border*type_count

    height_step = height/10
    layers_count = int(height/height_step + 1)

    angle_step = 2*np.pi / points_per_layer
    alpha_angle = np.arcsin(a / (2*R))
    beta_angle = np.arcsin(a / (2*r))
    angle = -alpha_angle # начальный угол
    phi_angle = np.pi/2 - 2*beta_angle

    idx = 0
    shell_idx = 0
    count_i = 0
    #count_j = 0
    for h in np.arange(0, height + height_step, height_step):
        #1
        for i in range(points_per_border):
            point = Vector4f(
                r*np.cos(beta_angle) + b,
                -a/2 + a * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #2    
        for i in range(points_per_border):
            point = Vector4f(
                r*np.cos(beta_angle) + b - b * i/points_per_border,
                a/2,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #3
        for i in range(points_per_border):
            point = Vector4f(
                r*np.cos(beta_angle + phi_angle * i/points_per_border), 
                r*np.sin(beta_angle + phi_angle * i/points_per_border), 
                h, 
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #4
        for i in range(points_per_border):
            point = Vector4f(
                a/2,
                r*np.cos(beta_angle) + b * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #5
        for i in range(points_per_border):
            point = Vector4f(
                a/2 - a * i/points_per_border,
                r*np.cos(beta_angle) + b,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #6
        for i in range(points_per_border):
            point = Vector4f(
                -a/2,
                r*np.cos(beta_angle) + b - b * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #7
        for i in range(points_per_border):
            point = Vector4f(
                -r*np.sin(beta_angle + phi_angle * i/points_per_border),
                r*np.cos(beta_angle + phi_angle * i/points_per_border),
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #8
        for i in range(points_per_border):
            point = Vector4f(
                -r*np.cos(beta_angle) - b * i/points_per_border,
                a/2,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #9
        for i in range(points_per_border):
            point = Vector4f(
                -r*np.cos(beta_angle) - b,
                a/2 - a * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #10
        for i in range(points_per_border):
            point = Vector4f(
                -r*np.cos(beta_angle) - b + b * i/points_per_border,
                -a/2,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #11
        for i in range(points_per_border):
            point = Vector4f(
                -r*np.cos(beta_angle + phi_angle * i/points_per_border),
                -r*np.sin(beta_angle + phi_angle * i/points_per_border),
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #12
        for i in range(points_per_border):
            point = Vector4f(
                -a/2,
                -r*np.cos(beta_angle) - b * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #13
        for i in range(points_per_border):
            point = Vector4f(
                -a/2 + a * i/points_per_border,
                -r*np.cos(beta_angle) - b,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #14
        for i in range(points_per_border):
            point = Vector4f(
                a/2,
                -r*np.cos(beta_angle) - b + b * i/points_per_border,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #15
        for i in range(points_per_border):
            point = Vector4f(
                r*np.sin(beta_angle + phi_angle * i/points_per_border),
                -r*np.cos(beta_angle + phi_angle * i/points_per_border),
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        #16
        for i in range(points_per_border):
            point = Vector4f(
                r*np.cos(beta_angle) + b * i/points_per_border,
                -a/2,
                h,
                idx)
            points.append(point)
            idx += 1
            angle += angle_step
            count_i += 1

        for angle in np.arange(0, 2*np.pi + angle_step, angle_step):
            point = Vector4f(R*np.cos(angle), R*np.sin(angle), h, shell_idx)
            shell_points.append(point)
            shell_idx += 1
    return points, shell_points, points_per_layer, layers_count
