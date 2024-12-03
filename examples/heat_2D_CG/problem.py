from multipinn.condition import Condition, ConditionExtra
from multipinn.condition.diff import grad, num_diff_random, unpack
from multipinn.geometry import *


def problem():
    input_dim = 3
    output_dim = 1

    C = 1

    R_L = 1
    Q_L_CIRCLE = 1
    G_L_CIRCLE = 5

    R_S = 0.4
    Q_S_CIRCLE = 1
    G_S_CIRCLE = 1

    Q_BOUNDARY = 1
    G_BOUNDARY = 0.1

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, y, t = unpack(arg)
        return f, u, x, y, t

    def inner(model, arg):
        f, u, x, y, t = basic_symbols(model, arg)
        u_x, u_y, u_t = unpack(grad(u, arg))
        u_xx, u_xy, u_xt = unpack(grad(u_x, arg))
        u_yx, u_yy, u_yt = unpack(grad(u_y, arg))
        eq1 = u_xx + u_yy
        return [u_t - eq1]

    def rect_bnd_f(model, arg, data):
        f, u, x, y, t = basic_symbols(model, arg)
        normal = data[0]
        (u_n,) = unpack(num_diff_random(model, arg, f, normal))

        return [-C * u_n - G_BOUNDARY + Q_BOUNDARY * u]

    def small_circle_bnd(model, arg, data):
        f, u, x, y, t = basic_symbols(model, arg)
        normal = data[0]
        (u_n,) = unpack(num_diff_random(model, arg, f, normal))

        return [-C * u_n - G_S_CIRCLE + Q_S_CIRCLE * u]

    def large_circle_bnd(model, arg, data):
        f, u, x, y, t = basic_symbols(model, arg)
        normal = data[0]
        (u_n,) = unpack(num_diff_random(model, arg, f, normal))

        return [-C * u_n - G_L_CIRCLE + Q_L_CIRCLE * u]

    def init(model, arg):
        f, u, x, y, t = basic_symbols(model, arg)
        return [u]

    l1 = Hypersphere([4, 3], R_L)
    l2 = Hypersphere([-4, -3], R_L)
    l3 = Hypersphere([-4, 3], R_L)
    l4 = Hypersphere([4, -3], R_L)
    l5 = Hypersphere([4, 9], R_L)
    l6 = Hypersphere([-4, -9], R_L)
    l7 = Hypersphere([4, -9], R_L)
    l8 = Hypersphere([-4, 9], R_L)
    l9 = Hypersphere([0, 0], R_L)
    l10 = Hypersphere([0, 6], R_L)
    l11 = Hypersphere([0, -6], R_L)

    s1 = Hypersphere([3.2, 6], R_S)
    s2 = Hypersphere([-3.2, -6], R_S)
    s3 = Hypersphere([3.2, -6], R_S)
    s4 = Hypersphere([-3.2, 6], R_S)
    s5 = Hypersphere([3.2, 0], R_S)
    s6 = Hypersphere([-3.2, 0], R_S)

    rect = Hypercube([-8, -12], [8, 12])

    large_circles = l1 | l2 | l3 | l4 | l5 | l6 | l7 | l8 | l9 | l10 | l11
    small_circles = s1 | s2 | s3 | s4 | s5 | s6

    large_circles_with_t = large_circles * Hypercube([0], [3])
    small_circles_with_t = small_circles * Hypercube([0], [3])

    rect_with_t = rect * Hypercube([0], [3])

    large_circles_with_0 = large_circles * Hypercube([0], [0])
    small_circles_with_0 = small_circles * Hypercube([0], [0])

    rect_with_0 = rect * Hypercube([0], [0])
    initial = rect_with_0 - (large_circles_with_0 | small_circles_with_0)

    domain_with_t = rect_with_t - (large_circles_with_t | small_circles_with_t)

    bottom = Hypercube([-8, -12, 0], [8, 12, 0])
    top = Hypercube([-8, -12, 3], [8, 12, 3])

    rect_bnd = Shell(rect_with_t) - (bottom | top)

    s_circles_shell = Shell(small_circles_with_t) - (bottom | top)
    l_ciclers_shell = Shell(large_circles_with_t) - (bottom | top)

    pde = [
        Condition(inner, domain_with_t),
        Condition(init, initial),
        ConditionExtra(rect_bnd_f, rect_bnd, ["normals"]),
        ConditionExtra(small_circle_bnd, s_circles_shell, ["normals"]),
        ConditionExtra(large_circle_bnd, l_ciclers_shell, ["normals"]),
    ]
    return pde, input_dim, output_dim
