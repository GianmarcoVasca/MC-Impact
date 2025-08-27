import numpy as np
from types import SimpleNamespace
from funzioni.funzioni_ausiliarie import deg2rad, rad2deg, calcola_errore, calcola_errore_angolare

# modello a 3gdl
def modello_3gdl(params, targets):
    errore_logico = 0
    V1_post = params.V1_post_Kmh / 3.6
    V2_post = params.V2_post_Kmh / 3.6

    th1_in_rad = deg2rad(params.theta1_in)
    th1_out_rad = deg2rad(params.theta1_out)
    th2_in_rad = deg2rad(params.theta2_in)
    th2_out_rad = deg2rad(params.theta2_out)

    V1x = V1_post * np.cos(th1_out_rad)
    V1y = V1_post * np.sin(th1_out_rad)
    V2x = V2_post * np.cos(th2_out_rad)
    V2y = V2_post * np.sin(th2_out_rad)

    A = -params.m1 * (V1y - np.tan(th1_in_rad) * V1x)
    B = -params.m2 * (V2y - np.tan(th2_in_rad) * V2x)

    mu = (np.tan(th2_in_rad) * A + np.tan(th1_in_rad) * B) / (A + B)
    In = (A + B) / (np.tan(th1_in_rad) - np.tan(th2_in_rad))

    V1x_pre = V1x - In / params.m1
    V1y_pre = V1y - mu * In / params.m1
    V2x_pre = V2x + In / params.m2
    V2y_pre = V2y + mu * In / params.m2

    V1_post_mod = np.linalg.norm([V1x, V1y])
    V2_post_mod = np.linalg.norm([V2x, V2y])
    V1_pre_mod = np.linalg.norm([V1x_pre, V1y_pre])
    V2_pre_mod = np.linalg.norm([V2x_pre, V2y_pre])

    omega1_pre = params.omega1_post - (In / params.J1) * params.y1 + (mu * In / params.J1) * params.x1
    omega2_pre = params.omega2_post + (In / params.J2) * params.y2 - (mu * In / params.J2) * params.x2

    Ec_pre = 0.5 * params.m1 * V1_pre_mod**2 + 0.5 * params.m2 * V2_pre_mod**2 + \
             0.5 * (params.J1 * omega1_pre**2 + params.J2 * omega2_pre**2)
    Ec_post = 0.5 * params.m1 * V1_post_mod**2 + 0.5 * params.m2 * V2_post_mod**2 + \
              0.5 * (params.J1 * params.omega1_post**2 + params.J2 * params.omega2_post**2)

    Ed = Ec_pre - Ec_post
    err_Ed = abs(Ed - targets.Ed_target) / 1000

    params.theta1_in %= 360
    params.theta1_out %= 360
    params.theta2_out %= 360
    params.theta2_in %= 360
    err_theta1_in = calcola_errore_angolare(params, targets, 'theta1_in')
    err_theta1_out = calcola_errore_angolare(params, targets, 'theta1_out')
    err_theta2_in = calcola_errore_angolare(params, targets, 'theta2_in')
    err_theta2_out = calcola_errore_angolare(params, targets, 'theta2_out')

    It = mu * In

    err_x1 = calcola_errore(params, targets, 'x1')
    err_y1 = calcola_errore(params, targets, 'y1')
    err_x2 = calcola_errore(params, targets, 'x2')
    err_y2 = calcola_errore(params, targets, 'y2')

    err_omega1_pre = calcola_errore(SimpleNamespace(omega1_pre=omega1_pre), targets, 'omega1_pre')
    err_omega2_pre = calcola_errore(SimpleNamespace(omega2_pre=omega2_pre), targets, 'omega2_pre')

    if V1_pre_mod < 0 or V2_pre_mod < 0:
        errore_logico = 1.1

    results = {
        'x1': params.x1, 'y1': params.y1, 'x2': params.x2, 'y2': params.y2,
        'theta1_in': params.theta1_in, 'theta1_out': params.theta1_out,
        'theta2_in': params.theta2_in, 'theta2_out': params.theta2_out,
        'V1_post': V1_post_mod * 3.6, 'V2_post': V2_post_mod * 3.6,
        'V1_pre': V1_pre_mod * 3.6, 'V2_pre': V2_pre_mod * 3.6,
        'omega1_post': params.omega1_post, 'omega2_post': params.omega2_post,
        'omega1_pre': omega1_pre, 'omega2_pre': omega2_pre,
        'Ed': Ed, 'In': In, 'It': It, 'mu': mu
    }

    errors = {
        'err_x1': err_x1, 'err_y1': err_y1, 'err_x2': err_x2, 'err_y2': err_y2,
        'err_theta1_in': err_theta1_in, 'err_theta1_out': err_theta1_out,
        'err_theta2_in': err_theta2_in, 'err_theta2_out': err_theta2_out,
        'err_omega1_pre': err_omega1_pre, 'err_omega2_pre': err_omega2_pre,
        'errore_Ed_[KJ]': err_Ed, 'errore_logico': errore_logico
    }

    return results, errors


# modello a 2gdl
def modello_2gdl(params, targets):
    errore_logico = 0
    V1_post = params.V1_post_Kmh / 3.6
    V2_post = params.V2_post_Kmh / 3.6

    V1_post_mod = V1_post
    V2_post_mod = V2_post

    th1_in_rad = deg2rad(params.theta1_in)
    th1_out_rad = deg2rad(params.theta1_out)
    th2_in_rad = deg2rad(params.theta2_in)
    th2_out_rad = deg2rad(params.theta2_out)

    m1 = params.m1
    m2 = params.m2

    a = m1 * np.cos(th1_in_rad)
    b = m2 * np.cos(th2_in_rad)
    c = m1 * np.cos(th1_out_rad)
    d = m2 * np.cos(th2_out_rad)
    e = m1 * np.sin(th1_in_rad)
    f = m2 * np.sin(th2_in_rad)
    g = m1 * np.sin(th1_out_rad)
    h = m2 * np.sin(th2_out_rad)

    # Numeratore e denominatore V2 (eq. 4.96)
    denom = f - (e * b / a)
    if abs(denom) < 1e-6:
        return  # evita divisione per zero
    V2_pre_mod = (1 / denom) * (V2_post * (h - (e * d / a)) + V1_post * (g - (e * c / a)))
    V1_pre_mod = (1 / a) * (d * V2_post + c * V1_post - b * V2_pre_mod)

    omega1_pre = 0
    omega2_pre = 0

    Ec_pre = 0.5 * params.m1 * V1_pre_mod**2 + 0.5 * params.m2 * V2_pre_mod**2 + \
             0.5 * (params.J1 * omega1_pre**2 + params.J2 * omega2_pre**2)
    Ec_post = 0.5 * params.m1 * V1_post_mod**2 + 0.5 * params.m2 * V2_post_mod**2 + \
              0.5 * (params.J1 * params.omega1_post**2 + params.J2 * params.omega2_post**2)

    Ed = Ec_pre - Ec_post
    err_Ed = abs(Ed - targets.Ed_target) / 1000

    params.theta1_in %= 360
    params.theta1_out %= 360
    params.theta2_out %= 360
    params.theta2_in %= 360
    err_theta1_in = calcola_errore_angolare(params, targets, 'theta1_in')
    err_theta1_out = calcola_errore_angolare(params, targets, 'theta1_out')
    err_theta2_in = calcola_errore_angolare(params, targets, 'theta2_in')
    err_theta2_out = calcola_errore_angolare(params, targets, 'theta2_out')

    if V1_pre_mod < 0 or V2_pre_mod < 0:
        errore_logico = 1.1

    results = {
        'x1': params.x1, 'y1': params.y1, 'x2': params.x2, 'y2': params.y2,
        'theta1_in': params.theta1_in, 'theta1_out': params.theta1_out,
        'theta2_in': params.theta2_in, 'theta2_out': params.theta2_out,
        'V1_post': V1_post_mod * 3.6, 'V2_post': V2_post_mod * 3.6,
        'V1_pre': V1_pre_mod * 3.6, 'V2_pre': V2_pre_mod * 3.6,
        'omega1_post': params.omega1_post, 'omega2_post': params.omega2_post,
        'omega1_pre': omega1_pre, 'omega2_pre': omega2_pre,
        'Ed': Ed
    }

    errors = {
        'err_theta1_in': err_theta1_in, 'err_theta1_out': err_theta1_out,
        'err_theta2_in': err_theta2_in, 'err_theta2_out': err_theta2_out,
        'errore_Ed_[KJ]': err_Ed, 'errore_logico': errore_logico
    }

    return results, errors
