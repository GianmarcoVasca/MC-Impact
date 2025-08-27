import numpy as np
from funzioni.funzioni_ausiliarie import deg2rad
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# === CICLOIDE ANALITICA ===
def ruota_attorno_a(x, y, angolo_rad, A):
    cos_a, sin_a = np.cos(angolo_rad), np.sin(angolo_rad)
    x_shift = x - A[0]
    y_shift = y - A[1]
    x_rot = cos_a * x_shift - sin_a * y_shift + A[0]
    y_rot = sin_a * x_shift + cos_a * y_shift + A[1]
    return x_rot, y_rot

def curva_cicloide(theta, R):
    t = np.linspace(0, theta, 20000)
    x = R * (t - np.sin(t))
    y = -R * (1 - np.cos(t))
    return x, y

def curva_cicloide_inversa(theta, R):
    t = np.linspace(0, theta, 20000)
    x = -R * (t - np.sin(t))
    y = -R * (1 - np.cos(t))
    # Rotazione di 180° oraria = -pi attorno all'origine
    x, y = ruota_attorno_a(x, y, -np.pi, (0, 0))
    return x, y

def calcola_rotazione_e_tipo(A, B, angolo_tang_deg):
    errore_ = 0

    # Angolo retta AB
    dx = B[0] - A[0]
    angolo_AB = np.rad2deg(np.arctan2(B[1] - A[1], dx)) % 360

    # Normalizza angolo tangente
    angolo_tang_norm = angolo_tang_deg % 360

    # Intervallo circolare tra angolo_AB e angolo_AB + 180
    limite_inf = angolo_AB
    limite_sup = (angolo_AB + 180) % 360

    def in_intervallo(a, inf, sup):
        if inf <= sup:
            return inf <= a <= sup
        else:
            return a >= inf or a <= sup

    usa_cicloide_inv = in_intervallo(angolo_tang_norm, limite_inf, limite_sup)

    # Selezione curva e angolo rotazione
    if not usa_cicloide_inv:
        base_curve = curva_cicloide
        angolo_rot_deg = (angolo_tang_deg + 90)
        if angolo_rot_deg < angolo_AB:
            errore_ = 1
    elif usa_cicloide_inv:
        base_curve = curva_cicloide_inversa
        angolo_rot_deg = (angolo_tang_deg - 90)
        if angolo_rot_deg > angolo_AB:
            errore_ = 1

    angolo_rot_rad = np.deg2rad(angolo_rot_deg)

    return angolo_rot_rad, base_curve, errore_

def calcolo_cicloide(A, B, angolo_tang, theta_target=np.pi * 2):

    angolo_rot_rad, base_curve, errore_ = calcola_rotazione_e_tipo(A, B, angolo_tang)

    if errore_ == 0:
        # Ottimizza R per far intercettare B
        def errore_R(R):
            x, y = base_curve(theta_target, R)
            x += A[0]
            y += A[1]
            x, y = ruota_attorno_a(x, y, angolo_rot_rad, A)
            distanze = np.sqrt((x - B[0])**2 + (y - B[1])**2)
            return np.min(distanze)

        R_opt = minimize_scalar(
            errore_R,
            bounds=(0.01, 100),
            method='bounded',
            options={'xatol': 1e-12}
        ).x

        def distanza_da_B(theta):
            x, y = base_curve(theta, R_opt)
            x, y = x[-1] + A[0], y[-1] + A[1]
            x, y = ruota_attorno_a(np.array([x]), np.array([y]), angolo_rot_rad, A)
            return np.linalg.norm(np.array([x[0], y[0]]) - B)

        theta_finale = minimize_scalar(distanza_da_B, bounds=(0.1, 2*np.pi), method='bounded', options={'xatol': 1e-12}).x

        # Genera curva finale
        x, y = base_curve(theta_finale, R_opt)
        x += A[0]
        y += A[1]
        x, y = ruota_attorno_a(x, y, angolo_rot_rad, A)

        # === Calcolo lunghezza arco ===
        def lunghezza_cicloide(theta, R):
            if theta <= 0 or R <= 0:
                return 0  # oppure solleva errore
            integranda = lambda t: np.sqrt(1 - np.cos(t))
            valore_integrale, _ = quad(integranda, 0, theta)
            return R * np.sqrt(2) * valore_integrale
        
        lunghezza = lunghezza_cicloide(theta_finale, R_opt)
        if lunghezza == 0:
            errore_ = 1

        soglia_theta = 0.5
        soglia_R = 99
        if theta_finale < soglia_theta and R_opt > soglia_R:
            lunghezza = 0
            x = y = None
            theta_finale = R_opt = None
            errore_ = 1

    elif errore_ == 1:
        lunghezza = 0
        x = y = None
        theta_finale = R_opt = None

    return lunghezza, errore_, x, y, theta_finale, R_opt

# === CICLOIDE SEMPLICE + RICHIAMO CICLOIDE AVANZATA===
def cicloide(veicolo, cicloide_avanzata_flag, theta_out_cic, cicloide_nota_flag):
    # Estrazione parametri
    if cicloide_avanzata_flag == 1:
        x = veicolo.x
        y = veicolo.y
        x_quiete = veicolo.x_quiete
        y_quiete = veicolo.y_quiete
    m = veicolo.m
    l = veicolo.l
    p = veicolo.p
    theta_post = deg2rad(veicolo.theta_post)
    d_post = veicolo.d_post
    f = veicolo.f
        
    # Costanti
    g = 9.81
    k = 0.1

    # Momenti d'inerzia
    J = 0.1478 * m * l * p

    # inizializzazione:
    errore_cicloide = 0
    punti_cicloide_x = 0
    punti_cicloide_y = 0
    cicloide_th = 0
    cicloide_R = 0
    cic_A = 0
    cic_B = 0

    # Lunghezze cicloide
    if cicloide_avanzata_flag == 0:
        lung_cicloide = d_post + k * theta_post * p / 2
    elif cicloide_avanzata_flag == 1:
        cic_A = np.array([x, y])
        cic_B = np.array([x_quiete, y_quiete])
        lung_cicloide, errore_cicloide, punti_cicloide_x, punti_cicloide_y, cicloide_th, cicloide_R = calcolo_cicloide(cic_A, cic_B, theta_out_cic)
        d_post = np.hypot(x_quiete - x, y_quiete - y)

    # cicloide nota
    if cicloide_nota_flag ==1:
        lung_cicloide = veicolo.lunghezza_cicloide

    # Velocità lineari post-urto
    V_post = np.sqrt((2 * m * g * f * lung_cicloide) / (m + J * theta_post**2 / d_post**2))
    V_post_Kmh = V_post * 3.6

    # Velocità angolari post-urto
    omega_post = (V_post * theta_post) / d_post



    return J, lung_cicloide, d_post, V_post_Kmh, omega_post, errore_cicloide, punti_cicloide_x, punti_cicloide_y, cicloide_th, cicloide_R, cic_A, cic_B

def calcolo_Ed_EES (veicolo):

    EES = veicolo.EES_Kmh / 3.6
    m = veicolo.m
    # Energia di deformazione attesa
    Ed = 0.5 * m * EES**2

    return Ed

