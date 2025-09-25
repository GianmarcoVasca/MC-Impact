import numpy as np
from scipy.optimize import fminbound

# Funzioni ausiliarie
def deg2rad(deg): return np.deg2rad(deg)
def rad2deg(rad): return np.rad2deg(rad)

def calcola_errore(valori, targets, key):
    val1 = getattr(valori, key)
    val2 = getattr(targets, key)
    # Se entrambi sono prossimi a zero, errore nullo
    if abs(val1) < 0.01 and abs(val2) < 0.01:
        return 0.0
    # Altrimenti errore percentuale
    return abs(val1 - val2) / max(abs(val1), abs(val2), np.finfo(float).eps) * 100

def calcola_errore_angolare(set_val, targets, key):
    ang1 = getattr(set_val, key) % 360
    ang2 = getattr(targets, key) % 360
    # Se entrambi sono prossimi a 0, errore nullo
    if abs(ang1) < 0.01 and abs(ang2) < 0.01:
        return 0.0
    # Calcolo dell'errore angolare in %
    delta = abs(ang1 - ang2) % 360
    errore = min(delta, 360 - delta) / 180 * 100
    return errore

def assegna_parametro(params, nome, valore):
    # esempio: assegna_parametro(params, 'm2', 200)
    if isinstance(params, dict):
        params[nome] = valore
    elif hasattr(params, '__dict__'):
        setattr(params, nome, valore)
    else:
        raise TypeError("Il tipo di 'params' non è supportato.")
    
def distanza_punto_retta(theta_deg, veicolo):
    """
    Calcola la distanza del punto (x0, y0) dalla retta passante per l'origine con inclinazione theta (in radianti).
    
    Parametri:
        theta_rad : inclinazione della retta in radianti
        x0, y0 : coordinate del punto
    
    Ritorna:
        distanza : float
    """
    x0 = veicolo.x
    y0 = veicolo.y
    theta_rad = np.deg2rad(theta_deg)  # converte in radianti
    return np.abs(np.sin(theta_rad) * x0 - np.cos(theta_rad) * y0)

def differenza_vett(V1, ang1, V2, ang2):
    """
    Calcola la differenza vettoriale tra V2 e V1.

    Parametri:
        V1, ang1 : modulo e angolo (in gradi) del primo vettore [km/h, °]
        V2, ang2 : modulo e angolo (in gradi) del secondo vettore [km/h, °]

    Ritorna:
        V3_mod : modulo del vettore risultante [km/h]
        V3_ang : angolo del vettore risultante [°]
    """

    # Conversione in radianti
    a1 = np.deg2rad(ang1)
    a2 = np.deg2rad(ang2)

    # Componenti cartesiane
    Vx1 = V1 * np.cos(a1)
    Vy1 = V1 * np.sin(a1)
    Vx2 = V2 * np.cos(a2)
    Vy2 = V2 * np.sin(a2)

    # Differenza vettoriale
    Vx3 = Vx2 - Vx1
    Vy3 = Vy2 - Vy1

    # Modulo e angolo del vettore risultante
    V3_mod = np.sqrt(Vx3**2 + Vy3**2)
    V3_ang = np.mod(np.rad2deg(np.arctan2(Vy3, Vx3)), 360)

    return V3_mod, V3_ang

def deltaV(veicolo_1, veicolo_2, Ed, epsilon):
    """
    Calcola le variazioni di velocità (ΔV) lungo la direzione dell'impulso,
    usando l’energia di deformazione, i momenti di inerzia, le distanze dalla PDOF
    e il coefficiente di restituzione.

    Parametri:
        Ed : energia di deformazione totale [J]
        m1, m2 : masse dei veicoli [kg]
        J1, J2 : momenti di inerzia dei veicoli
        h1, h2 : distanze baricentro - PDOF [m]
        epsilon : coefficiente di restituzione [-]

    Ritorna:
        deltaV1, deltaV2 : variazioni di velocità [km/h]
    """
    # dichiarazione variabili
    m1 = veicolo_1.m
    m2 = veicolo_2.m
    J1 = veicolo_1.J
    J2 = veicolo_2.J
    h1 = veicolo_1.h
    h2 = veicolo_2.h

    # Raggi di inerzia
    k1 = np.sqrt(J1 / m1)
    k2 = np.sqrt(J2 / m2)

    # Fattori di riduzione della massa
    gamma1 = k1**2 / (k1**2 + h1**2)
    gamma2 = k2**2 / (k2**2 + h2**2)

    # Denominatore comune
    denom = m2 / gamma1 + m1 / gamma2

    # ΔV1 con coeff. restituzione (formula 4.157)
    deltaV1 = np.sqrt((2 * Ed * m2 * (1 + epsilon)) / (m1 * denom * (1 - epsilon)))
    deltaV2 = np.sqrt((2 * Ed * m1 * (1 + epsilon)) / (m2 * denom * (1 - epsilon)))

    # Conversione in km/h
    deltaV1 *= 3.6
    deltaV2 *= 3.6

    return deltaV1, deltaV2

def stima_epsilon(veicolo_1, veicolo_2, Ed, deltaV1_target, deltaV2_target):
    """
    Trova il valore ottimale di epsilon che minimizza l'errore rispetto ai deltaV target.
    """

    def somma_errori(epsilon):
        try:
            deltaV1_e, deltaV2_e = deltaV(veicolo_1, veicolo_2, Ed, epsilon)
            errore = abs(deltaV1_e - deltaV1_target) + abs(deltaV2_e - deltaV2_target)
            if np.isnan(errore) or np.isinf(errore):
                return 1e6
            return errore
        except:
            return 1e6

    # Minimizzazione nell'intervallo [0, 0.999]
    epsilon = fminbound(somma_errori, 0, 0.999)

    # Correzione ai limiti
    if epsilon < 0.01:
        epsilon = 0.01
    elif epsilon > 0.99:
        epsilon = 0.99

    return epsilon

def ottimizza_deltaV_grafico(veicolo , deltaV_target):
    """
    Ottimizza i valori di V_post e theta_in/out per far combaciare ΔV grafico con ΔV energetico.

    Parametri:
        V_pre         — velocità pre-urto [km/h]
        V_post        — velocità post-urto [km/h]
        theta_in      — angolo ingresso [°]
        theta_out     — angolo uscita [°]
        deltaV_target — valore ΔV energetico [km/h]

    Ritorna:
        best_set — [V_post_mod, θ_in, θ_out, ΔV_graf, errore]
        min_err  — errore minimo trovato [km/h]
    """

    V_pre = veicolo.V_pre_Kmh
    V_post = veicolo.V_post_Kmh
    theta_in = veicolo.theta_in
    theta_out= veicolo.theta_out

    soglia_errore = 0.5  # km/h
    V_range = np.arange(V_post - 5, V_post + 6, 1)
    theta_in_range = np.arange(theta_in - 3, theta_in + 4, 1)
    theta_out_range = np.arange(theta_out - 3, theta_out + 4, 1)

    min_err = np.inf
    best_set = [np.nan] * 5

    for Vp in V_range:
        for tin in theta_in_range:
            for tout in theta_out_range:
                dV_mod, _ = differenza_vett(V_pre, tin, Vp, tout)
                err = np.abs(dV_mod - deltaV_target)
                if err < min_err:
                    min_err = err
                    best_set = [Vp, tin, tout, dV_mod, err]

    if min_err > soglia_errore:
        best_set = [np.nan] * 5
        min_err = np.nan

    return best_set, min_err


