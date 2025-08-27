import numpy as np
from types import SimpleNamespace
from funzioni.funzioni_ausiliarie import ottimizza_deltaV_grafico, deltaV, stima_epsilon, differenza_vett, calcola_errore_angolare, assegna_parametro, distanza_punto_retta
from funzioni.modelli_impulsivi import modello_3gdl, modello_2gdl
from funzioni.operazioni_predefinite import cicloide, calcolo_Ed_EES


def calcoli(params, targets):

    results = {}
    errors = {}

    # creazione veicoli
    veicolo_1 = SimpleNamespace(
        x = params.x1,
        y = params.y1,
        x_quiete = params.x1_quiete,
        y_quiete = params.y1_quiete,
        m = params.m1,
        l = params.l1,
        p = params.p1,
        d_post = params.d_post1,
        f = params.f1,
        theta_post = params.theta_post1,
        EES_Kmh = params.EES1_Kmh,
        lunghezza_cicloide = params.lunghezza_cicloide_1
    )

    veicolo_2 = SimpleNamespace(
        x = params.x2,
        y = params.y2,
        x_quiete = params.x2_quiete,
        y_quiete = params.y2_quiete,
        m = params.m2,
        l = params.l2,
        p = params.p2,
        d_post = params.d_post2,
        f = params.f2,
        theta_post = params.theta_post2,
        EES_Kmh = params.EES2_Kmh,
        lunghezza_cicloide = params.lunghezza_cicloide_2
    )

    # post-urto
    cicloide_flag = params.cicloide
    cicloide_avanzata_flag = params.cicloide_avanzata
    cicloide_nota_flag = params.cicloide_nota
    if cicloide_flag == 1:
        theta1_out_cic = params.theta1_out
        J1, cicloide1, d_post1, V1_post_Kmh, omega1_post, errore_cicloide1, punti_cicloide_x1, punti_cicloide_y1, cicloide_th1, cicloide_R1, cic_A1, cic_B1 = cicloide(veicolo_1, cicloide_avanzata_flag, theta1_out_cic, cicloide_nota_flag)
        assegna_parametro(params, 'J1', J1)
        assegna_parametro(params, 'cicloide1', cicloide1)
        assegna_parametro(params, 'V1_post_Kmh', V1_post_Kmh)
        assegna_parametro(params, 'omega1_post', omega1_post)
        theta2_out_cic = params.theta2_out
        J2, cicloide2, d_post2, V2_post_Kmh, omega2_post, errore_cicloide2, punti_cicloide_x2, punti_cicloide_y2, cicloide_th2, cicloide_R2, cic_A2, cic_B2 = cicloide(veicolo_2, cicloide_avanzata_flag, theta2_out_cic, cicloide_nota_flag)
        assegna_parametro(params, 'J2', J2)
        assegna_parametro(params, 'cicloide2', cicloide2)
        assegna_parametro(params, 'V2_post_Kmh', V2_post_Kmh)
        assegna_parametro(params, 'omega2_post', omega2_post)
    elif cicloide_flag == 0:
        J1 = 0.1478 * params.m1 * params.l1 * params.p1
        J2 = 0.1478 * params.m2 * params.l2 * params.p2
        assegna_parametro(params, 'J1', J1)
        assegna_parametro(params, 'J2', J2)


    # calcolo energia target:
    energia_EES = params.energia_EES
    if energia_EES == 1:
        Ed1 = calcolo_Ed_EES(veicolo_1)
        Ed2 = calcolo_Ed_EES(veicolo_2)
        Ed_target = Ed1 + Ed2
        assegna_parametro(targets, 'Ed_target', Ed_target)
    elif energia_EES == 0:
        Ed_target = params.Ed_target
        assegna_parametro(targets, 'Ed_target', Ed_target)

    # modello impulsivo urto
    gdl = params.gdl
    if gdl == 2:
        results, errors = modello_2gdl(params, targets)
    elif gdl == 3:
        results, errors = modello_3gdl(params, targets)
    else:
         raise ValueError(f"Modello con {gdl} gradi di libertà non supportato.")

    # calcolo direzione stimata PDOF dai vettori deltaV
    stima_PDOF = params.stima_PDOF
    if stima_PDOF == 1:
        delta1, PDOF1 = differenza_vett(results['V1_pre'], results['theta1_in'], results['V1_post'], results['theta1_out'])
        delta2, PDOF2 = differenza_vett(results['V2_pre'], results['theta2_in'], results['V2_post'], results['theta2_out'])
        PDOF_stima = np.degrees(np.arctan2(
            np.sin(np.radians(PDOF1 + 180)) + np.sin(np.radians(PDOF2)),
            np.cos(np.radians(PDOF1 + 180)) + np.cos(np.radians(PDOF2))
        ))
        PDOF_stima = np.mod(PDOF_stima, 360)
        assegna_parametro(results, 'PDOF_stima', PDOF_stima)
        err_PDOF = calcola_errore_angolare(SimpleNamespace(PDOF=PDOF_stima), targets, 'PDOF')
        assegna_parametro(errors, 'err_PDOF', err_PDOF)

    # altri parametri da riportare nei risultati
    assegna_parametro(results, 'f1', params.f1)
    assegna_parametro(results, 'f2', params.f2)
    assegna_parametro(results, 'J1', params.J1)
    assegna_parametro(results, 'J2', params.J2)
    if cicloide_flag == 1:
        assegna_parametro(results, 'd_post1', d_post1)
        assegna_parametro(results, 'cicloide1', cicloide1)
        assegna_parametro(results, 'd_post2', d_post2)
        assegna_parametro(results, 'cicloide2', cicloide2)
    assegna_parametro(results, 'Ed_target', targets.Ed_target)
    if cicloide_avanzata_flag == 1:
        assegna_parametro(results, 'punti_cicloide_x1', punti_cicloide_x1)
        assegna_parametro(results, 'punti_cicloide_y1', punti_cicloide_y1)
        assegna_parametro(results, 'cicloide_th1', cicloide_th1)
        assegna_parametro(results, 'cicloide_R1', cicloide_R1)
        assegna_parametro(results, 'punti_cicloide_x2', punti_cicloide_x2)
        assegna_parametro(results, 'punti_cicloide_y2', punti_cicloide_y2)
        assegna_parametro(results, 'cicloide_th2', cicloide_th2)
        assegna_parametro(results, 'cicloide_R2', cicloide_R2)
        assegna_parametro(results, 'cic_A1', cic_A1)
        assegna_parametro(results, 'cic_B1', cic_B1)
        assegna_parametro(results, 'cic_A2', cic_A2)
        assegna_parametro(results, 'cic_B2', cic_B2)
        if errore_cicloide1 == 1 or errore_cicloide2 == 1:
            errore_cicloide = 10
            assegna_parametro(errors, 'errore_cicloide', errore_cicloide)


    # chiusura triangoli
    chiusura = params.chiusura_triangoli
    if chiusura == 1:
        # assegnazione parametri output del modello impulsivo
        PDOF = targets.PDOF
        Ed = Ed_target
        h1 = distanza_punto_retta(PDOF, veicolo_1)
        h2 = distanza_punto_retta(PDOF, veicolo_2)
        assegna_parametro(veicolo_1, 'h', h1)
        assegna_parametro(veicolo_2, 'h', h2)
        assegna_parametro(veicolo_1, 'J', J1)
        assegna_parametro(veicolo_2, 'J', J2)
        assegna_parametro(veicolo_1, 'theta_in', results['theta1_in'])
        assegna_parametro(veicolo_1, 'theta_out', results['theta1_out'])
        assegna_parametro(veicolo_1, 'V_post_Kmh', results['V1_post'])
        assegna_parametro(veicolo_1, 'V_pre_Kmh', results['V1_pre'])
        assegna_parametro(veicolo_1, 'omega_post', results['omega1_post'])
        assegna_parametro(veicolo_1, 'omega_pre', results['omega1_pre'])
        assegna_parametro(veicolo_2, 'theta_in', results['theta2_in'])
        assegna_parametro(veicolo_2, 'theta_out', results['theta2_out'])
        assegna_parametro(veicolo_2, 'V_post_Kmh', results['V2_post'])
        assegna_parametro(veicolo_2, 'V_pre_Kmh', results['V2_pre'])
        assegna_parametro(veicolo_2, 'omega_post', results['omega2_post'])
        assegna_parametro(veicolo_2, 'omega_pre', results['omega2_pre'])

        # calcolo dei deltaV in base ai risultati del modello impulsivo
        V1_pre = veicolo_1.V_pre_Kmh
        ang1_pre = veicolo_1.theta_in
        V1_post = veicolo_1.V_post_Kmh
        ang1_post = veicolo_1.theta_out
        deltaV1_target, ang_deltaV1_target = differenza_vett(V1_pre, ang1_pre, V1_post, ang1_post)
        V2_pre = veicolo_2.V_pre_Kmh
        ang2_pre = veicolo_2.theta_in
        V2_post = veicolo_2.V_post_Kmh
        ang2_post = veicolo_2.theta_out
        deltaV2_target, ang_deltaV2_target = differenza_vett(V2_pre, ang2_pre, V2_post, ang2_post)

        # stimo epsilon ottimale che minimizza l'errore tra i deltaV del modello impulsivo e quelli target calcolati dall'energia di deformazione
        epsilon = stima_epsilon(veicolo_1, veicolo_2, Ed, deltaV1_target, deltaV2_target)
        assegna_parametro(results, 'epsilon', epsilon)

        # calcolo effettivamente i deltaV target dall'energia di deformazione
        deltaV1, deltaV2 = deltaV(veicolo_1, veicolo_2, Ed, epsilon)

        # faccio variare moduli e angoli per far combaciare ΔV grafico con ΔV energetico
        best1, min_err1 = ottimizza_deltaV_grafico(veicolo_1 , deltaV1)
        best2, min_err2 = ottimizza_deltaV_grafico(veicolo_2 , deltaV2)

        # errore di chiusura dei triangoli fuori dai range di variazione impostati in ottimizza_deltaV_grafico
        errore_chiusura = 0 #basta minore di 1
        assegna_parametro(errors, 'errore_chiusura', errore_chiusura)
        if not best1 or np.isnan(best1[0]) or np.isnan(best1[1]) or not best2 or np.isnan(best2[0]) or np.isnan(best2[1]):
            errore_chiusura = 10 #basta maggiore di 1
            assegna_parametro(errors, 'errore_chiusura', errore_chiusura)

        # Aggiorna i triangoli di velocità
        V1_post = best1[0]
        ang1_pre = np.mod(best1[1], 360)
        ang1_post = np.mod(best1[2], 360)
        deltaV1_graf, ang_deltaV1_graf = differenza_vett(V1_pre, ang1_pre, V1_post, ang1_post)
        err_deltaV1 = np.abs(deltaV1_graf - deltaV1)
        assegna_parametro(results, 'V1_post_t', V1_post)
        assegna_parametro(results, 'theta1_in_t', ang1_pre)
        assegna_parametro(results, 'theta1_out_t', ang1_post)
        V2_post = best2[0]
        ang2_pre = np.mod(best2[1], 360)
        ang2_post = np.mod(best2[2], 360)
        deltaV2_graf, ang_deltaV2_graf = differenza_vett(V2_pre, ang2_pre, V2_post, ang2_post)
        err_deltaV2 = np.abs(deltaV2_graf - deltaV2)
        assegna_parametro(results, 'V2_post_t', V2_post)
        assegna_parametro(results, 'theta2_in_t', ang2_pre)
        assegna_parametro(results, 'theta2_out_t', ang2_post)

        # Verifica univocità retta su cui giacciono i deltaV
        differenza_angolo = np.mod(np.abs(ang_deltaV1_graf - ang_deltaV2_graf), 360)
        differenza_angolo = min(differenza_angolo, 180 - differenza_angolo)
        errore_parallelismo = np.abs(differenza_angolo)/5
        assegna_parametro(errors, 'errore_parallelismo', errore_parallelismo)

        # Calcolo nuova PDOF come media angolare
        PDOF_angolo = np.degrees(np.arctan2(
            np.sin(np.radians(ang_deltaV1_graf + 180)) + np.sin(np.radians(ang_deltaV2_graf)),
            np.cos(np.radians(ang_deltaV1_graf + 180)) + np.cos(np.radians(ang_deltaV2_graf))
        ))
        PDOF_angolo = np.mod(PDOF_angolo, 360)
        assegna_parametro(results, 'PDOF_eff', PDOF_angolo)

        # Calcolo delle distanze baricentro–nuova_PDOF proiettate
        h1_eff = distanza_punto_retta(PDOF_angolo, veicolo_1)
        h2_eff = distanza_punto_retta(PDOF_angolo, veicolo_2)
        assegna_parametro(results, 'h1', h1)
        assegna_parametro(results, 'h2', h2)

        # verifica scostamenti massimi di 0.1 rispetto ai valori attesi
        scostamento1 = abs(h1 - h1_eff)
        scostamento2 = abs(h2 - h2_eff)
        scostamento_max = np.maximum(scostamento1, scostamento2)
        errore_scostamento = scostamento_max/0.1
        assegna_parametro(errors, 'errore_scostamento', errore_scostamento)

    return results, errors
