import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from funzioni.funzioni_ausiliarie import differenza_vett


# creare il grafico
def mostra_interfaccia_grafico(df_validi, riga_minima, variabili):
    # trova le colonne errore_
    colonne_errore = [col for col in df_validi.columns if col.lower().startswith("errore_")]

    def aggiorna_grafico():
        x = combo_x.get()
        y = combo_y.get()
        col_errore = combo_c.get()

        fig.clear()
        ax = fig.add_subplot(111)

        scatter = ax.scatter(df_validi[x], df_validi[y], c=df_validi[col_errore], cmap='viridis', alpha=0.7)

        ax.scatter(riga_minima[x], riga_minima[y], color='red', edgecolor='black', s=100, label='Errore medio minimo')

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} in funzione di {x}")
        ax.legend()
        ax.grid(True)

        fig.colorbar(scatter, ax=ax, label=col_errore)
        canvas.draw()

    root = tk.Tk()
    root.title("Grafico Montecarlo interattivo")

    frame = tk.Frame(root)
    frame.pack()

    # Combo box per asse X
    tk.Label(frame, text="Asse X:").grid(row=0, column=0, padx=5, pady=5)
    combo_x = ttk.Combobox(frame, values=variabili)
    combo_x.set(variabili[0])
    combo_x.grid(row=0, column=1, padx=5, pady=5)

    # Combo box per asse Y
    tk.Label(frame, text="Asse Y:").grid(row=1, column=0, padx=5, pady=5)
    combo_y = ttk.Combobox(frame, values=variabili)
    combo_y.set(variabili[1])
    combo_y.grid(row=1, column=1, padx=5, pady=5)

    # Combo box per colonna colore
    tk.Label(frame, text="Colorbar:").grid(row=2, column=0, padx=5, pady=5)
    combo_c = ttk.Combobox(frame, values=colonne_errore)
    combo_c.set(colonne_errore[0])
    combo_c.grid(row=2, column=1, padx=5, pady=5)

    # Pulsante
    btn = ttk.Button(frame, text="Genera grafico", command=aggiorna_grafico)
    btn.grid(row=3, columnspan=2, pady=10)

    # Grafico
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()

def draw_arrow(start_pt, vec, color):
    arrow_length = np.linalg.norm(vec)
    if arrow_length == 0:
        return plt.plot(start_pt[0], start_pt[1], marker='o', color=color)[0]

    arrow_head_length = 0.06
    arrow_head_width = 0.04

    dir_vec = vec / arrow_length
    ort_vec = np.array([-dir_vec[1], dir_vec[0]])
    end_pt = start_pt + (arrow_length - arrow_head_length) * dir_vec

    # Corpo della freccia
    h = plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]],
                 color=color, linewidth=1.5)[0]

    # Punta triangolare piena
    arrow_tip = start_pt + vec
    base_left = end_pt + (arrow_head_width / 2) * ort_vec
    base_right = end_pt - (arrow_head_width / 2) * ort_vec

    plt.fill([arrow_tip[0], base_left[0], base_right[0]],
             [arrow_tip[1], base_left[1], base_right[1]],
             color=color, edgecolor=color)
    
    return h

def plot_triangoli(V1_pre, th1_in, V1_post, th1_out,
                        V2_pre, th2_in, V2_post, th2_out,
                        x1, y1, x2, y2, PDOF_angolo):

    scala = 0.04  # 1 unità grafica = 25 km/h

    # Conversione angoli in radianti
    a1_in = np.deg2rad(th1_in)
    a1_out = np.deg2rad(th1_out)
    a2_in = np.deg2rad(th2_in)
    a2_out = np.deg2rad(th2_out)

    # Vettori scalati
    v1pre = scala * V1_pre * np.array([np.cos(a1_in), np.sin(a1_in)])
    v1post = scala * V1_post * np.array([np.cos(a1_out), np.sin(a1_out)])
    v2pre = scala * V2_pre * np.array([np.cos(a2_in), np.sin(a2_in)])
    v2post = scala * V2_post * np.array([np.cos(a2_out), np.sin(a2_out)])
    dV1 = v1post - v1pre
    dV2 = v2post - v2pre

    # Setup grafico
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Rappresentazione vettoriale (1 unità grafica = 25 km/h)', pad=20)

    # Origine
    plt.plot(0, 0, 'ko')

    # PDOF
    PDOF_rad = np.deg2rad(PDOF_angolo)
    raggio = 5
    xPDOF = np.array([-np.cos(PDOF_rad), np.cos(PDOF_rad)]) * raggio
    yPDOF = np.array([-np.sin(PDOF_rad), np.sin(PDOF_rad)]) * raggio
    pPDOF = plt.plot(xPDOF, yPDOF, 'k--', linewidth=2)[0]
    plt.text(np.cos(PDOF_rad)*raggio*1.05, np.sin(PDOF_rad)*raggio*1.05,
             f'{PDOF_angolo:.1f}°', fontweight='bold', ha='center')

    # Baricentri
    plt.plot(x1, y1, 'bo')
    plt.text(x1 + 0.05, y1 + 0.05, 'G1', color='b', fontweight='bold')
    plt.plot(x2, y2, 'ro')
    plt.text(x2 + 0.05, y2 + 0.05, 'G2', color='r', fontweight='bold')

    # Vettori veicolo 1
    h0 = draw_arrow(np.array([x1, y1]), v1pre, [0.4, 0.7, 1])  # azzurro chiaro
    h1 = draw_arrow(np.array([x1, y1]), v1post, [0, 0, 1])     # blu
    h3 = draw_arrow(np.array([x1, y1]) + v1pre, dV1, [0, 0.6, 0])  # verde

    # Vettori veicolo 2
    h4 = draw_arrow(np.array([x2, y2]), v2pre, [1, 0.5, 0.5])  # rosso chiaro
    h2 = draw_arrow(np.array([x2, y2]), v2post, [1, 0, 0])     # rosso
    h5 = draw_arrow(np.array([x2, y2]) + v2pre, dV2, [0, 0.6, 0])  # verde

    # Legenda
    plt.legend([h0, h1, h4, h2, h3, pPDOF],
               ['V1 pre-urto', 'V1 post-urto', 'V2 pre-urto', 'V2 post-urto', 'ΔV', 'PDOF'],
               loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.show(block=False)

def draw_angle_arc(center, angle_deg, color, radius=0.5, label=None):
    theta1, theta2 = (0, angle_deg) if angle_deg >= 0 else (angle_deg, 0)

    arc = patches.Arc(center, width=2*radius, height=2*radius,
                      angle=0, theta1=theta1, theta2=theta2,
                      color=color, linewidth=1.5)  # stesso color
    plt.gca().add_patch(arc)

    # posizionamento dell'etichetta
    angle_rad = np.deg2rad(abs(angle_deg)) / 2
    label_x = center[0] + radius * 1.1 * np.cos(angle_rad)
    label_y = center[1] + radius * 1.1 * np.sin(angle_rad)
    if label:
        plt.text(label_x, label_y, f"{angle_deg:.0f}°", color=color, fontsize=9)

def plot_vettori(V1_pre, th1_in, V1_post, th1_out,
                 V2_pre, th2_in, V2_post, th2_out,
                 x1, y1, x2, y2):

    scala = 0.04  # 1 unità grafica = 25 km/h

    # Conversione angoli in radianti
    a1_in = np.deg2rad(th1_in)
    a1_out = np.deg2rad(th1_out)
    a2_in = np.deg2rad(th2_in)
    a2_out = np.deg2rad(th2_out)

    # Vettori scalati
    v1pre = scala * V1_pre * np.array([np.cos(a1_in), np.sin(a1_in)])
    v1post = scala * V1_post * np.array([np.cos(a1_out), np.sin(a1_out)])
    v2pre = scala * V2_pre * np.array([np.cos(a2_in), np.sin(a2_in)])
    v2post = scala * V2_post * np.array([np.cos(a2_out), np.sin(a2_out)])

    # Setup grafico
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Rappresentazione vettoriale (1 unità grafica = 25 km/h)', pad=20)

    # Origine
    plt.plot(0, 0, 'ko')

    # Baricentri
    plt.plot(x1, y1, 'bo')
    plt.text(x1 + 0.05, y1 + 0.05, 'G1', color='b', fontweight='bold')
    plt.plot(x2, y2, 'ro')
    plt.text(x2 + 0.05, y2 + 0.05, 'G2', color='r', fontweight='bold')

    # Vettori veicolo 1
    h0 = draw_arrow(np.array([x1, y1]), v1pre, [0.4, 0.7, 1])
    h1 = draw_arrow(np.array([x1, y1]), v1post, [0, 0, 1])
    draw_angle_arc([x1, y1], th1_in, color=[0.4, 0.7, 1], radius=0.5, label=f"{th1_in:.0f}")
    draw_angle_arc([x1, y1], th1_out, color=[0, 0, 1], radius=0.7, label=f"{th1_out:.0f}")

    # Vettori veicolo 2
    h4 = draw_arrow(np.array([x2, y2]), v2pre, [1, 0.5, 0.5])
    h2 = draw_arrow(np.array([x2, y2]), v2post, [1, 0, 0])
    draw_angle_arc([x2, y2], th2_in, color=[1, 0.5, 0.5], radius=0.5, label=f"{th2_in:.0f}")
    draw_angle_arc([x2, y2], th2_out, color=[1, 0, 0], radius=0.7, label=f"{th2_out:.0f}")

    # Legenda
    plt.legend([h0, h1, h4, h2],
            ['V1 pre-urto', 'V1 post-urto', 'V2 pre-urto', 'V2 post-urto'],
            loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=4)

    plt.show(block=False)

def plot_vettori_con_PDOF(V1_pre, th1_in, V1_post, th1_out,
                 V2_pre, th2_in, V2_post, th2_out,
                 x1, y1, x2, y2, PDOF_stima):

    scala = 0.04  # 1 unità grafica = 25 km/h

    # Conversione angoli in radianti
    a1_in = np.deg2rad(th1_in)
    a1_out = np.deg2rad(th1_out)
    a2_in = np.deg2rad(th2_in)
    a2_out = np.deg2rad(th2_out)

    # Vettori scalati
    v1pre = scala * V1_pre * np.array([np.cos(a1_in), np.sin(a1_in)])
    v1post = scala * V1_post * np.array([np.cos(a1_out), np.sin(a1_out)])
    v2pre = scala * V2_pre * np.array([np.cos(a2_in), np.sin(a2_in)])
    v2post = scala * V2_post * np.array([np.cos(a2_out), np.sin(a2_out)])

    # Setup grafico
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Rappresentazione vettoriale (1 unità grafica = 25 km/h)', pad=20)

    # Origine
    plt.plot(0, 0, 'ko')

    # PDOF
    PDOF_rad = np.deg2rad(PDOF_stima)
    raggio = 5
    xPDOF = np.array([-np.cos(PDOF_rad), np.cos(PDOF_rad)]) * raggio
    yPDOF = np.array([-np.sin(PDOF_rad), np.sin(PDOF_rad)]) * raggio
    pPDOF = plt.plot(xPDOF, yPDOF, linestyle='--', color='magenta', linewidth=2)[0]
    plt.text(np.cos(PDOF_rad)*raggio*1.05,
         np.sin(PDOF_rad)*raggio*1.05,
         f'{PDOF_stima:.1f}°',
         fontweight='bold', ha='center', color='magenta')

    # Baricentri
    plt.plot(x1, y1, 'bo')
    plt.text(x1 + 0.05, y1 + 0.05, 'G1', color='b', fontweight='bold')
    plt.plot(x2, y2, 'ro')
    plt.text(x2 + 0.05, y2 + 0.05, 'G2', color='r', fontweight='bold')

    # Vettori veicolo 1
    h0 = draw_arrow(np.array([x1, y1]), v1pre, [0.4, 0.7, 1])
    h1 = draw_arrow(np.array([x1, y1]), v1post, [0, 0, 1])
    draw_angle_arc([x1, y1], th1_in, color=[0.4, 0.7, 1], radius=0.5, label=f"{th1_in:.0f}")
    draw_angle_arc([x1, y1], th1_out, color=[0, 0, 1], radius=0.7, label=f"{th1_out:.0f}")

    # Vettori veicolo 2
    h4 = draw_arrow(np.array([x2, y2]), v2pre, [1, 0.5, 0.5])
    h2 = draw_arrow(np.array([x2, y2]), v2post, [1, 0, 0])
    draw_angle_arc([x2, y2], th2_in, color=[1, 0.5, 0.5], radius=0.5, label=f"{th2_in:.0f}")
    draw_angle_arc([x2, y2], th2_out, color=[1, 0, 0], radius=0.7, label=f"{th2_out:.0f}")

    # Legenda
    plt.legend([h0, h1, h4, h2, pPDOF],
            ['V1 pre-urto', 'V1 post-urto', 'V2 pre-urto', 'V2 post-urto', 'PDOF stimata'],
            loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=5)

    plt.show(block=False)

def draw_sector(ax, theta1_rad, theta2_rad, r, color):
    def plot_single_sector(t1, t2):
        t1 %= 2 * np.pi
        t2 %= 2 * np.pi

        if np.isclose(t1, t2):
            return

        if t2 < t1:
            # Settore attraversa lo zero → due archi
            width1 = 2 * np.pi - t1
            width2 = t2
            ax.bar(x=t1 + width1 / 2, height=r, width=width1, bottom=0,
                   color=color, alpha=0.25, edgecolor=None, linewidth=0)
            ax.bar(x=width2 / 2, height=r, width=width2, bottom=0,
                   color=color, alpha=0.25, edgecolor=None, linewidth=0)
        else:
            # Settore normale
            width = t2 - t1
            ax.bar(x=t1 + width / 2, height=r, width=width, bottom=0,
                   color=color, alpha=0.25, edgecolor=None, linewidth=0)

    # Settore principale
    plot_single_sector(theta1_rad, theta2_rad)

    # Settore opposto (rotazione di π)
    opp1 = (theta1_rad + np.pi) % (2 * np.pi)
    opp2 = (theta2_rad + np.pi) % (2 * np.pi)
    plot_single_sector(opp1, opp2)

def angolo_cardinale_nel_verso(PDOF, verso, theta_altro_baricentro):
    PDOF = PDOF % (2 * np.pi)
    theta_altro = theta_altro_baricentro % (2 * np.pi)

    # Cardinali netti (0, 90, 180, 270)
    cardinali = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

    # Trova il quadrante di theta_altro
    def trova_quadrante(theta):
        if 0 <= theta < np.pi/2:
            return 0
        elif np.pi/2 <= theta < np.pi:
            return 1
        elif np.pi <= theta < 3*np.pi/2:
            return 2
        else:
            return 3

    q = trova_quadrante(theta_altro)
    estremo_sx = cardinali[q]
    estremo_dx = cardinali[(q + 1) % 4]

    # Scegli quale dei due estremi viene incontrato per primo nel verso indicato
    if verso == 1:  # antiorario
        delta_sx = (estremo_sx - PDOF) % (2 * np.pi)
        delta_dx = (estremo_dx - PDOF) % (2 * np.pi)
    else:  # orario
        delta_sx = (PDOF - estremo_sx) % (2 * np.pi)
        delta_dx = (PDOF - estremo_dx) % (2 * np.pi)

    if delta_sx < delta_dx:
        return estremo_sx
    else:
        return estremo_dx

def stimaPDOF(params, targets, pdof):

    pdof_target = np.deg2rad(pdof.PDOF) % (2 * np.pi)
    PDOF_stima  = np.deg2rad(targets.PDOF_stima) % (2 * np.pi)

    # braccio minimo per consentire il post-urto
    V1_post = targets.V1_post / 3.6
    V1_pre  = targets.V1_pre / 3.6
    V2_post = targets.V2_post / 3.6
    V2_pre  = targets.V2_pre / 3.6

    theta1_in = targets.theta1_in
    theta1_out = targets.theta1_out
    theta2_in = targets.theta2_in
    theta2_out = targets.theta2_out

    deltaV1, ang_deltaV1 = differenza_vett(V1_pre, theta1_in, V1_post, theta1_out)
    deltaV2, ang_deltaV2 = differenza_vett(V2_pre, theta2_in, V2_post, theta2_out)

    I_max1 = params.m1 * deltaV1
    I_max2 = params.m2 * deltaV2

    h1_min = abs(targets.J1 * (targets.omega1_pre - targets.omega1_post) / I_max1)
    h2_min = abs(targets.J2 * (targets.omega2_pre - targets.omega2_post) / I_max2)

    # baricentri
    r1 = np.hypot(targets.x1, targets.y1)
    r2 = np.hypot(targets.x2, targets.y2)
    theta_1 = np.arctan2(targets.y1, targets.x1)
    theta_2 = np.arctan2(targets.y2, targets.x2)

    # angoli tra i raggi vettore baricentrici e la PDOF
    alfa1 = np.arcsin(min(1, h1_min / r1))
    alfa2 = np.arcsin(min(1, h2_min / r2))

    # PDOF in radianti
    PDOF1 = theta_1 + np.sign(targets.omega1_post) * -alfa1
    PDOF2 = theta_2 + np.sign(targets.omega2_post) * -alfa2

    # Versi opposti
    verso1 = -np.sign(targets.omega1_post)
    verso2 = -np.sign(targets.omega2_post)

    # Calcolo effettivo dei due angoli cardinali
    a = angolo_cardinale_nel_verso(PDOF1, verso1, theta_2)
    b = angolo_cardinale_nel_verso(PDOF2, verso2, theta_1)
    cardinali_rad = np.array([a, b])

    # Verso opposto alla rotazione post-urto
    verso1 = -np.sign(targets.omega1_post)
    verso2 = -np.sign(targets.omega2_post)

    def angolo_cardinale(PDOF, verso):
        if verso == 0:
            return PDOF
        if verso == -1:
            delta = (PDOF - cardinali_rad) % (2 * np.pi)
        else:
            delta = (cardinali_rad - PDOF) % (2 * np.pi)
        return cardinali_rad[np.argmin(delta)]

    # Calcolo estremi dei settori
    range1 = angolo_cardinale(PDOF1, verso1)
    range2 = angolo_cardinale(PDOF2, verso2)

    # normalizzazione
    PDOF1 = PDOF1 % (2 * np.pi)
    PDOF2 = PDOF2 % (2 * np.pi)
    range1 = range1 % (2 * np.pi)
    range2 = range2 % (2 * np.pi)

    # peso per media ponderata
    w1 = abs(I_max1)
    w2 = abs(I_max2)

    # Dati per il plot
    G1_angle = theta_1
    G2_angle = theta_2
    G1_r = r1
    G2_r = r2

    a1 = PDOF1
    a2 = PDOF2
    a1_opposto = (a1 + np.pi) % (2 * np.pi)
    a2_opposto = (a2 + np.pi) % (2 * np.pi)

    r1_plot = min(2, 2 * w1 / max(w1, w2))
    r2_plot = min(2, 2 * w2 / max(w1, w2))
    r_max = max([r1_plot, r2_plot, G1_r, G2_r]) + 0.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Settori verdi
    draw_sector(ax, range1, PDOF1, r=2.5, color='green')
    draw_sector(ax, range2, PDOF2, r=2.5, color='green')

    ax.plot([a1_opposto, a1], [r_max, r_max], color=(0.8, 0.85, 1.0), linewidth=2)
    ax.plot([a2_opposto, a2], [r_max, r_max], color=(1.0, 0.8, 0.8), linewidth=2)

    vet1, = ax.plot([0, a1], [0, r1_plot], 'b-', linewidth=2)
    vet2, = ax.plot([0, a2], [0, r2_plot], 'r-', linewidth=2)

    # Etichette angoli principali
    ax.text(a1, r_max + 0.35, f"{np.rad2deg(a1)%360:.1f}°", ha='center', color='b', weight='bold', fontsize=10)
    ax.text(a2, r_max + 0.35, f"{np.rad2deg(a2)%360:.1f}°", ha='center', color='r', weight='bold', fontsize=10)

    # Etichette angoli opposti (una sola volta!)
    opp1_deg = (np.rad2deg(a1) + 180) % 360
    opp2_deg = (np.rad2deg(a2) + 180) % 360

    ax.text(a1_opposto, r_max + 0.35, f"{opp1_deg:.1f}°", ha='center', color='b', weight='bold', fontsize=10)
    ax.text(a2_opposto, r_max + 0.35, f"{opp2_deg:.1f}°", ha='center', color='r', weight='bold', fontsize=10)

    # Baricentri
    ax.plot(G1_angle, G1_r, 'bo', markersize=8)
    ax.plot(G2_angle, G2_r, 'ro', markersize=8)
    ax.text(G1_angle, G1_r + 0.2, 'G1', color='b', weight='bold', ha='center')
    ax.text(G2_angle, G2_r + 0.2, 'G2', color='r', weight='bold', ha='center')

    # Retta tratteggiata nera per pdof_target e sua opposta
    pdof_opp = (pdof_target + np.pi) % (2 * np.pi)
    pdof_targ, = ax.plot([0, pdof_target], [0, r_max], linestyle='--', color='black', linewidth=1.5)
    ax.plot([0, pdof_opp], [0, r_max], linestyle='--', color='black', linewidth=1.5)
    # Etichette agli estremi delle due linee
    ax.text(pdof_target, r_max + 0.35, f"{np.rad2deg(pdof_target)%360:.1f}°",
            ha='center', color='black', fontsize=10, weight='bold')
    ax.text(pdof_opp, r_max + 0.35, f"{(np.rad2deg(pdof_target) + 180)%360:.1f}°",
            ha='center', color='black', fontsize=10, weight='bold')
    
    # Retta tratteggiata nera per PDOF_stima e sua opposta
    pdof_stimata_opp = (PDOF_stima + np.pi) % (2 * np.pi)
    pdof_stimata, = ax.plot([0, PDOF_stima], [0, r_max], linestyle='--', color='magenta', linewidth=1.5)
    ax.plot([0, pdof_stimata_opp], [0, r_max], linestyle='--', color='magenta', linewidth=1.5)
    # Etichette agli estremi delle due linee
    ax.text(PDOF_stima, r_max + 0.35, f"{np.rad2deg(PDOF_stima)%360:.1f}°",
            ha='center', color='magenta', fontsize=10, weight='bold')
    ax.text(pdof_stimata_opp, r_max + 0.35, f"{(np.rad2deg(PDOF_stima) + 180)%360:.1f}°",
            ha='center', color='magenta', fontsize=10, weight='bold')

    ax.set_title("Cono di validità della PDOF", pad=40)
    ax.set_rlim(0, r_max)
    ax.legend([vet1, vet2, pdof_targ, pdof_stimata], ['PDOF veicolo 1', 'PDOF veicolo 2', 'PDOF target', 'PDOF stimata'],
          loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=4)

    plt.show(block=False)

def stimaPDOF_triangoli(params, targets):

    pdof_target = np.deg2rad(targets.PDOF_eff) % (2 * np.pi)

    # braccio minimo per consentire il post-urto
    V1_post = targets.V1_post / 3.6
    V1_pre  = targets.V1_pre / 3.6
    V2_post = targets.V2_post / 3.6
    V2_pre  = targets.V2_pre / 3.6

    theta1_in = targets.theta1_in
    theta1_out = targets.theta1_out
    theta2_in = targets.theta2_in
    theta2_out = targets.theta2_out

    deltaV1, ang_deltaV1 = differenza_vett(V1_pre, theta1_in, V1_post, theta1_out)
    deltaV2, ang_deltaV2 = differenza_vett(V2_pre, theta2_in, V2_post, theta2_out)

    I_max1 = params.m1 * deltaV1
    I_max2 = params.m2 * deltaV2

    h1_min = abs(targets.J1 * (targets.omega1_pre - targets.omega1_post) / I_max1)
    h2_min = abs(targets.J2 * (targets.omega2_pre - targets.omega2_post) / I_max2)

    # baricentri
    r1 = np.hypot(targets.x1, targets.y1)
    r2 = np.hypot(targets.x2, targets.y2)
    theta_1 = np.arctan2(targets.y1, targets.x1)
    theta_2 = np.arctan2(targets.y2, targets.x2)

    # angoli tra i raggi vettore baricentrici e la PDOF
    alfa1 = np.arcsin(min(1, h1_min / r1))
    alfa2 = np.arcsin(min(1, h2_min / r2))

    # PDOF in radianti
    PDOF1 = theta_1 + np.sign(targets.omega1_post) * -alfa1
    PDOF2 = theta_2 + np.sign(targets.omega2_post) * -alfa2

    # Verso opposto alla rotazione post-urto
    verso1 = -np.sign(targets.omega1_post)
    verso2 = -np.sign(targets.omega2_post)

    # Calcolo effettivo dei due angoli cardinali
    a = angolo_cardinale_nel_verso(PDOF1, verso1, theta_2)
    b = angolo_cardinale_nel_verso(PDOF2, verso2, theta_1)
    cardinali_rad = np.array([a, b])

    def angolo_cardinale(PDOF, verso):
        if verso == 0:
            return PDOF
        if verso == -1:
            delta = (PDOF - cardinali_rad) % (2 * np.pi)
        else:
            delta = (cardinali_rad - PDOF) % (2 * np.pi)
        return cardinali_rad[np.argmin(delta)]

    # Calcolo estremi dei settori
    range1 = angolo_cardinale(PDOF1, verso1)
    range2 = angolo_cardinale(PDOF2, verso2)

    # normalizzazione
    PDOF1 = PDOF1 % (2 * np.pi)
    PDOF2 = PDOF2 % (2 * np.pi)
    range1 = range1 % (2 * np.pi)
    range2 = range2 % (2 * np.pi)

    # peso per media ponderata
    w1 = abs(I_max1)
    w2 = abs(I_max2)

    # Dati per il plot
    G1_angle = theta_1
    G2_angle = theta_2
    G1_r = r1
    G2_r = r2

    a1 = PDOF1
    a2 = PDOF2
    a1_opposto = (a1 + np.pi) % (2 * np.pi)
    a2_opposto = (a2 + np.pi) % (2 * np.pi)

    r1_plot = min(2, 2 * w1 / max(w1, w2))
    r2_plot = min(2, 2 * w2 / max(w1, w2))
    r_max = max([r1_plot, r2_plot, G1_r, G2_r]) + 0.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Settori verdi
    draw_sector(ax, range1, PDOF1, r=2.5, color='green')
    draw_sector(ax, range2, PDOF2, r=2.5, color='green')

    ax.plot([a1_opposto, a1], [r_max, r_max], color=(0.8, 0.85, 1.0), linewidth=2)
    ax.plot([a2_opposto, a2], [r_max, r_max], color=(1.0, 0.8, 0.8), linewidth=2)

    vet1, = ax.plot([0, a1], [0, r1_plot], 'b-', linewidth=2)
    vet2, = ax.plot([0, a2], [0, r2_plot], 'r-', linewidth=2)

    # Etichette angoli principali
    ax.text(a1, r_max + 0.35, f"{np.rad2deg(a1)%360:.1f}°", ha='center', color='b', weight='bold', fontsize=10)
    ax.text(a2, r_max + 0.35, f"{np.rad2deg(a2)%360:.1f}°", ha='center', color='r', weight='bold', fontsize=10)

    # Etichette angoli opposti (una sola volta!)
    opp1_deg = (np.rad2deg(a1) + 180) % 360
    opp2_deg = (np.rad2deg(a2) + 180) % 360

    ax.text(a1_opposto, r_max + 0.35, f"{opp1_deg:.1f}°", ha='center', color='b', weight='bold', fontsize=10)
    ax.text(a2_opposto, r_max + 0.35, f"{opp2_deg:.1f}°", ha='center', color='r', weight='bold', fontsize=10)

    # Baricentri
    ax.plot(G1_angle, G1_r, 'bo', markersize=8)
    ax.plot(G2_angle, G2_r, 'ro', markersize=8)
    ax.text(G1_angle, G1_r + 0.2, 'G1', color='b', weight='bold', ha='center')
    ax.text(G2_angle, G2_r + 0.2, 'G2', color='r', weight='bold', ha='center')

    # Retta tratteggiata nera per pdof_target e sua opposta
    pdof_opp = (pdof_target + np.pi) % (2 * np.pi)
    pdof_targ, = ax.plot([0, pdof_target], [0, r_max], linestyle='--', color='black', linewidth=1.5)
    ax.plot([0, pdof_opp], [0, r_max], linestyle='--', color='black', linewidth=1.5)

    # Etichette agli estremi delle due linee
    ax.text(pdof_target, r_max + 0.35, f"{np.rad2deg(pdof_target)%360:.1f}°",
            ha='center', color='black', fontsize=10, weight='bold')
    ax.text(pdof_opp, r_max + 0.35, f"{(np.rad2deg(pdof_target) + 180)%360:.1f}°",
            ha='center', color='black', fontsize=10, weight='bold')

    ax.set_title("Cono di validità della PDOF", pad=40)
    ax.set_rlim(0, r_max)
    ax.legend([vet1, vet2, pdof_targ], ['PDOF veicolo 1', 'PDOF veicolo 2', 'PDOF effettiva'],
          loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3)

    plt.show(block=False)

def plotta_due_cicloidi(A1, B1, x1, y1, theta1, R1, lung1, ang1,
                        A2, B2, x2, y2, theta2, R2, lung2, ang2):

    plt.figure(figsize=(10, 7))

    # Prima cicloide
    plt.plot(x1, y1, label=f'Cicloide 1: theta={theta1:.2f}, R={R1:.2f}, L={lung1:.2f}')
    plt.plot(*A1, 'ko')
    plt.plot(*B1, 'ro')
    plt.text(*A1, 'A1', fontsize=9, ha='right', va='top')
    plt.text(*B1, 'B1', fontsize=9, ha='right', va='bottom')

    # Seconda cicloide
    plt.plot(x2, y2, label=f'Cicloide 2: theta={theta2:.2f}, R={R2:.2f}, L={lung2:.2f}')
    plt.plot(*A2, 'ks')
    plt.plot(*B2, 'rs')
    plt.text(*A2, 'A2', fontsize=9, ha='left', va='top')
    plt.text(*B2, 'B2', fontsize=9, ha='left', va='bottom')

    # Retta tangente 1
    L_retta = 3
    ang1_rad = np.deg2rad(ang1)
    retta1_x = [A1[0] - L_retta * np.cos(ang1_rad), A1[0] + L_retta * np.cos(ang1_rad)]
    retta1_y = [A1[1] - L_retta * np.sin(ang1_rad), A1[1] + L_retta * np.sin(ang1_rad)]
    plt.plot(retta1_x, retta1_y, 'g--', label=f'Retta tangente 1: α={ang1:.2f}°')

    # Retta tangente 2
    ang2_rad = np.deg2rad(ang2)
    retta2_x = [A2[0] - L_retta * np.cos(ang2_rad), A2[0] + L_retta * np.cos(ang2_rad)]
    retta2_y = [A2[1] - L_retta * np.sin(ang2_rad), A2[1] + L_retta * np.sin(ang2_rad)]
    plt.plot(retta2_x, retta2_y, 'm--', label=f'Retta tangente 2: α={ang2:.2f}°')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Cicloidi post-urto", pad=20)
    plt.show(block=False)
