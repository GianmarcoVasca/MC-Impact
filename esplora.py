import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funzioni.operazioni import calcoli
from funzioni.lettura_e_scrittura import leggi_range_da_file, genera_input_casuale
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def filtra_valori_positivi(nome_variabile, valori):
    variabili_da_filtrare = [
        "V1_pre", "V2_pre", "V1_post", "V2_post",
        "Ed", "f1", "f2", "cicloide1", "cicloide2"
    ]
    if nome_variabile in variabili_da_filtrare:
        return [v for v in valori if v >= 0]
    return valori

# --- GUI per selezionare i parametri da espandere ---
def seleziona_parametri_da_espandere(parametri):
    variabili_candidabili = [k for k, v in parametri.items() if isinstance(v, tuple) and len(v) == 2 and v[0] != v[1]]
    selezione = []

    def conferma():
        for var, stato in selezioni.items():
            if stato.get():
                selezione.append(var)
        root.destroy()

    root = tk.Tk()
    root.title("Seleziona parametri")

    label = tk.Label(root, text="Range delle variabili da incrementare:")
    label.pack(anchor='w', padx=10, pady=(10, 0))

    selezioni = {k: tk.BooleanVar(value=False) for k in variabili_candidabili}
    for k in variabili_candidabili:
        cb = tk.Checkbutton(root, text=k, variable=selezioni[k])
        cb.pack(anchor='w', padx=20, pady=2)

    btn = ttk.Button(root, text="Conferma", command=conferma)
    btn.pack(pady=10)

    root.mainloop()
    return selezione

# --- GUI per selezionare la variabile da monitorare ---
def seleziona_variabile_da_monitorare(listato):
    opzioni = list(listato.keys())
    root = tk.Tk()
    root.title("Seleziona variabile da monitorare")

    selezione = tk.StringVar(master=root)
    selezione.set(opzioni[0])

    tk.Label(root, text="Variabile da monitorare:").pack(padx=20, pady=(10, 0))
    menu = ttk.Combobox(root, textvariable=selezione, values=opzioni, state="readonly")
    menu.pack(padx=20, pady=10)
    ttk.Button(root, text="Conferma", command=root.quit).pack(pady=10)

    root.mainloop()
    root.destroy()
    return selezione.get()

def espandi_range(ranges, passo=1.0):
    nuovi_ranges = {}
    for k, (v_min, v_max) in ranges.items():
        if k in ['f1', 'f2']:
            nuovo_min = max(0.3, v_min - 0.01)
            nuovo_max = min(1.0, v_max + 0.01)
        elif k in ['d_post1', 'd_post2']:
            nuovo_min = max(0.1, v_min - 0.1)
            nuovo_max = v_max + 0.1
        elif k in ['V1_post_Kmh', 'V2_post_Kmh', 'EES1_Kmh', 'EES2_Kmh']:
            nuovo_min = max(0.0, v_min - 1.0)
            nuovo_max = v_max + 1.0
        elif k in ['m1', 'm2']:
            nuovo_min = max(0.0, v_min - 10.0)
            nuovo_max = v_max + 10.0
        else:
            nuovo_min = v_min - passo / 2
            nuovo_max = v_max + passo / 2
        nuovi_ranges[k] = (nuovo_min, nuovo_max)
    return nuovi_ranges

def aggiorna_parametri(parametri_base, nuovi_ranges):
    parametri_mod = parametri_base.copy()
    for k, v in nuovi_ranges.items():
        parametri_mod[k] = v
    return parametri_mod

def model_montecarlo(parametri_modificati, targets, N=1000):
    risultati = []
    for _ in range(N):
        p = genera_input_casuale(parametri_modificati)
        t = genera_input_casuale(targets)
        try:
            res, err = calcoli(p, t)
            if all(val < 1 for key, val in err.items() if 'errore_' in key.lower()):
                risultati.append({**res, **err})
        except:
            continue
    return risultati

def mostra_grafico_interattivo(output):
    if not output:
        print("\u26a0\ufe0f Nessun dato valido per generare il grafico interattivo.")
        return

    df = pd.DataFrame(output)
    if df.empty:
        print("\u26a0\ufe0f DataFrame vuoto: impossibile generare il grafico.")
        return

    variabili = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    colonne_errore = [col for col in df.columns if col.lower().startswith("errore_")]

    if not colonne_errore:
        print("\u26a0\ufe0f Nessuna colonna 'errore_' trovata.")
        return

    media_errori = df[colonne_errore].mean(axis=1)
    if media_errori.isnull().all():
        print("\u26a0\ufe0f Tutti i valori nelle colonne 'errore_' sono NaN.")
        return

    riga_minima = df.loc[media_errori.idxmin()]

    def aggiorna_grafico():
        x = combo_x.get()
        y = combo_y.get()
        colore = combo_c.get()

        fig.clear()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(df[x], df[y], c=df[colore], cmap='viridis', alpha=0.7)
        ax.scatter(riga_minima[x], riga_minima[y], color='red', edgecolor='black', s=100, label='Errore minimo')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} in funzione di {x}")
        ax.legend()
        ax.grid(True)
        fig.colorbar(scatter, ax=ax, label=colore)
        canvas.draw()

    root = tk.Tk()
    root.title("Grafico Montecarlo interattivo")

    frame = tk.Frame(root)
    frame.pack()

    tk.Label(frame, text="Asse X:").grid(row=0, column=0, padx=5, pady=5)
    combo_x = ttk.Combobox(frame, values=variabili)
    combo_x.set(variabili[0])
    combo_x.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(frame, text="Asse Y:").grid(row=1, column=0, padx=5, pady=5)
    combo_y = ttk.Combobox(frame, values=variabili)
    combo_y.set(variabili[1] if len(variabili) > 1 else variabili[0])
    combo_y.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(frame, text="Colorbar:").grid(row=2, column=0, padx=5, pady=5)
    combo_c = ttk.Combobox(frame, values=colonne_errore)
    combo_c.set(colonne_errore[0])
    combo_c.grid(row=2, column=1, padx=5, pady=5)

    btn = ttk.Button(frame, text="Genera grafico", command=aggiorna_grafico)
    btn.grid(row=3, columnspan=2, pady=10)

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()

def plot_devstd_vs_iterazione(risultati):
    iterazioni = [r['iter'] for r in risultati if not np.isnan(r['deviazione_std'])]
    std_vals = [r['deviazione_std'] for r in risultati if not np.isnan(r['deviazione_std'])]

    plt.figure(figsize=(8, 4))
    plt.plot(iterazioni, std_vals, marker='o')
    plt.xlabel("Iterazione")
    plt.ylabel(f"Deviazione standard {OUTPUT_METRICA}")
    plt.title("Variazione della deviazione standard in funzione delle iterazioni")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def esplora_variabilita_angoli(parametri_base, targets, max_iter=20, passo=1.0, soglia_salto=3.0, variabili_da_espandere=None):
    if variabili_da_espandere is None:
        variabili_da_espandere = list(parametri_base.keys())

    risultati = []
    angoli_variabili = {k: tuple(parametri_base[k]) for k in variabili_da_espandere if k in parametri_base and len(parametri_base[k]) == 2 and parametri_base[k][0] != parametri_base[k][1]}
    if not angoli_variabili:
        print("\u26a0\ufe0f Nessun parametro variabile trovato.")
        return []

    ranges_correnti = angoli_variabili.copy()
    salto_rilevato = False
    iterazioni_post_salto = 0
    ultimo_output_valido = None

    for iterazione in range(max_iter):
        parametri_attuali = aggiorna_parametri(parametri_base, ranges_correnti)

        N_pre = 10000
        pre_output = model_montecarlo(parametri_attuali, targets, N=N_pre)
        valori_pre = [r[OUTPUT_METRICA] for r in pre_output if OUTPUT_METRICA in r and not np.isnan(r[OUTPUT_METRICA])]
        valori_pre = filtra_valori_positivi(OUTPUT_METRICA, valori_pre)
        sigma_pre = np.std(valori_pre)
        media_pre = np.mean(valori_pre)

        z = 1.96
        errore_abs = 0.025
        if sigma_pre == 0 or np.isnan(sigma_pre) or np.isnan(media_pre):
            N_dinamico = N_pre
        else:
            N_dinamico = int((z * sigma_pre / errore_abs) ** 2)
            N_dinamico = max(N_dinamico, N_pre)
            N_dinamico = min(N_dinamico, 200000)

        print(f"[Iter {iterazione+1}]")
        print(f"   → N dinamico calcolato: {N_dinamico} (media={media_pre:.3f}, std={sigma_pre:.3f})")

        output = model_montecarlo(parametri_attuali, targets, N=N_dinamico)

        if output:
            ultimo_output_valido = output

        # Estrazione e filtro valori monitorati
        valori_output = [r[OUTPUT_METRICA] for r in output if OUTPUT_METRICA in r and not np.isnan(r[OUTPUT_METRICA])]
        valori_output_filtrati = filtra_valori_positivi(OUTPUT_METRICA, valori_output)
        std = np.std(valori_output_filtrati) if valori_output_filtrati else float('nan')
        media = np.mean(valori_output_filtrati) if valori_output_filtrati else float('nan')

        print(f"   Deviazione std {OUTPUT_METRICA}: {std:.2f}" if not np.isnan(std) else "   Nessun dato valido")

        risultati.append({
            "iter": iterazione + 1,
            "ranges": ranges_correnti.copy(),
            "deviazione_std": std,
            "media_output": media,
            "campioni_validi": len(valori_output_filtrati),
            "output": output,
            "valori_filtrati": valori_output_filtrati
        })

        if len(risultati) >= 2 and not salto_rilevato:
            std_prec = risultati[-2]['deviazione_std']
            if not np.isnan(std) and not np.isnan(std_prec) and std_prec >= 1e-6:
                incremento = std / std_prec
                if incremento > soglia_salto:
                    salto_rilevato = True
                    print(f"\n\u26a0\ufe0f   SALTO NETTO RILEVATO: da std = {std_prec:.2f} a std = {std:.2f} (x{incremento:.2f})\n")

        if salto_rilevato:
            iterazioni_post_salto += 1
            if iterazioni_post_salto >= 15:
                break

        ranges_correnti = espandi_range(ranges_correnti, passo=passo)


    plot_devstd_vs_iterazione(risultati)

    print("\n--- RIEPILOGO ---")
    for r in risultati:
        media = r['media_output']
        std = r['deviazione_std']
        low = media - 3 * std
        high = media + 3 * std

        # Correzione per variabili che non possono essere negative
        variabili_positive = {'V1_pre', 'V2_pre', 'V1_post', 'V2_post', 'Ed', 'f1', 'f2', 'cicloide1', 'cicloide2'}
        if OUTPUT_METRICA in variabili_positive:
            low = max(0, low)

        if not np.isnan(std) and not np.isnan(media):
            print(f"Iter {r['iter']:>2} | std = {std:.2f} | {OUTPUT_METRICA}: {media:.2f} ± {3*std:.2f} → [{low:.2f}, {high:.2f}]")
        else:
            print(f"Iter {r['iter']:>2} | std = {std:.2f} | {OUTPUT_METRICA}: nessun dato valido")

    print()
    if salto_rilevato and len(risultati) >= 16:
        print("------ ULTIMA SIMULAZIONE PRIMA DEL SALTO ------\n")
        ultima_simulazione = risultati[-16]
    else:
        print("------ ULTIMA SIMULAZIONE (NESSUN SALTO RILEVATO) ------\n")
        ultima_simulazione = risultati[-1]

    media = ultima_simulazione['media_output']
    std = ultima_simulazione['deviazione_std']
    low = max(0, media - 3 * std)
    high = media + 3 * std

    print("Valore Atteso della variabile testata:\n")
    if low == 0:
        print(f"{OUTPUT_METRICA:<25} = {media:12.2f} + {3*std:.2f}\n")
    else:
        print(f"{OUTPUT_METRICA:<25} = {media:12.2f} ± {3*std:.2f} → [{low:.2f}, {high:.2f}]\n")

    print("Precisione (μ ± 3σ)       =        99.73%")
    print("Livello di Confidenza (μ) =        95.00%\n")
    print("Range angoli (valore centrale ± semiampiezza):\n")

    for k, (v_min, v_max) in ultima_simulazione['ranges'].items():
        centro = (v_min + v_max) / 2
        semiamp = (v_max - v_min) / 2
        print(f" - {k:<23}= {centro:10.2f} ± {semiamp:.2f}")

    # --- RIEPILOGO CON FILTRO SU DEVIAZIONE STANDARD ---
    SOGLIA_STD = 5.0
    filtrati = [r for r in risultati if not np.isnan(r['deviazione_std']) and r['deviazione_std'] <= SOGLIA_STD]

    if filtrati:
        print(f"\n\n------- ULTIMA SIMULAZIONE con dev_std = {SOGLIA_STD} -------:\n")
        migliore = min(filtrati, key=lambda r: r['deviazione_std'])

        media = migliore['media_output']
        std = migliore['deviazione_std']
        low = max(0, media - 3 * std)
        high = media + 3 * std

        if low == 0:
            print(f"{OUTPUT_METRICA:<25} = {media:12.2f} + {3 * std:.2f}\n")
        else:
            print(f"{OUTPUT_METRICA:<25} = {media:12.2f} ± {3 * std:.2f} → [{low:.2f}, {high:.2f}]\n")

        print("Precisione (μ ± 3σ)       =        99.73%")
        print("Livello di Confidenza (μ) =        95.00%\n")
        print("Range angoli (valore centrale ± semiampiezza):\n")

        for k, (v_min, v_max) in migliore['ranges'].items():
            centro = (v_min + v_max) / 2
            semiamp = (v_max - v_min) / 2
            print(f" - {k:<23}= {centro:10.2f} ± {semiamp:.2f}")


    if ultimo_output_valido:
        mostra_grafico_interattivo(ultimo_output_valido)

    return risultati

# --- Avvio ---
if __name__ == "__main__":
    parametri = leggi_range_da_file("dati.txt")
    variabili_da_espandere = seleziona_parametri_da_espandere(parametri)
    targets = leggi_range_da_file("targets.txt")
    listato = leggi_range_da_file("listato.txt")
    OUTPUT_METRICA = seleziona_variabile_da_monitorare(listato)

    risultati = esplora_variabilita_angoli(
        parametri_base=parametri,
        targets=targets,
        variabili_da_espandere=variabili_da_espandere,
        max_iter=int(parametri["max_iter"][0]),
        passo=int(parametri["passo"][0]),
        soglia_salto=int(parametri["soglia_salto"][0])
    )
