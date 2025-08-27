import pandas as pd
from types import SimpleNamespace
from funzioni.lettura_e_scrittura import (
    carica_backup_da_cartella,
    scrivi_range_su_file, scrivi_risultati_formattati
)
from funzioni.plot_risultati import (
    plotta_due_cicloidi, mostra_interfaccia_grafico, plot_triangoli,
    plot_vettori, stimaPDOF, stimaPDOF_triangoli, plot_vettori_con_PDOF
)

# === Carica il backup selezionato ===
risultati_validi, riga_minima = carica_backup_da_cartella()
df_validi = pd.DataFrame(risultati_validi)

# scrivi su risultati
scrivi_risultati_formattati(df_validi, riga_minima, "backup/_risultati.txt")

# === Estrai parametri, targets, listato da riga_minima ===
raw_params = {k[2:]: v for k, v in riga_minima.items() if k.startswith("p_")}
raw_targets = {k[2:]: v for k, v in riga_minima.items() if k.startswith("t_")}
raw_listato = {k[2:]: v for k, v in riga_minima.items() if k.startswith("l_")}

# Converti valori tuple in float (solo primo elemento)
dati_params = {k: float(v[0]) if isinstance(v, (tuple, list)) else float(v) for k, v in raw_params.items()}
dati_targets = {k: float(v[0]) if isinstance(v, (tuple, list)) else float(v) for k, v in raw_targets.items()}
dati_listato = {k: float(v[0]) if isinstance(v, (tuple, list)) else float(v) for k, v in raw_listato.items()}

# === Sovrascrivi i file txt ===
scrivi_range_su_file("backup/_dati.txt", dati_params)
scrivi_range_su_file("backup/_targets.txt", dati_targets)
scrivi_range_su_file("backup/_listato.txt", dati_listato)

# === Convertili in SimpleNamespace ===
dati_params = SimpleNamespace(**dati_params)
dati_targets = SimpleNamespace(**dati_targets)
dati_listato = SimpleNamespace(**dati_listato)

# === Flag principali ===
chiusura = int(dati_params.chiusura_triangoli)
gradi = int(dati_params.gdl)
cicloide_avanzata = int(dati_params.cicloide_avanzata)
stima_PDOF = int(dati_params.stima_PDOF)

# === Plot ===
if chiusura == 1:
    plot_vettori_con_PDOF(
        riga_minima['V1_pre'], riga_minima['theta1_in'],
        riga_minima['V1_post'], riga_minima['theta1_out'],
        riga_minima['V2_pre'], riga_minima['theta2_in'],
        riga_minima['V2_post'], riga_minima['theta2_out'],
        riga_minima['x1'], riga_minima['y1'],
        riga_minima['x2'], riga_minima['y2'],
        riga_minima['PDOF_stima']
    )
    plot_triangoli(
        riga_minima['V1_pre'], riga_minima['theta1_in_t'],
        riga_minima['V1_post_t'], riga_minima['theta1_out_t'],
        riga_minima['V2_pre'], riga_minima['theta2_in_t'],
        riga_minima['V2_post_t'], riga_minima['theta2_out_t'],
        riga_minima['x1'], riga_minima['y1'],
        riga_minima['x2'], riga_minima['y2'],
        riga_minima['PDOF_eff']
    )
    stimaPDOF_triangoli(dati_params, dati_listato)

elif chiusura == 0:
    if stima_PDOF == 1:
        plot_vettori_con_PDOF(
            riga_minima['V1_pre'], riga_minima['theta1_in'],
            riga_minima['V1_post'], riga_minima['theta1_out'],
            riga_minima['V2_pre'], riga_minima['theta2_in'],
            riga_minima['V2_post'], riga_minima['theta2_out'],
            riga_minima['x1'], riga_minima['y1'],
            riga_minima['x2'], riga_minima['y2'],
            riga_minima['PDOF_stima']
        )
        stimaPDOF(dati_params, dati_listato, dati_targets)
    else:
        plot_vettori(
            riga_minima['V1_pre'], riga_minima['theta1_in'],
            riga_minima['V1_post'], riga_minima['theta1_out'],
            riga_minima['V2_pre'], riga_minima['theta2_in'],
            riga_minima['V2_post'], riga_minima['theta2_out'],
            riga_minima['x1'], riga_minima['y1'],
            riga_minima['x2'], riga_minima['y2']
        )

# === Plot cicloidi, se presenti ===
campi_cicloide = [
    'punti_cicloide_x1', 'punti_cicloide_y1', 'cicloide_th1', 'cicloide_R1', 'cicloide1',
    'punti_cicloide_x2', 'punti_cicloide_y2', 'cicloide_th2', 'cicloide_R2', 'cicloide2',
    'cic_A1', 'cic_B1', 'cic_A2', 'cic_B2'
]

mancanti = [k for k in campi_cicloide if k not in riga_minima]
if cicloide_avanzata == 1 and not mancanti:
    plotta_due_cicloidi(
        riga_minima['cic_A1'], riga_minima['cic_B1'],
        riga_minima['punti_cicloide_x1'], riga_minima['punti_cicloide_y1'], riga_minima['cicloide_th1'],
        riga_minima['cicloide_R1'], riga_minima['cicloide1'], riga_minima['theta1_out'],
        riga_minima['cic_A2'], riga_minima['cic_B2'],
        riga_minima['punti_cicloide_x2'], riga_minima['punti_cicloide_y2'], riga_minima['cicloide_th2'],
        riga_minima['cicloide_R2'], riga_minima['cicloide2'], riga_minima['theta2_out']
    )
elif cicloide_avanzata == 1 and mancanti:
    print(f"⚠️  Le cicloidi non erano presenti nel backup selezionato. Campi mancanti: {mancanti}")

# === Variabili plottabili (escludi colonne di errore) ===
variabili = [col for col in df_validi.columns if not col.lower().startswith(('errore_', 'err_'))]

# === Plot interattivo ===
mostra_interfaccia_grafico(df_validi, riga_minima, variabili)
