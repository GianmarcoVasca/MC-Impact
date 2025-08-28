import pandas as pd
from funzioni.operazioni import calcoli
from funzioni.lettura_e_scrittura import salva_backup_versionato, leggi_range_da_file, genera_input_casuale, scrivi_risultati_su_file, barra_avanzamento, scrivi_risultati_formattati
from funzioni.plot_risultati import plotta_due_cicloidi, mostra_interfaccia_grafico, plot_triangoli, plot_vettori, stimaPDOF, stimaPDOF_triangoli, plot_vettori_con_PDOF
from types import SimpleNamespace
import time
pd.options.mode.chained_assignment = None  # Disabilita warning

# lettura dati da file
parametri = leggi_range_da_file("dati.txt")
targets = leggi_range_da_file("targets.txt")

# Esecuzione Montecarlo
N = int(parametri["N"][0])
risultati_validi = []

start_time = time.time()

for i in range(N):
    p = genera_input_casuale(parametri)
    t = genera_input_casuale(targets)
    try:
        res, err = calcoli(p, t)
        if all(val < 1 for key, val in err.items() if 'errore_' in key.lower()):
            risultati_validi.append({**res, **err})
    except:
        continue

    if i % 500 == 0 or i == N - 1:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed  # iterazioni/secondo
        remaining = (N - i - 1) / rate
        eta_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
        barra_avanzamento(i + 1, N, eta_str)

print()  # newline finale

if not risultati_validi:
    print("⚠️   Nessuna combinazione soddisfa i criteri")

# Conversione in DataFrame
df_validi = pd.DataFrame(risultati_validi)

# Calcolo dell'errore medio %
colonne_errori = [key for key in risultati_validi[0].keys()
                  if key.startswith("err_")]
df_validi['err_medio%'] = df_validi[colonne_errori].mean(axis=1)

# Trova la riga con errore medio più basso
stima_PDOF = int(parametri["stima_PDOF"][0])
if stima_PDOF == 1:
    riga_minima = df_validi.loc[df_validi['err_PDOF'].idxmin()]
elif stima_PDOF == 0:
    riga_minima = df_validi.loc[df_validi['err_medio%'].idxmin()]

# scrivi i risultati
scrivi_risultati_su_file(df_validi, riga_minima)
scrivi_risultati_formattati(df_validi, riga_minima)

#leggi files testo per grafici
# lettura dati da file
dati_listato = SimpleNamespace(**leggi_range_da_file("listato.txt"))
dati_params = SimpleNamespace(**leggi_range_da_file("dati.txt"))
dati_targets = SimpleNamespace(**leggi_range_da_file("targets.txt"))
# Conversione in scalare
dati_listato = SimpleNamespace(**{k: float(v[0]) if isinstance(v, tuple) else float(v)
                                    for k, v in vars(dati_listato).items()})
dati_params = SimpleNamespace(**{k: float(v[0]) if isinstance(v, tuple) else float(v)
                                    for k, v in vars(dati_params).items()})
dati_targets = SimpleNamespace(**{k: float(v[0]) if isinstance(v, tuple) else float(v)
                                    for k, v in vars(dati_targets).items()})

chiusura = int(parametri["chiusura_triangoli"][0])
gradi = int(parametri["gdl"][0])
cicloide_avanzata = int(parametri["cicloide_avanzata"][0])
if chiusura == 1:
    plot_vettori_con_PDOF(
        riga_minima['V1_pre'],         riga_minima['theta1_in'],
        riga_minima['V1_post'],        riga_minima['theta1_out'],
        riga_minima['V2_pre'],         riga_minima['theta2_in'],
        riga_minima['V2_post'],        riga_minima['theta2_out'],
        riga_minima['x1'],             riga_minima['y1'],
        riga_minima['x2'],             riga_minima['y2'],
        riga_minima['PDOF_stima']
    )
    plot_triangoli(
        riga_minima['V1_pre'],         riga_minima['theta1_in_t'],
        riga_minima['V1_post_t'],        riga_minima['theta1_out_t'],
        riga_minima['V2_pre'],         riga_minima['theta2_in_t'],
        riga_minima['V2_post_t'],        riga_minima['theta2_out_t'],
        riga_minima['x1'],             riga_minima['y1'],
        riga_minima['x2'],             riga_minima['y2'],
        riga_minima['PDOF_eff']
    )
    stimaPDOF_triangoli(dati_params, dati_listato)
elif chiusura == 0:
    if stima_PDOF == 1:
        plot_vettori_con_PDOF(
            riga_minima['V1_pre'],         riga_minima['theta1_in'],
            riga_minima['V1_post'],        riga_minima['theta1_out'],
            riga_minima['V2_pre'],         riga_minima['theta2_in'],
            riga_minima['V2_post'],        riga_minima['theta2_out'],
            riga_minima['x1'],             riga_minima['y1'],
            riga_minima['x2'],             riga_minima['y2'],
            riga_minima['PDOF_stima']
        )
        stimaPDOF(dati_params, dati_listato, dati_targets)
    elif stima_PDOF == 0:
        plot_vettori(
            riga_minima['V1_pre'],         riga_minima['theta1_in'],
            riga_minima['V1_post'],        riga_minima['theta1_out'],
            riga_minima['V2_pre'],         riga_minima['theta2_in'],
            riga_minima['V2_post'],        riga_minima['theta2_out'],
            riga_minima['x1'],             riga_minima['y1'],
            riga_minima['x2'],             riga_minima['y2']
        )
if cicloide_avanzata == 1:
    plotta_due_cicloidi(riga_minima['cic_A1'], riga_minima['cic_B1'], 
                        riga_minima['punti_cicloide_x1'], riga_minima['punti_cicloide_y1'], riga_minima['cicloide_th1'], 
                        riga_minima['cicloide_R1'], riga_minima['cicloide1'], riga_minima['theta1_out'],
                        riga_minima['cic_A2'], riga_minima['cic_B2'], 
                        riga_minima['punti_cicloide_x2'], riga_minima['punti_cicloide_y2'], riga_minima['cicloide_th2'], 
                        riga_minima['cicloide_R2'], riga_minima['cicloide2'], riga_minima['theta2_out'])

# Lista variabili plottabili
variabili = [col for col in df_validi.columns if not col.lower().startswith(('errore_', 'err_'))]

# plot interattivo
mostra_interfaccia_grafico(df_validi, riga_minima, variabili)