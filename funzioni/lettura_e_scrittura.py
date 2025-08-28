import random
from types import SimpleNamespace
import pandas as pd
import os
import numpy as np
from datetime import datetime

#leggere dati in input da file
def leggi_range_da_file(percorso):
    ranges = {}
    with open(percorso, 'r') as f:
        for riga in f:
            riga = riga.strip()
            if not riga or riga.lstrip().startswith('#'):
                continue
            if '=' not in riga:
                continue
            nome, valori_str = riga.split('=')
            nome = nome.strip()
            valori_str = valori_str.strip()

            # Salta righe con array lunghi (probabile export numpy)
            if '[' in valori_str and '...' in valori_str:
                continue

            parts = [v.strip() for v in valori_str.split(',')]
            try:
                if len(parts) == 1:
                    val = float(parts[0])
                    ranges[nome] = (val, val)
                elif len(parts) == 2:
                    ranges[nome] = (float(parts[0]), float(parts[1]))
                else:
                    raise ValueError(f"Formato non valido in dati.txt per la voce '{nome}'")
            except ValueError:
                # print(f"[DEBUG] Voce ignorata: {nome} = {valori_str}")
                continue

    return ranges

# generatore di combinazioni in input con passo 0.01
def genera_input_casuale(ranges):
    valori = {
        k: random.choice(np.arange(v[0], v[1] + 0.001, 0.01))
        for k, v in ranges.items()
    }
    params = SimpleNamespace(**valori)
    return params

#scrivere i risultati
def scrivi_risultati_su_file(df, riga, nome_file="listato.txt"):
    with open(nome_file, "w", encoding="utf-8") as f:
        f.write("Combinazione con errore medio più basso:\n")
        for col in df.columns:
            valore = riga[col]
            if isinstance(valore, (int, float)):
                valore = round(valore, 2)
            f.write(f"{col} = {valore}\n")

def barra_avanzamento(i, N, eta_str):
    percentuale = int((i / N) * 100)
    barra = '=' * (percentuale // 2) + ' ' * (50 - percentuale // 2)
    print(f"\r[{barra}] {percentuale}% - Tempo stimato: {eta_str}", end='')

def scrivi_risultati_formattati(df, riga, nome_file="risultati.txt"):
    with open(nome_file, "w", encoding="utf-8") as f:
        f.write("Combinazione con errore medio più basso:\n\n")

        ignora_prefix = ("punti_cicloide_",)
        ignora_nomi = {
            "cicloide_th", "cicloide_R",
            "cic_A1", "cic_B1", "cic_A2", "cic_B2", 
            "cicloide_th1", "cicloide_th2", "cicloide_R1", "cicloide_R2"
        }

        count = 0
        for col in df.columns:
            if col.startswith("err"):
                continue
            if any(col.startswith(p) for p in ignora_prefix) or col in ignora_nomi:
                continue
            if col in riga:
                valore = riga[col]

                # Gestione sicura del tipo di valore
                if isinstance(valore, (int, float)):
                    valore_str = f"{round(valore, 2):>10}"
                elif hasattr(valore, "__iter__") and not isinstance(valore, str):
                    valore_str = str(list(valore))
                else:
                    valore_str = str(valore)

                f.write(f"{col:<15}= \t{valore_str}\n")
                count += 1
                if count % 4 == 0:
                    f.write("\n")

def salva_backup_versionato(risultati_validi, riga_minima, backup_dir="backup"):
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_validi = os.path.join(backup_dir, f"risultati_validi_{timestamp}.pkl")
    file_minima = os.path.join(backup_dir, f"riga_minima_{timestamp}.pkl")

    with open(file_validi, "wb") as f:
        pickle.dump(risultati_validi, f)
    with open(file_minima, "wb") as f:
        pickle.dump(riga_minima, f)

    print(f"✅ Backup salvato in '{backup_dir}/' con timestamp {timestamp}")

def carica_backup_da_cartella(backup_dir="backup"):
    files = sorted([f for f in os.listdir(backup_dir) if f.endswith(".pkl")])
    if not files:
        print(f"❌ Nessun file .pkl trovato nella cartella '{backup_dir}/'")
        return [], {}

    coppie = [(f1, f2) for f1 in files for f2 in files
              if f1.startswith("risultati_validi_") and f2.startswith("riga_minima_")
              and f1[-19:] == f2[-19:]]
    if not coppie:
        print("⚠️  Nessuna coppia valida trovata.")
        return [], {}

    coppie = sorted(coppie, key=lambda x: x[0][-19:], reverse=True)

    print(f"\n📂 Backup disponibili in '{backup_dir}/':")
    for idx, (f1, f2) in enumerate(coppie):
        print(f"  [{idx}] {f1} + {f2}")

    while True:
        scelta = input(f"\nSeleziona il numero del backup da caricare [0-{len(coppie)-1}]: ")
        if scelta.isdigit() and 0 <= int(scelta) < len(coppie):
            break
        print("❌ Input non valido.")

    file_validi, file_minima = coppie[int(scelta)]

    with open(os.path.join(backup_dir, file_validi), "rb") as f:
        risultati_validi = pickle.load(f)
    with open(os.path.join(backup_dir, file_minima), "rb") as f:
        riga_minima = pickle.load(f)

    print(f"\n✅ Backup '{file_validi}' caricato correttamente.")
    return risultati_validi, riga_minima

def scrivi_range_su_file(nome_file, dizionario):
    with open(nome_file, "w") as f:
        for k, v in dizionario.items():
            if isinstance(v, (tuple, list)):
                val_str = ", ".join(str(x) for x in v)
            else:
                val_str = str(v)
            f.write(f"{k} = {val_str}\n")




