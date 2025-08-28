MC-Impact — Interfaccia Web minima

Requisiti
- Python 3.10+
- Dipendenze: Flask, pandas, numpy (già usate dal progetto)

Installazione dipendenze (opzionale)
```
pip install flask pandas numpy
```

Avvio
```
python app.py
```

Uso
- Apri il browser su http://127.0.0.1:5000
- Spunta "Usa i file predefiniti" per usare `dati.txt` e `targets.txt` esistenti nella cartella del progetto, oppure carica i due file.
- (Facoltativo) imposta un override per `N`.
- Clicca "Esegui Monte Carlo" per ottenere:
  - Anteprima delle prime 50 righe valide
  - Migliore combinazione stimata
  - Link per scaricare `risultati.txt` e `listato.txt`

Note
- Questa interfaccia non mostra i grafici interattivi di `plot_risultati.py` né richiede input da terminale (backup). La logica dei calcoli è identica a `main.py` per la parte Monte Carlo e report testuali.
- Le stime potrebbero richiedere tempo in base a `N`.

