MC-Impact

Requisiti
- Python 3.10+
- Dipendenze elencate in `requirements.txt` (Flask, pandas, numpy, ecc.)

Installazione dipendenze (opzionale)
```
pip install -r requirements.txt
```

Avvio
Prima di avviare l'applicazione è necessario impostare la variabile d'ambiente `SECRET_KEY`.
Per scegliere il backend di persistenza del rate limiter è possibile impostare anche
`RATELIMIT_STORAGE_URI` (di default `memory://`, ad esempio `redis://localhost:6379` in produzione):
```
export SECRET_KEY=9f6c7a42d8b19e34a1c56fbb0e2a48cf7fbe13d5c1a27c89de47b0f32f9a8c3d
export RATELIMIT_STORAGE_URI=memory://
python app.py
```

Uso
- Apri il browser su http://127.0.0.1:5000`.

