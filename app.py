from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
import pickle
import json
import math

# Reuse existing project logic
import pandas as pd
from funzioni.lettura_e_scrittura import (
    leggi_range_da_file,
    genera_input_casuale,
    scrivi_risultati_su_file,
    scrivi_risultati_formattati,
)
from funzioni.operazioni import calcoli

app = Flask(__name__)
app.secret_key = "mc-impact-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def run_simulation(dati_path: str, targets_path: str, override_N: int | None = None):
    """Run the same Monte Carlo flow as main.py but without plots or prompts.

    Returns (df_validi, riga_minima) where df_validi is a pandas DataFrame
    and riga_minima is a pandas Series (best row).
    Also writes listato.txt and risultati.txt in project root.
    """
    # Read ranges
    parametri = leggi_range_da_file(dati_path)
    targets = leggi_range_da_file(targets_path)

    # Number of iterations
    if override_N is not None:
        N = int(override_N)
    else:
        N = int(parametri.get("N", (1000, 1000))[0])  # default fallback

    risultati_validi = []

    for i in range(N):
        p = genera_input_casuale(parametri)
        t = genera_input_casuale(targets)
        try:
            res, err = calcoli(p, t)
            # accept only if all custom error flags < 1
            if all(val < 1 for key, val in err.items() if "errore_" in key.lower()):
                risultati_validi.append({**res, **err})
        except Exception:
            # ignore failing combinations just like main.py
            continue

    if not risultati_validi:
        raise RuntimeError("Nessuna combinazione soddisfa i criteri nei limiti impostati.")

    # Build DataFrame and compute mean error
    df_validi = pd.DataFrame(risultati_validi)
    colonne_errori = [k for k in risultati_validi[0].keys() if k.startswith("err_")]
    if colonne_errori:
        df_validi["err_medio%"] = df_validi[colonne_errori].mean(axis=1)

    # Selection rule based on stima_PDOF flag in dati.txt
    stima_PDOF = int(parametri.get("stima_PDOF", (0, 0))[0])
    if stima_PDOF == 1 and "err_PDOF" in df_validi.columns:
        riga_minima = df_validi.loc[df_validi["err_PDOF"].idxmin()]
    else:
        # fallback to mean error if available, else first row
        if "err_medio%" in df_validi.columns:
            riga_minima = df_validi.loc[df_validi["err_medio%"].idxmin()]
        else:
            riga_minima = df_validi.iloc[0]

    # Write textual outputs (match local behavior)
    scrivi_risultati_su_file(df_validi, riga_minima)
    scrivi_risultati_formattati(df_validi, riga_minima)

    return df_validi, riga_minima, parametri, targets


def to_jsonable(obj):
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    # Pandas series
    if hasattr(obj, 'to_dict'):
        return to_jsonable(obj.to_dict())
    return str(obj)


@app.route("/")
def index():
    # Expose defaults and a minimal form
    defaults = {
        "has_default_dati": os.path.exists(os.path.join(BASE_DIR, "dati.txt")),
        "has_default_targets": os.path.exists(os.path.join(BASE_DIR, "targets.txt")),
    }
    return render_template("index.html", defaults=defaults)


@app.route("/run", methods=["POST"]) 
def run():
    override_N = request.form.get("N")
    override_N = int(override_N) if (override_N and override_N.isdigit()) else None

    # Decide which files to use
    use_defaults = request.form.get("use_defaults") == "on"

    if use_defaults:
        dati_path = os.path.join(BASE_DIR, "dati.txt")
        targets_path = os.path.join(BASE_DIR, "targets.txt")
        if not (os.path.exists(dati_path) and os.path.exists(targets_path)):
            flash("File dati.txt/targets.txt non trovati nella cartella del progetto.", "error")
            return redirect(url_for("index"))
    else:
        dati_file = request.files.get("dati_file")
        targets_file = request.files.get("targets_file")
        if not dati_file or not targets_file or dati_file.filename == "" or targets_file.filename == "":
            flash("Carica sia dati.txt che targets.txt o seleziona l'opzione predefinita.", "error")
            return redirect(url_for("index"))
        # Save uploads
        dati_name = secure_filename(dati_file.filename) or "dati.txt"
        targets_name = secure_filename(targets_file.filename) or "targets.txt"
        dati_path = os.path.join(UPLOAD_DIR, dati_name)
        targets_path = os.path.join(UPLOAD_DIR, targets_name)
        dati_file.save(dati_path)
        targets_file.save(targets_path)

    try:
        df_validi, riga_minima, parametri, targets = run_simulation(dati_path, targets_path, override_N)
    except Exception as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    # Compute helper lists
    colonne_errori = [c for c in df_validi.columns if c.lower().startswith("errore_") or c.lower().startswith("err_")]
    variabili = [c for c in df_validi.columns if not (c.lower().startswith("errore_") or c.lower().startswith("err_"))]

    # Persist run on disk for interactive API usage
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "df.pkl"), "wb") as f:
        pickle.dump(df_validi, f)
    with open(os.path.join(run_dir, "best.pkl"), "wb") as f:
        pickle.dump(riga_minima, f)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "variabili": variabili,
            "colonne_errori": colonne_errori,
            "parametri": to_jsonable(parametri),
            "targets": to_jsonable(targets),
        }, f)

    # Prepare minimal summary for the page
    ignora_prefix = ("punti_cicloide_",)
    ignora_nomi = {
        "cicloide_th", "cicloide_R",
        "cic_A1", "cic_B1", "cic_A2", "cic_B2",
        "cicloide_th1", "cicloide_th2", "cicloide_R1", "cicloide_R2"
    }
    best = {}
    for k, v in riga_minima.items():
        ks = str(k)
        if ks.startswith("err_") or ks.startswith("errore_"):
            continue
        if any(ks.startswith(p) for p in ignora_prefix) or ks in ignora_nomi:
            continue
        if isinstance(v, (int, float)):
            best[k] = round(v, 2)
        else:
            best[k] = v

    # Build graph inputs (only necessary fields)
    def get_val(name, default=None):
        return riga_minima[name] if name in riga_minima else default

    graph_inputs = {
        "vectors": {
            "V1_pre": get_val('V1_pre'),
            "theta1_in": get_val('theta1_in'),
            "V1_post": get_val('V1_post'),
            "theta1_out": get_val('theta1_out'),
            "V2_pre": get_val('V2_pre'),
            "theta2_in": get_val('theta2_in'),
            "V2_post": get_val('V2_post'),
            "theta2_out": get_val('theta2_out'),
            "x1": get_val('x1'),
            "y1": get_val('y1'),
            "x2": get_val('x2'),
            "y2": get_val('y2'),
            "PDOF_stima": get_val('PDOF_stima'),
            "J1": get_val('J1'),
            "J2": get_val('J2'),
            "omega1_pre": get_val('omega1_pre'),
            "omega1_post": get_val('omega1_post'),
            "omega2_pre": get_val('omega2_pre'),
            "omega2_post": get_val('omega2_post')
        },
        "triangoli": {
            "V1_pre": get_val('V1_pre'),
            "theta1_in_t": get_val('theta1_in_t'),
            "V1_post_t": get_val('V1_post_t'),
            "theta1_out_t": get_val('theta1_out_t'),
            "V2_pre": get_val('V2_pre'),
            "theta2_in_t": get_val('theta2_in_t'),
            "V2_post_t": get_val('V2_post_t'),
            "theta2_out_t": get_val('theta2_out_t'),
            "x1": get_val('x1'),
            "y1": get_val('y1'),
            "x2": get_val('x2'),
            "y2": get_val('y2'),
            "PDOF_eff": get_val('PDOF_eff')
        },
        "cicloidi": {},
        "pdof": {}
    }

    # cicloidi avanzate se disponibili
    if 'punti_cicloide_x1' in riga_minima and 'punti_cicloide_y1' in riga_minima:
        try:
            graph_inputs["cicloidi"] = {
                "x1": to_jsonable(get_val('punti_cicloide_x1')),
                "y1": to_jsonable(get_val('punti_cicloide_y1')),
                "A1": to_jsonable(get_val('cic_A1')),
                "B1": to_jsonable(get_val('cic_B1')),
                "theta1": get_val('cicloide_th1'),
                "R1": get_val('cicloide_R1'),
                "L1": get_val('cicloide1'),
                "ang1": get_val('theta1_out'),
                "x2": to_jsonable(get_val('punti_cicloide_x2')),
                "y2": to_jsonable(get_val('punti_cicloide_y2')),
                "A2": to_jsonable(get_val('cic_A2')),
                "B2": to_jsonable(get_val('cic_B2')),
                "theta2": get_val('cicloide_th2'),
                "R2": get_val('cicloide_R2'),
                "L2": get_val('cicloide2'),
                "ang2": get_val('theta2_out')
            }
        except Exception:
            pass

    # pdof target/stima semplice (polar)
    try:
        pdof_target = None
        if 'PDOF' in targets:
            tval = targets['PDOF']
            if isinstance(tval, (tuple, list)):
                pdof_target = float(tval[0])
        graph_inputs["pdof"] = {
            "target": pdof_target,
            "stima": get_val('PDOF_stima'),
            "eff": get_val('PDOF_eff')
        }
    except Exception:
        pass

    # Limit preview rows to avoid overloading the page
    preview = df_validi.head(50).to_dict(orient="records")

    # Persist graph_inputs for fullscreen pages
    try:
        with open(os.path.join(run_dir, "graph_inputs.json"), "w", encoding="utf-8") as gf:
            json.dump(to_jsonable(graph_inputs), gf)
    except Exception:
        pass

    return render_template(
        "results.html",
        best=best,
        rows=preview,
        total=len(df_validi),
        run_id=run_id,
        variabili=variabili,
        colonne_errori=colonne_errori,
        graph_inputs=json.dumps(to_jsonable(graph_inputs)),
        files_available={"risultati.txt": os.path.exists(os.path.join(BASE_DIR, "risultati.txt")),
                         "listato.txt": os.path.exists(os.path.join(BASE_DIR, "listato.txt"))}
    )


@app.route("/download/<path:filename>")
def download(filename):
    # Serve output files from project root
    return send_from_directory(BASE_DIR, filename, as_attachment=True)

@app.route("/api/run/<run_id>/graph_inputs")
def api_graph_inputs(run_id):
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    path = os.path.join(run_dir, "graph_inputs.json")
    if not os.path.exists(path):
        return jsonify({"error": "run non trovato"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # also include meta for parametri/targets if present
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        data["_meta"] = meta
    return jsonify(data)

@app.route("/plot/<kind>/<run_id>")
def plot_full(kind, run_id):
    # kind: scatter | vectors | triangoli | cicloidi | pdof | pdof_tri
    return render_template("plot_full.html", kind=kind, run_id=run_id)


@app.route("/api/run/<run_id>/scatter")
def api_scatter(run_id):
    x = request.args.get("x")
    y = request.args.get("y")
    c = request.args.get("c")
    limit = int(request.args.get("limit", "10000"))
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    pkl = os.path.join(run_dir, "df.pkl")
    if not os.path.exists(pkl):
        return jsonify({"error": "run non trovato"}), 404
    with open(pkl, "rb") as f:
        df = pickle.load(f)
    if x not in df.columns or y not in df.columns or c not in df.columns:
        return jsonify({"error": "colonna non trovata"}), 400
    df2 = df[[x, y, c]].dropna().head(limit)
    return jsonify({
        "x": to_jsonable(df2[x].tolist()),
        "y": to_jsonable(df2[y].tolist()),
        "c": to_jsonable(df2[c].tolist())
    })


if __name__ == "__main__":
    # Debug server for local use
    app.run(debug=True, port=5000)
