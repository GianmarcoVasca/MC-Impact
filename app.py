from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
import json
import math
import time
import threading
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Reuse existing project logic
import pandas as pd
from funzioni.lettura_e_scrittura import (
    leggi_range_da_file,
    genera_input_casuale,
    scrivi_risultati_su_file,
    scrivi_risultati_formattati,
)
from funzioni.operazioni import calcoli

from multiprocessing import Manager, freeze_support

def create_manager():
    return Manager()

app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    app.logger.critical("SECRET_KEY not set")
    raise RuntimeError("SECRET_KEY not set")
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

ALLOWED_UPLOADS = {".txt"}
ALLOWED_PAYLOAD = {".json"}
ALLOWED_DOWNLOADS = {"risultati.txt", "listato.txt"}
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "700000"))
SIMULATION_TIMEOUT = int(os.environ.get("SIMULATION_TIMEOUT", "60"))
MAX_WORKERS = int(os.environ.get("SIM_WORKERS", "2"))
MAX_QUEUE = int(os.environ.get("SIM_QUEUE", "4"))
UPLOAD_TTL_SECONDS = 3600  # 1 hour

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

manager = Manager()
PROGRESS: dict[str, dict] = manager.dict()
PROGRESS_EXPLORA: dict[str, dict] = manager.dict()
EXECUTOR = ProcessPoolExecutor(max_workers=MAX_WORKERS)
PENDING_FUTURES: dict[str, object] = {}
PENDING_LOCK = threading.Lock()

# Unita' di misura di base per alcune variabili note
UNITS: dict[str, str] = {
    # Veicoli
    "m1": "kg", "m2": "kg",
    "l1": "m", "l2": "m", "p1": "m", "p2": "m",
    # Post-urto
    "d_post1": "m", "d_post2": "m",
    "theta_post1": "deg", "theta_post2": "deg",
    "f1": "", "f2": "",
    # Cicloide nota / avanzata
    "lunghezza_cicloide_1": "m", "lunghezza_cicloide_2": "m",
    "x1_quiete": "m", "y1_quiete": "m", "x2_quiete": "m", "y2_quiete": "m",
    # No-cicloide
    "V1_post_Kmh": "km/h", "V2_post_Kmh": "km/h",
    # VelocitÃ  in km/h (best summary)
    "V1_pre": "Km/h", "V2_pre": "Km/h", "V1_post": "Km/h", "V2_post": "Km/h",
    "omega1_post": "rad/s", "omega2_post": "rad/s",
    "omega1_pre": "rad/s", "omega2_pre": "rad/s",
    # Energia
    "EES1_Kmh": "km/h", "EES2_Kmh": "km/h", "Ed_target": "J", "Ed": "J",
    # Momenti d'inerzia
    "J1": "Kg*m^2", "J2": "Kg*m^2",
    # Angoli
    "theta1_in": "deg", "theta1_out": "deg", "theta2_in": "deg", "theta2_out": "deg",
    # Coordinate e lunghezze
    "x1": "m", "y1": "m", "x2": "m", "y2": "m",
    "cicloide1": "m", "cicloide2": "m",
    # PDOF
    "PDOF_stima": "deg", "PDOF_eff": "deg",
}

def cleanup_uploads(keep_run_id: str | None = None):
    """Remove old uploaded files and run directories except the one to keep."""
    if not os.path.isdir(UPLOAD_DIR):
        return
    now = time.time()
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isdir(path) and name.startswith("run_"):
                rid = name[4:]
                if keep_run_id and rid == keep_run_id:
                    continue
                if now - os.path.getmtime(path) > UPLOAD_TTL_SECONDS:
                    shutil.rmtree(path)
            elif os.path.isfile(path) and name.endswith((".txt", ".json")):
                if now - os.path.getmtime(path) > UPLOAD_TTL_SECONDS:
                    os.remove(path)
        except Exception:
            pass


def _valid_run_id(run_id: str) -> bool:
    try:
        uuid.UUID(run_id)
        return True
    except Exception:
        return False


def _format_used_params(params: dict | None, targets: dict | None):
    """Merge params and targets and format values for display.

    - If a value is a 2-tuple/list with equal endpoints, show only the first.
    - For specific flag keys, show 'attivo' for 1 and 'disattivo' for 0
    - Ignore keys not relevant to summary.
    """
    params = params or {}
    targets = targets or {}

    ignore_keys = {"max_iter", "passo", "soglia_salto"}
    flag_keys = {
        "cicloide", "cicloide_avanzata", "cicloide_nota",
        "energia_EES", "stima_PDOF", "chiusura_triangoli",
    }

    merged: dict[str, object] = {}
    # Prefer values from params when duplicated
    for src in (params, targets):
        for k, v in src.items():
            if k in ignore_keys:
                continue
            merged.setdefault(str(k), v)

    def fmt_value(k: str, v: object) -> str:
        # Normalize tuples/lists
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            a, b = v[0], v[1]
            try:
                equal = (a == b)
            except Exception:
                equal = False
            v0 = a
            if k in flag_keys:
                try:
                    flag = int(a)
                except Exception:
                    flag = 0
                return "attivo" if flag == 1 else "disattivo"
            if equal:
                return str(v0)
            return f"({a}, {b})"
        # Single value
        if k in flag_keys:
            try:
                flag = int(v)
            except Exception:
                flag = 0
            return "attivo" if flag == 1 else "disattivo"
        return str(v)

    formatted = {k: fmt_value(k, v) for k, v in merged.items()}
    # Keep deterministic ordering (alphabetical)
    return dict(sorted(formatted.items(), key=lambda kv: kv[0].lower()))


def _group_used_params(formatted: dict[str, str]):
    """Return a list of (section, OrderedDict) preserving a manual-like order."""
    from collections import OrderedDict

    GROUPS: list[tuple[str, list[str]]] = [
        ("Simulazione", ["N", "gdl"]),
        ("Flag", [
            "cicloide", "cicloide_avanzata", "cicloide_nota",
            "energia_EES", "stima_PDOF", "chiusura_triangoli",
        ]),
        ("Post-urto (cicloide)", [
            "d_post1", "d_post2", "theta_post1", "theta_post2", "f1", "f2"
        ]),
        ("Post-urto avanzato", [
            "x1_quiete", "y1_quiete", "x2_quiete", "y2_quiete"
        ]),
        ("Post-urto (no-cicloide)", [
            "V1_post_Kmh", "V2_post_Kmh", "omega1_post", "omega2_post"
        ]),
        ("Cicloide nota", [
            "lunghezza_cicloide_1", "lunghezza_cicloide_2"
        ]),
        ("Energia EES", ["EES1_Kmh", "EES2_Kmh"]),
        ("Coordinate baricentriche", ["x1", "y1", "x2", "y2"]),
        ("Targets fissi", ["PDOF", "omega1_pre", "omega2_pre"]),
    ]

    used = set()
    out: list[tuple[str, OrderedDict[str, str]]] = []
    for title, keys in GROUPS:
        section = OrderedDict()
        for k in keys:
            if k in formatted:
                section[k] = formatted[k]
                used.add(k)
        if section:
            out.append((title, section))

    # Remaining keys go into "Altri"
    remaining = OrderedDict((k, v) for k, v in formatted.items() if k not in used)
    if remaining:
        out.append(("Altri", remaining))
    return out


def run_simulation(
    dati_path: str,
    targets_path: str,
    override_N: int | None = None,
    progress_key: str | None = None,
    timeout_seconds: int = SIMULATION_TIMEOUT,
):
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
    if N > MAX_ITERATIONS:
        raise ValueError(f"N massimo consentito {MAX_ITERATIONS}")

    risultati_validi = []

    start_ts = time.time()
    for i in range(N):
        if timeout_seconds and (time.time() - start_ts) > timeout_seconds:
            raise TimeoutError("Tempo massimo di esecuzione superato")
        # Check for async cancellation
        if progress_key and PROGRESS.get(progress_key, {}).get("cancel"):
            PROGRESS[progress_key] = {**PROGRESS.get(progress_key, {}), "done": True, "canceled": True}
            raise RuntimeError("Esecuzione annullata dall'utente")
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

        # Aggiorna stato di avanzamento
        if progress_key:
            now = time.time()
            done = i + 1
            elapsed = max(0.0, now - start_ts)
            if done > 0 and elapsed > 2.0:  # ignora stima nei primi 2 secondi
                rate = done / elapsed
                remaining = (N - done) / rate if rate > 0 else None
            else:
                remaining = None
            if done >= N:
                remaining = 0
            PROGRESS[progress_key] = {
                "current": done,
                "total": N,
                "elapsed": elapsed,
                "eta_seconds": remaining,
                "started": start_ts,
            }

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


@app.route("/home")
def index():
    # Expose defaults and a minimal form
    defaults = {
        "has_default_dati": os.path.exists(os.path.join(BASE_DIR, "dati.txt")),
        "has_default_targets": os.path.exists(os.path.join(BASE_DIR, "targets.txt")),
    }
    return render_template("index.html", defaults=defaults)

@app.route("/")
def intro():
    return render_template("intro.html")


@app.route("/logo.png")
def logo():
    """Backward-compat route: serve the historical single logo."""
    path = os.path.join(BASE_DIR, "templates")
    return send_from_directory(path, "logo.png", as_attachment=False)

@app.route("/logo_scritta.png")
def logo_scritta():
    """Serve the left brand image (scritta)."""
    path = os.path.join(BASE_DIR, "templates")
    return send_from_directory(path, "logo_scritta.png", as_attachment=False)

@app.route("/logo_cars.png")
def logo_cars():
    """Serve the right brand image (cars)."""
    path = os.path.join(BASE_DIR, "templates")
    return send_from_directory(path, "logo_cars.png", as_attachment=False)

@app.route("/bg.png")
def bg_image():
    """Serve the background sample image placed under templates/sfondo.png"""
    path = os.path.join(BASE_DIR, "templates")
    return send_from_directory(path, "sfondo.png", as_attachment=False)

# Favicon for all pages (served from templates/favicon.jpg)
@app.route("/favicon.ico")
def favicon():
    # Many browsers auto-request /favicon.ico; redirect to our JPEG favicon
    return redirect(url_for('favicon_jpg'))

@app.route("/favicon.jpg")
def favicon_jpg():
    path = os.path.join(BASE_DIR, "templates")
    return send_from_directory(path, "favicon.jpg", as_attachment=False, mimetype="image/jpeg")

@app.route("/api/cancel/<run_id>", methods=["POST"])
def api_cancel(run_id):
    if not _valid_run_id(run_id) or run_id not in PROGRESS:
        return jsonify({"error": "run non trovato"}), 404
    PROGRESS[run_id] = {**PROGRESS.get(run_id, {}), "cancel": True}
    return jsonify({"status": "cancelling"})


@app.route("/run", methods=["POST"])
def run():
    override_N = request.form.get("N")
    override_N = int(override_N) if (override_N and override_N.isdigit()) else None
    run_id = str(uuid.uuid4())
    cleanup_uploads(keep_run_id=run_id)
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    input_mode = request.form.get("input_mode", "upload")
    if input_mode == "manual":
        try:
            dati_tmp, targets_tmp = _build_manual_files_from_form(request.form)
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
        dati_path = os.path.join(run_dir, "dati.txt")
        targets_path = os.path.join(run_dir, "targets.txt")
        shutil.move(dati_tmp, dati_path)
        shutil.move(targets_tmp, targets_path)
    else:
        # Richiede esplicitamente upload dei file
        dati_file = request.files.get("dati_file")
        targets_file = request.files.get("targets_file")
        if not dati_file or not targets_file or not dati_file.filename or not targets_file.filename:
            flash("Carica sia dati.txt che targets.txt.", "error")
            return redirect(url_for("index"))
        dati_name = secure_filename(dati_file.filename)
        targets_name = secure_filename(targets_file.filename)
        if not dati_name or os.path.splitext(dati_name)[1].lower() not in ALLOWED_UPLOADS:
            flash("Formato dati non consentito", "error")
            return redirect(url_for("index"))
        if not targets_name or os.path.splitext(targets_name)[1].lower() not in ALLOWED_UPLOADS:
            flash("Formato targets non consentito", "error")
            return redirect(url_for("index"))
        dati_path = os.path.join(run_dir, "dati.txt")
        targets_path = os.path.join(run_dir, "targets.txt")
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
    with open(os.path.join(run_dir, "df.json"), "w", encoding="utf-8") as f:
        df_validi.to_json(f, orient="records")
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(riga_minima.to_dict()), f)
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
        if name == 'PDOF_eff':
            for alt in ('PDOF_eff', 'PDOF_effettiva', 'PDOF_effettivo'):
                if alt in riga_minima:
                    return riga_minima[alt]
            return default
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
        with open(os.path.join(run_dir, "graph_inputs.json"), "w", encoding="utf-8") as gf:
            json.dump(to_jsonable(graph_inputs), gf)
    except Exception:
        pass

    # Limit preview rows to avoid overloading the page
    preview = df_validi.head(50).to_dict(orient="records")

    # Prepara elenco variabili espandibili (range definiti)
    expandable_vars = [
        k for k, v in parametri.items()
        if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] != v[1]
    ]

    used_params = _format_used_params(to_jsonable(parametri), to_jsonable(targets))
    # Evita duplicati: escludi variabili già mostrate in "Migliore combinazione"
    try:
        used_params = {k: v for k, v in used_params.items() if k not in best}
    except Exception:
        pass
    used_params_grouped = _group_used_params(used_params)

    def _is_active(d: dict, key: str) -> bool:
        v = d.get(key)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        try:
            return int(v) == 1
        except Exception:
            return False

    flags = {
        'chiusura_triangoli': _is_active(parametri, 'chiusura_triangoli'),
        'cicloide_avanzata': _is_active(parametri, 'cicloide_avanzata'),
        'stima_PDOF': _is_active(parametri, 'stima_PDOF'),
    }

    rows_full = df_validi.to_dict(orient="records")
    payload = {
        "best": best,
        "rows": rows_full,
        "total": len(df_validi),
        "run_id": run_id,
        "variabili": variabili,
        "colonne_errori": colonne_errori,
        "graph_inputs": json.dumps(to_jsonable(graph_inputs)),
        "units": UNITS,
        "expandable_vars": expandable_vars,
        "used_params": used_params,
        "used_params_grouped": used_params_grouped,
        "flags": flags,
        "parametri": to_jsonable(parametri),
        "targets": to_jsonable(targets),
    }

    payload_path = os.path.join(run_dir, "results_payload.json")
    with open(payload_path, "w", encoding="utf-8") as pf:
        json.dump(to_jsonable(payload), pf)

    files_available = {
        "risultati.txt": os.path.exists(os.path.join(BASE_DIR, "risultati.txt")),
        "listato.txt": os.path.exists(os.path.join(BASE_DIR, "listato.txt")),
        "results_payload.json": os.path.exists(payload_path),
    }
    payload["files_available"] = files_available
    with open(payload_path, "w", encoding="utf-8") as pf:
        json.dump(to_jsonable(payload), pf)

    # Use only preview rows for immediate rendering
    payload["rows"] = preview
    return render_template("results.html", **payload)


@app.route("/download/<path:filename>")
def download(filename):
    # Serve only whitelisted output files from project root
    name = secure_filename(filename)
    if name not in ALLOWED_DOWNLOADS:
        return jsonify({"error": "file non disponibile"}), 404
    return send_from_directory(BASE_DIR, name, as_attachment=True)

@app.route("/download_payload/<run_id>")
def download_payload(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    path = os.path.join(run_dir, "results_payload.json")
    if not os.path.exists(path):
        return jsonify({"error": "run non trovato"}), 404
    return send_from_directory(run_dir, "results_payload.json", as_attachment=True)


@app.route("/load_payload", methods=["POST"])
def load_payload():
    """Load a previously saved results payload and show the results page."""
    payload_file = request.files.get("payload_file")
    if not payload_file or not payload_file.filename:
        flash("Carica un file results_payload.json", "error")
        return redirect(url_for("index"))
    fname = secure_filename(payload_file.filename)
    if os.path.splitext(fname)[1].lower() not in ALLOWED_PAYLOAD:
        flash("Formato file non consentito", "error")
        return redirect(url_for("index"))
    try:
        payload = json.load(payload_file)
    except Exception:
        flash("File results_payload.json non valido", "error")
        return redirect(url_for("index"))
    
    run_id = payload.get("run_id") or str(uuid.uuid4())
    cleanup_uploads(keep_run_id=run_id)
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results_payload.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)

    rows = payload.get("rows") or []
    if rows:
        df = pd.DataFrame(rows)
        with open(os.path.join(run_dir, "df.json"), "w", encoding="utf-8") as f:
            df.to_json(f, orient="records")
        payload["rows"] = rows[:50]
        payload["total"] = len(rows)
    else:
        payload["rows"] = []
        payload["total"] = 0
    best = payload.get("best") or {}
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(best), f)
    meta = {
        "variabili": payload.get("variabili", []),
        "colonne_errori": payload.get("colonne_errori", []),
        "parametri": payload.get("parametri", {}),
        "targets": payload.get("targets", {}),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    graph_inputs = payload.get("graph_inputs") or {}
    if isinstance(graph_inputs, str):
        try:
            graph_inputs = json.loads(graph_inputs)
        except Exception:
            graph_inputs = {}
    with open(os.path.join(run_dir, "graph_inputs.json"), "w", encoding="utf-8") as f:
        json.dump(graph_inputs, f)

        # Override availability of exportable files when loading a payload.
    # The original payload might have been saved when risultati.txt and
    # listato.txt were present, but after uploading a payload we don't have
    # those text files on disk. Force these buttons to stay hidden.
    files_available = payload.get("files_available") or {}
    files_available["risultati.txt"] = False
    files_available["listato.txt"] = False
    files_available["results_payload.json"] = True
    
    payload["files_available"] = files_available
    payload["run_id"] = run_id
    return render_template("results.html", **payload)


@app.route("/api/run/<run_id>/graph_inputs")
def api_graph_inputs(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
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
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    x = request.args.get("x")
    y = request.args.get("y")
    c = request.args.get("c")
    limit = int(request.args.get("limit", "10000"))
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    json_path = os.path.join(run_dir, "df.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "run non trovato"}), 404
    df = pd.read_json(json_path)
    if x not in df.columns or y not in df.columns or c not in df.columns:
        return jsonify({"error": "colonna non trovata"}), 400
    df2 = df[[x, y, c]].dropna().head(limit)
    return jsonify({
        "x": to_jsonable(df2[x].tolist()),
        "y": to_jsonable(df2[y].tolist()),
        "c": to_jsonable(df2[c].tolist())
    })


# --- Nuovo flusso asincrono con barra di avanzamento ---
def _background_run(run_id: str, dati_path: str, targets_path: str, override_N: int | None):
    key = run_id
    try:
        df_validi, riga_minima, parametri, targets = run_simulation(dati_path, targets_path, override_N, progress_key=key)
        # Compute helper lists
        colonne_errori = [c for c in df_validi.columns if c.lower().startswith("errore_") or c.lower().startswith("err_")]
        variabili = [c for c in df_validi.columns if not (c.lower().startswith("errore_") or c.lower().startswith("err_"))]

        run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "df.json"), "w", encoding="utf-8") as f:
            df_validi.to_json(f, orient="records")
        with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(riga_minima.to_dict()), f)
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "variabili": variabili,
                "colonne_errori": colonne_errori,
                "parametri": to_jsonable(parametri),
                "targets": to_jsonable(targets),
            }, f)

        # Prepara graph_inputs come in /run
        def get_val(name, default=None):
            if name == 'PDOF_eff':
                for alt in ('PDOF_eff', 'PDOF_effettiva', 'PDOF_effettivo'):
                    if alt in riga_minima:
                        return riga_minima[alt]
                return default
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
            run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
            with open(os.path.join(run_dir, "graph_inputs.json"), "w", encoding="utf-8") as gf:
                json.dump(to_jsonable(graph_inputs), gf)
        except Exception:
            pass
        
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

        expandable_vars = [
            k for k, v in parametri.items()
            if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] != v[1]
        ]

        used_params = _format_used_params(to_jsonable(parametri), to_jsonable(targets))
        try:
            used_params = {k: v for k, v in used_params.items() if k not in best}
        except Exception:
            pass
        used_params_grouped = _group_used_params(used_params)

        def _is_active(d: dict, key: str) -> bool:
            v = d.get(key)
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            try:
                return int(v) == 1
            except Exception:
                return False

        flags = {
            'chiusura_triangoli': _is_active(parametri, 'chiusura_triangoli'),
            'cicloide_avanzata': _is_active(parametri, 'cicloide_avanzata'),
            'stima_PDOF': _is_active(parametri, 'stima_PDOF'),
        }

        rows_full = df_validi.to_dict(orient="records")

        payload = {
            "best": best,
            "rows": rows_full,
            "total": len(df_validi),
            "run_id": run_id,
            "variabili": variabili,
            "colonne_errori": colonne_errori,
            "graph_inputs": json.dumps(to_jsonable(graph_inputs)),
            "units": UNITS,
            "expandable_vars": expandable_vars,
            "used_params": used_params,
            "used_params_grouped": used_params_grouped,
            "flags": flags,
            "parametri": to_jsonable(parametri),
            "targets": to_jsonable(targets),
        }
        payload_path = os.path.join(run_dir, "results_payload.json")
        with open(payload_path, "w", encoding="utf-8") as pf:
            json.dump(to_jsonable(payload), pf)
        files_available = {
            "risultati.txt": os.path.exists(os.path.join(BASE_DIR, "risultati.txt")),
            "listato.txt": os.path.exists(os.path.join(BASE_DIR, "listato.txt")),
            "results_payload.json": os.path.exists(payload_path),
        }
        payload["files_available"] = files_available
        with open(payload_path, "w", encoding="utf-8") as pf:
            json.dump(to_jsonable(payload), pf)

        PROGRESS[key] = { **PROGRESS.get(key, {}), "done": True }
    except Exception as e:
        PROGRESS[key] = { **PROGRESS.get(key, {}), "error": str(e), "done": True }


@app.route("/start", methods=["POST"])
def start_async():
    override_N = request.form.get("N")
    override_N = int(override_N) if (override_N and str(override_N).isdigit()) else None
    run_id = str(uuid.uuid4())
    cleanup_uploads(keep_run_id=run_id)
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    input_mode = request.form.get("input_mode", "upload")
    if input_mode == "manual":
        try:
            dati_tmp, targets_tmp = _build_manual_files_from_form(request.form)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        dati_path = os.path.join(run_dir, "dati.txt")
        targets_path = os.path.join(run_dir, "targets.txt")
        shutil.move(dati_tmp, dati_path)
        shutil.move(targets_tmp, targets_path)
    else:
        # Richiede upload dei file
        dati_file = request.files.get("dati_file")
        targets_file = request.files.get("targets_file")
        if not dati_file or not targets_file or not dati_file.filename or not targets_file.filename:
            return jsonify({"error": "Carica sia dati.txt che targets.txt."}), 400
        dati_name = secure_filename(dati_file.filename)
        targets_name = secure_filename(targets_file.filename)
        if not dati_name or os.path.splitext(dati_name)[1].lower() not in ALLOWED_UPLOADS:
            return jsonify({"error": "Formato dati non consentito"}), 400
        if not targets_name or os.path.splitext(targets_name)[1].lower() not in ALLOWED_UPLOADS:
            return jsonify({"error": "Formato targets non consentito"}), 400
        dati_path = os.path.join(run_dir, "dati.txt")
        targets_path = os.path.join(run_dir, "targets.txt")
        dati_file.save(dati_path)
        targets_file.save(targets_path)

    PROGRESS[run_id] = {"current": 0, "total": 0, "started": time.time(), "eta_seconds": None, "done": False}

    with PENDING_LOCK:
        if len(PENDING_FUTURES) >= MAX_QUEUE:
            return jsonify({"error": "coda piena"}), 429
        future = EXECUTOR.submit(_background_run, run_id, dati_path, targets_path, override_N)
        PENDING_FUTURES[run_id] = future

    def _done_cb(fut, rid=run_id):
        with PENDING_LOCK:
            PENDING_FUTURES.pop(rid, None)

    future.add_done_callback(_done_cb)

    return jsonify({"run_id": run_id, "status": "queued"})


@app.route("/api/progress/<run_id>")
def api_progress(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    info = PROGRESS.get(run_id)
    if not info:
        return jsonify({"error": "run non trovato"}), 404
    return jsonify(info)


@app.route("/api/esplora_progress/<run_id>")
def api_esplora_progress(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    info = PROGRESS_EXPLORA.get(run_id)
    if not info:
        return jsonify({"error": "run non trovato"}), 404
    return jsonify(info)


@app.route("/result/<run_id>")
def result(run_id):
    # Ricostruisce la pagina risultati da file salvati
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    json_df = os.path.join(run_dir, "df.json")
    json_best = os.path.join(run_dir, "best.json")
    meta_path = os.path.join(run_dir, "meta.json")
    gi_path = os.path.join(run_dir, "graph_inputs.json")
    if not (os.path.exists(json_df) and os.path.exists(json_best) and os.path.exists(meta_path) and os.path.exists(gi_path)):
        return jsonify({"error": "risultati non pronti"}), 404
    with open(json_best, "r", encoding="utf-8") as f:
        riga_minima = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(gi_path, "r", encoding="utf-8") as f:
        graph_inputs = json.load(f)

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

    # Prepara elenco variabili espandibili (range definiti)
    params = meta.get("parametri", {})
    expandable_vars = [
        k for k, v in params.items()
        if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] != v[1]
    ]

    used_params = _format_used_params(meta.get("parametri", {}), meta.get("targets", {}))
    # Evita duplicati rispetto a "Migliore combinazione"
    try:
        used_params = {k: v for k, v in used_params.items() if k not in best}
    except Exception:
        pass
    used_params_grouped = _group_used_params(used_params)

    def _is_active_meta(d: dict, key: str) -> bool:
        v = d.get(key)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        try:
            return int(v) == 1
        except Exception:
            return False

    flags = {
        'chiusura_triangoli': _is_active_meta(params, 'chiusura_triangoli'),
        'cicloide_avanzata': _is_active_meta(params, 'cicloide_avanzata'),
        'stima_PDOF': _is_active_meta(params, 'stima_PDOF'),
    }

    return render_template(
        "results.html",
        best=best,
        rows=[],
        total=0,
        run_id=run_id,
        variabili=meta.get("variabili", []),
        colonne_errori=meta.get("colonne_errori", []),
        graph_inputs=json.dumps(to_jsonable(graph_inputs)),
        units=UNITS,
        expandable_vars=expandable_vars,
        used_params=used_params,
        used_params_grouped=used_params_grouped,
        flags=flags,
        files_available={
            "risultati.txt": os.path.exists(os.path.join(BASE_DIR, "risultati.txt")),
            "listato.txt": os.path.exists(os.path.join(BASE_DIR, "listato.txt")),
            "results_payload.json": os.path.exists(os.path.join(run_dir, "results_payload.json")),
        }
    )


# --- Esposizione semplificata di esplora.py lato web ---
def _filtra_valori_positivi(nome_variabile, valori):
    variabili_da_filtrare = [
        "V1_pre", "V2_pre", "V1_post", "V2_post",
        "Ed", "f1", "f2", "cicloide1", "cicloide2"
    ]
    if nome_variabile in variabili_da_filtrare:
        return [v for v in valori if v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= 0]
    return [v for v in valori if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _espandi_range(ranges: dict, passo: float = 1.0):
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


def _aggiorna_parametri(parametri_base: dict, nuovi_ranges: dict):
    parametri_mod = dict(parametri_base)
    for k, v in nuovi_ranges.items():
        parametri_mod[k] = v
    return parametri_mod


def _model_montecarlo(parametri_modificati: dict, targets: dict, N: int = 1000):
    risultati = []
    for _ in range(N):
        p = genera_input_casuale(parametri_modificati)
        t = genera_input_casuale(targets)
        try:
            res, err = calcoli(p, t)
            if all(val < 1 for key, val in err.items() if 'errore_' in key.lower()):
                risultati.append({**res, **err})
        except Exception:
            continue
    return risultati


@app.route("/api/esplora/<run_id>", methods=["POST"])
def api_esplora(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    # Parametri di input
    body = request.get_json(silent=True) or {}
    metric = body.get("metrica")
    max_iter = int(body.get("max_iter", 20))
    passo = float(body.get("passo", 1.0))
    soglia_salto = float(body.get("soglia_salto", 3.0))
    user_vars = body.get("vars")  # lista di variabili che l'utente vuole espandere

    run_dir = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    meta_path = os.path.join(run_dir, "meta.json")
    parametri = {}
    targets = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        parametri = meta.get("parametri", {})
        targets = meta.get("targets", {})
    else:
        payload_path = os.path.join(run_dir, "results_payload.json")
        if not os.path.exists(payload_path):
            return jsonify({"error": "run non trovato"}), 404
        with open(payload_path, "r", encoding="utf-8") as pf:
            payload = json.load(pf)
        parametri = payload.get("parametri", {})
        targets = payload.get("targets", {})

    if not parametri or not targets:
        return jsonify({"error": "parametri o targets mancanti"}), 400
    
    # Per l'esplorazione forza sempre alcuni flag a zero
    for k in ("cicloide_avanzata", "chiusura_triangoli"):
        parametri[k] = [0, 0]

    if not metric:
        return jsonify({"error": "metrica non specificata"}), 400

    # Prepara variabili da espandere: solo quelle con range variabile
    candidates = [k for k, v in parametri.items() if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] != v[1]]
    if isinstance(user_vars, list) and len(user_vars) > 0:
        variabili_da_espandere = [k for k in user_vars if k in candidates]
    else:
        variabili_da_espandere = candidates
    if not variabili_da_espandere:
        return jsonify({"error": "nessun parametro variabile"}), 400

    ranges_correnti = {k: tuple(parametri[k]) for k in variabili_da_espandere}
    risultati_iter = []
    report_lines: list[str] = []
    salto_rilevato = False
    iterazioni_post_salto = 0
    ultimo_output_valido = None

    import numpy as np  # lazy import

    # inizializza stato avanzamento
    PROGRESS_EXPLORA[run_id] = {
        "current": 0,
        "total": max_iter,
        "done": False,
        "error": None,
        "last": None,
        "started": time.time(),
        "cancel": False,
    }

    for iterazione in range(max_iter):
        # check cancel richiesta dall'utente
        try:
            if PROGRESS_EXPLORA.get(run_id, {}).get("cancel"):
                PROGRESS_EXPLORA[run_id] = { **PROGRESS_EXPLORA.get(run_id, {}), "done": True, "canceled": True }
                break
        except Exception:
            pass
        parametri_attuali = _aggiorna_parametri(parametri, ranges_correnti)
        N_pre = 10000
        pre_output = _model_montecarlo(parametri_attuali, targets, N=N_pre)
        valori_pre = [r.get(metric) for r in pre_output if metric in r]
        valori_pre = _filtra_valori_positivi(metric, [v for v in valori_pre if v is not None and not (isinstance(v, float) and math.isnan(v))])
        sigma_pre = float(np.std(valori_pre)) if valori_pre else float('nan')
        media_pre = float(np.mean(valori_pre)) if valori_pre else float('nan')

        z = 1.96
        errore_abs = 0.025
        if not valori_pre or math.isnan(sigma_pre) or math.isnan(media_pre) or sigma_pre == 0:
            N_dinamico = N_pre
        else:
            N_dinamico = int((z * sigma_pre / errore_abs) ** 2)
            N_dinamico = max(N_dinamico, N_pre)
            N_dinamico = min(N_dinamico, 200000)

        # Log intestazione e N dinamico
        report_lines.append(f"[Iter {iterazione+1}]")
        try:
            report_lines.append(f"   N dinamico calcolato: {N_dinamico} (media={media_pre:.3f}, std={sigma_pre:.3f})")
        except Exception:
            report_lines.append(f"   N dinamico calcolato: {N_dinamico}")

        output = _model_montecarlo(parametri_attuali, targets, N=N_dinamico)
        if output:
            ultimo_output_valido = output

        valori_output = [r.get(metric) for r in output if metric in r]
        valori_output_filtrati = _filtra_valori_positivi(metric, [v for v in valori_output if v is not None and not (isinstance(v, float) and math.isnan(v))])
        std = float(np.std(valori_output_filtrati)) if valori_output_filtrati else float('nan')
        media = float(np.mean(valori_output_filtrati)) if valori_output_filtrati else float('nan')

        # Log deviazione standard/assenza dati
        if not math.isnan(std):
            report_lines.append(f"   Deviazione std {metric}: {std:.2f}")
        else:
            report_lines.append("   Nessun dato valido")

        step = {
            "iter": iterazione + 1,
            "ranges": ranges_correnti.copy(),
            "deviazione_std": std,
            "media_output": media,
            "campioni_validi": len(valori_output_filtrati)
        }
        risultati_iter.append(step)

        # aggiorna barra avanzamento
        try:
            PROGRESS_EXPLORA[run_id] = {
                **PROGRESS_EXPLORA.get(run_id, {}),
                "current": iterazione + 1,
                "total": max_iter,
                "done": False,
                "error": None,
                "last": step,
            }
        except Exception:
            pass

        if len(risultati_iter) >= 2 and not salto_rilevato:
            std_prec = risultati_iter[-2]['deviazione_std']
            if not math.isnan(std) and not math.isnan(std_prec) and std_prec >= 1e-6:
                incremento = std / std_prec
                if incremento > soglia_salto:
                    salto_rilevato = True
                    report_lines.append("")
                    report_lines.append(f"âš ï¸   SALTO NETTO RILEVATO: da std = {std_prec:.2f} a std = {std:.2f} (x{incremento:.2f})")
                    report_lines.append("")

        if salto_rilevato:
            iterazioni_post_salto += 1
            if iterazioni_post_salto >= 15:
                break

        ranges_correnti = _espandi_range(ranges_correnti, passo=passo)

    # chiude stato avanzamento
    try:
        PROGRESS_EXPLORA[run_id] = {
            **PROGRESS_EXPLORA.get(run_id, {}),
            "current": len(risultati_iter),
            "total": max_iter,
            "done": True,
            "error": None,
        }
    except Exception:
        pass

    # --- RIEPILOGO ---
    report_lines.append("")
    # --- RIEPILOGO --- (rimosso su richiesta)
    variabili_positive = {'V1_pre', 'V2_pre', 'V1_post', 'V2_post', 'Ed', 'f1', 'f2', 'cicloide1', 'cicloide2'}
    for r in []:
        media = r.get('media_output')
        std = r.get('deviazione_std')
        if media is None or std is None or math.isnan(media) or math.isnan(std):
            report_lines.append(f"Iter {r.get('iter'):>2} | std = {std:.2f} | {metric}: nessun dato valido")
            continue
        low = media - 3*std
        high = media + 3*std
        if metric in variabili_positive:
            low = max(0, low)
        try:
            report_lines.append(f"Iter {r.get('iter'):>2} | std = {std:.2f} | {metric}: {media:.2f} +/- {3*std:.2f}  [{low:.2f}, {high:.2f}]")
        except Exception:
            # fallback without f-string formatting of low/high as placeholders
            report_lines.append(f"Iter {r.get('iter'):>2} | std = {std:.2f} | {metric}: {media:.2f}")

    report_lines.append("")
    if salto_rilevato and len(risultati_iter) >= 16:
        report_lines.append("------ ULTIMA SIMULAZIONE PRIMA DEL SALTO ------")
        report_lines.append("")
        ultima_simulazione = risultati_iter[-16]
    else:
        report_lines.append("------ ULTIMA SIMULAZIONE (NESSUN SALTO RILEVATO) ------")
        report_lines.append("")
        ultima_simulazione = risultati_iter[-1] if risultati_iter else None

    if ultima_simulazione:
        media = ultima_simulazione.get('media_output')
        std = ultima_simulazione.get('deviazione_std')
        low = max(0, media - 3*std) if (media is not None and std is not None) else None
        high = (media + 3*std) if (media is not None and std is not None) else None
        report_lines.append("Valore Atteso della variabile testata:")
        report_lines.append("")
        if low is not None and low == 0:
            report_lines.append(f"{metric:<25} = {media:12.2f} + {3*std:.2f}")
        elif low is not None and high is not None:
            report_lines.append(f"{metric:<25} = {media:12.2f} +/- {3*std:.2f}  [{low:.2f}, {high:.2f}]")
        report_lines.append("")
        report_lines.append("Precisione (+/- 3 sigma)       =        99.73%")
        report_lines.append("Livello di Confidenza (sigma) =        95.00%")
        report_lines.append("")
        report_lines.append("Range valori (valore centrale +/- semiampiezza):")
        report_lines.append("")
        for k, (v_min, v_max) in (ultima_simulazione.get('ranges') or {}).items():
            try:
                centro = (v_min + v_max) / 2
                semiamp = (v_max - v_min) / 2
                report_lines.append(f" - {k:<23}= {centro:10.2f} +/- {semiamp:.2f}")
            except Exception:
                continue

    # RIEPILOGO con filtro su deviazione standard
    SOGLIA_STD = 5.0
    filtrati = [r for r in risultati_iter if r.get('deviazione_std') is not None and not math.isnan(r.get('deviazione_std')) and r.get('deviazione_std') <= SOGLIA_STD]
    if filtrati:
        report_lines.append("")
        report_lines.append(f"------- ULTIMA SIMULAZIONE con dev_std = {SOGLIA_STD} -------:")
        report_lines.append("")
        migliore = min(filtrati, key=lambda r: r['deviazione_std'])
        media = migliore.get('media_output')
        std = migliore.get('deviazione_std')
        low = max(0, media - 3*std) if (media is not None and std is not None) else None
        high = (media + 3*std) if (media is not None and std is not None) else None
        if low is not None and low == 0:
            report_lines.append(f"{metric:<25} = {media:12.2f} + {3*std:.2f}")
        elif low is not None and high is not None:
            report_lines.append(f"{metric:<25} = {media:12.2f} +/- {3*std:.2f}  [{low:.2f}, {high:.2f}]")
        report_lines.append("")
        report_lines.append("Precisione (+/- 3 sigma)       =        99.73%")
        report_lines.append("Livello di Confidenza (sigma) =        95.00%")
        report_lines.append("")
        report_lines.append("Range valori (valore centrale +/- semiampiezza):")
        report_lines.append("")
        for k, (v_min, v_max) in (migliore.get('ranges') or {}).items():
            try:
                centro = (v_min + v_max) / 2
                semiamp = (v_max - v_min) / 2
                report_lines.append(f" - {k:<23}= {centro:10.2f} +/- {semiamp:.2f}")
            except Exception:
                continue

    # RIEPILOGO aggiuntivo ben formattato (come terminale)
    report_lines.append("")
    # report_lines.append("--- RIEPILOGO (formattato) ---")
    variabili_positive = {'V1_pre', 'V2_pre', 'V1_post', 'V2_post', 'Ed', 'f1', 'f2', 'cicloide1', 'cicloide2'}
    for r in []:
        media = r.get('media_output')
        std = r.get('deviazione_std')
        if media is None or std is None or math.isnan(media) or math.isnan(std):
            report_lines.append(f"Iter {r.get('iter'):>2} | std = {std if std is not None else float('nan'):.2f} | {metric}: nessun dato valido")
            continue
        low = media - 3*std
        high = media + 3*std
        if metric in variabili_positive:
            low = max(0, low)
        report_lines.append(f"Iter {r.get('iter'):>2} | std = {std:.2f} | {metric}: {media:.2f} +/- {3*std:.2f}  [{low:.2f}, {high:.2f}]")

    response = {
        "metric": metric,
        "iterazioni": risultati_iter,
        "salto_rilevato": salto_rilevato,
        "canceled": PROGRESS_EXPLORA.get(run_id, {}).get("canceled", False),
        "report": "\n".join(report_lines),
    }
    return jsonify(response)


@app.route("/api/esplora_cancel/<run_id>", methods=["POST"])
def api_esplora_cancel(run_id):
    if not _valid_run_id(run_id):
        return jsonify({"error": "run non trovato"}), 404
    if run_id not in PROGRESS_EXPLORA:
        PROGRESS_EXPLORA[run_id] = {"cancel": True}
    else:
        PROGRESS_EXPLORA[run_id] = { **PROGRESS_EXPLORA.get(run_id, {}), "cancel": True }
    return jsonify({"status": "cancelling"})


def _coalesce_val(v: str | None):
    return v if v is not None and str(v).strip() != "" else None


def _build_manual_files_from_form(form) -> tuple[str, str]:
    """Create temporary dati.txt and targets.txt from manual form inputs.

    Returns (dati_path, targets_path).
    Raises ValueError on invalid/missing required inputs.
    """
    # Manual DATI
    dati_lines: list[str] = []
    written_names: set[str] = set()

    # N (single only)
    man_N = _coalesce_val(form.get("man_N"))
    if man_N:
        dati_lines.append(f"N = {man_N}")
        written_names.add("N")

    # gdl select (2 or 3)
    man_gdl = _coalesce_val(form.get("man_gdl"))
    if man_gdl in {"2", "3"}:
        dati_lines.append(f"gdl = {man_gdl}")
        written_names.add("gdl")

    # Booleans as dropdowns
    bool_keys = [
        "cicloide", "cicloide_avanzata", "cicloide_nota",
        "energia_EES", "stima_PDOF", "chiusura_triangoli",
    ]
    for k in bool_keys:
        val = form.get(f"man_{k}")
        if val in {"0", "1"}:  # map attivo/non attivo to 1/0 in UI
            dati_lines.append(f"{k} = {val}")
            written_names.add(k)

    # Generic manual rows for dati
    names = form.getlist("man_dati_name[]")
    modes = form.getlist("man_dati_mode[]")  # 'single' | 'range'
    v1s = form.getlist("man_dati_v1[]")
    v2s = form.getlist("man_dati_v2[]")
    provided_names = set()
    for i in range(len(names)):
        name = (names[i] or "").strip()
        if not name:
            continue
        provided_names.add(name)
        # avoid duplicating reserved keys handled above
        if name in {"N", "gdl", *bool_keys}:
            continue
        v1 = (v1s[i] if i < len(v1s) else "").strip()
        v2 = (v2s[i] if i < len(v2s) else "").strip()
        mode = (modes[i] if i < len(modes) else "single").strip()
        if not v1:
            continue
        if mode == "range" and v2:
            dati_lines.append(f"{name} = {v1}, {v2}")
            written_names.add(name)
        else:
            dati_lines.append(f"{name} = {v1}")
            written_names.add(name)

    # Manual TARGETS
    target_lines: list[str] = []
    t_names = form.getlist("man_target_name[]")
    t_modes = form.getlist("man_target_mode[]")
    t_v1s = form.getlist("man_target_v1[]")
    t_v2s = form.getlist("man_target_v2[]")
    for i in range(len(t_names)):
        name = (t_names[i] or "").strip()
        if not name:
            continue
        v1 = (t_v1s[i] if i < len(t_v1s) else "").strip()
        v2 = (t_v2s[i] if i < len(t_v2s) else "").strip()
        mode = (t_modes[i] if i < len(t_modes) else "single").strip()
        if not v1:
            continue
        if mode == "range" and v2:
            target_lines.append(f"{name} = {v1}, {v2}")
        else:
            target_lines.append(f"{name} = {v1}")

    # Auto-compile default coordinates when not requested by UI
    try:
        man_gdl = _coalesce_val(form.get("man_gdl"))
        ctri = form.get("man_chiusura_triangoli") == "1"
        cav = form.get("man_cicloide_avanzata") == "1"
        sp = form.get("man_stima_PDOF") == "1"
        if man_gdl == "2" and not ctri and not cav and not sp:
            # Sezione coordinate nascosta: assegna default richiesti
            if "x1" not in written_names:
                dati_lines.append("x1 = 2")
                written_names.add("x1")
            if "y1" not in written_names:
                dati_lines.append("y1 = 2")
                written_names.add("y1")
            if "x2" not in written_names:
                dati_lines.append("x2 = -2")
                written_names.add("x2")
            if "y2" not in written_names:
                dati_lines.append("y2 = -2")
                written_names.add("y2")
    except Exception:
        pass

    # Per tutte le altre variabili non compilate assegna 0
    # (copriamo i parametri usati direttamente nei modelli/calcoli)
    required_defaults = [
        'x1','y1','x2','y2',
        'x1_quiete','y1_quiete','x2_quiete','y2_quiete',
        'm1','l1','p1','d_post1','f1','theta_post1','EES1_Kmh','lunghezza_cicloide_1',
        'm2','l2','p2','d_post2','f2','theta_post2','EES2_Kmh','lunghezza_cicloide_2',
        'theta1_in','theta1_out','theta2_in','theta2_out',
        'Ed_target','omega1_post','omega2_post','J1','J2'
    ]
    # Booleans: se non specificati, default 0
    for k in bool_keys:
        if k not in written_names:
            dati_lines.append(f"{k} = 0")
            written_names.add(k)
    for k in required_defaults:
        if k not in written_names:
            dati_lines.append(f"{k} = 0")
            written_names.add(k)

    # Require at least some content in both files
    if not dati_lines:
        raise ValueError("Inserisci tutti i parametri necessari.")
    if not target_lines:
        raise ValueError("Inserisci tutti i parametri necessari.")

    # Write to unique files
    run_id = uuid.uuid4().hex
    dati_path = os.path.join(UPLOAD_DIR, f"manual_dati_{run_id}.txt")
    targets_path = os.path.join(UPLOAD_DIR, f"manual_targets_{run_id}.txt")
    with open(dati_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dati_lines) + "\n")
    with open(targets_path, "w", encoding="utf-8") as f:
        f.write("\n".join(target_lines) + "\n")

    return dati_path, targets_path


if __name__ == "__main__":
    freeze_support()
    manager = create_manager()
    cleanup_uploads()
    app.run(debug=True)  # o False in produzione
