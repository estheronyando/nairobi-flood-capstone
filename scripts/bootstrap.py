"""One command from a fresh copy of this repository to a working app.

    python scripts/bootstrap.py

It does five things, in order, and stops with a plain explanation if any of
them cannot be done:

  1. checks the interpreter is new enough for the pinned wheels
  2. creates .venv and installs the API and UI requirements
  3. finds the 50 input rasters, extracting them from the data archive if needed
  4. runs the pipeline if there is no trained model yet
  5. starts the API, waits for it to report healthy, starts the UI, opens it

Nothing here trains differently, changes the split, or touches the model: it is
setup plumbing around the same `scripts/run_pipeline.py` you would run by hand.

Useful flags:
    --data-zip PATH   the archive holding the GeoTIFFs, if it is somewhere odd
    --no-venv         install into the current environment instead of .venv
    --skip-install    environment is already prepared
    --no-ui           API only
    --port / --ui-port
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nairobi_flood_data"
MODEL = ROOT / "models" / "flood_model.joblib"
EXPECTED_TIFS = 50
MIN_PY = (3, 12)   # rasterio/pyproj/xgboost/shap/numpy pins publish no older wheels

WIN = platform.system() == "Windows"


# ---------------------------------------------------------------- reporting
def say(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def die(msg: str, *, fix: str | None = None) -> None:
    print(f"\nSTOPPED: {msg}", file=sys.stderr)
    if fix:
        print(f"\nWhat to do:\n{fix}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------- 1. python
def check_interpreter(for_venv: bool) -> None:
    v = sys.version_info
    say("1/5", f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()}")
    if not for_venv:
        return
    if (v.major, v.minor) < MIN_PY:
        die(
            f"Python {v.major}.{v.minor} is too old for this project's pinned "
            f"packages (need {MIN_PY[0]}.{MIN_PY[1]} or newer).",
            fix=(
                "  rasterio, pyproj, xgboost, shap and numpy publish no wheels for\n"
                "  3.11 or earlier at the versions pinned here, so pip would try to\n"
                "  compile them from source and fail.\n\n"
                "  If you have Anaconda:\n"
                "      conda create -n flood python=3.13 -y\n"
                "      conda activate flood\n"
                "      python scripts/bootstrap.py --no-venv\n\n"
                "  Or install Python 3.13 from python.org, then:\n"
                "      py -3.13 scripts\\bootstrap.py"
            ),
        )


# ---------------------------------------------------------------- 2. install
def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


def prepare_env(args: argparse.Namespace) -> Path:
    if args.no_venv:
        py = Path(sys.executable)
        say("2/5", f"using the current environment ({py})")
    else:
        venv = ROOT / ".venv"
        py = venv_python(venv)
        if py.exists():
            say("2/5", f"reusing {venv.name}")
        else:
            say("2/5", f"creating {venv.name} (this takes a moment)")
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    if args.skip_install:
        say("2/5", "skipping installation as asked")
        return py

    for req in (ROOT / "requirements.txt", ROOT / "app" / "requirements.txt"):
        if not req.exists():
            continue
        say("2/5", f"installing {req.relative_to(ROOT)}")
        proc = subprocess.run(
            [str(py), "-m", "pip", "install", "--quiet", "-r", str(req)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
            die(
                f"installing {req.name} failed.",
                fix="pip said:\n\n  " + "\n  ".join(tail),
            )
    say("2/5", "dependencies installed")
    return py


# ---------------------------------------------------------------- 3. data
def count_tifs(d: Path) -> int:
    return len(list(d.glob("*.tif"))) if d.is_dir() else 0


def candidate_archives(explicit: str | None) -> list[Path]:
    if explicit:
        return [Path(explicit).expanduser()]
    homes = [Path.home() / "Downloads", Path.home(), ROOT, ROOT.parent]
    found: list[Path] = []
    for h in homes:
        if not h.is_dir():
            continue
        try:
            for z in sorted(h.glob("*.zip")):
                name = z.name.lower()
                if "nairobi" in name or "flood" in name:
                    found.append(z)
        except OSError:
            continue
    return found


def extract_rasters(archive: Path) -> int:
    """Pull the .tif members out of an archive into data/nairobi_flood_data.

    The archive is treated as untrusted input: member names are taken apart
    and only the basename is used, so a crafted entry cannot write outside
    the data directory. Only .tif files are taken.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(".tif")]
        if not members:
            die(f"{archive.name} contains no .tif files.",
                fix="Point --data-zip at the archive holding the 50 GeoTIFFs.")
        for m in members:
            target = DATA_DIR / Path(m.filename.replace("\\", "/")).name
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written


def ensure_data(args: argparse.Namespace) -> None:
    have = count_tifs(DATA_DIR)
    if have >= EXPECTED_TIFS:
        say("3/5", f"{have} rasters already in place")
        return

    archives = candidate_archives(args.data_zip)
    if not archives:
        die(
            f"only {have} of {EXPECTED_TIFS} rasters are present and no data "
            "archive was found.",
            fix=(
                f"  Put the 50 GeoTIFFs in:\n      {DATA_DIR}\n\n"
                "  or point this script at the archive:\n"
                "      python scripts/bootstrap.py --data-zip "
                '"C:\\path\\to\\nairobi_flood_mvp.zip"'
            ),
        )

    archive = archives[0]
    say("3/5", f"extracting rasters from {archive.name}")
    n = extract_rasters(archive)
    total = count_tifs(DATA_DIR)
    say("3/5", f"wrote {n} rasters ({total} in place)")
    if total < EXPECTED_TIFS:
        die(f"{total} rasters present, expected {EXPECTED_TIFS}.",
            fix="The archive is missing layers. The audit step would stop anyway, "
                "so fix the data rather than continuing.")


# ---------------------------------------------------------------- 4. pipeline
def ensure_model(py: Path) -> None:
    if MODEL.exists():
        say("4/5", f"trained model already present ({MODEL.name})")
        return
    say("4/5", "no model yet - running the pipeline (about 20 seconds)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([str(py), str(ROOT / "scripts" / "run_pipeline.py")],
                          cwd=ROOT, env=env)
    if proc.returncode != 0:
        die("the pipeline failed, so there is no model to serve.",
            fix="Read the error above. The audit stage stops the run on a grid or "
                "CRS mismatch rather than resampling silently.")
    if not MODEL.exists():
        die("the pipeline finished but wrote no model file.")
    say("4/5", "model trained")


# ---------------------------------------------------------------- 5. serve
def wait_for_health(url: str, proc: subprocess.Popen, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            die(f"the API exited with code {proc.returncode} before becoming healthy.",
                fix="Its output is above.")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                import json
                body = json.load(r)
                if body.get("model_loaded"):
                    return body
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(2)
    die(f"the API did not report healthy within {timeout}s.")
    return {}


def serve(py: Path, args: argparse.Namespace) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    say("5/5", f"starting the API on port {args.port}")
    api = subprocess.Popen(
        [str(py), "-m", "uvicorn", "floodrisk.api:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=ROOT, env=env,
    )
    procs = [api]
    try:
        body = wait_for_health(f"http://127.0.0.1:{args.port}/health", api)
        say("5/5", f"API healthy, serving {body.get('model_name')}")

        if not args.no_ui:
            env_ui = {**env, "FLOOD_API_URL": f"http://127.0.0.1:{args.port}"}
            say("5/5", f"starting the UI on port {args.ui_port}")
            ui = subprocess.Popen(
                [str(py), "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                 "--server.port", str(args.ui_port),
                 "--server.headless", "true",
                 "--browser.gatherUsageStats", "false"],
                cwd=ROOT, env=env_ui,
            )
            procs.append(ui)
            time.sleep(6)

        print("\n" + "=" * 68)
        if not args.no_ui:
            print(f"  App   ->  http://localhost:{args.ui_port}")
        print(f"  API   ->  http://localhost:{args.port}/docs")
        print("\n  Try Mathare valley:  lat -1.2630   lon 36.8600")
        print("  Expect a probability near 0.113.")
        print("\n  Ctrl+C here stops both.")
        print("=" * 68 + "\n")

        if not args.no_ui:
            try:
                import webbrowser
                webbrowser.open(f"http://localhost:{args.ui_port}")
            except Exception:
                pass

        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()   # portable; Windows maps this to TerminateProcess
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-zip")
    ap.add_argument("--no-venv", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    args = ap.parse_args()

    print("\nNairobi flood susceptibility - setup\n" + "-" * 38)
    check_interpreter(for_venv=not args.no_venv)
    py = prepare_env(args)
    ensure_data(args)
    ensure_model(py)
    serve(py, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
