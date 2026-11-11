"""Kaggle entry point. build.py inlines the demo scripts into FILES and writes run_all.py.

Installs openpi, runs the DROID "inside the brain" demo, then the LIBERO simulation demo.
Each stage logs to /kaggle/working/logs and a failure in one stage does not stop the next.
"""

import os
import pathlib
import socket
import subprocess
import time

FILES = {}  # filled in by build.py

W = pathlib.Path("/kaggle/working")
LOGS = W / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
REPO = pathlib.Path("/tmp/openpi")
ENV = {**os.environ, "OPENPI_DATA_HOME": "/tmp/openpi_cache", "GIT_LFS_SKIP_SMUDGE": "1",
       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85", "PATH": f"/root/.local/bin:{os.environ['PATH']}"}
ENV.pop("UV_SYSTEM_PYTHON", None)  # Kaggle sets this, which made uv install into the system Python 3.13
LIBERO_PY = REPO / "examples/libero/.venv/bin/python"


def sh(name, cmd, env=None, timeout=3 * 3600):
    print(f"\n===== {name} =====", flush=True)
    t = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        try:
            rc = subprocess.run(["bash", "-lc", cmd], cwd=REPO if REPO.exists() else W, env=env or ENV,
                                stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    tail = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()[-25:]
    print("\n".join(tail))
    print(f"----- {name}: rc={rc} in {time.time() - t:.0f}s", flush=True)
    return rc == 0


for fname, src in FILES.items():
    (W / "scripts").mkdir(exist_ok=True)
    (W / "scripts" / fname).write_text(src)

sh("00_system", "nvidia-smi; df -h / /tmp /kaggle/working; python --version; free -g")
sh("01_clone", "cd /tmp && git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git && cd openpi && git log -1 --format='%h %cd'")
ok = sh("02_install", "env | grep -i '^uv_' ; pip install -q uv && uv sync --python 3.11 && uv run python -c 'import jax; print(jax.devices())'")


def fetch_checkpoint(name):
    """openpi downloads with gsutil, which fails on Kaggle without compiled crcmod; skip the hash check."""
    dst = f"/tmp/openpi_cache/openpi-assets/checkpoints/{name}"
    return sh(f"ckpt_{name}", f"mkdir -p {dst} && gsutil -q -o 'GSUtil:check_hashes=never' -m cp -r "
                              f"'gs://openpi-assets/checkpoints/{name}/*' {dst}/ && du -sh {dst}")

# ---------------- Layer 3: DROID inside-the-brain ----------------
if ok and fetch_checkpoint("pi05_droid"):
    sh("03_droid_demo", f"uv run python {W}/scripts/droid_demo.py {W}/droid_out", timeout=2 * 3600)
    sh("04_free_disk", "rm -rf /tmp/openpi_cache/openpi-assets/checkpoints/pi05_droid; df -h /tmp")

# ---------------- Layer 2: LIBERO simulation ----------------
lib_ok = sh("05_libero_install", f"""
uv venv --python 3.8 examples/libero/.venv
uv pip install --python {LIBERO_PY} -r examples/libero/requirements.txt -r third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python {LIBERO_PY} -e packages/openpi-client -e third_party/libero
""")

render_env = None
if lib_ok:
    # LIBERO asks an interactive question on first import; answer N to use default paths.
    base = {**ENV, "PYTHONPATH": str(REPO / "third_party/libero")}
    test = f"echo N | {LIBERO_PY} -c \"from libero.libero import benchmark; print('libero import ok')\" && " \
           f"{LIBERO_PY} -c \"import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); r=mujoco.Renderer(m); print('render ok')\""
    for gl in ["egl", "osmesa"]:
        if gl == "osmesa":
            sh("06_osmesa_apt", "apt-get update -qq && apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3")
        e = {**base, "MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl}
        if sh(f"06_render_test_{gl}", test, env=e):
            render_env = e
            break

if ok and render_env and fetch_checkpoint("pi05_libero"):
    server = subprocess.Popen(["bash", "-lc", "uv run scripts/serve_policy.py --env LIBERO"], cwd=REPO, env=ENV,
                              stdout=open(LOGS / "07_server.log", "w"), stderr=subprocess.STDOUT)
    ready = False
    for _ in range(90):  # model load + compile can take a while
        try:
            socket.create_connection(("localhost", 8000), timeout=2).close()
            ready = True
            break
        except OSError:
            if server.poll() is not None:
                break
            time.sleep(10)
    if ready:
        sh("08_libero_demo", f"{LIBERO_PY} {W}/scripts/libero_demo.py {W}/libero_out", env=render_env, timeout=3 * 3600)
    else:
        print("policy server not up, skipping LIBERO rollouts")
    server.kill()
    print("\n".join((LOGS / "07_server.log").read_text(errors="replace").splitlines()[-15:]))

sh("99_summary", f"ls -la {W}/droid_out {W}/libero_out; du -sh {W}")
