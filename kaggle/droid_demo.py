"""Layer 3 demo: look inside pi0.5's brain.

Feeds real DROID robot camera frames + different text prompts to the open-source
pi05_droid checkpoint and plots the action chunk it predicts.
Run inside the openpi uv env:  uv run python droid_demo.py OUT_DIR
"""

import json
import pathlib
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REPO = "lerobot/droid_100"
CAMS = ["observation.images.exterior_image_1_left", "observation.images.wrist_image_left"]
# Franka "ready" pose. droid_100 only stores the gripper's xyz/rotation, not joint angles,
# so we give the model this standard pose as its joint state.
READY_JOINTS = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])
SAMPLES_PER_PROMPT = 3
JOINT_NAMES = [f"joint {i + 1}" for i in range(7)] + ["gripper"]


def grab_frame(video, t):
    """Return the RGB frame at time t (seconds) from an AV1 video."""
    try:
        import av

        with av.open(str(video)) as c:
            s = c.streams.video[0]
            c.seek(int(max(t - 1.0, 0) / s.time_base), stream=s)
            for f in c.decode(s):
                if f.time is not None and f.time >= t - 1e-3:
                    return f.to_ndarray(format="rgb24")
    except Exception as e:  # PyAV may lack an AV1 decoder; fall back to ffmpeg
        print("PyAV failed:", e)
    png = OUT / "_tmp.png"
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(png)], check=True)
    return np.asarray(Image.open(png).convert("RGB"))


def square(img, size=224):
    """Pad to square then resize, like openpi's resize_with_pad."""
    h, w = img.shape[:2]
    m = max(h, w)
    canvas = np.zeros((m, m, 3), np.uint8)
    canvas[(m - h) // 2 : (m - h) // 2 + h, (m - w) // 2 : (m - w) // 2 + w] = img
    return np.asarray(Image.fromarray(canvas).resize((size, size)))


# ---- 1. pick a few real DROID scenes -------------------------------------------------
episodes = pd.read_parquet(hf_hub_download(REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
print("episode columns:", list(episodes.columns))
videos = {cam: hf_hub_download(REPO, f"videos/{cam}/chunk-000/file-000.mp4", repo_type="dataset") for cam in CAMS}

scenes, seen = [], set()
for _, ep in episodes.iterrows():
    task = ep["tasks"][0] if len(ep["tasks"]) else ""
    if not task or task in seen:
        continue
    seen.add(task)
    imgs = {cam: grab_frame(videos[cam], float(ep[f"videos/{cam}/from_timestamp"]) + 0.5) for cam in CAMS}
    scenes.append({"episode": int(ep["episode_index"]), "task": task, "images": imgs})
    if len(scenes) == 4:
        break

# Same scene, different instructions: the real one + two others.
OTHER_PROMPTS = ["open the drawer", "pick up the cup and move it to the left"]

# ---- 2. load pi0.5 --------------------------------------------------------------------
t0 = time.time()
config = _config.get_config("pi05_droid")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, ckpt)
load_s = time.time() - t0
print(f"model loaded in {load_s:.0f}s")

results = {"load_seconds": load_s, "infer_seconds": [], "scenes": []}
for si, sc in enumerate(scenes):
    prompts = [sc["task"]] + OTHER_PROMPTS
    obs = {
        "observation/exterior_image_1_left": square(sc["images"][CAMS[0]]),
        "observation/wrist_image_left": square(sc["images"][CAMS[1]]),
        "observation/joint_position": READY_JOINTS,
        "observation/gripper_position": np.array([0.0]),
    }
    chunks = {}
    for p in prompts:
        chunks[p] = []
        for _ in range(SAMPLES_PER_PROMPT):  # flow matching starts from random noise, so samples differ
            t = time.time()
            a = policy.infer({**obs, "prompt": p})["actions"]
            results["infer_seconds"].append(time.time() - t)
            chunks[p].append(np.asarray(a))
    means = {p: np.mean(c, 0) for p, c in chunks.items()}
    within = float(np.mean([np.std(c, 0).mean() for c in chunks.values()]))
    across = float(np.mean([np.abs(means[a] - means[b]).mean() for i, a in enumerate(prompts) for b in prompts[i + 1 :]]))
    results["scenes"].append(
        {"episode": sc["episode"], "real_task": sc["task"], "prompts": prompts,
         "chunk_shape": list(chunks[prompts[0]][0].shape),
         "spread_same_prompt": within, "difference_between_prompts": across,
         "actions": {p: [c.tolist() for c in v] for p, v in chunks.items()}}
    )

    # ---- 3. figure: what the robot sees -> what it plans, per prompt -------------------
    fig = plt.figure(figsize=(16, 4.2 * len(prompts)))
    gs = fig.add_gridspec(len(prompts), 3, width_ratios=[1, 1, 2.2])
    for r, p in enumerate(prompts):
        for c, cam in enumerate(CAMS):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(obs["observation/exterior_image_1_left" if c == 0 else "observation/wrist_image_left"])
            ax.set_title(["front camera", "wrist camera"][c])
            ax.axis("off")
        ax = fig.add_subplot(gs[r, 2])
        m = means[p]
        steps = np.arange(m.shape[0]) / 15.0
        for j in range(7):
            ax.plot(steps, m[:, j], label=JOINT_NAMES[j])
        ax.plot(steps, m[:, 7], "k--", lw=2.5, label="gripper (1 = close)")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("seconds into the future")
        ax.set_ylabel("joint velocity command")
        ax.set_title(f'prompt: "{p}"' + ("  (the real task)" if r == 0 else ""), fontsize=12, weight="bold")
        if r == 0:
            ax.legend(ncol=4, fontsize=8, loc="lower left")
    fig.suptitle(f"pi0.5 action chunk ({m.shape[0]} steps x {m.shape[1]} numbers), same scene, different instructions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"scene{si + 1}_prompts.png", dpi=110)
    plt.close(fig)
    print(f"scene {si + 1}: '{sc['task']}' spread(same prompt)={within:.3f} diff(between prompts)={across:.3f}")

inf = results["infer_seconds"]
results["first_infer_seconds"] = inf[0]
results["median_infer_seconds"] = float(np.median(inf[1:]))
print(f"first inference (includes compile) {inf[0]:.1f}s, then median {results['median_infer_seconds'] * 1000:.0f} ms")
(OUT / "droid_results.json").write_text(json.dumps(results, indent=1))
