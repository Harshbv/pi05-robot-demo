"""Layer 2 demo: pi0.5 controlling a simulated robot arm in LIBERO.

Based on openpi/examples/libero/main.py. Talks to the policy server
(scripts/serve_policy.py --env LIBERO) over a websocket, records labelled videos.
Run with the LIBERO py3.8 venv:  python libero_demo.py OUT_DIR
"""

import collections
import json
import math
import pathlib
import sys

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
REPLAN = 5
WAIT = 10
SEED = 7
MAX_STEPS = {"libero_goal": 300, "libero_spatial": 220, "libero_object": 280, "libero_10": 520}

# Re-worded instructions the model never saw in training (language robustness test).
PARAPHRASES = {
    "open the middle drawer of the cabinet": "pull open the cabinet's middle drawer",
    "put the bowl on the plate": "place the bowl onto the plate",
    "turn on the stove": "switch the stove on",
}


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def label(frame, wrist, text, status):
    """Side-by-side front + wrist view, upscaled, with the prompt written on top."""
    both = np.concatenate([frame, wrist], 1)
    both = cv2.resize(both, (both.shape[1] * 2, both.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    bar = np.full((70, both.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f'"{text}"', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    color = (0, 140, 0) if status == "SUCCESS" else (0, 0, 200) if status == "FAILED" else (90, 90, 90)
    cv2.putText(bar, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color[::-1], 2)
    return np.concatenate([bar, both], 0)


def rollout(client, task, init_state, prompt, max_steps, video_path):
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=RES, camera_widths=RES)
    env.seed(SEED)
    env.reset()
    obs = env.set_init_state(init_state)
    plan, frames, done, t = collections.deque(), [], False, 0
    while t < max_steps + WAIT:
        if t < WAIT:  # let objects settle
            obs, _, done, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # rotate 180 to match training
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
        frames.append((img, wrist))
        if not plan:
            chunk = client.infer({
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": prompt,
            })["actions"]
            plan.extend(chunk[:REPLAN])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
        t += 1
    env.close()
    status = "SUCCESS" if done else "FAILED"
    video = [label(f, w, prompt, "running...") for f, w in frames]
    video += [label(*frames[-1], prompt, status)] * 15  # hold the last frame
    imageio.mimwrite(video_path, video, fps=10)
    return done, len(frames)


def main():
    client = _websocket_client_policy.WebsocketClientPolicy("localhost", 8000)
    suite_name = "libero_goal"  # every task in LIBERO-Goal shares the same kitchen scene
    suite = benchmark.get_benchmark_dict()[suite_name]()
    log = []

    def run(task_id, prompt, trial, tag):
        task = suite.get_task(task_id)
        init = suite.get_task_init_states(task_id)[trial]
        name = f"{tag}_t{task_id}_trial{trial}_" + prompt.replace(" ", "_")[:60] + ".mp4"
        ok, steps = rollout(client, task, init, prompt, MAX_STEPS[suite_name], OUT / name)
        rec = {"tag": tag, "task_id": task_id, "official_task": task.language, "prompt": prompt,
               "trial": trial, "success": bool(ok), "steps": steps, "video": name}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        (OUT / "libero_results.json").write_text(json.dumps(log, indent=1))

    # A. Same kitchen, 10 different instructions, 2 tries each.
    for tid in range(suite.n_tasks):
        for trial in range(2):
            run(tid, suite.get_task(tid).language, trial, "A_same_scene")

    # B. Paraphrased instructions (wording it never trained on).
    for tid in range(suite.n_tasks):
        official = suite.get_task(tid).language
        if official in PARAPHRASES:
            for trial in range(2):
                run(tid, PARAPHRASES[official], trial, "B_paraphrase")

    ok = sum(r["success"] for r in log if r["tag"] == "A_same_scene")
    n = sum(r["tag"] == "A_same_scene" for r in log)
    print(f"LIBERO-Goal success: {ok}/{n}")


if __name__ == "__main__":
    main()
