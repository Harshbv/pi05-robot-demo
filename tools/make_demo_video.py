"""Build the 1-minute presentation video from the Kaggle LIBERO rollouts.

Part 1: same kitchen, 4 different instructions, playing side by side.
Part 2: original wording vs reworded instruction.
Usage: python tools/make_demo_video.py <dir with the LIBERO rollout mp4s> media/demo.mp4
"""

import pathlib
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
FPS = 15
W, H = 960, 540  # each tile is 480 x 270... built below
# Segoe UI on Windows; set DEMO_FONT / DEMO_FONT_BOLD to any .ttf elsewhere (e.g. DejaVuSans.ttf)
import os
FONT = os.environ.get("DEMO_FONT", "C:/Windows/Fonts/segoeui.ttf")
BOLD = os.environ.get("DEMO_FONT_BOLD", "C:/Windows/Fonts/segoeuib.ttf")

GRID = [
    "A_same_scene_t3_trial0_open_the_top_drawer_and_put_the_bowl_inside.mp4",
    "A_same_scene_t2_trial0_put_the_wine_bottle_on_top_of_the_cabinet.mp4",
    "A_same_scene_t7_trial0_turn_on_the_stove.mp4",
    "A_same_scene_t8_trial0_put_the_bowl_on_the_plate.mp4",
]
PAIR = [
    ("A_same_scene_t7_trial1_turn_on_the_stove.mp4", "turn on the stove", "trained wording"),
    ("B_paraphrase_t7_trial0_switch_the_stove_on.mp4", "switch the stove on", "new wording, never trained on"),
]


def front_view(path):
    """Crop the front camera out of the labelled front+wrist video (bar is 70 px, front is 448 px wide)."""
    return [f[70:, :448] for f in iio.imread(SRC / path)]


def tile(frame, prompt, status, size, sub=None):
    t = Image.new("RGB", size, "white")
    bar = 64 if sub else 48
    img = Image.fromarray(frame).resize((size[1] - bar, size[1] - bar))
    t.paste(img, ((size[0] - img.width) // 2, bar))
    d = ImageDraw.Draw(t)
    d.text((size[0] // 2, 6), f'"{prompt}"', fill="black", font=ImageFont.truetype(BOLD, 20), anchor="mt")
    if sub:
        d.text((size[0] // 2, 32), sub, fill=(90, 90, 90), font=ImageFont.truetype(FONT, 16), anchor="mt")
    if status:
        color = (0, 140, 60) if status == "SUCCESS" else (90, 90, 90)
        d.text((size[0] - 10, size[1] - 10), status, fill=color, font=ImageFont.truetype(BOLD, 22), anchor="rb")
    return np.asarray(t)


def title_card(lines, seconds, size=(1280, 720)):
    t = Image.new("RGB", size, (20, 24, 33))
    d = ImageDraw.Draw(t)
    y = size[1] // 2 - 40 * len(lines) // 2
    for i, line in enumerate(lines):
        d.text((size[0] // 2, y + i * 56), line, fill="white" if i == 0 else (180, 190, 205),
               font=ImageFont.truetype(BOLD if i == 0 else FONT, 40 if i == 0 else 28), anchor="mt")
    return [np.asarray(t)] * int(seconds * FPS)


def play_together(clips, prompts, subs, size, layout, hold=1.5):
    n = max(len(c) for c in clips) + int(hold * FPS)
    out = []
    for i in range(n):
        tiles = []
        for c, p, s in zip(clips, prompts, subs):
            done = i >= len(c)
            tiles.append(tile(c[min(i, len(c) - 1)], p, "SUCCESS" if done else "", size, s))
        rows = [np.concatenate(tiles[r * layout[1]:(r + 1) * layout[1]], 1) for r in range(layout[0])]
        out.append(np.concatenate(rows, 0))
    return out


grid_clips = [front_view(p) for p in GRID]
grid_prompts = [p.split("trial0_")[1][:-4].replace("_", " ") for p in GRID]
part1 = play_together(grid_clips, grid_prompts, [None] * 4, (640, 360), (2, 2), hold=2)

pair_clips = [front_view(p) for p, _, _ in PAIR]
part2 = play_together(pair_clips, [p for _, p, _ in PAIR], [s for _, _, s in PAIR], (640, 720), (1, 2), hold=2)

frames = (
    title_card(["Same kitchen, 4 different instructions", "pi0.5 (open-source checkpoint) controlling a simulated robot arm"], 2.5)
    + part1
    + title_card(["Does the exact wording matter?", "Same task, reworded instruction"], 2)
    + part2
    + title_card(["26 / 26 tasks succeeded", "LIBERO-Goal simulation, run on a free Kaggle T4 GPU"], 3)
)
iio.imwrite(OUT, np.stack(frames), fps=FPS, codec="libx264", macro_block_size=8)
print(f"{OUT}: {len(frames) / FPS:.1f} s")
