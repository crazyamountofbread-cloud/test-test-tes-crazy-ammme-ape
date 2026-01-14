from flask import Flask, request, jsonify, send_file
import os
import re
import textwrap
import unicodedata
import tempfile
import subprocess
import urllib.request
import base64
import json
import random

app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True})


# ======================================================
# /health
# ======================================================
@app.get("/health")
def health():
    return jsonify({"ok": True})


# ======================================================
# /get  (Instagram Reel -> direct mp4 CDN url)
# ======================================================
@app.get("/get")
def get_direct():
    return jsonify({
        "error": "disabled",
        "details": "yt-dlp removed from this service. Provide a direct CDN mp4 to /still or upload mp4 to /render_binary."
    }), 501


# ======================================================
# /still  (directUrl -> jpg)
# ======================================================
@app.get("/still")
def still():
    direct_url = request.args.get("directUrl")
    if not direct_url:
        return jsonify({"error": "missing directUrl"}), 400

    tmp_dir = tempfile.mkdtemp(prefix="still_")
    out_path = os.path.join(tmp_dir, "frame.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", direct_url,
        "-frames:v", "1",
        "-q:v", "2",
        out_path
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not os.path.exists(out_path):
            return jsonify({
                "error": "ffmpeg still failed",
                "stderr": (r.stderr or "")[-1200:]
            }), 500
        return send_file(out_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": "exception", "details": str(e)}), 500


# ======================================================
# Text helpers
# ======================================================
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ff_escape_text(s: str) -> str:
    # Robust escaping for ffmpeg drawtext
    s = s.replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace(",", r"\,")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    s = s.replace("\r", "")
    s = s.replace("\n", r"\n")
    s = s.replace("[", r"\[").replace("]", r"\]")
    return s


# ======================================================
# Helpers (crop + auto-fit)
# ======================================================
from dataclasses import dataclass
from PIL import ImageFont
import numpy as np
import cv2


def resolve_font_path(candidates: list[str], test_size: int = 48) -> str:
    for p in candidates:
        if not p:
            continue
        p = p.strip()
        if not p:
            continue
        if os.path.exists(p):
            try:
                ImageFont.truetype(p, test_size)
                return p
            except Exception:
                continue
    raise RuntimeError("No usable TTF font found.")


def escape_filter_path(p: str) -> str:
    return p.replace("\\", "\\\\").replace(":", r"\:").replace(",", r"\,")


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


# ======================================================
# /render_binary  (upload mp4 -> final mp4)
# ======================================================
@app.post("/render_binary")
def render_binary():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    f = request.files["file"]
    caption = normalize_text(request.form.get("caption", ""))

    vid_id = re.sub(r"[^a-zA-Z0-9_-]", "", request.form.get("id", "video"))
    tmp_dir = tempfile.mkdtemp(prefix="render_")

    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    f.save(in_path)

    CANVAS_W = 1080
    CANVAS_H = 1920

    font_candidates = [
        os.environ.get("FONTFILE", ""),
        "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "./GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    fontfile = resolve_font_path(font_candidates)
    font_ff = escape_filter_path(fontfile)

    # Simplified bbox: full frame
    cap = cv2.VideoCapture(in_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    bbox = BBox(0, 0, w, h)

    out_cw = CANVAS_W
    out_ch = int(CANVAS_W * h / w)

    x0 = 0
    y0 = (CANVAS_H - out_ch) // 2

    random.seed(vid_id)
    jx = random.randint(-2, 2)
    jy = random.randint(-2, 2)

    x0j = max(0, min(CANVAS_W - out_cw, x0 + jx))
    y0j = max(0, min(CANVAS_H - out_ch, y0 + jy))

    text = ff_escape_text(caption)

    # ======================================================
    # FILTER GRAPH (FIXED)
    # ======================================================
    fc = f"[0:v]split=2[vbgsrc][vfgsrc];"

    fc += (
        f"[vbgsrc]"
        f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
        f"scale={CANVAS_W}:{CANVAS_H}:flags=lanczos,"
        f"gblur=sigma=18,"
        f"format=rgba,"
        f"colorchannelmixer=aa=0.35"
        f"[bg];"
    )

    fc += (
        f"[vfgsrc]"
        f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
        f"scale={out_cw}:{out_ch}:flags=lanczos,"
        f"pad={CANVAS_W}:{CANVAS_H}:{x0j}:{y0j}:color=black@0,"
        f"format=rgba"
        f"[fg];"
    )

    fc += (
        f"[bg][fg]overlay=0:0,"
        f"drawtext=fontfile='{font_ff}':"
        f"text='{text}':"
        f"x=(w-text_w)/2:y=h*0.78:"
        f"fontsize=64:"
        f"fontcolor=white:"
        f"borderw=4:bordercolor=black,"
        f"fps=30"
        f"[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0 or not os.path.exists(out_path):
        return jsonify({
            "error": "ffmpeg failed",
            "stderr_tail": (r.stderr or "")[-2000:]
        }), 500

    return send_file(out_path, mimetype="video/mp4")
