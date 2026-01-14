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
import time
import uuid
import fcntl

app = Flask(__name__)

# ======================================================
# Global single-file queue lock (1 request at a time)
# Works across gunicorn workers/processes.
# ======================================================
LOCK_PATH = os.environ.get("RENDER_LOCK_PATH", "/tmp/render_binary.lock")
LOCK_WAIT_S = int(os.environ.get("RENDER_LOCK_WAIT_S", "3600"))  # max wait in queue

def acquire_render_lock():
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    f = os.fdopen(fd, "r+")
    start = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # write current holder info (debug)
            f.seek(0)
            f.truncate()
            f.write(f"pid={os.getpid()} ts={time.time()}\n")
            f.flush()
            return f
        except BlockingIOError:
            if time.time() - start > LOCK_WAIT_S:
                try:
                    f.close()
                except:
                    pass
                raise TimeoutError("render queue lock timeout")
            time.sleep(0.25)

def release_render_lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_UN)
    except:
        pass
    try:
        f.close()
    except:
        pass

FFMPEG_TIMEOUT_S = int(os.environ.get("FFMPEG_TIMEOUT_S", "900"))

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
    # yt-dlp removido do projeto. Este endpoint permanece apenas para não quebrar integrações.
    # Use /still passando directUrl (CDN mp4) ou envie o mp4 direto para /render_binary.
    return jsonify({
        "error": "disabled",
        "details": "yt-dlp removed from this service. Provide...to /still via directUrl, or upload the mp4 to /render_binary."
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


def wrap_caption_two_lines(s: str, width: int = 26):
    """Wrap caption into at most 2 lines. If longer, ellipsis."""
    s = normalize_text(s)
    if not s:
        return "", ""

    lines = textwrap.wrap(s, width=width)
    if len(lines) <= 2:
        if len(lines) == 1:
            return lines[0], ""
        return lines[0], lines[1]

    # Merge rest into 2nd line and ellipsis
    second = " ".join(lines[1:])
    second = normalize_text(second)
    # Hard cut if still huge
    if len(second) > width * 2:
        second = second[: width * 2].rstrip() + "…"

    # Ensure second line doesn't exceed width too badly
    lines2 = textwrap.wrap(second, width=width)
    if not lines2:
        return lines[0], ""

    if len(lines2) >= 1:
        line2 = lines2[0]
        # Add ellipsis if there were more than one wrapped segment
        if len(lines2) > 1:
            # ensure ellipsis fits
            if len(line2) >= width:
                line2 = line2[: max(0, width - 1)].rstrip()
            line2 = line2.rstrip() + "…"
        return lines[0], line2

    return lines[0], ""


def wrap_caption_two_lines_for_ffmpeg(s: str, width: int = 26):
    """Return two lines, each max width, suitable for drawtext with newline."""
    l1, l2 = wrap_caption_two_lines(s, width=width)
    if l2:
        return f"{l1}\n{l2}"
    return l1


def wrap_caption_two_lines_strict(s: str, width: int = 26):
    """Strict 2 lines: if line2 too long, ellipsis it."""
    s = normalize_text(s)
    if not s:
        return "", ""

    lines = textwrap.wrap(s, width=width)
    if len(lines) == 1:
        return lines[0], ""

    line1 = lines[0]
    line2 = " ".join(lines[1:])
    line2 = normalize_text(line2)

    # Ellipsis if it wraps further
    line2_parts = textwrap.wrap(line2, width=width)
    if len(line2_parts) > 1:
        last = line2_parts[0]
        if len(last) >= width:
            last = last[: max(0, width - 1)].rstrip()
        line2 = last.rstrip() + "…"
    else:
        line2 = line2_parts[0] if line2_parts else ""

    return line1, line2


def ff_escape_text(s: str) -> str:
    # Robust escaping for ffmpeg drawtext (prevents filter graph breaks)
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
# Helpers (new crop + auto-fit)
# ======================================================
from dataclasses import dataclass
from PIL import ImageFont
import numpy as np
import cv2

# ---- Font helpers (Pillow on some Linux builds can't load variable fonts) ----
def resolve_font_path(candidates: list[str], test_size: int = 48) -> str:
    """Return first font path that exists and Pillow can load."""
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
    raise RuntimeError("No usable TTF font found from candidates.")


def escape_filter_path(p: str) -> str:
    """Escape fontfile path for ffmpeg filter (Windows not needed on Render, but safe)."""
    return p.replace("\\", "\\\\").replace(":", r"\:").replace(",", r"\,")


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


def detect_foreground_bbox(video_path: str, max_frames: int = 20, sample_stride: int = 10) -> BBox:
    """
    Detect foreground bbox by motion + edges. Robust against black bars and static background.
    Returns bbox in original video coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for bbox detection")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        frame_count = 999999

    # sample indices
    idxs = []
    start = min(5, frame_count - 1)
    step = max(1, sample_stride)
    for k in range(start, min(frame_count, start + max_frames * step), step):
        idxs.append(k)

    prev_gray = None
    agg = np.zeros((h, w), dtype=np.float32)

    for target in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray_blur
            continue

        diff = cv2.absdiff(gray_blur, prev_gray)
        prev_gray = gray_blur

        # edges
        edges = cv2.Canny(gray_blur, 70, 140)

        # combine diff + edges
        m = cv2.normalize(diff.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        e = cv2.normalize(edges.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

        combo = 0.65 * m + 0.35 * e
        agg += combo

    cap.release()

    if np.max(agg) <= 1e-6:
        # fallback: full frame
        return BBox(0, 0, w, h)

    agg = agg / (np.max(agg) + 1e-9)
    # threshold
    mask = (agg > 0.15).astype(np.uint8) * 255

    # morph to fill holes
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return BBox(0, 0, w, h)

    cnt = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # expand slightly
    pad_x = int(bw * 0.06)
    pad_y = int(bh * 0.06)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)

    # sanity: minimum coverage
    if bw < int(w * 0.3) or bh < int(h * 0.3):
        return BBox(0, 0, w, h)

    return BBox(x, y, bw, bh)


def compute_scale_to_fit(w: int, h: int, target_w: int, target_h: int):
    """Return scale (out_w, out_h) to fit inside target while preserving aspect ratio."""
    if w <= 0 or h <= 0:
        return target_w, target_h

    ar = w / h
    tar = target_w / target_h

    if ar >= tar:
        # wider -> fit width
        out_w = target_w
        out_h = int(target_w / ar)
    else:
        # taller -> fit height
        out_h = target_h
        out_w = int(target_h * ar)

    return out_w, out_h


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ======================================================
# /render_binary  (upload mp4 -> final mp4)
# ======================================================
@app.post("/render_binary")
def render_binary():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    f = request.files["file"]
    caption = normalize_text(request.form.get("caption", ""))

    request_id = uuid.uuid4().hex[:12]
    lock_f = None
    try:
        app.logger.info("[render_binary %s] waiting queue lock", request_id)
        lock_f = acquire_render_lock()
        app.logger.info("[render_binary %s] acquired queue lock", request_id)
    except Exception as e:
        app.logger.error("[render_binary %s] queue lock error: %s", request_id, str(e))
        return jsonify({"error": "queue lock error", "details": str(e)[:2000]}), 503

    vid_id = re.sub(r"[^a-zA-Z0-9_-]", "", request.form.get("id", request_id))
    tmp_dir = tempfile.mkdtemp(prefix=f"render_{request_id}_")

    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    f.save(in_path)

    app.logger.info("[render_binary] START %s", in_path)

    try:
        CANVAS_W = 1080
        CANVAS_H = 1920

        # Caption styling
        caption_width = int(request.form.get("caption_width", "26") or "26")
        line1, line2 = wrap_caption_two_lines_strict(caption, width=caption_width)

        # CTA (optional)
        cta = normalize_text(request.form.get("cta", ""))
        cta_font_size = int(request.form.get("cta_font_size", "54") or "54")

        # Speed tweak (audio)
        atempo = float(request.form.get("atempo", "1.0") or "1.0")
        atempo = max(0.5, min(2.0, atempo))

        # Encoding speed/quality knobs (defaults are tuned for Render stability)
        preset = (request.form.get("preset") or os.environ.get("X264_PRESET", "ultrafast")).strip() or "ultrafast"
        try:
            crf = int(request.form.get("crf") or os.environ.get("X264_CRF", "23"))
        except Exception:
            crf = 23
        crf = max(16, min(35, crf))

        # Random micro-jitter deterministic by vid_id
        random.seed(vid_id)

        # --- Determine bbox (foreground), compute background sizing ---
        bbox = detect_foreground_bbox(in_path)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video for reading dims")
        iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # foreground crop dims
        cw = bbox.w
        ch = bbox.h

        # Fit foreground inside 9:16 canvas
        out_cw, out_ch = compute_scale_to_fit(cw, ch, CANVAS_W, CANVAS_H)
        x0 = (CANVAS_W - out_cw) // 2
        y0 = (CANVAS_H - out_ch) // 2

        # micro jitter
        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        x0j = clamp(x0 + jx, 0, CANVAS_W - out_cw)
        y0j = clamp(y0 + jy, 0, CANVAS_H - out_ch)

        # Font path (prefer bundled GoogleSans, fallback to DejaVu)
        font_candidates = [
            os.environ.get("FONTFILE", ""),
            "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
            "./GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        fontfile = resolve_font_path(font_candidates)
        font_ff = escape_filter_path(fontfile)

        # Build texts
        caption_text = line1 + ("\n" + line2 if line2 else "")
        caption_esc = ff_escape_text(caption_text)

        cta_esc = ff_escape_text(cta) if cta else ""
        x_cta = int(request.form.get("cta_x", "60") or "60")
        y_cta = int(request.form.get("cta_y", "1550") or "1550")

        # Caption position
        caption_font_size = int(request.form.get("caption_font_size", "64") or "64")
        caption_y = int(request.form.get("caption_y", "1490") or "1490")

        # =======================
        # FILTER GRAPH (FAST BG)
        # =======================
        fc = f"[0:v]split=2[vbgsrc][vfgsrc];"

        # Background: blur at half-res then upscale (MUCH faster)
        fc += (
            f"[vbgsrc]"
            f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
            f"scale={CANVAS_W}:{CANVAS_H}:flags=bicubic,"
            f"scale=540:960:flags=bicubic,"
            f"gblur=sigma=12,"
            f"scale={CANVAS_W}:{CANVAS_H}:flags=bicubic,"
            f"format=rgba,"
            f"colorchannelmixer=aa=0.35"
            f"[bg];"
        )

        fc += (
            f"[vfgsrc]"
            f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
            f"scale={out_cw}:{out_ch}:flags=lanczos,"
            f"setsar=1,setdar=9/16,"
            f"pad={CANVAS_W}:{CANVAS_H}:{x0j}:{y0j}:color=black@0,"
            f"format=rgba"
            f"[fg];"
        )

        fc += f"[bg][fg]overlay=0:0[v0];"

        v_in = "v0"

        # Caption
        if caption_esc:
            fc += (
                f"[{v_in}]drawtext=fontfile='{font_ff}':text='{caption_esc}':"
                f"fontsize={caption_font_size}:x=(w-text_w)/2:y={caption_y}:"
                f"fontcolor=white:"
                f"bordercolor=black:borderw=4:"
                f"line_spacing=10:"
                f"box=0"
                f"[v1];"
            )
            v_in = "v1"

        # CTA
        if cta_esc:
            fc += (
                f"[{v_in}]drawtext=fontfile='{font_ff}':text='{cta_esc}':"
                f"fontsize={cta_font_size}:x={x_cta}:y={y_cta}:"
                f"fontcolor=white:"
                f"bordercolor=black:borderw=4:"
                f"box=0,"
                f"fps=30[vout]"
            )
        else:
            fc += f"[{v_in}]fps=30[vout]"

        app.logger.info("[render_binary] running ffmpeg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", "0.35",
            "-i", in_path,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-map", "0:a?",
            "-af", f"atempo={atempo},volume=1.02",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-g", "90", "-keyint_min", "90", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-aspect", "9:16",
            out_path
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)

        # fallback 1: se falhar, tenta SEM -ss
        if r.returncode != 0 or not os.path.exists(out_path):
            if "-ss" in cmd:
                cmd2 = []
                skip_next = False
                for tok in cmd:
                    if skip_next:
                        skip_next = False
                        continue
                    if tok == "-ss":
                        skip_next = True
                        continue
                    cmd2.append(tok)
                app.logger.warning("[render_binary] ffmpeg failed w/ -ss, retrying without -ss")
                r = subprocess.run(cmd2, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)

        # fallback 2: se falhar, tenta SEM áudio (remove -map 0:a? e -af e setar -an)
        if r.returncode != 0 or not os.path.exists(out_path):
            cmd3 = []
            skip_next = False
            for j, tok in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if tok in ("-af",):
                    skip_next = True
                    continue
                if tok == "-map" and j + 1 < len(cmd) and cmd[j+1] == "0:a?":
                    skip_next = True
                    continue
                if tok in ("-c:a", "-b:a"):
                    skip_next = True
                    continue
                cmd3.append(tok)
            if "-an" not in cmd3:
                cmd3.insert(cmd3.index(out_path), "-an")
            app.logger.warning("[render_binary] ffmpeg failed w/ audio, retrying without audio")
            r = subprocess.run(cmd3, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)

        if r.returncode != 0 or not os.path.exists(out_path):
            app.logger.error("[render_binary] ffmpeg FAILED")
            app.logger.error((r.stderr or "")[-2000:])
            return jsonify({
                "error": "ffmpeg failed",
                "stderr_tail": (r.stderr or "")[-2000:]
            }), 500

        app.logger.info("[render_binary] DONE, sending file")
        return send_file(
            out_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=f"{vid_id}F.mp4"
        )

    except subprocess.TimeoutExpired:
        app.logger.error("[render_binary] ffmpeg TIMEOUT")
        return jsonify({"error": "ffmpeg timeout"}), 504

    except Exception as e:
        app.logger.exception("[render_binary] EXCEPTION")
        return jsonify({"error": "render exception", "details": str(e)[:2000]}), 500

    finally:
        # release queue lock
        if 'lock_f' in locals() and lock_f is not None:
            release_render_lock(lock_f)

        # cleanup tmp dir
        if 'tmp_dir' not in locals() or not tmp_dir:
            pass
        try:
            for fn in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, fn))
                except:
                    pass
            os.rmdir(tmp_dir)
        except:
            pass


# ======================================================
# App entry
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
