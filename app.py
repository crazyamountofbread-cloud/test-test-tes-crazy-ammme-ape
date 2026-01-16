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
    # yt-dlp removido do projeto. Este endpoint permanece apenas para não quebrar integrações.
    # Use /still passando directUrl (CDN mp4) ou envie o mp4 direto para /render_binary.
    return jsonify({
        "error": "disabled",
        "details": "yt-dlp removed from this service. Provide a direct mp4 URL to /still via directUrl, or upload the mp4 to /render_binary."
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
    out_path = os.path.join(tmp_dir, "still.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", "0.2",
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
        return jsonify({"error": "still exception", "details": str(e)[:1200]}), 500
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            os.rmdir(tmp_dir)
        except:
            pass


# ======================================================
# Helpers (render)
# ======================================================
def sanitize_caption(s: str) -> str:
    s = s.replace("\r", "").lstrip("\ufeff")
    s = unicodedata.normalize("NFKC", s)

    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] == "C":
            continue
        if cat == "So":  # emojis removidos
            continue
        out.append(ch)

    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_caption_lines(caption: str, width: int = 28, max_lines: int = 2):
    s = sanitize_caption(caption)
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        if len(last) >= width:
            last = last[: max(0, width - 1)].rstrip()
        lines[-1] = last.rstrip() + "…"

    line1 = lines[0] if len(lines) > 0 else ""
    line2 = lines[1] if len(lines) > 1 else ""
    return line1, line2


def ff_escape_text(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    return s


# ======================================================
# Helpers (new crop + auto-fit)
# ======================================================

from dataclasses import dataclass
from PIL import ImageFont

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
    raise RuntimeError(
        "No usable TTF font found. Provide a static .ttf via FONTFILE env or include one in ./fonts."
    )

import numpy as np
import cv2

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

def ffprobe_dims(path: str):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    data = json.loads(p.stdout)
    st = data["streams"][0]
    return int(st["width"]), int(st["height"])

def extract_frame(in_video: str, out_png: str, t: float) -> bool:
    cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", in_video, "-vframes", "1", "-q:v", "2", out_png]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out_png)

def detect_bg_color(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    band = max(8, h // 50)
    top = img_bgr[:band, :, :]
    bot = img_bgr[h-band:, :, :]

    b = np.concatenate([top[...,0].astype(np.float32), bot[...,0].astype(np.float32)], axis=0)
    g = np.concatenate([top[...,1].astype(np.float32), bot[...,1].astype(np.float32)], axis=0)
    r = np.concatenate([top[...,2].astype(np.float32), bot[...,2].astype(np.float32)], axis=0)
    luma = 0.114*b + 0.587*g + 0.299*r
    m = float(np.mean(luma))
    return (255,255,255) if m >= 128 else (0,0,0)

def smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(5, k | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")

def largest_contiguous_segment(idx: np.ndarray):
    if idx.size == 0:
        return None
    d = np.diff(idx)
    breaks = np.where(d > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends   = np.r_[idx[breaks], idx[-1]]
    lengths = ends - starts + 1
    j = int(np.argmax(lengths))
    return int(starts[j]), int(ends[j])

def build_content_mask(img_bgr: np.ndarray, tol: int) -> np.ndarray:
    bg = np.array(detect_bg_color(img_bgr), dtype=np.int16)
    img16 = img_bgr.astype(np.int16)
    diff = np.max(np.abs(img16 - bg[None, None, :]), axis=2)
    content = (diff > tol).astype(np.uint8)

    h, w = img_bgr.shape[:2]
    k = max(3, (min(h, w) // 280) | 1)
    kernel = np.ones((k, k), np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN, kernel, iterations=1)
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel, iterations=2)
    return content

def find_bbox_ignore_overlay(img_bgr: np.ndarray, tol: int = 28, row_thresh: float = 0.08):
    # y_only: corta só topo/baixo, mantém largura total
    h, w = img_bgr.shape[:2]
    content = build_content_mask(img_bgr, tol=tol)
    row_frac = np.mean(content, axis=1)
    row_frac_s = smooth1d(row_frac, k=max(11, h // 120))
    rows = np.where(row_frac_s > row_thresh)[0]
    seg = largest_contiguous_segment(rows)
    if seg is None:
        raise RuntimeError("bbox not found")
    y1, y2 = seg
    return BBox(x=0, y=y1, w=w, h=(y2 - y1 + 1))

def median_bbox(bboxes):
    xs = np.array([b.x for b in bboxes], dtype=np.int32)
    ys = np.array([b.y for b in bboxes], dtype=np.int32)
    ws = np.array([b.w for b in bboxes], dtype=np.int32)
    hs = np.array([b.h for b in bboxes], dtype=np.int32)
    return BBox(int(np.median(xs)), int(np.median(ys)), int(np.median(ws)), int(np.median(hs)))

def escape_filter_path(p: str) -> str:
    return p.replace("\\", "\\\\").replace(":", r"\:").replace(",", r"\,")

@dataclass
class FitResult:
    font_size: int
    lines: list
    text_h: int
    line_h: int
    line_ws: list

def wrap_text_to_width(text: str, font: ImageFont.FreeTypeFont, max_w: int):
    words = text.strip().split()
    if not words:
        return [""]
    def width(s: str) -> int:
        bb = font.getbbox(s)
        return bb[2] - bb[0]
    lines = []
    cur = words[0]
    for w in words[1:]:
        t = cur + " " + w
        if width(t) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    fixed = []
    for ln in lines:
        if width(ln) <= max_w:
            fixed.append(ln)
            continue
        buf = ""
        for ch in ln:
            tt = buf + ch
            if width(tt) <= max_w or not buf:
                buf = tt
            else:
                fixed.append(buf)
                buf = ch
        if buf:
            fixed.append(buf)
    return fixed

def measure_lines(lines, font: ImageFont.FreeTypeFont, line_spacing: float):
    b = font.getbbox("Ag")
    base_line_h = (b[3] - b[1])
    line_h = int(round(base_line_h * line_spacing))
    widths = []
    for ln in lines:
        bb = font.getbbox(ln)
        widths.append(bb[2] - bb[0])
    text_h = line_h * len(lines) - (line_h - base_line_h)
    return text_h, line_h, widths

def fit_text_top(text: str, font_path: str, max_w: int, max_h: int,
                 max_font: int = 96, min_font: int = 34,
                 line_spacing: float = 1.15, max_lines: int = 3):
    text = sanitize_caption(text)
    for size in range(max_font, min_font - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_text_to_width(text, font, max_w)
        if len(lines) > max_lines:
            continue
        text_h, line_h, line_ws = measure_lines(lines, font, line_spacing)
        if text_h <= max_h:
            return FitResult(size, lines, text_h, line_h, line_ws)
    font = ImageFont.truetype(font_path, min_font)
    lines = wrap_text_to_width(text, font, max_w)[:max_lines]
    text_h, line_h, line_ws = measure_lines(lines, font, line_spacing)
    return FitResult(min_font, lines, min(text_h, max_h), line_h, line_ws)


# ======================================================
# /render_binary  (MAIN RENDER ENDPOINT)
# ======================================================
@app.post("/render_binary")
def render_binary():
    app.logger.info("[render_binary] START")

    if "file" in request.files:
        f = request.files["file"]
    elif len(request.files) > 0:
        f = next(iter(request.files.values()))
    else:
        app.logger.error("[render_binary] missing file")
        return jsonify({
            "error": "missing file",
            "files_keys": list(request.files.keys())
        }), 400

    caption = request.form.get("caption", "")
    vid_id = str(request.form.get("id", "video"))

    app.logger.info(f"[render_binary] id={vid_id} caption_len={len(caption)}")
    app.logger.info(f"[render_binary] files_keys={list(request.files.keys())}")

    # Font: try user-provided FONTFILE first, then common repo paths, then system fonts.
    font_candidates = [
        os.environ.get("FONTFILE", "").strip(),
        "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "./GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    fontfile = resolve_font_path(font_candidates)

    # (logo removida do render; mantemos LOGO_PATH só por compat, mas não exigimos arquivo)
    logo_path = os.environ.get("LOGO_PATH", "./Logo.png")

    if not os.path.exists(fontfile):
        app.logger.error(f"[render_binary] fontfile missing: {fontfile}")
        return jsonify({"error": "missing fontfile on server", "path": fontfile}), 500

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    try:
        # save upload
        f.save(in_path)
        app.logger.info("[render_binary] file saved, detecting bbox")

        # --- detect bbox (ignore overlay text) using a few frames ---
        sample_times = [0.5, 1.2, 2.0, 3.0]
        frames = []
        for i, t in enumerate(sample_times):
            fp = os.path.join(tmp_dir, f"___f{i}.png")
            if extract_frame(in_path, fp, t):
                frames.append(fp)

        bboxes = []
        for fp in frames:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                bboxes.append(find_bbox_ignore_overlay(img, tol=28, row_thresh=0.08))
            except Exception:
                continue

        if not bboxes:
            # fallback: assume full frame
            w, h = ffprobe_dims(in_path)
            bbox = BBox(0, 0, w, h)
            app.logger.warning("[render_binary] bbox detection failed, using full frame")
        else:
            bbox = median_bbox(bboxes)

        src_w, src_h = ffprobe_dims(in_path)
        # clamp bbox
        bbox.x = max(0, min(bbox.x, src_w - 1))
        bbox.y = max(0, min(bbox.y, src_h - 1))
        bbox.w = max(1, min(bbox.w, src_w - bbox.x))
        bbox.h = max(1, min(bbox.h, src_h - bbox.y))

        # --- fixed 9:16 canvas ---
        CANVAS_W, CANVAS_H = 1080, 1920

        # Foreground: scale cropped to fit canvas
        scale = min(CANVAS_W / bbox.w, CANVAS_H / bbox.h)
        out_cw = int(round(bbox.w * scale))
        out_ch = int(round(bbox.h * scale))
        x0 = (CANVAS_W - out_cw) // 2
        y0 = (CANVAS_H - out_ch) // 2

        # --- TOP TEXT auto-fit, bottom aligned to (y0 - 5) ---
        top_gap = 5
        space_above = max(10, y0 - top_gap)
        top_box_w = int(round(CANVAS_W * 0.82))
        fit = fit_text_top(
            caption,
            fontfile,
            max_w=top_box_w,
            max_h=space_above,
            max_font=96,
            min_font=34,
            line_spacing=1.15,
            max_lines=3,
        )
        y_block = (y0 - top_gap) - fit.text_h
        if y_block < 0:
            y_block = 0

        line_xs = []
        for lw in fit.line_ws:
            line_xs.append(max(0, (CANVAS_W - lw) // 2))
        line_ys = [y_block + i * fit.line_h for i in range(len(fit.lines))]

        # --- bottom CTA ---
        cta_text = "Siga @SuperEmAlta"
        cta_font_size = 48
        font_obj = ImageFont.truetype(fontfile, cta_font_size)
        bb = font_obj.getbbox(cta_text)
        cta_w = bb[2] - bb[0]
        cta_h = bb[3] - bb[1]
        cta_gap = 40
        y_cta = y0 + out_ch + cta_gap
        if y_cta + cta_h > CANVAS_H:
            y_cta = max(0, CANVAS_H - cta_h)
        x_cta = max(0, (CANVAS_W - cta_w) // 2)

        # --- ffmpeg filter_complex ---
        font_ff = escape_filter_path(fontfile)

        # ======================================================
        # Anti-fingerprint micro-variations (imperceptible)
        # ======================================================
        try:
            random.seed(f"{vid_id}-{os.environ.get('SEED_SALT','0')}")
        except Exception:
            pass

        # 1) Jitter in positions
        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        x0j = max(0, min(CANVAS_W - out_cw, x0 + jx))
        y0j = max(0, min(CANVAS_H - out_ch, y0 + jy))

        # 2) Tiny noise
        noise_strength = random.choice([1, 2, 3])

        # 3) Tiny audio tempo shift
        atempo = random.choice([0.99, 1.0, 1.01])

        # Background + Foreground with split=2 (CRITICO)
        fc = ""
        fc += f"[0:v]split=2[vbgsrc][vfgsrc];"

        # BG chain (fill + crop + blur)
        fc += (
            f"[vbgsrc]"
            f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=increase,"
            f"crop={CANVAS_W}:{CANVAS_H},"
            f"boxblur=luma_radius=10:luma_power=1:chroma_radius=10:chroma_power=1"
            f"[bg];"
        )

        # FG chain (crop detectado + scale fit)
        fc += (
            f"[vfgsrc]"
            f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
            f"scale={out_cw}:{out_ch}:flags=lanczos,"
            f"setsar=1"
            f"[fg];"
        )

        fc += f"[bg][fg]overlay={x0j}:{y0j}[v0];"

        # Top text (stroke 4px)
        v_in = "v0"
        for i, ln in enumerate(fit.lines):
            ln_esc = ff_escape_text(ln)
            v_out = f"vt{i}"
            fc += (
                f"[{v_in}]drawtext=fontfile='{font_ff}':text='{ln_esc}':"
                f"fontsize={fit.font_size}:x={line_xs[i]}:y={line_ys[i]}:"
                f"fontcolor=white:borderw=4:bordercolor=black"
                f"[{v_out}];"
            )
            v_in = v_out

        # CTA (stroke 4px) + noise + fps -> vout
        cta_esc = ff_escape_text(cta_text)
        fc += (
            f"[{v_in}]drawtext=fontfile='{font_ff}':text='{cta_esc}':"
            f"fontsize={cta_font_size}:x={x_cta}:y={y_cta}:"
            f"fontcolor=white:borderw=4:bordercolor=black,"
            f"noise=alls={noise_strength}:allf=t,"
            f"fps=30[vout]"
        )

        app.logger.info("[render_binary] running ffmpeg (new motor)")
        cmd = [
            "ffmpeg", "-y",
            "-ss", "0.35",
            "-i", in_path,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-map", "0:a?",
            "-af", f"atempo={atempo},volume=1.02",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-g", "90", "-keyint_min", "90", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-aspect", "9:16",
            out_path
        ]

        app.logger.error("[render_binary] FILTER_COMPLEX ↓↓↓\n" + fc)
        app.logger.error("[render_binary] CMD ↓↓↓\n" + " ".join(cmd))


        r = subprocess.run(cmd, capture_output=True, text=True, timeout=280)

        # fallback 1: se falhar, tenta SEM -ss
        if r.returncode != 0 or not os.path.exists(out_path):
            if "-ss" in cmd:
                cmd2 = cmd.copy()
                i = cmd2.index("-ss")
                del cmd2[i:i+2]
                r = subprocess.run(cmd2, capture_output=True, text=True, timeout=280)

        # fallback 2: se falhar, tenta SEM áudio
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
                cmd3.insert(cmd3.index("-c:v"), "-an")
                
            r = subprocess.run(cmd3, capture_output=True, text=True, timeout=280)

        if r.returncode != 0 or not os.path.exists(out_path):
            app.logger.error("[render_binary] ffmpeg FAILED")
            app.logger.error((r.stderr or "")[-8000:])
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
