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
# /mp3  (MP4 upload -> MP3 download)
# ======================================================
@app.post("/mp3")
def mp4_to_mp3():
    app.logger.info("[mp3] START")

    if "file" in request.files:
        f = request.files["file"]
    elif len(request.files) > 0:
        f = next(iter(request.files.values()))
    else:
        return jsonify({"error": "missing file", "files_keys": list(request.files.keys())}), 400

    vid_id = str(request.form.get("id", "audio"))
    tmp_dir = tempfile.mkdtemp(prefix="mp3_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}.mp3")

    try:
        f.save(in_path)

        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            out_path
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        # fallback: se libmp3lame não existir no build, tenta mp3 encoder "mp3" (raramente necessário)
        if (r.returncode != 0 or not os.path.exists(out_path)):
            cmd2 = [
                "ffmpeg", "-y",
                "-i", in_path,
                "-vn",
                "-codec:a", "mp3",
                "-b:a", "192k",
                "-ar", "44100",
                out_path
            ]
            r = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)

        if r.returncode != 0 or not os.path.exists(out_path):
            return jsonify({
                "error": "ffmpeg mp3 failed",
                "stderr_tail": (r.stderr or "")[-2000:]
            }), 500

        return send_file(
            out_path,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name=f"{vid_id}.mp3"
        )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "ffmpeg timeout"}), 504

    except Exception as e:
        app.logger.exception("[mp3] EXCEPTION")
        return jsonify({"error": "mp3 exception", "details": str(e)[:2000]}), 500

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
    s = s.replace("'", "’")
    s = s.replace('"', "”")
    

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
from PIL import Image, ImageDraw, ImageFont

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

# ======================================================
# Timed captions (EAST subtitle box + white box + PIL text)
# Mirrors the validated standalone script. Only used when `timed_captions`
# is present and not equal to "0".
# ======================================================

# EAST config
EAST_W, EAST_H = 320, 320
EAST_SCORE_THRESH = float(os.environ.get("EAST_SCORE_THRESH", "0.55"))
EAST_NMS_THRESH = float(os.environ.get("EAST_NMS_THRESH", "0.30"))
EAST_IGNORE_TOP = float(os.environ.get("EAST_IGNORE_TOP", str(1/3)))

# White box + layout
TC_PAD_X = int(os.environ.get("TC_PAD_X", "30"))
TC_PAD_Y = int(os.environ.get("TC_PAD_Y", "60"))
TC_HEIGHT_SHRINK_RATIO = float(os.environ.get("TC_HEIGHT_SHRINK_RATIO", "0.07"))

TC_TEXT_RGB = (255, 255, 0)  # yellow
TC_STROKE_RGB = (0, 0, 0)    # black
TC_INNER_PAD_X = int(os.environ.get("TC_INNER_PAD_X", "28"))
TC_INNER_PAD_Y = int(os.environ.get("TC_INNER_PAD_Y", "22"))
TC_STROKE_W = int(os.environ.get("TC_STROKE_W", "6"))
TC_LINE_SPACING = float(os.environ.get("TC_LINE_SPACING", "1.10"))


def _tc_time_to_seconds(mm: str, ss: str, ms: str) -> float:
    return (int(mm) * 60.0) + float(int(ss)) + (int(ms) / 1000.0)


_TC_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)\s*-->\s*(\d+):(\d+):(\d+)")


def parse_timed_captions(text: str):
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    items = []
    for b in blocks:
        lines = [l.rstrip("\r") for l in b.splitlines() if l.strip() != ""]
        if len(lines) < 3:
            continue
        m = _TC_TIME_RE.search(lines[1])
        if not m:
            continue
        start = _tc_time_to_seconds(m.group(1), m.group(2), m.group(3))
        end = _tc_time_to_seconds(m.group(4), m.group(5), m.group(6))
        caption = "\n".join(lines[2:]).strip()
        if caption:
            items.append([start, end, caption])
    items.sort(key=lambda x: x[0])
    return items


def extend_caption_ends_midpoint(items):
    if not items:
        return items
    for i in range(len(items) - 1):
        end_i = items[i][1]
        start_next = items[i + 1][0]
        if start_next > end_i:
            items[i][1] = (end_i + start_next) / 2.0
        else:
            items[i][1] = end_i
    return items


def active_caption(items, t: float):
    for s, e, txt in items:
        if s <= t <= e:
            return txt
    return None


def _east_decode(scores, geometry):
    rects, confs = [], []
    h, w = scores.shape[2:4]
    for y in range(h):
        for x in range(w):
            score = scores[0, 0, y, x]
            if score < EAST_SCORE_THRESH:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos, sin = np.cos(angle), np.sin(angle)
            h_box = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w_box = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = int(offsetX + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            endY = int(offsetY - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
            startX = int(endX - w_box)
            startY = int(endY - h_box)
            rects.append((startX, startY, endX, endY))
            confs.append(float(score))
    return rects, confs


def _east_nms_simple(boxes, scores):
    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = scores.argsort()[::-1]
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = w * h
        idxs = idxs[1:][overlap <= EAST_NMS_THRESH]
    return boxes[pick]


def detect_text_boxes_east(frame_bgr: np.ndarray, net):
    H, W = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1.0, (EAST_W, EAST_H),
        (123.68, 116.78, 103.94),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ])
    rects, confs = _east_decode(scores, geometry)
    boxes = _east_nms_simple(rects, confs)
    rW, rH = W / EAST_W, H / EAST_H
    out = []
    for x1, y1, x2, y2 in boxes:
        ax1 = int(x1 * rW); ay1 = int(y1 * rH)
        ax2 = int(x2 * rW); ay2 = int(y2 * rH)
        ax1 = max(0, min(W - 1, ax1)); ax2 = max(0, min(W - 1, ax2))
        ay1 = max(0, min(H - 1, ay1)); ay2 = max(0, min(H - 1, ay2))
        if ax2 > ax1 and ay2 > ay1:
            out.append((ax1, ay1, ax2, ay2))
    return out


def filter_ignore_top(boxes, H: int, ignore_top: float):
    y_min = int(H * ignore_top)
    return [b for b in boxes if b[3] > y_min]


def rect_area(r):
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def intersection_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def expand_rect(base, other):
    return (
        min(base[0], other[0]),
        min(base[1], other[1]),
        max(base[2], other[2]),
        max(base[3], other[3]),
    )


def apply_padding(rect_xyxy, W: int, H: int):
    x1, y1, x2, y2 = rect_xyxy
    return (
        max(0, x1 - TC_PAD_X),
        max(0, y1 - TC_PAD_Y),
        min(W - 1, x2 + TC_PAD_X),
        min(H - 1, y2 + TC_PAD_Y),
    )


def shrink_height(rect_xyxy, ratio: float, H: int):
    x1, y1, x2, y2 = rect_xyxy
    h = max(1, y2 - y1)
    shrink = int(h * ratio)
    ny1 = min(H - 2, y1 + shrink)
    ny2 = max(ny1 + 1, y2 - shrink)
    return (x1, ny1, x2, ny2)


def _tc_wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int):
    out_lines = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ") if raw_line.strip() else [""]
        cur = ""
        for w in words:
            test = (cur + " " + w).strip() if cur else w
            if font.getlength(test) <= max_w:
                cur = test
            else:
                if cur.strip():
                    out_lines.append(cur.strip())
                    cur = w
                else:
                    out_lines.append(w)
                    cur = ""
        out_lines.append(cur.strip())
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return out_lines


def _tc_measure(lines, font: ImageFont.FreeTypeFont):
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * TC_LINE_SPACING)
    widths = [int(font.getlength(l)) for l in lines] if lines else [0]
    return max(widths) if widths else 0, line_h * len(lines), line_h


def _tc_fit_font(text: str, font_path: str, box_w: int, box_h: int):
    lo, hi = 10, 140
    best = 32
    for _ in range(14):
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        lines = _tc_wrap_text(text, f, box_w)
        _, total_h, _ = _tc_measure(lines, f)
        if total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def draw_caption_pil(frame_bgr: np.ndarray, rect_xyxy, text: str, font_path: str):
    text = unicodedata.normalize("NFC", text)
    x1, y1, x2, y2 = rect_xyxy
    ix1 = x1 + TC_INNER_PAD_X
    iy1 = y1 + TC_INNER_PAD_Y
    ix2 = x2 - TC_INNER_PAD_X
    iy2 = y2 - TC_INNER_PAD_Y
    box_w = max(10, ix2 - ix1)
    box_h = max(10, iy2 - iy1)
    font_px = _tc_fit_font(text, font_path, box_w, box_h)
    font = ImageFont.truetype(font_path, font_px)
    lines = _tc_wrap_text(text, font, box_w)
    _, total_h, line_h = _tc_measure(lines, font)

    overlay = Image.new("RGBA", (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start_y = iy1 + max(0, (box_h - total_h) // 2)

    for i, line in enumerate(lines):
        tw = int(font.getlength(line))
        x = ix1 + max(0, (box_w - tw) // 2)
        y = start_y + i * line_h
        draw.text(
            (x, y),
            line,
            font=font,
            fill=TC_TEXT_RGB + (255,),
            stroke_width=TC_STROKE_W,
            stroke_fill=TC_STROKE_RGB + (255,),
        )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(frame_rgb).convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def ensure_east_model(tmp_dir: str) -> str:
    candidates = [
        os.environ.get("EAST_MODEL", "").strip(),
        "./frozen_east_text_detection.pb",
        "/app/frozen_east_text_detection.pb",
        "./models/frozen_east_text_detection.pb",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"[render_binary] EAST model found at: {p}", flush=True)
            return p

    url = os.environ.get(
        "EAST_MODEL_URL",
        "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/frozen_east_text_detection.pb",
    )
    outp = os.path.join(tmp_dir, "frozen_east_text_detection.pb")
    print(f"[render_binary] EAST model not found. Downloading from: {url}", flush=True)

    try:
        urllib.request.urlretrieve(url, outp)
    except Exception as e:
        print(f"[render_binary] EAST download FAILED: {e}", flush=True)
        raise

    if not os.path.exists(outp):
        raise RuntimeError("EAST model download finished but file missing")

    print(f"[render_binary] EAST model downloaded to: {outp}", flush=True)
    return outp


def apply_timed_captions_overlay(base_video: str, out_video: str, box_xyxy, items, font_path: str):
    print(f"[render_binary] [tc] overlay start | base={base_video}", flush=True)
    print(f"[render_binary] [tc] overlay box={box_xyxy}", flush=True)
    print(f"[render_binary] [tc] overlay items={len(items)}", flush=True)
    print(f"[render_binary] [tc] overlay font={font_path}", flush=True)

    cap = cv2.VideoCapture(base_video)
    if not cap.isOpened():
        raise RuntimeError("failed to open base video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # try h264 first
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
        if not out.isOpened():
            raise RuntimeError("VideoWriter not opened")

    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(W - 2, int(x1)))
    x2 = max(x1 + 1, min(W - 1, int(x2)))
    y1 = max(0, min(H - 2, int(y1)))
    y2 = max(y1 + 1, min(H - 1, int(y2)))

    frame_idx = 0
    last_print = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # white box ALWAYS (enquanto timed_captions estiver habilitado)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        t = frame_idx / fps
        txt = active_caption(items, t)
        if txt is not None:
            frame = draw_caption_pil(frame, (x1, y1, x2, y2), txt, font_path)

        out.write(frame)

        frame_idx += 1
        if frame_idx - last_print >= 60:
            last_print = frame_idx
            print(f"[render_binary] [tc] overlay progress frame={frame_idx} t={t:.2f}s", flush=True)

    cap.release()
    out.release()
    print(f"[render_binary] [tc] overlay done frames={frame_idx}", flush=True)


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
    # logs garantidos no Render
    def rb(msg: str):
        print(msg, flush=True)
        try:
            app.logger.info(msg)
        except Exception:
            pass

    rb("[render_binary] START")

    if "file" in request.files:
        f = request.files["file"]
    elif len(request.files) > 0:
        f = next(iter(request.files.values()))
    else:
        rb("[render_binary] ERROR missing file")
        return jsonify({"error": "missing file", "files_keys": list(request.files.keys())}), 400

    caption = request.form.get("caption", "")
    vid_id = str(request.form.get("id", "video"))

    # timed captions: aceita nomes alternativos também (pra não depender 100% do node)
    timed_captions_raw = (request.form.get("timed_captions", "") or "").strip()
    if not timed_captions_raw:
        timed_captions_raw = (request.form.get("timedCaptions", "") or "").strip()
    if not timed_captions_raw:
        timed_captions_raw = (request.form.get("timestamps", "") or "").strip()

    timed_captions_enabled = bool(timed_captions_raw) and timed_captions_raw != "0"

    rb(f"[render_binary] id={vid_id} caption_len={len(caption)}")
    rb(f"[render_binary] files_keys={list(request.files.keys())}")
    rb(f"[render_binary] timed_enabled={timed_captions_enabled} timed_len={len(timed_captions_raw)}")
    if timed_captions_enabled:
        rb(f"[render_binary] timed_head={timed_captions_raw[:80]!r}")

    # Font
    font_candidates = [
        os.environ.get("FONTFILE", "").strip(),
        "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "./GoogleSans-VariableFont_GRAD,opsz,wght.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    fontfile = resolve_font_path(font_candidates)

    if not os.path.exists(fontfile):
        rb(f"[render_binary] ERROR fontfile missing: {fontfile}")
        return jsonify({"error": "missing fontfile on server", "path": fontfile}), 500

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    try:
        f.save(in_path)
        rb("[render_binary] upload saved")

        # --- bbox detect (seu código original) ---
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
            w, h = ffprobe_dims(in_path)
            bbox = BBox(0, 0, w, h)
            rb("[render_binary] WARN bbox detection failed, using full frame")
        else:
            bbox = median_bbox(bboxes)

        src_w, src_h = ffprobe_dims(in_path)
        bbox.x = max(0, min(bbox.x, src_w - 1))
        bbox.y = max(0, min(bbox.y, src_h - 1))
        bbox.w = max(1, min(bbox.w, src_w - bbox.x))
        bbox.h = max(1, min(bbox.h, src_h - bbox.y))

        rb(f"[render_binary] bbox={bbox}")

        # --- canvas (original) ---
        CANVAS_W, CANVAS_H = 720, 1280
        scale = min(CANVAS_W / bbox.w, CANVAS_H / bbox.h)
        out_cw = int(round(bbox.w * scale))
        out_ch = int(round(bbox.h * scale))
        x0 = (CANVAS_W - out_cw) // 2
        y0 = (CANVAS_H - out_ch) // 2

        FG_TRIM_X = 10
        out_cw_fg = max(2, out_cw - (FG_TRIM_X * 2))

        # --- TOP TEXT (original) ---
        top_gap = 15
        space_above = max(10, y0 - top_gap)
        top_box_w = int(round(CANVAS_W * 0.82))
        fit = fit_text_top(
            caption, fontfile,
            max_w=top_box_w,
            max_h=space_above,
            max_font=58,
            min_font=28,
            line_spacing=1.06,
            max_lines=3,
        )

        TOP_MARGIN = 28
        y_block = (y0 - top_gap) - fit.text_h
        if y_block < TOP_MARGIN:
            y_block = TOP_MARGIN

        line_xs = [max(0, (CANVAS_W - lw) // 2) for lw in fit.line_ws]
        line_ys = [y_block + i * fit.line_h for i in range(len(fit.lines))]

        # --- CTA (original) ---
        cta_text = "Siga @SuperEmAlta"
        cta_font_size = 48
        font_obj = ImageFont.truetype(fontfile, cta_font_size)
        bb = font_obj.getbbox(cta_text)
        cta_w = bb[2] - bb[0]
        cta_h = bb[3] - bb[1]

        CTA_PAD_X = 28
        CTA_PAD_Y = 14
        CTA_BOTTOM_PAD = 18

        y_cta = y0 + out_ch - cta_h - CTA_BOTTOM_PAD
        if y_cta < 0:
            y_cta = 0
        x_cta = max(0, (CANVAS_W - cta_w) // 2)

        box_w = cta_w + (CTA_PAD_X * 2)
        box_h = cta_h + (CTA_PAD_Y * 2)
        x_box = max(0, (CANVAS_W - box_w) // 2)
        y_box = max(0, y_cta - CTA_PAD_Y)

        font_ff = escape_filter_path(fontfile)

        # jitter/noise/audio (original)
        try:
            random.seed(f"{vid_id}-{os.environ.get('SEED_SALT','0')}")
        except Exception:
            pass

        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        x0j = max(0, min(CANVAS_W - out_cw_fg, (x0 + jx) + FG_TRIM_X))
        y0j = max(0, min(CANVAS_H - out_ch, y0 + jy))

        noise_strength = random.choice([1, 2, 3])
        atempo = random.choice([0.99, 1.0, 1.01])

        # ======================================================
        # Timed captions (EAST) — FULL fg scan, BEFORE hflip
        # ======================================================
        tc_box_canvas = None
        tc_items = []

        if timed_captions_enabled:
            rb("[render_binary] [tc] START parsing timed captions")
            try:
                tc_text = timed_captions_raw.replace("\\n", "\n") if "\\n" in timed_captions_raw else timed_captions_raw
                tc_text = unicodedata.normalize("NFC", tc_text)
                tc_items = parse_timed_captions(tc_text)
                tc_items = extend_caption_ends_midpoint(tc_items)
                rb(f"[render_binary] [tc] parsed items={len(tc_items)}")
                if tc_items:
                    rb(f"[render_binary] [tc] first={tc_items[0][0]:.3f}->{tc_items[0][1]:.3f} '{tc_items[0][2][:40]}'")

                if not tc_items:
                    rb("[render_binary] [tc] parsed=0 -> disable timed overlay")
                    timed_captions_enabled = False
            except Exception as e:
                rb(f"[render_binary] [tc] parse FAILED: {e}")
                timed_captions_enabled = False
                tc_items = []

        if timed_captions_enabled:
            rb("[render_binary] [tc] Resolving EAST model...")
            model_path = ensure_east_model(tmp_dir)
            rb(f"[render_binary] [tc] EAST model path={model_path}")

            rb("[render_binary] [tc] Building fg_noflip temp video...")
            fg_path = os.path.join(tmp_dir, "__fg_noflip.mp4")

            # mesmo recorte que alimenta FG, mas SEM hflip
            # (30 fps, sem áudio)
            vf_fg = (
                f"crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},"
                f"scale={out_cw}:{out_ch}:flags=lanczos,"
                f"fps=30,setsar=1"
            )
            cmd_fg = [
                "ffmpeg", "-y",
                "-ss", "0.35",
                "-i", in_path,
                "-vf", vf_fg,
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-movflags", "+faststart",
                fg_path
            ]
            rb("[render_binary] [tc] CMD fg_noflip=" + " ".join(cmd_fg))
            rfg = subprocess.run(cmd_fg, capture_output=True, text=True, timeout=180)
            if rfg.returncode != 0 or not os.path.exists(fg_path):
                rb("[render_binary] [tc] fg_noflip FAILED")
                rb((rfg.stderr or "")[-2000:])
                timed_captions_enabled = False
                tc_items = []
            else:
                rb("[render_binary] [tc] fg_noflip OK")

        if timed_captions_enabled:
            rb("[render_binary] [tc] Scanning FULL fg_noflip with EAST...")
            net = cv2.dnn.readNet(model_path)

            cap_fg = cv2.VideoCapture(fg_path)
            if not cap_fg.isOpened():
                rb("[render_binary] [tc] ERROR open fg_noflip")
                timed_captions_enabled = False
                tc_items = []
            else:
                fg_fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0
                fgW = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
                fgH = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
                rb(f"[render_binary] [tc] fg dims={fgW}x{fgH} fps={fg_fps:.2f}")

                global_rect = None
                frame_idx = 0
                last_print = 0

                # stride leve pra performance, mas ainda “vídeo inteiro”
                stride = int(os.environ.get("TC_STRIDE", "2"))
                stride = max(1, stride)

                while True:
                    ok, frame = cap_fg.read()
                    if not ok:
                        break

                    if frame_idx % stride == 0:
                        boxes = _east_filter_ignore_top(_east_detect_text_boxes(frame, net), fgH, EAST_IGNORE_TOP)
                        if boxes:
                            if global_rect is None:
                                global_rect = (
                                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                                )
                            else:
                                for b in boxes:
                                    inter = _intersection_area(global_rect, b)
                                    if _rect_area(b) > 0 and (inter / _rect_area(b)) < 0.5:
                                        global_rect = _expand_rect(global_rect, b)

                    frame_idx += 1
                    if frame_idx - last_print >= 150:
                        last_print = frame_idx
                        rb(f"[render_binary] [tc] scan progress frame={frame_idx}")

                cap_fg.release()

                if global_rect is None:
                    rb("[render_binary] [tc] NO BOX FOUND -> disable timed overlay")
                    timed_captions_enabled = False
                    tc_items = []
                else:
                    rb(f"[render_binary] [tc] raw global_rect={global_rect}")

                    global_rect = _apply_padding(global_rect, fgW, fgH)
                    global_rect = _shrink_height(global_rect, TC_HEIGHT_SHRINK_RATIO, fgH)
                    gx1, gy1, gx2, gy2 = global_rect

                    rb(f"[render_binary] [tc] padded+shrink rect={global_rect}")

                    # map pre-hflip -> post-hflip
                    hx1 = fgW - gx2
                    hx2 = fgW - gx1

                    tc_box_canvas = (
                        x0j + hx1,
                        y0j + gy1,
                        x0j + hx2,
                        y0j + gy2,
                    )
                    rb(f"[render_binary] [tc] tc_box_canvas={tc_box_canvas}")

        # ======================================================
        # ffmpeg render original (sem timed)
        # ======================================================
        fc = ""
        fc += (f"[0:v]crop={bbox.w}:{bbox.h}:{bbox.x}:{bbox.y},setsar=1[vcrop];")
        fc += f"[vcrop]split=2[vbgsrc][vfgsrc];"
        fc += (
            f"[vbgsrc]"
            f"scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=increase,"
            f"crop={CANVAS_W}:{CANVAS_H},"
            f"boxblur=luma_radius=10:luma_power=1:chroma_radius=10:chroma_power=1"
            f"[bg];"
        )
        fc += (
            f"[vfgsrc]"
            f"scale={out_cw}:{out_ch}:flags=lanczos,"
            f"hflip,"
            f"setsar=1"
            f"[fg];"
        )
        fc += f"[bg][fg]overlay={x0j}:{y0j}[v0];"

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

        cta_esc = ff_escape_text(cta_text)
        fc += (
            f"[{v_in}]"
            f"drawbox=x={x_box}:y={y_box}:w={box_w}:h={box_h}:color=black@0.5:t=fill,"
            f"drawtext=fontfile='{font_ff}':text='{cta_esc}':"
            f"fontsize={cta_font_size}:x={x_cta}:y={y_cta}:"
            f"fontcolor=white:borderw=4:bordercolor=black,"
            f"noise=alls={noise_strength}:allf=t,"
            f"fps=30[vout]"
        )

        rb("[render_binary] running ffmpeg base render")
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
        rb("[render_binary] CMD=" + " ".join(cmd))

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=280)

        if r.returncode != 0 or not os.path.exists(out_path):
            rb("[render_binary] ffmpeg base FAILED")
            rb((r.stderr or "")[-4000:])
            return jsonify({"error": "ffmpeg failed", "stderr_tail": (r.stderr or "")[-2000:]}), 500

        rb("[render_binary] ffmpeg base OK")

        # ======================================================
        # Timed overlay pass (white + timed captions)
        # ======================================================
        if timed_captions_enabled and tc_box_canvas is not None and tc_items:
            rb("[render_binary] [tc] APPLY OVERLAY PASS")
            overlay_noaudio = os.path.join(tmp_dir, f"{vid_id}__tc_noaudio.mp4")
            overlay_final = os.path.join(tmp_dir, f"{vid_id}__tc_final.mp4")

            apply_timed_captions_overlay(out_path, overlay_noaudio, tc_box_canvas, tc_items, fontfile)

            rb("[render_binary] [tc] mux audio back")
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", overlay_noaudio,
                "-i", out_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                overlay_final
            ]
            rb("[render_binary] [tc] CMD mux=" + " ".join(cmd_mux))
            rmux = subprocess.run(cmd_mux, capture_output=True, text=True, timeout=280)
            if rmux.returncode != 0 or not os.path.exists(overlay_final):
                rb("[render_binary] [tc] mux FAILED")
                rb((rmux.stderr or "")[-4000:])
            else:
                rb("[render_binary] [tc] overlay OK -> replacing out_path")
                out_path = overlay_final
        else:
            rb(f\"[render_binary] [tc] SKIP overlay | enabled={timed_captions_enabled} box={tc_box_canvas is not None} items={len(tc_items)}\")

        rb("[render_binary] DONE, sending file")
        return send_file(out_path, mimetype="video/mp4", as_attachment=True, download_name=f\"{vid_id}F.mp4\")

    except subprocess.TimeoutExpired:
        rb("[render_binary] ERROR ffmpeg timeout")
        return jsonify({"error": "ffmpeg timeout"}), 504

    except Exception as e:
        rb(f\"[render_binary] EXCEPTION: {e}\")
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
