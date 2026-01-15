from flask import Flask, request, jsonify, send_file
import os
import tempfile
import subprocess
import random
import re
import unicodedata

app = Flask(__name__)

# =========================
# Helpers
# =========================
def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def ff_escape_text(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
         .replace(":", r"\:")
         .replace(",", r"\,")
         .replace("'", r"\'")
         .replace("%", r"\%")
         .replace("\n", r"\n")
    )

# =========================
# RENDER ONLY
# =========================
@app.post("/render_binary")
def render_binary():

    # ---- file ----
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    f = request.files["file"]
    caption = normalize_text(request.form.get("caption", ""))
    vid_id = re.sub(r"[^a-zA-Z0-9_-]", "", request.form.get("id", "video"))

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")
    f.save(in_path)

    # ---- constants (UNCHANGED EDIT) ----
    CANVAS_W = 1080
    CANVAS_H = 1920

    caption_font_size = int(request.form.get("caption_font_size", "64"))
    caption_y = int(request.form.get("caption_y", "1490"))

    random.seed(vid_id)

    # ---- FULL FRAME (NO BBOX LOGIC) ----
    # input is already vertical in your use-case
    out_cw = CANVAS_W
    out_ch = CANVAS_H
    x0 = 0
    y0 = 0

    caption_esc = ff_escape_text(caption)

    # ---- FILTER GRAPH (SAME EDIT STRUCTURE) ----
    print("FILTER GRAPH SAME EDIT STRUCTURE")
    fc = (
        f"[0:v]split=2[vbg][vfg];"
        f"[vbg]"
        f"scale={CANVAS_W}:{CANVAS_H}:flags=lanczos,"
        f"gblur=sigma=18,"
        f"format=rgba,"
        f"colorchannelmixer=aa=0.35"
        f"[bg];"
        f"[vfg]"
        f"scale={out_cw}:{out_ch}:flags=lanczos,"
        f"format=rgba"
        f"[fg];"
        f"[bg][fg]overlay=0:0"
    )
    print("MEXENDO COM FONTE")
    if caption_esc:
        fc += (
            f",drawtext=text='{caption_esc}':"
            f"fontcolor=white:"
            f"fontsize={caption_font_size}:"
            f"x=(w-text_w)/2:y={caption_y}:"
            f"borderw=4:bordercolor=black"
        )

    fc += ",fps=30[vout]"

    print("CONFIGURANDO FFMPEG")

    # ---- FFMPEG ----
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        out_path
    ]
    print("COMECA SUBPROCESS")
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0 or not os.path.exists(out_path):
        return jsonify({
            "error": "ffmpeg failed",
            "stderr": (r.stderr or "")[-2000:]
        }), 500

    return send_file(
        out_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"{vid_id}F.mp4"
    )


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
