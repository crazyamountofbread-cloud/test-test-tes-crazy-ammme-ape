from flask import Flask, request, send_file, jsonify
import os
import tempfile
import subprocess
import re

app = Flask(__name__)

# ======================================================
# Helpers
# ======================================================
def ff_escape_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace(",", r"\,")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    s = s.replace("\n", r"\n")
    return s


# ======================================================
# Render endpoint (ONLY THIS)
# ======================================================
@app.post("/render_binary")
def render_binary():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    video_file = request.files["file"]
    caption = ff_escape_text(request.form.get("caption", ""))

    # sanitize id
    vid_id = re.sub(r"[^a-zA-Z0-9_-]", "", request.form.get("id", "video"))

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}_out.mp4")

    video_file.save(in_path)

    CANVAS_W = 1080
    CANVAS_H = 1920

    # Simple 9:16 center fit, black background
    filter_chain = (
        f"[0:v]"
        f"scale='if(gt(a,9/16),{CANVAS_W},-2)':'if(gt(a,9/16),-2,{CANVAS_H})',"
        f"pad={CANVAS_W}:{CANVAS_H}:(ow-iw)/2:(oh-ih)/2:black"
        f"[v0]"
    )

    if caption:
        filter_chain += (
            f";[v0]drawtext="
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"text='{caption}':"
            f"fontsize=64:"
            f"fontcolor=white:"
            f"x=(w-text_w)/2:"
            f"y=h*0.78:"
            f"borderw=3:bordercolor=black"
            f"[vout]"
        )
        map_video = "[vout]"
    else:
        map_video = "[v0]"

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-filter_complex", filter_chain,
        "-map", map_video,
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
        if r.returncode != 0 or not os.path.exists(out_path):
            return jsonify({
                "error": "ffmpeg failed",
                "stderr": (r.stderr or "")[-2000:]
            }), 500

        return send_file(
            out_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=f"{vid_id}.mp4"
        )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "ffmpeg timeout"}), 504

    finally:
        try:
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
        except:
            pass


# ======================================================
# App entry
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
