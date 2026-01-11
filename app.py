from flask import Flask, request, jsonify, send_file
import yt_dlp
import os
import re
import textwrap
import unicodedata
import tempfile
import subprocess
import urllib.request

app = Flask(__name__)

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
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "missing url"}), 400

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            direct = info.get("url")
            if not direct:
                return jsonify({"error": "no direct url returned"}), 500
            return jsonify({"directUrl": direct})
    except Exception as e:
        return jsonify({"error": "yt-dlp failed", "details": str(e)[:1200]}), 500


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

    fontfile = os.environ.get(
        "FONTFILE",
        "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf"
    )
    logo_path = os.environ.get("LOGO_PATH", "./Logo.png")

    if not os.path.exists(fontfile):
        app.logger.error(f"[render_binary] fontfile missing: {fontfile}")
        return jsonify({"error": "missing fontfile on server", "path": fontfile}), 500

    if not os.path.exists(logo_path):
        app.logger.error(f"[render_binary] logo missing: {logo_path}")
        return jsonify({"error": "missing Logo.png on server", "path": logo_path}), 500

    line1, line2 = make_caption_lines(caption)
    font_ff = fontfile.replace(",", r"\,").replace(":", r"\:")

    filter_complex = (
        "[0:v]"
        "scale=1280:720:force_original_aspect_ratio=increase,"
        "crop=1280:720,"
        "scale=1080:-1,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        "[base];"
        "[1:v]scale=220:-1[logo];"
        "[base][logo]overlay=x=(W-w)/2:y=H*0.04[withlogo];"
        "[withlogo]"
        f"drawtext=text='{ff_escape_text(line1)}':fontfile={font_ff}:"
        "fontsize=56:fontcolor=white:borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.25-36,"
        f"drawtext=text='{ff_escape_text(line2)}':fontfile={font_ff}:"
        "fontsize=56:fontcolor=white:borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.25+36,"
        "drawtext=text='Siga @SuperEmAlta':"
        f"fontfile={font_ff}:"
        "fontsize=38:fontcolor=white:borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.70"
        "[vout]"
    )

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    try:
        f.save(in_path)
        app.logger.info("[render_binary] file saved, running ffmpeg")

        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-i", logo_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            out_path
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=280)

        if r.returncode != 0 or not os.path.exists(out_path):
            app.logger.error("[render_binary] ffmpeg FAILED")
            app.logger.error((r.stderr or "")[-1200:])
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
