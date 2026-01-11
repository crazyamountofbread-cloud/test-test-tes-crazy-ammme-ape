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


# -----------------------
# Existing: /health
# -----------------------
@app.get("/health")
def health():
    return jsonify({"ok": True})


# -----------------------
# Existing: /get  (Instagram Reel -> direct mp4 CDN url)
# -----------------------
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


# -----------------------
# Existing: /still (directUrl -> jpg)
# -----------------------
@app.get("/still")
def still():
    # recebe a URL direta do mp4 (CDN)
    direct_url = request.args.get("directUrl")
    if not direct_url:
        return jsonify({"error": "missing directUrl"}), 400

    # gera um jpg do frame ~0.2s (evita frame preto)
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
            return jsonify({"error": "ffmpeg failed", "details": (r.stderr or "")[-1200:]}), 500
        return send_file(out_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": "still failed", "details": str(e)[:1200]}), 500
    finally:
        # best-effort cleanup (não muda resposta)
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            os.rmdir(tmp_dir)
        except:
            pass


# -----------------------
# NEW: helpers for /render
# -----------------------
def sanitize_caption(s: str) -> str:
    # Remove CR/BOM, normaliza unicode, remove controles invisíveis e emojis (So).
    s = s.replace("\r", "").lstrip("\ufeff")
    s = unicodedata.normalize("NFKC", s)

    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] == "C":  # control/format/etc
            continue
        if cat == "So":    # símbolos pictográficos (emoji) -> você decidiu ignorar
            continue
        out.append(ch)

    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_caption_lines(caption: str, width: int = 28, max_lines: int = 2) -> tuple[str, str]:
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
    # Escapes necessários pro drawtext text='...'
    s = s.replace("\\", "\\\\")
    s = s.replace(":", r"\:")
    s = s.replace("'", r"\'")
    s = s.replace("%", r"\%")
    return s


def download_to_file(url: str, out_path: str, timeout_sec: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(out_path, "wb") as f:
        f.write(resp.read())


def run(cmd: list[str], timeout: int = 240) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


# -----------------------
# NEW: /render (1-pass 5 -> 5F)
# -----------------------
@app.post("/render")
def render():
    """
    Body JSON:
      {
        "video_url": "https://....mp4"   (pode ser o directUrl)
        "caption": "texto..."
        "id": 5
      }

    Returns: mp4 binário (idF.mp4)
    """
    data = request.get_json(silent=True) or {}

    video_url = data.get("video_url") or data.get("directUrl") or data.get("url")
    caption = data.get("caption", "")
    vid_id = str(data.get("id", "video"))

    if not video_url:
        return jsonify({"error": "missing video_url"}), 400

    fontfile = os.environ.get("FONTFILE", "./fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf")
    logo_path = os.environ.get("LOGO_PATH", "./Logo.png")

    if not os.path.exists(logo_path):
        return jsonify({"error": "missing Logo.png on server", "path": logo_path}), 500
    if not os.path.exists(fontfile):
        return jsonify({"error": "missing fontfile on server", "path": fontfile}), 500

    # 2 linhas max (pipeline)
    line1, line2 = make_caption_lines(caption, width=28, max_lines=2)

    # CRÍTICO: escapar vírgulas e ":" no caminho da fonte dentro do filtergraph
    font_ff = fontfile.replace(",", r"\,").replace(":", r"\:")

    # 1-pass: crop horizontal -> pad vertical -> logo -> textos (2 linhas + CTA)
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
        "fontsize=56:fontcolor=white:"
        "borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.25-36,"
        f"drawtext=text='{ff_escape_text(line2)}':fontfile={font_ff}:"
        "fontsize=56:fontcolor=white:"
        "borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.25+36,"
        "drawtext=text='Siga @SuperEmAlta':"
        f"fontfile={font_ff}:"
        "fontsize=38:fontcolor=white:"
        "borderw=3:bordercolor=black@0.90:"
        "shadowx=2:shadowy=2:shadowcolor=black@0.35:"
        "x=(w-text_w)/2:y=h*0.70"
        "[vout]"
    )

    tmp_dir = tempfile.mkdtemp(prefix="render_")
    in_path = os.path.join(tmp_dir, f"{vid_id}.mp4")
    out_path = os.path.join(tmp_dir, f"{vid_id}F.mp4")

    try:
        app.logger.info(f"[render] downloading video -> {in_path}")
        download_to_file(video_url, in_path, timeout_sec=60)

        app.logger.info("[render] ffmpeg 1-pass render start")
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

        code, _, err = run(cmd, timeout=240)
        if code != 0 or not os.path.exists(out_path):
            return jsonify({"error": "render failed", "details": err[-2000:]}), 500

        app.logger.info(f"[render] done -> {out_path}")
        return send_file(out_path, mimetype="video/mp4", as_attachment=True, download_name=f"{vid_id}F.mp4")

    except Exception as e:
        return jsonify({"error": "render failed", "details": str(e)[:2000]}), 500

    finally:
        # best-effort cleanup
        try:
            for fn in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, fn))
                except:
                    pass
            os.rmdir(tmp_dir)
        except:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
