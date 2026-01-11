from flask import Flask, request, jsonify, send_file
import yt_dlp
import os, tempfile, subprocess

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
