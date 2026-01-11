from flask import Flask, request, jsonify
import yt_dlp

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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
