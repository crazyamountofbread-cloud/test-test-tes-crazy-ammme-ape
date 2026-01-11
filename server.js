const express = require("express");
const { spawn } = require("child_process");

const app = express();

/**
 * Health check
 */
app.get("/health", (_, res) => {
  res.json({ ok: true });
});

/**
 * Retorna URL direto do vídeo (leve, sem baixar no Render)
 */
app.get("/get", (req, res) => {
  const url = req.query.url;
  if (!url) {
    return res.status(400).json({ error: "missing url" });
  }

  const p = spawn("yt-dlp", ["-g", "--no-playlist", url]);

  let stdout = "";
  let stderr = "";

  p.stdout.on("data", (d) => {
    stdout += d.toString();
  });

  p.stderr.on("data", (d) => {
    stderr += d.toString();
  });

  // evita crash se yt-dlp não existir
  p.on("error", (err) => {
    return res.status(500).json({
      error: "yt-dlp spawn error",
      details: String(err),
    });
  });

  p.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: "yt-dlp failed",
        details: stderr.slice(-1000),
      });
    }

    const lines = stdout
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    if (!lines.length) {
      return res.status(500).json({ error: "no video url returned" });
    }

    // primeira URL normalmente funciona
    res.json({
      directUrl: lines[0],
      allUrls: lines,
    });
  });
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log("Server running on port", PORT);
});
