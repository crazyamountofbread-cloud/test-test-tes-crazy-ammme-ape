const express = require("express");
const { spawn } = require("child_process");

const app = express();

app.get("/health", (_, res) => res.json({ ok: true }));

app.get("/get", (req, res) => {
  const url = req.query.url;
  if (!url) return res.status(400).json({ error: "missing url" });

  // executa o yt-dlp instalado no user bin
  const cmd = "yt-dlp";
  const args = ["-g", "--no-playlist", url];

  const p = spawn(cmd, args);

  let stdout = "";
  let stderr = "";

  p.stdout.on("data", (d) => (stdout += d.toString()));
  p.stderr.on("data", (d) => (stderr += d.toString()));

  p.on("error", (err) => {
    return res.status(500).json({
      error: "spawn error",
      details: String(err),
    });
  });

  p.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: "yt-dlp failed",
        details: stderr.slice(-1200),
      });
    }

    const lines = stdout
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    if (!lines.length) return res.status(500).json({ error: "no direct url returned" });

    res.json({ directUrl: lines[0], allUrls: lines });
  });
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log("Server running on port", PORT));
