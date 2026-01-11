const express = require("express");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const app = express();

app.get("/health", (_, res) => res.json({ ok: true }));

app.get("/download-file", (req, res) => {
  const url = req.query.url;
  if (!url) return res.status(400).send("missing url");

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "reel-"));
  const outFile = path.join(tmpDir, "video.mp4");

  const args = [
    "--no-playlist",
    "--sleep-interval", "5",
    "--max-sleep-interval", "12",
    "--referer", "https://www.instagram.com/",
    "--user-agent", "Mozilla/5.0",
    "-f", "bestvideo+bestaudio/best",
    "--merge-output-format", "mp4",
    "-o", outFile,
    url
  ];

  const p = spawn("yt-dlp", args, { stdio: ["ignore", "ignore", "pipe"] });
  let err = "";
  p.stderr.on("data", d => err += d.toString());

  p.on("close", (code) => {
    if (code !== 0 || !fs.existsSync(outFile)) {
      return res.status(500).json({ error: "yt-dlp failed", details: err.slice(-1200) });
    }
    res.setHeader("Content-Type", "video/mp4");
    fs.createReadStream(outFile).pipe(res);
  });
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log("running on", PORT));
