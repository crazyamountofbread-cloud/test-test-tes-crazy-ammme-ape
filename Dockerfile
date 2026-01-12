FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg fonts-dejavu-core ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# força yt-dlp NIGHTLY (instagram quebra o stable o tempo todo)
RUN python -m pip install --no-cache-dir -U --pre "yt-dlp[default]"
RUN python -m pip show yt-dlp && yt-dlp --version

# App + assets
COPY app.py .
COPY Logo.png .
COPY fonts ./fonts

ENV PORT=10000
EXPOSE 10000

# /render pode demorar mais que 30s
CMD ["gunicorn", "--timeout", "180", "-b", "0.0.0.0:10000", "app:app"]
