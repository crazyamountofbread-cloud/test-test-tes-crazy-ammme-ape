FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# App + assets
COPY app.py .
COPY Logo.png .
COPY fonts ./fonts

ENV PORT=10000
EXPOSE 10000

# /render pode demorar mais que 30s
CMD ["gunicorn", "--timeout", "180", "-b", "0.0.0.0:10000", "app:app"]
