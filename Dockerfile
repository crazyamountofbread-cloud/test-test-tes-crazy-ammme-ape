FROM python:3.11-slim

WORKDIR /app

# deps mínimas
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=10000
EXPOSE 10000

CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
