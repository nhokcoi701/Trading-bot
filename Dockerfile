FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Fly.io sets PORT; default 8080 for many platforms
CMD ["bash", "-lc", "uvicorn iphone_webapp.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
