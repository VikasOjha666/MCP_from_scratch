# Dockerfile
FROM python:3.11-slim

# Keep container small: install pip and essential build tools only if needed
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy app
COPY app /app/app

# Expose configured port (default 8001)
EXPOSE 8000

# run
CMD ["python", "app/main.py"]
