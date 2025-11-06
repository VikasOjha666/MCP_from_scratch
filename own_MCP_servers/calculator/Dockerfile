# Dockerfile
FROM python:3.11-slim

# Avoid creation of .pyc files and enable stdout/stderr buffering (helps logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set defaults for host/port; can be overridden at run time
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app

# system deps (if any) - keep minimal; expand if you need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY app/main.py /app/server.py

# Expose port for documentation (not required at runtime)
EXPOSE 8000

# Default command: run the Python server which reads HOST/PORT from env
CMD ["python", "server.py"]
