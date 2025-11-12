# Dockerfile (cpu)
FROM python:3.10-slim

# Install system deps needed to build some Python packages (llama-cpp-python can compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    pkg-config \
    libopenblas-dev \
    libgomp1 \
    libcurl4-openssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user (but do not switch to it yet)
RUN useradd --create-home appuser

WORKDIR /home/appuser/app

# Copy only requirements first for better cache
COPY requirements.txt .

# Use pip wheel cache installation - keep pip from caching wheels to disk
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Install Python deps AS ROOT (so site-packages are writable)
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r requirements.txt

# Quick sanity-check: ensure llama_index.tools.mcp is importable at build time.
# This will make the build fail early with a clear error if the installed packages are wrong.
RUN python -c "from llama_index.tools.mcp import BasicMCPClient, McpToolSpec; print('llama_index.tools.mcp import OK')"

# Now copy app code and switch to non-root user
COPY --chown=appuser:appuser . .
USER appuser

# Expose the port your FastAPI app runs on
EXPOSE 3000

# Default environment variables (can be overridden)
ENV MODEL_PATH=/home/appuser/app/gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf
ENV MCP_SERVER_URLS="http://mcp_calculator:8000/sse,http://mcp_converter:8000/sse"

# Run uvicorn using the module path you provided
CMD ["python","-m","uvicorn","llm_client:app","--host","0.0.0.0","--port","3000","--workers","1"]
