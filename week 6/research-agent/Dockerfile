# ── Build stage ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
# The entire research-agent folder is the context root
COPY . .

# Uvicorn port
EXPOSE 8000

# Health-check so Docker (and compose) knows when the app is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run as non-root for security
RUN useradd -m appuser
USER appuser

CMD ["python3", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]