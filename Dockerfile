# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

# 1) Env & perf
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSFRPROTECTION=false

WORKDIR /app

# 2) OS deps minimales (faiss cpu ok via wheels; ajoute build-essential si besoin de compiles)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc wget git \
 && rm -rf /var/lib/apt/lists/*

# 3) Dépendances Python (cache des layers)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=120 --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kagglehub

# 4) Code (on copie après pour garder le cache deps)
#    ⚠️ évite de copier des datasets lourds inutiles à l’exécution
COPY app/ /app/
COPY data/my_custom_dataset /app/data/my_custom_dataset
# si tu as d’autres fichiers légers utiles à l’exécution, copie-les explicitement :
COPY data/feedback.csv /app/data/feedback.csv
COPY images/ /app/images/

# 5) (Optionnel) Pré-télécharger les modèles pour éviter un gros téléchargement au démarrage
# Crée un script app/preload.py qui charge les modèles (DPR/SBERT) une fois.
# RUN python /app/preload.py || true

# 6) Non-root user (bonne pratique)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 7) Healthcheck (facultatif)
# HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
#   CMD wget -qO- http://127.0.0.1:${PORT:-8501}/_stcore/health || exit 1

EXPOSE 8501

# 8) Streamlit écoute sur 0.0.0.0:$PORT (Vercel fournit $PORT)
CMD streamlit run /app/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
