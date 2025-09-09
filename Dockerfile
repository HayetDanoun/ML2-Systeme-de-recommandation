FROM python:3.10-slim
WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt
# (optionnel) seulement si tu en as besoin au runtime :
# RUN pip install kagglehub

# Code + données nécessaires à l'exécution
COPY app/ /app/
COPY data/movies_metadata_clean.csv /app/data/movies_metadata_clean.csv
COPY data/my_custom_dataset /app/data/my_custom_dataset
COPY data/feedback.csv /app/data/feedback.csv
COPY images/ /app/images/

EXPOSE 8501
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
