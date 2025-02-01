# Utiliser une image légère de Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances et installer
COPY requirements.txt .
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# Installer explicitement kagglehub
RUN pip install kagglehub

# Copier tout le code source
COPY app/ /app/

# Copie du dossier data si besoin :
COPY data/my_custom_dataset /app/data/my_custom_dataset
# COPY data/my_custom_dataset2 /app/data/my_custom_dataset2
COPY data/ /app/data/

# Exposer le port de Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
