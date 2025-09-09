FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt
RUN pip install kagglehub
COPY app/ /app/
COPY data/my_custom_dataset /app/data/my_custom_dataset
# COPY data/my_custom_dataset2 /app/data/my_custom_dataset2
COPY data/ /app/data/
EXPOSE 8501
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
