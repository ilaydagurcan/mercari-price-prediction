FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyaları
COPY src/ src/
COPY models/ models/

# Portlar
EXPOSE 8000 8501

# Varsayılan: API başlat
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
