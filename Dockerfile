FROM python:3.9-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bağımlılıkları yükle
COPY backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY backend ./backend

EXPOSE 5000
CMD ["python", "backend/main_app.py"]