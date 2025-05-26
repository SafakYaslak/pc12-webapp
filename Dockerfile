FROM python:3.9-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Gereken dosyaları kopyala ve bağımlılıkları yükle
COPY backend/requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Tüm backend (ve public içeriği dahil) dosyalarını kopyala
COPY backend ./backend

EXPOSE 5000
CMD ["python", "backend/main_app.py"]
