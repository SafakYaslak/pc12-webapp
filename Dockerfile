# --- 1. Aşama: Frontend'i derle ---
FROM node:18-alpine AS build-frontend
WORKDIR /app

# package ve config dosyalarını kopyala
COPY package.json package-lock.json vite.config.ts tsconfig.json ./
COPY public/ ./public
COPY src/ ./src

# bağımlılıkları yükle ve build al
RUN npm ci && npm run build

# --- 2. Aşama: Python backend + statik dosya sunumu ---
FROM python:3.10.16-slim
WORKDIR /app

# derleme için gerekli OS paketleri
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Frontend'den oluşan dist klasörünü kopyala
COPY --from=build-frontend /app/dist ./dist

# Backend kodunu kopyala
COPY backend/ ./

# Render'ın dinleyeceği port
EXPOSE 10000

# Uygulamayı çalıştır
CMD ["gunicorn", "main_app:app", "--bind", "0.0.0.0:10000"]
