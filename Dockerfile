# ======================
# 🔹 STAGE 1: Frontend Build (Vite + React + Tailwind)
# ======================
FROM node:20-alpine AS frontend-builder

WORKDIR /app

# package.json ve lock dosyalarını kopyala
COPY package*.json ./

# Diğer config dosyalarını kopyala
COPY vite.config.ts tsconfig*.json postcss.config.js tailwind.config.js ./

# public ve src klasörlerini kopyala
COPY public ./public
COPY src ./src

# node modüllerini yükle
RUN npm install

# frontend build
RUN npm run build


# ======================
# 🔹 STAGE 2: Backend (Python App)
# ======================
FROM python:3.11-slim AS backend

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend klasörünü kopyala
COPY backend ./backend

# Backend requirements.txt'yi kopyala ve yükle
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Frontend build çıktısını backend/public klasörüne kopyala
# Böylece backend uygulaman public klasöründeki frontend build dosyalarına erişebilir
COPY --from=frontend-builder /app/dist ./backend/public

# Flask için port aç
EXPOSE 5000

# Flask uygulamasını başlat
CMD ["python", "backend/main_app.py"]
