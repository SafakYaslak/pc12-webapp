# ============================
# 🔹 STAGE 1: Frontend Build
# ============================
FROM node:20-alpine AS frontend-builder

# Frontend için izole bir çalışma dizini
WORKDIR /frontend

# Sadece frontend’e ait dosyaları kopyala
COPY package.json package-lock.json vite.config.ts index.html ./
COPY postcss.config.js tailwind.config.js tsconfig*.json ./
COPY public ./public
COPY src ./src

# Bağımlılıkları yükle ve üretim derlemesi yap
RUN npm install
RUN npm run build


# ============================
# 🔹 STAGE 2: Backend (Python + Flask)
# ============================
FROM python:3.10-slim AS backend

# Sistem kütüphaneleri (OpenCV gibi şeyler gerekiyorsa)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Backend için çalışma dizini
WORKDIR /app

# Python bağımlılıklarını yükle
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend kodunu kopyala
COPY backend ./backend

# Frontend build çıktısını backend/public klasörüne kopyala
COPY --from=frontend-builder /frontend/dist ./backend/public

# Flask portu
EXPOSE 5000

# Uygulamayı başlat
CMD ["python", "backend/main_app.py"]
