# ============================
# 🔹 STAGE 1: Frontend Build (Vite/React)
# ============================
FROM node:20-alpine AS frontend-builder

# 1) Çalışma dizini
WORKDIR /app

# 2) Önce bağımlılıkları kopyala (cache optimizasyonu için)
COPY package.json package-lock.json ./
COPY vite.config.ts ./
RUN npm install

# 3) Geri kalan dosyaları kopyala
COPY . .

# 4) Build işlemi (çıktı direkt backend/public'e gidecek)
RUN npm run build

# ============================
# 🔹 STAGE 2: Backend (Python + Flask)
# ============================
FROM python:3.10-slim

# 1) Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2) Çalışma dizini
WORKDIR /app

# 3) Python bağımlılıklarını yükle
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4) Backend kodunu kopyala
COPY backend ./backend

# 5) Frontend build çıktısını kopyala (vite.config.ts'de belirtilen dizin)
COPY --from=frontend-builder /app/backend/public ./backend/public

# 6) Port aç
EXPOSE 5000

# 7) Uygulamayı başlat
CMD ["python", "backend/main_app.py"]