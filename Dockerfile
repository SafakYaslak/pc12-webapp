# ====================================
# 🔹 STAGE 1: Frontend Build (Node 20)
# ====================================
FROM node:20-alpine AS frontend-builder

WORKDIR /build

# 1) Proje kökünü (index.html, package.json, vite.config.ts, public/, src/) kopyala
COPY . .

# 2) Frontend bağımlılıklarını Legacy peer deps ile yükle
RUN npm ci --legacy-peer-deps

# 3) Gerekirse browserslist DB güncelle
RUN npx update-browserslist-db@latest --update-db

# 4) Üretim derlemesini al
RUN npm run build


# ====================================
# 🔹 STAGE 2: Backend (Python 3.9 + Flask)
# ====================================
FROM python:3.9-slim

# 1) Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
      build-essential \
      libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) requirements.txt’yi kopyala ve yükle
COPY backend/requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 3) Backend kodunu kopyala
COPY backend ./backend

# 4) Frontend build çıktısını backend/public altına al
COPY --from=frontend-builder /build/dist ./backend/public

# 5) Uygulamayı dinleyecek port
EXPOSE 5000

# 6) Başlatma komutu
CMD ["python", "backend/main_app.py"]
