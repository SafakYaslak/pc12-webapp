# ======================
# 🔹 STAGE 1: Frontend Build (Vite + React + Tailwind)
# ======================
FROM node:20-alpine AS frontend-builder

# Çalışma dizinini ayarla
WORKDIR /app

# package.json ve lock dosyasını kopyala
COPY package*.json ./
COPY vite.config.ts tsconfig*.json postcss.config.js tailwind.config.js ./
COPY public ./public
COPY src ./src

# Gerekli node modüllerini yükle
RUN npm install

# Frontend'i üretime hazır olarak derle
RUN npm run build


# ======================
# 🔹 STAGE 2: Backend (Python App)
# ======================
FROM python:3.11-slim AS backend

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Backend dosyalarını kopyala
COPY backend ./backend

# Frontend build çıktısını backend/public altına kopyala
COPY --from=frontend-builder /app/dist ./backend/public

# Backend bağımlılıklarını yükle
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Gerekirse portu aç (örneğin Flask için)
EXPOSE 5000

# Uygulama başlangıcı
CMD ["python", "backend/main_app.py"]
