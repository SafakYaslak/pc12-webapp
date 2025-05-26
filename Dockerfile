# ============================
# 🔹 STAGE 1: Frontend Build
# ============================
FROM node:20-alpine AS frontend-builder

# Konteyner içi çalışma dizini
WORKDIR /app

# Proje kökünü tamamen kopyala (backend dahil ama build etme)
COPY . .

# Sadece frontend bağımlılıklarını yükle
# (backend/requirements.txt veya Dockerfile’ın başka kısımları bu aşamayı etkilemez)
RUN npm install

# Build
RUN npm run build


# ============================
# 🔹 STAGE 2: Backend (Python)
# ============================
FROM python:3.10-slim AS backend

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bağımlılıkları
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Backend kodu
COPY backend ./backend

# Frontend build çıktılarını kopyala
COPY --from=frontend-builder /app/dist ./backend/public

EXPOSE 5000

CMD ["python", "backend/main_app.py"]
