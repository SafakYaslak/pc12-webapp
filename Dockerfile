# ============================
# 🔹 STAGE 1: Frontend Build
# ============================
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY . .
RUN npx update-browserslist-db@latest
RUN npm run build

# ============================
# 🔹 STAGE 2: Backend (Python + Flask)
# ============================
FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pip'i güncelle ve bağımlılıkları yükle
COPY backend/requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY backend ./backend
COPY --from=frontend-builder /app/backend/public ./backend/public

EXPOSE 5000
CMD ["python", "backend/main_app.py"]