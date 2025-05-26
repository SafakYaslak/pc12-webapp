# ============================
# 🔹 STAGE 1: Frontend Build
# ============================
FROM node:20-alpine AS frontend-builder

# 1) Çalışma dizini
WORKDIR /build

# 2) Proje kökünü (safak/ içindeki her şeyi) kopyala
COPY . .

# 3) Frontend bağımlılıklarını yükle ve build al
RUN npm install
RUN npm run build


# ============================
# 🔹 STAGE 2: Backend (Python + Flask)
# ============================
FROM python:3.10-slim AS backend

# 1) Sistem paketleri (OpenCV lib gibi gerekirse)
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

# 5) Frontend build çıktısını backend/public altına kopyala
#    (builder aşamasında /build/dist varsa, getirsin)
COPY --from=frontend-builder /build/dist ./backend/public

# 6) Port aç
EXPOSE 5000

# 7) Uygulamayı başlat
CMD ["python", "backend/main_app.py"]
