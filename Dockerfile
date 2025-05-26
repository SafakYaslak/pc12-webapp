# ======================
# 🔹 STAGE 1: Frontend Build
# ======================
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json vite.config.ts index.html postcss.config.js tailwind.config.js tsconfig*.json ./
COPY public ./public
COPY src ./src
RUN npm install
RUN npm run build

# ======================
# 🔹 STAGE 2: Backend
# ======================
FROM python:3.10-slim AS backend
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY --from=frontend-builder /app/dist ./backend/public
EXPOSE 5000
CMD ["python", "backend/main_app.py"]
