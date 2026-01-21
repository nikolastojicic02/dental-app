# --- STAGE 1: Build Frontenda ---
FROM node:20 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- STAGE 2: Build Backenda ---
FROM python:3.10-slim

# Instalacija sistemskih biblioteka za Open3D i grafiku
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Napravi korisnika "user" (Hugging Face zahteva UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Kopiraj requirements i instaliraj
COPY --chown=user backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --upgrade -r backend/requirements.txt

# Kopiraj CEO projekat (uključujući i weights folder)
COPY --chown=user . .

# Kopiraj buildovan frontend iz Stage 1
COPY --from=frontend-build --chown=user /app/frontend/dist ./frontend/dist

# Promeni folder na backend da bi uvicorn video main.py
WORKDIR /app/backend

# Port 7860 je obavezan za Hugging Face
EXPOSE 7860

# Pokreni aplikaciju
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]