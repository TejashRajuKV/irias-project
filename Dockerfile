# ── Stage 1: Python dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS python-deps

WORKDIR /app
COPY ml-service/requirements.txt ./ml-service/requirements.txt
RUN pip install --no-cache-dir -r ml-service/requirements.txt

# ── Stage 2: Build Next.js ────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy all source files
COPY . .

# Increase Node memory for build (fixes OOM errors on Render/Cloud Run)
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ── Stage 3: Production image ─────────────────────────────────────────────────
FROM node:20-slim AS runner

# Install Python 3
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

# Copy Python packages from python-deps stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Symlink python3
RUN ln -sf /usr/bin/python3 /usr/bin/python3

# Copy Next.js standalone output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Copy ML service (models + scripts)
COPY --from=builder /app/ml-service ./ml-service

EXPOSE 8080

CMD ["node", "server.js"]
