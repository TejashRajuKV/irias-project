# ── Stage 1: Build Next.js frontend ──────────────────────────────────────────
FROM node:20-slim AS next-builder
WORKDIR /app
COPY package*.json ./
# Install ALL deps including devDependencies (tailwindcss, postcss, etc. needed for build)
RUN npm install --include=dev --legacy-peer-deps
COPY . .
# Increase Node heap to handle large Next.js builds; build the app
RUN NODE_OPTIONS="--max-old-space-size=4096" npm run build

# ── Stage 2: Train ML Models ──────────────────────────────────────────────────
FROM python:3.11-slim AS ml-builder
WORKDIR /app
COPY ml-service/requirements.txt ./ml-service/requirements.txt
RUN pip install --no-cache-dir -r ml-service/requirements.txt
COPY ml-service ./ml-service
COPY upload ./upload
# Run the training pipeline to generate the model artifacts (.pkl & .json)
RUN python ml-service/train_models.py

# ── Stage 3: Assemble Production Runner ───────────────────────────────────────
FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

# Install Python runtime and pip in runner image
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from ml-builder stage
COPY --from=ml-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
RUN ln -sf /usr/bin/python3 /usr/bin/python3

# Copy Next.js standalone server and static assets from next-builder stage
COPY --from=next-builder /app/.next/standalone ./
COPY --from=next-builder /app/.next/static ./.next/static
COPY --from=next-builder /app/public ./public

# Copy pre-trained ML models and services from ml-builder stage
COPY --from=ml-builder /app/ml-service ./ml-service

EXPOSE 8080
CMD ["node", "server.js"]

