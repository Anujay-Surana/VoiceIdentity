# Frontend Dockerfile for Railway
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the app (will use VITE_API_URL from build args)
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Production stage - serve with a simple static server
FROM node:20-alpine AS runner

WORKDIR /app

# Install serve for static hosting
RUN npm install -g serve

# Copy built assets
COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
