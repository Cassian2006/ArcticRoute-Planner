# Multi-stage build for ARO-UI (Vite + React)
FROM node:18-alpine AS build
WORKDIR /app

# --- build-time API base injection ---
ARG VITE_API_BASE_URL
ARG VITE_FUEL_BASE_URL
# Provide sane defaults if not passed
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL:-http://fuel:8121}
ENV VITE_FUEL_BASE_URL=${VITE_FUEL_BASE_URL:-http://fuel:8121}

COPY package*.json ./
COPY tsconfig*.json ./
COPY vite.config.* ./
COPY src ./src
COPY index.html ./
COPY postcss.config.js ./
COPY tailwind.config.js ./

# install & build
RUN npm ci --no-audit --no-fund && \
    echo "Using VITE_API_BASE_URL=$VITE_API_BASE_URL VITE_FUEL_BASE_URL=$VITE_FUEL_BASE_URL" && \
    npm run build

# Serve with nginx
FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
# Basic nginx config (gzip on, index, SPA fallback)
RUN printf '\nserver {\n  listen 80;\n  server_name _;\n  root /usr/share/nginx/html;\n  index index.html;\n  gzip on;\n  gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;\n  location / {\n    try_files $uri /index.html;\n  }\n}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]