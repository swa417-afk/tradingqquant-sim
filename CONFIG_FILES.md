# Configuration Files - Copy These

## 1. Backend package.json

Save as: `server/package.json`

```json
{
  "name": "trading-quant-backend",
  "version": "1.0.0",
  "description": "Backend API for mobile trading app",
  "type": "module",
  "main": "dist/server/index.js",
  "scripts": {
    "dev": "tsx watch src/server/index.ts",
    "build": "tsc",
    "start": "node dist/server/index.js",
    "lint": "eslint src --ext .ts"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "@types/express": "^4.17.20",
    "@types/node": "^20.8.0",
    "typescript": "^5.2.2",
    "tsx": "^3.13.0"
  }
}
```

---

## 2. Frontend package.json

Save as: `mobile-app/package.json`

```json
{
  "name": "trading-quant-mobile",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.8.3",
    "lucide-react": "^0.263.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.31",
    "@types/react-dom": "^18.2.14",
    "@vitejs/plugin-react": "^4.1.0",
    "vite": "^4.5.0",
    "typescript": "^5.2.2",
    "tailwindcss": "^3.3.5",
    "postcss": "^8.4.31",
    "autoprefixer": "^10.4.16"
  }
}
```

---

## 3. Backend .env

Save as: `server/.env`

```bash
PORT=5000
NODE_ENV=development
DATABASE_URL=postgresql://user:password@localhost:5432/trading_quant
LOG_LEVEL=debug
```

---

## 4. Frontend .env.local

Save as: `mobile-app/.env.local`

```bash
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_ENV=development
```

---

## 5. Backend tsconfig.json

Save as: `server/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020"],
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "outDir": "./dist",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

---

## 6. Frontend vite.config.ts

Save as: `mobile-app/vite.config.ts`

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
```

---

## 7. Frontend tailwind.config.ts

Save as: `mobile-app/tailwind.config.ts`

```typescript
import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config
```

---

## 8. Frontend postcss.config.js

Save as: `mobile-app/postcss.config.js`

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

---

## 9. Frontend tsconfig.json

Save as: `mobile-app/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## 10. .gitignore (Both)

Save as: `server/.gitignore` and `mobile-app/.gitignore`

```
node_modules
dist
.env
.env.local
*.log
.DS_Store
.idea
.vscode/settings.json
```

---

## 11. docker-compose.yml

Save as: `docker-compose.yml` (in root)

```yaml
version: '3.8'

services:
  backend:
    build: ./server
    ports:
      - "5000:5000"
    environment:
      PORT: 5000
      NODE_ENV: development
    volumes:
      - ./server/src:/app/src

  frontend:
    build: ./mobile-app
    ports:
      - "5173:5173"
    environment:
      REACT_APP_API_URL: http://localhost:5000/api
    depends_on:
      - backend
    volumes:
      - ./mobile-app/src:/app/src
```

---

## 12. Dockerfile (Backend)

Save as: `server/Dockerfile`

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY src ./src
COPY tsconfig.json ./

RUN npm run build

ENV NODE_ENV=production
EXPOSE 5000

CMD ["npm", "start"]
```

---

## 13. Dockerfile (Frontend)

Save as: `mobile-app/Dockerfile`

```dockerfile
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 14. Frontend index.css

Save as: `mobile-app/src/index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: #030712;
  color: #ffffff;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

---

## 15. Frontend main.tsx

Save as: `mobile-app/src/main.tsx`

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App.tsx'
import './index.css'

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
```

---

## 16. Frontend index.html

Save as: `mobile-app/index.html`

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="Mobile Trading App - Real-time market scanner and portfolio tracker" />
    <title>TradingQ Mobile</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

---

## Installation Steps

1. **Backend**:
   ```bash
   mkdir server
   cd server
   # Copy all backend files above
   npm install
   npm run dev
   ```

2. **Frontend**:
   ```bash
   mkdir mobile-app
   cd mobile-app
   # Copy all frontend files above
   npm install
   npm run dev
   ```

3. **Verify**:
   - Backend: http://localhost:5000/api/mobile/dashboard
   - Frontend: http://localhost:5173

---

## Production Environment Variables

### Backend
```bash
PORT=5000
NODE_ENV=production
DATABASE_URL=postgresql://prod_user:prod_pass@prod_host:5432/trading
LOG_LEVEL=info
```

### Frontend
```bash
REACT_APP_API_URL=https://api.yourdomain.com/api
REACT_APP_ENV=production
```

---

**All configuration files ready to use! ✅**
