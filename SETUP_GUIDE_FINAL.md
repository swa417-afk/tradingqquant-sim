# Mobile Trading App - Complete Setup Guide

## ⚡ 15-Minute Quick Start

### Backend Setup
```bash
# 1. Copy API file
cp mobile_api_endpoints.ts server/src/routes/mobile.ts

# 2. Update server/src/server/index.ts - ADD these lines:
import mobileRoutes from './routes/mobile';
app.use('/api/mobile', mobileRoutes);

# 3. Install dependencies (if needed)
cd server
npm install express cors @tanstack/react-query

# 4. Start server
npm run dev
```

### Frontend Setup
```bash
# 1. Create React app
npm create vite@latest mobile-app -- --template react-ts
cd mobile-app

# 2. Install dependencies
npm install \
  @tanstack/react-query \
  lucide-react \
  tailwindcss \
  postcss \
  autoprefixer

# 3. Copy source files
cp ../mobile_app_frontend.tsx src/App.tsx
cp ../api_hooks_services.ts src/lib/api.ts
cp ../utilities_helpers.ts src/lib/utils.ts

# 4. Setup Tailwind
npx tailwindcss init -p

# 5. Create .env.local
echo "REACT_APP_API_URL=http://localhost:5000/api" > .env.local

# 6. Start dev server
npm run dev
```

### Verify Setup
```bash
# Terminal 1: Backend
cd server
npm run dev
# Should show: Server running on port 5000

# Terminal 2: Frontend  
cd mobile-app
npm run dev
# Should show: Local: http://localhost:5173

# Terminal 3: Test API
curl http://localhost:5000/api/mobile/dashboard
# Should return JSON
```

Visit: **http://localhost:5173** ✅

---

## 📋 Directory Structure

```
your-project/
├── server/
│   ├── src/
│   │   ├── routes/
│   │   │   └── mobile.ts ← PASTE mobile_api_endpoints.ts
│   │   ├── server/
│   │   │   └── index.ts (UPDATE: add mobile routes)
│   │   └── ...
│   ├── package.json
│   └── tsconfig.json
│
└── mobile-app/
    ├── src/
    │   ├── App.tsx ← PASTE mobile_app_frontend.tsx
    │   ├── lib/
    │   │   ├── api.ts ← PASTE api_hooks_services.ts
    │   │   ├── utils.ts ← PASTE utilities_helpers.ts
    │   │   └── types.ts (auto-generated)
    │   ├── main.tsx
    │   └── index.css
    ├── .env.local ← CREATE THIS
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── package.json
```

---

## 🔧 Environment Variables

### Backend `.env`
```bash
PORT=5000
NODE_ENV=development
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_quant
```

### Frontend `.env.local`
```bash
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_ENV=development
```

---

## 📦 Dependencies

### Backend (Express)
```
express: ^4.18.2
cors: ^2.8.5
@tanstack/react-query: ^5.8.3
typescript: ^5.2.2
```

### Frontend (React + Vite)
```
react: ^18.2.0
react-dom: ^18.2.0
@tanstack/react-query: ^5.8.3
lucide-react: ^0.263.1
tailwindcss: ^3.3.5
vite: ^4.5.0
typescript: ^5.2.2
```

---

## 🧪 Test Each Endpoint

```bash
# Scanner
curl http://localhost:5000/api/mobile/scanner

# Watchlist
curl http://localhost:5000/api/mobile/watchlist

# Portfolio
curl http://localhost:5000/api/mobile/portfolio

# Dashboard
curl http://localhost:5000/api/mobile/dashboard

# Add to watchlist
curl -X POST http://localhost:5000/api/mobile/watchlist \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USD"}'
```

All should return JSON without errors.

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# Kill it
kill -9 <PID>
```

### CORS Error
```bash
# Install cors package
npm install cors

# Add to server/src/server/index.ts
import cors from 'cors';
app.use(cors());
```

### Module Not Found
```bash
# Verify npm packages installed
npm ls

# If missing, install all
npm install

# Check file paths match
ls -la src/lib/api.ts  # Should exist
```

### API Returns 404
```bash
# 1. Verify route is registered in server/index.ts
# 2. Verify correct API URL in .env.local
# 3. Verify TypeScript compiled (no errors in terminal)
# 4. Restart server: npm run dev
```

### Data Not Showing in UI
```bash
# 1. Open DevTools (F12 → Network tab)
# 2. Click each tab and watch for API calls
# 3. Check response status (should be 200)
# 4. Check response format matches expected types
# 5. Check console for errors
```

---

## 📱 Mobile Optimization

The app includes:
- ✅ Mobile-first design
- ✅ Bottom navigation (easy thumb reach)
- ✅ Large touch targets (44px minimum)
- ✅ Responsive grid layouts
- ✅ Dark theme (OLED battery friendly)
- ✅ Touch-optimized spacing

For Mobiroller:
1. Build: `npm run build`
2. Export to PWA
3. Add to home screen
4. Works offline (with caching)

---

## 🚀 Deployment

### Vercel (Frontend)
```bash
npm run build
# Deploy dist/ folder to Vercel
```

### Render (Backend)
```bash
# Push to GitHub
git push origin main

# Connect Render.com
# Select repository
# Set start command: npm start
# Add environment variables
# Deploy
```

### Docker (Both)
```bash
docker-compose up
# Frontend: http://localhost:5173
# Backend: http://localhost:5000
```

---

## ✅ Checklist

- [ ] Backend running on port 5000
- [ ] Frontend running on port 5173
- [ ] `.env.local` created
- [ ] API endpoint responds (curl test)
- [ ] UI loads at http://localhost:5173
- [ ] Dashboard tab shows data
- [ ] Scanner tab loads
- [ ] Watchlist add/remove works
- [ ] Portfolio shows positions
- [ ] No errors in console

---

## 📚 API Endpoints

### Scanner
- `GET /api/mobile/scanner?limit=20` - Get top momentum stocks
- `GET /api/mobile/scanner/symbol/:symbol` - Get one symbol

### Watchlist
- `GET /api/mobile/watchlist` - Get your watchlist
- `POST /api/mobile/watchlist` - Add symbol
- `DELETE /api/mobile/watchlist/:symbol` - Remove symbol

### Portfolio
- `GET /api/mobile/portfolio` - Current positions & P&L
- `GET /api/mobile/portfolio/equity-curve?days=30` - Equity chart

### Runs (Backtests)
- `GET /api/mobile/runs?limit=10` - List runs
- `GET /api/mobile/runs/:id` - Get details
- `POST /api/mobile/runs` - Create backtest
- `GET /api/mobile/runs/:id/signals` - Signal execution history
- `GET /api/mobile/runs/:id/trades` - Get trades

### Alerts
- `GET /api/mobile/alerts` - Get alerts
- `POST /api/mobile/alerts` - Create alert

### Dashboard
- `GET /api/mobile/dashboard` - Quick overview

---

## 🎓 Key Files

| File | Purpose | Size |
|------|---------|------|
| mobile_api_endpoints.ts | Backend API | 17 KB |
| mobile_app_frontend.tsx | React UI | 19 KB |
| api_hooks_services.ts | React Query hooks | 11 KB |
| utilities_helpers.ts | Utility functions | 12 KB |

---

## 💡 Next Steps

1. **Immediate**: Follow Quick Start above
2. **Today**: Verify all endpoints work
3. **Tomorrow**: Customize UI colors/layout
4. **This week**: Add real data from your quant-sim
5. **Next week**: Deploy to production

---

**You're ready to go! 🚀**
