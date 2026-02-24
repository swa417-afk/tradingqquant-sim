const http = require('http');
const { URL } = require('url');

// Robust port parsing and fallback
const definePort = () => {
  const rawPort = process.env.PORT;
  const parsedPort = Number.parseInt(rawPort, 10);
  return Number.isInteger(parsedPort) && parsedPort >= 1 && parsedPort <= 65535 ? parsedPort : 8080;
};
const port = definePort();
const host = '0.0.0.0';

const server = http.createServer((req, res) => {
  const { method, url } = req;
  const pathname = new URL(url, 'http://localhost').pathname;
  res.on('finish', () => {
    console.log(`[request] ${method} ${pathname} -> ${res.statusCode}`);
  });

  if (method === 'GET' && pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><html><body><h1>tradingq up ✅</h1></body></html>');
    return;
  }

  if (method === 'GET' && pathname === '/healthz') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end('Not Found');
});

server
  .listen(port, host, () => {
    console.log(`[startup] server listening on http://${host}:${port}`);
  })
  .on('error', (err) => {
    console.error('[startup] failed to start server:', err);
    process.exit(1);
  });

const shutdownSignals = ['SIGTERM', 'SIGINT'];
const shutdownTimeoutMs = 10000;

shutdownSignals.forEach((signal) => {
  process.on(signal, () => {
    console.log(`[shutdown] received ${signal}, closing server`);

    const forceShutdownTimeout = setTimeout(() => {
      console.error('[shutdown] forcing shutdown after timeout');
      process.exit(1);
    }, shutdownTimeoutMs);

    if (typeof forceShutdownTimeout.unref === 'function') {
      forceShutdownTimeout.unref();
    }

    server.close((err) => {
      if (err) {
        console.error('[shutdown] error while closing server:', err);
        process.exit(1);
      }

      console.log('[shutdown] server closed gracefully');
      process.exit(0);
    });
  });
});
