const http = require('http');

const rawPort = process.env.PORT;
const parsedPort = rawPort !== undefined ? Number.parseInt(rawPort, 10) : NaN;
const port =
  Number.isInteger(parsedPort) && parsedPort >= 1 && parsedPort <= 65535
    ? parsedPort
    : 8080;
const host = '0.0.0.0';

const server = http.createServer((req, res) => {
  const { method, url } = req;
  res.on('finish', () => {
    console.log(`[request] ${method} ${url} -> ${res.statusCode}`);
  });

  if (method === 'GET' && url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><html><body><h1>tradingq up ✅</h1></body></html>');
    return;
  }

  if (method === 'GET' && url === '/healthz') {
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
