FROM node:20-alpine3.19

WORKDIR /app

COPY index.js package.json ./
RUN npm install --production

EXPOSE 8080

CMD ["node", "index.js"]
