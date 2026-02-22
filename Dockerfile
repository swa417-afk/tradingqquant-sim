FROM node:20-alpine

WORKDIR /app

RUN chown -R node:node /app

USER node

COPY index.js package.json ./

RUN npm install --production

EXPOSE 8080

CMD ["node", "index.js"]
