# base stage
FROM node:25 as build

# define the working directory
WORKDIR /app

# copy package.json and install dependencies
COPY package*.json ./

# clean install - makes sure it adheres to the package-lock.json
RUN npm ci

# copy source files
COPY . .

# build frontend
RUN npm run build

# production stage
FROM nginx:alpine as production

# Copy custom Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

ENTRYPOINT [ "nginx", "-c", "/etc/nginx/nginx.conf" ]
CMD [ "-g", "daemon off;"]
