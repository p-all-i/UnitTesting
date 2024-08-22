#!/bin/sh
chmod +x "$0"
docker build --build-arg belts=belt1 -t assembly .
docker-compose up -d --remove-orphans