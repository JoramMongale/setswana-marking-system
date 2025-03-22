#!/bin/bash

# Allow X server connections
xhost +local:docker

# Run docker-compose with env variables
docker-compose up --build

# Disallow X server connections when done
xhost -local:docker