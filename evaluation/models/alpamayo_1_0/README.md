## Overview
1. Get access to the NVIDIA PhysicalAI AV Dataset on HuggingFace
2. Get HuggingFace token and export as an env variable
3. Download the dataset

## Commands
Build

```shell
docker compose -f path/to/docker-compose.yml --env-file path/to/dev.env build
```

Download model
```shell
docker compose -f path/to/docker-compose.yml --env-file path/to/dev.env --profile setup up -d
```

Run inference
```shell
docker compose -f path/to/docker-compose.yml --env-file path/to/dev.env run [options] <service_name> <command>

```

Logs
```shell
docker logs <container-name>
```

Interactive container
```shell
docker exec -it <container-name> /bin/bash
```
