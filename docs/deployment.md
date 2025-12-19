\# Deployment Guide



Complete guide for deploying the Alzheimer's Disease Classification system to production.



\## Table of Contents

\- \[Overview](#overview)

\- \[Prerequisites](#prerequisites)

\- \[Local Deployment](#local-deployment)

\- \[Docker Deployment](#docker-deployment)

\- \[Cloud Deployment](#cloud-deployment)

\- \[Kubernetes Deployment](#kubernetes-deployment)

\- \[Monitoring \& Logging](#monitoring--logging)

\- \[Security](#security)

\- \[Scaling](#scaling)

\- \[Troubleshooting](#troubleshooting)



---



\## Overview



This guide covers multiple deployment strategies from local development to production-grade cloud deployment with high availability and auto-scaling.



\### Deployment Options Comparison



| Method | Complexity | Cost | Scalability | Use Case |

|--------|------------|------|-------------|----------|

| \*\*Local\*\* | ⭐ | Free | Single machine | Development |

| \*\*Docker\*\* | ⭐⭐ | Low | Single/few machines | Small-scale production |

| \*\*Docker Compose\*\* | ⭐⭐ | Low | Multiple containers | Medium-scale |

| \*\*Kubernetes\*\* | ⭐⭐⭐⭐ | Medium-High | Auto-scaling | Large-scale production |

| \*\*Cloud (AWS/Azure)\*\* | ⭐⭐⭐ | Medium | Auto-scaling | Enterprise |



---



\## Prerequisites



\### System Requirements



\*\*Minimum (CPU-only):\*\*

\- CPU: 4 cores

\- RAM: 8GB

\- Storage: 50GB

\- OS: Ubuntu 20.04+, Windows 10+, macOS 10.15+



\*\*Recommended (GPU):\*\*

\- CPU: 8+ cores

\- RAM: 16GB+

\- GPU: NVIDIA GPU with 8GB+ VRAM (for fast inference)

\- Storage: 100GB SSD

\- OS: Ubuntu 20.04 LTS



\### Software Requirements



```bash

\# Required

Python 3.8+

Docker 20.10+

Docker Compose 2.0+



\# Optional

Kubernetes 1.20+

Helm 3.0+

NVIDIA Docker (for GPU)

```



\### Pre-deployment Checklist



\- \[ ] Model weights trained and saved

\- \[ ] Configuration files prepared

\- \[ ] Environment variables set

\- \[ ] SSL certificates obtained (for HTTPS)

\- \[ ] Domain name configured (if applicable)

\- \[ ] Monitoring tools set up

\- \[ ] Backup strategy defined



---



\## Local Deployment



\### Quick Start (Development)



```bash

\# 1. Clone repository

git clone https://github.com/yourusername/alzheimers-mri-setnn.git

cd alzheimers-mri-setnn



\# 2. Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate



\# 3. Install dependencies

pip install -r requirements.txt

pip install -e .



\# 4. Download/prepare model weights

\# Place trained model in: results/experiments/setnn\_v1/models/



\# 5. Start API server

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload



\# API now available at http://localhost:8000

```



\### Environment Variables



Create `.env` file:



```bash

\# Application

MODEL\_PATH=results/experiments/setnn\_v1/models

LOG\_LEVEL=INFO

WORKERS=4



\# API Configuration

API\_HOST=0.0.0.0

API\_PORT=8000

API\_RELOAD=false



\# Model Configuration

CONFIDENCE\_THRESHOLD=0.5

BATCH\_SIZE=32



\# Monitoring

ENABLE\_METRICS=true

METRICS\_PORT=9090

```



Load environment:

```bash

export $(cat .env | xargs)

```



\### Running with Gunicorn (Production WSGI)



```bash

gunicorn api.app:app \\

&nbsp; --workers 4 \\

&nbsp; --worker-class uvicorn.workers.UvicornWorker \\

&nbsp; --bind 0.0.0.0:8000 \\

&nbsp; --timeout 300 \\

&nbsp; --access-logfile logs/access.log \\

&nbsp; --error-logfile logs/error.log

```



---



\## Docker Deployment



\### Building Docker Image



```bash

\# Build image

docker build -t alzheimers-setnn:latest -f docker/Dockerfile .



\# Verify image

docker images | grep alzheimers-setnn



\# Test image locally

docker run -p 8000:8000 alzheimers-setnn:latest

```



\### Running Container



```bash

\# Basic run

docker run -d \\

&nbsp; --name alzheimers-api \\

&nbsp; -p 8000:8000 \\

&nbsp; -v $(pwd)/data:/app/data:ro \\

&nbsp; -v $(pwd)/results:/app/results \\

&nbsp; alzheimers-setnn:latest



\# With GPU support

docker run -d \\

&nbsp; --name alzheimers-api \\

&nbsp; --gpus all \\

&nbsp; -p 8000:8000 \\

&nbsp; -v $(pwd)/data:/app/data:ro \\

&nbsp; -v $(pwd)/results:/app/results \\

&nbsp; alzheimers-setnn:latest



\# With environment variables

docker run -d \\

&nbsp; --name alzheimers-api \\

&nbsp; -p 8000:8000 \\

&nbsp; -e MODEL\_PATH=/app/results/models \\

&nbsp; -e LOG\_LEVEL=INFO \\

&nbsp; -v $(pwd)/results:/app/results \\

&nbsp; alzheimers-setnn:latest

```



\### Docker Compose Deployment



```bash

\# Start all services

docker-compose -f docker/docker-compose.yml up -d



\# View logs

docker-compose -f docker/docker-compose.yml logs -f



\# Stop services

docker-compose -f docker/docker-compose.yml down



\# Rebuild and restart

docker-compose -f docker/docker-compose.yml up -d --build

```



\### Docker Best Practices



```dockerfile

\# Multi-stage build for smaller images

FROM python:3.10-slim as builder

\# ... build dependencies



FROM python:3.10-slim

\# ... copy from builder



\# Use non-root user

RUN useradd -m appuser

USER appuser



\# Health check

HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health



\# Resource limits

\# Set in docker-compose or kubernetes

```



---



\## Cloud Deployment



\### AWS Deployment



\#### Option 1: AWS Elastic Beanstalk



```bash

\# Install EB CLI

pip install awsebcli



\# Initialize

eb init -p docker alzheimers-setnn



\# Create environment

eb create alzheimers-prod \\

&nbsp; --instance-type t3.large \\

&nbsp; --envvars MODEL\_PATH=/app/results/models



\# Deploy

eb deploy



\# Open in browser

eb open

```



\#### Option 2: AWS ECS (Elastic Container Service)



```bash

\# 1. Build and push to ECR

aws ecr create-repository --repository-name alzheimers-setnn



\# Get login token

aws ecr get-login-password --region us-east-1 | \\

&nbsp; docker login --username AWS --password-stdin \\

&nbsp; ACCOUNT\_ID.dkr.ecr.us-east-1.amazonaws.com



\# Tag and push

docker tag alzheimers-setnn:latest \\

&nbsp; ACCOUNT\_ID.dkr.ecr.us-east-1.amazonaws.com/alzheimers-setnn:latest



docker push ACCOUNT\_ID.dkr.ecr.us-east-1.amazonaws.com/alzheimers-setnn:latest



\# 2. Create ECS task definition (task-definition.json)

{

&nbsp; "family": "alzheimers-setnn",

&nbsp; "containerDefinitions": \[{

&nbsp;   "name": "api",

&nbsp;   "image": "ACCOUNT\_ID.dkr.ecr.us-east-1.amazonaws.com/alzheimers-setnn:latest",

&nbsp;   "memory": 4096,

&nbsp;   "cpu": 2048,

&nbsp;   "essential": true,

&nbsp;   "portMappings": \[{

&nbsp;     "containerPort": 8000,

&nbsp;     "hostPort": 8000

&nbsp;   }]

&nbsp; }]

}



\# 3. Register task definition

aws ecs register-task-definition \\

&nbsp; --cli-input-json file://task-definition.json



\# 4. Create service

aws ecs create-service \\

&nbsp; --cluster default \\

&nbsp; --service-name alzheimers-api \\

&nbsp; --task-definition alzheimers-setnn \\

&nbsp; --desired-count 2 \\

&nbsp; --launch-type FARGATE

```



\#### Option 3: AWS Lambda + API Gateway (Serverless)



For lower traffic, use serverless:



```python

\# lambda\_handler.py

import json

from mangum import Mangum

from api.app import app



handler = Mangum(app)

```



Deploy with AWS SAM:

```bash

sam build

sam deploy --guided

```



\### Azure Deployment



\#### Azure Container Instances



```bash

\# Login

az login



\# Create resource group

az group create --name alzheimers-rg --location eastus



\# Create container

az container create \\

&nbsp; --resource-group alzheimers-rg \\

&nbsp; --name alzheimers-api \\

&nbsp; --image alzheimers-setnn:latest \\

&nbsp; --cpu 4 \\

&nbsp; --memory 8 \\

&nbsp; --ports 8000 \\

&nbsp; --dns-name-label alzheimers-api \\

&nbsp; --environment-variables \\

&nbsp;   MODEL\_PATH=/app/results/models \\

&nbsp;   LOG\_LEVEL=INFO



\# Get public IP

az container show \\

&nbsp; --resource-group alzheimers-rg \\

&nbsp; --name alzheimers-api \\

&nbsp; --query ipAddress.fqdn

```



\#### Azure Kubernetes Service (AKS)



```bash

\# Create AKS cluster

az aks create \\

&nbsp; --resource-group alzheimers-rg \\

&nbsp; --name alzheimers-cluster \\

&nbsp; --node-count 3 \\

&nbsp; --node-vm-size Standard\_D4s\_v3 \\

&nbsp; --enable-addons monitoring \\

&nbsp; --generate-ssh-keys



\# Get credentials

az aks get-credentials \\

&nbsp; --resource-group alzheimers-rg \\

&nbsp; --name alzheimers-cluster



\# Deploy (see Kubernetes section)

kubectl apply -f k8s/

```



\### Google Cloud Platform (GCP)



\#### Cloud Run



```bash

\# Build and push

gcloud builds submit --tag gcr.io/PROJECT\_ID/alzheimers-setnn



\# Deploy

gcloud run deploy alzheimers-api \\

&nbsp; --image gcr.io/PROJECT\_ID/alzheimers-setnn \\

&nbsp; --platform managed \\

&nbsp; --region us-central1 \\

&nbsp; --memory 4Gi \\

&nbsp; --cpu 2 \\

&nbsp; --timeout 300 \\

&nbsp; --allow-unauthenticated

```



---



\## Kubernetes Deployment



\### Kubernetes Manifests



\#### Deployment (`k8s/deployment.yaml`)



```yaml

apiVersion: apps/v1

kind: Deployment

metadata:

&nbsp; name: alzheimers-api

&nbsp; labels:

&nbsp;   app: alzheimers-api

spec:

&nbsp; replicas: 3

&nbsp; selector:

&nbsp;   matchLabels:

&nbsp;     app: alzheimers-api

&nbsp; template:

&nbsp;   metadata:

&nbsp;     labels:

&nbsp;       app: alzheimers-api

&nbsp;   spec:

&nbsp;     containers:

&nbsp;     - name: api

&nbsp;       image: alzheimers-setnn:latest

&nbsp;       ports:

&nbsp;       - containerPort: 8000

&nbsp;       resources:

&nbsp;         requests:

&nbsp;           memory: "4Gi"

&nbsp;           cpu: "2"

&nbsp;         limits:

&nbsp;           memory: "8Gi"

&nbsp;           cpu: "4"

&nbsp;       env:

&nbsp;       - name: MODEL\_PATH

&nbsp;         value: "/app/results/models"

&nbsp;       - name: LOG\_LEVEL

&nbsp;         value: "INFO"

&nbsp;       livenessProbe:

&nbsp;         httpGet:

&nbsp;           path: /health

&nbsp;           port: 8000

&nbsp;         initialDelaySeconds: 30

&nbsp;         periodSeconds: 10

&nbsp;       readinessProbe:

&nbsp;         httpGet:

&nbsp;           path: /health

&nbsp;           port: 8000

&nbsp;         initialDelaySeconds: 20

&nbsp;         periodSeconds: 5

&nbsp;       volumeMounts:

&nbsp;       - name: model-storage

&nbsp;         mountPath: /app/results

&nbsp;     volumes:

&nbsp;     - name: model-storage

&nbsp;       persistentVolumeClaim:

&nbsp;         claimName: model-pvc

```



\#### Service (`k8s/service.yaml`)



```yaml

apiVersion: v1

kind: Service

metadata:

&nbsp; name: alzheimers-api-service

spec:

&nbsp; type: LoadBalancer

&nbsp; selector:

&nbsp;   app: alzheimers-api

&nbsp; ports:

&nbsp; - protocol: TCP

&nbsp;   port: 80

&nbsp;   targetPort: 8000

```



\#### Horizontal Pod Autoscaler (`k8s/hpa.yaml`)



```yaml

apiVersion: autoscaling/v2

kind: HorizontalPodAutoscaler

metadata:

&nbsp; name: alzheimers-api-hpa

spec:

&nbsp; scaleTargetRef:

&nbsp;   apiVersion: apps/v1

&nbsp;   kind: Deployment

&nbsp;   name: alzheimers-api

&nbsp; minReplicas: 2

&nbsp; maxReplicas: 10

&nbsp; metrics:

&nbsp; - type: Resource

&nbsp;   resource:

&nbsp;     name: cpu

&nbsp;     target:

&nbsp;       type: Utilization

&nbsp;       averageUtilization: 70

&nbsp; - type: Resource

&nbsp;   resource:

&nbsp;     name: memory

&nbsp;     target:

&nbsp;       type: Utilization

&nbsp;       averageUtilization: 80

```



\### Deploy to Kubernetes



```bash

\# Apply all manifests

kubectl apply -f k8s/



\# Check deployment

kubectl get deployments

kubectl get pods

kubectl get services



\# View logs

kubectl logs -f deployment/alzheimers-api



\# Scale manually

kubectl scale deployment alzheimers-api --replicas=5



\# Get service URL

kubectl get service alzheimers-api-service

```



\### Helm Deployment



Create Helm chart (`helm/values.yaml`):



```yaml

replicaCount: 3



image:

&nbsp; repository: alzheimers-setnn

&nbsp; tag: latest

&nbsp; pullPolicy: IfNotPresent



service:

&nbsp; type: LoadBalancer

&nbsp; port: 80

&nbsp; targetPort: 8000



resources:

&nbsp; requests:

&nbsp;   memory: "4Gi"

&nbsp;   cpu: "2"

&nbsp; limits:

&nbsp;   memory: "8Gi"

&nbsp;   cpu: "4"



autoscaling:

&nbsp; enabled: true

&nbsp; minReplicas: 2

&nbsp; maxReplicas: 10

&nbsp; targetCPUUtilizationPercentage: 70

```



Deploy with Helm:

```bash

helm install alzheimers-api ./helm

```



---



\## Monitoring \& Logging



\### Prometheus + Grafana



```yaml

\# docker-compose.yml monitoring stack

services:

&nbsp; prometheus:

&nbsp;   image: prom/prometheus

&nbsp;   volumes:

&nbsp;     - ./prometheus.yml:/etc/prometheus/prometheus.yml

&nbsp;   ports:

&nbsp;     - "9090:9090"

&nbsp; 

&nbsp; grafana:

&nbsp;   image: grafana/grafana

&nbsp;   ports:

&nbsp;     - "3000:3000"

&nbsp;   environment:

&nbsp;     - GF\_SECURITY\_ADMIN\_PASSWORD=admin

```



\### Application Metrics



```python

\# Add to api/app.py

from prometheus\_client import Counter, Histogram, generate\_latest



\# Metrics

prediction\_counter = Counter('predictions\_total', 'Total predictions')

prediction\_duration = Histogram('prediction\_duration\_seconds', 'Prediction duration')



@app.get("/metrics")

async def metrics():

&nbsp;   return Response(generate\_latest(), media\_type="text/plain")

```



\### Logging Configuration



```python

\# Production logging

import logging

from logging.handlers import RotatingFileHandler



handler = RotatingFileHandler(

&nbsp;   'logs/api.log',

&nbsp;   maxBytes=10485760,  # 10MB

&nbsp;   backupCount=10

)



logging.basicConfig(

&nbsp;   level=logging.INFO,

&nbsp;   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

&nbsp;   handlers=\[handler, logging.StreamHandler()]

)

```



\### ELK Stack (Elasticsearch, Logstash, Kibana)



```yaml

\# Log aggregation

services:

&nbsp; elasticsearch:

&nbsp;   image: elasticsearch:8.5.0

&nbsp;   environment:

&nbsp;     - discovery.type=single-node

&nbsp;   ports:

&nbsp;     - "9200:9200"

&nbsp; 

&nbsp; kibana:

&nbsp;   image: kibana:8.5.0

&nbsp;   ports:

&nbsp;     - "5601:5601"

&nbsp;   depends\_on:

&nbsp;     - elasticsearch

```



---



\## Security



\### SSL/TLS Configuration



\#### Nginx SSL Reverse Proxy



```nginx

\# nginx.conf

server {

&nbsp;   listen 80;

&nbsp;   server\_name api.example.com;

&nbsp;   return 301 https://$server\_name$request\_uri;

}



server {

&nbsp;   listen 443 ssl http2;

&nbsp;   server\_name api.example.com;



&nbsp;   ssl\_certificate /etc/nginx/ssl/cert.pem;

&nbsp;   ssl\_certificate\_key /etc/nginx/ssl/key.pem;

&nbsp;   ssl\_protocols TLSv1.2 TLSv1.3;

&nbsp;   ssl\_ciphers HIGH:!aNULL:!MD5;



&nbsp;   location / {

&nbsp;       proxy\_pass http://api:8000;

&nbsp;       proxy\_set\_header Host $host;

&nbsp;       proxy\_set\_header X-Real-IP $remote\_addr;

&nbsp;   }

}

```



\### API Authentication



```python

\# Add JWT authentication

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials



security = HTTPBearer()



@app.post("/predict")

async def predict(

&nbsp;   file: UploadFile,

&nbsp;   credentials: HTTPAuthorizationCredentials = Depends(security)

):

&nbsp;   token = credentials.credentials

&nbsp;   # Verify token

&nbsp;   ...

```



\### Rate Limiting



```python

from slowapi import Limiter

from slowapi.util import get\_remote\_address



limiter = Limiter(key\_func=get\_remote\_address)

app.state.limiter = limiter



@app.post("/predict")

@limiter.limit("10/minute")

async def predict(request: Request, file: UploadFile):

&nbsp;   ...

```



---



\## Scaling



\### Horizontal Scaling



```bash

\# Docker Swarm

docker service scale alzheimers-api=5



\# Kubernetes

kubectl scale deployment alzheimers-api --replicas=5



\# Auto-scaling (Kubernetes HPA)

kubectl autoscale deployment alzheimers-api \\

&nbsp; --min=2 --max=10 \\

&nbsp; --cpu-percent=70

```



\### Load Balancing



\*\*Nginx Load Balancer:\*\*

```nginx

upstream api\_backend {

&nbsp;   least\_conn;

&nbsp;   server api1:8000;

&nbsp;   server api2:8000;

&nbsp;   server api3:8000;

}



server {

&nbsp;   listen 80;

&nbsp;   location / {

&nbsp;       proxy\_pass http://api\_backend;

&nbsp;   }

}

```



\### Caching Strategy



```python

\# Redis caching

import redis

from functools import wraps



redis\_client = redis.Redis(host='localhost', port=6379)



def cache\_prediction(func):

&nbsp;   @wraps(func)

&nbsp;   async def wrapper(file\_hash, \*args, \*\*kwargs):

&nbsp;       # Check cache

&nbsp;       cached = redis\_client.get(file\_hash)

&nbsp;       if cached:

&nbsp;           return json.loads(cached)

&nbsp;       

&nbsp;       # Compute and cache

&nbsp;       result = await func(\*args, \*\*kwargs)

&nbsp;       redis\_client.setex(file\_hash, 3600, json.dumps(result))

&nbsp;       return result

&nbsp;   return wrapper

```



---



\## Troubleshooting



\### Common Issues



\*\*Issue:\*\* Container exits immediately

```bash

\# Check logs

docker logs alzheimers-api



\# Common causes:

\# - Model not found

\# - Port already in use

\# - Insufficient memory

```



\*\*Issue:\*\* High memory usage

```bash

\# Solution: Limit memory in docker-compose

services:

&nbsp; api:

&nbsp;   deploy:

&nbsp;     resources:

&nbsp;       limits:

&nbsp;         memory: 8G

```



\*\*Issue:\*\* Slow predictions

```bash

\# Check CPU/GPU usage

docker stats alzheimers-api



\# Solutions:

\# - Add GPU support

\# - Increase workers

\# - Use batch processing

\# - Enable model quantization

```



\### Health Checks



```bash

\# Check API health

curl http://localhost:8000/health



\# Check specific endpoint

curl -X POST http://localhost:8000/predict \\

&nbsp; -F "file=@test.nii.gz"



\# Monitor logs

tail -f logs/api.log

```



---



\## Backup \& Disaster Recovery



\### Backup Strategy



```bash

\#!/bin/bash

\# backup.sh



\# Backup models

tar -czf backup-$(date +%Y%m%d).tar.gz \\

&nbsp; results/models/ \\

&nbsp; configs/



\# Upload to S3

aws s3 cp backup-$(date +%Y%m%d).tar.gz \\

&nbsp; s3://backups/alzheimers-setnn/

```



\### Restore Procedure



```bash

\# Download backup

aws s3 cp s3://backups/alzheimers-setnn/backup-20250101.tar.gz .



\# Extract

tar -xzf backup-20250101.tar.gz



\# Restart services

docker-compose down

docker-compose up -d

```



---



\## Maintenance



\### Rolling Updates



```bash

\# Kubernetes rolling update

kubectl set image deployment/alzheimers-api \\

&nbsp; api=alzheimers-setnn:v2.0.0



\# Docker Compose

docker-compose up -d --no-deps --build api

```



\### Health Monitoring



```bash

\# Kubernetes

kubectl get pods -w



\# Check pod health

kubectl describe pod alzheimers-api-xxx



\# View events

kubectl get events --sort-by='.lastTimestamp'

```



---



\## Production Checklist



\- \[ ] SSL/TLS configured

\- \[ ] Authentication enabled

\- \[ ] Rate limiting active

\- \[ ] Monitoring set up (Prometheus/Grafana)

\- \[ ] Logging centralized (ELK/CloudWatch)

\- \[ ] Backups automated

\- \[ ] Auto-scaling configured

\- \[ ] Load balancer deployed

\- \[ ] Health checks working

\- \[ ] Documentation updated

\- \[ ] Disaster recovery plan tested

\- \[ ] Security audit completed



---



\## Support



For deployment issues:

\- \*\*GitHub Issues:\*\* https://github.com/yourusername/alzheimers-mri-setnn/issues

\- \*\*Documentation:\*\* https://github.com/yourusername/alzheimers-mri-setnn/docs

\- \*\*Email:\*\* your.email@example.com

