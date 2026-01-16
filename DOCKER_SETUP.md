# Docker Setup Guide for Is It Real

## Prerequisites
- Docker and Docker Compose installed on your system
- Model file (`model/best_model.h5`) in the project directory

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f is_it_real

# Stop the container
docker-compose down
```

### Option 2: Manual Docker Build
```bash
# Build the image
docker build -t is_it_real:latest .

# Run the container
docker run -d \
  --name is_it_real \
  -p 4142:8000 \
  -v ./file_dir:/app/file_dir \
  -v ./model:/app/model \
  -e TF_ENABLE_ONEDNN_OPTS=0 \
  is_it_real:latest

# View logs
docker logs -f is_it_real

# Stop the container
docker stop is_it_real
```

## API Access
Once the container is running, access the API at:
- **Swagger UI**: http://localhost:4142/docs
- **ReDoc**: http://localhost:4142/redoc
- **API Base**: http://localhost:4142

## API Usage Example
```bash
# Predict an image
curl -X POST "http://localhost:8000/predict/folder_name/image.jpg"
```

## File Structure Requirements
```
project/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model/
│   └── best_model.h5
├── file_dir/           # Your images directory
│   └── subfolder/
│       └── image.jpg
└── is_it_real/
```

## Environment Configuration
Before running, update `app.py` if needed:
- `MODEL_PATH`: Path to the trained model (default: `model\best_model.h5`)
- `FILES_DIR`: Directory containing images for prediction (default: `file_dir`)

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs is_it_real

# Rebuild without cache
docker-compose build --no-cache
```

### Model not found error
- Ensure `model/best_model.h5` exists in your project directory
- The file is mounted in the container automatically

### Port 4142 already in use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "4143:8000"  # Use 4143 instead
```

## Performance Notes
- First prediction may take longer as TensorFlow initializes
- Use GPU support for faster inference:
  ```yaml
  # In docker-compose.yml, add under services.is_it_real:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  ```

## Cleanup
```bash
# Remove container
docker-compose down

# Remove image
docker rmi is_it_real:latest

# Remove all dangling resources
docker system prune -a
```
