FROM golang:1.24-alpine AS builder

WORKDIR /app

# Copy go.mod and go.sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o ml-prediction-service .

# Create final image with Python and Go binary
FROM python:3.10-slim

WORKDIR /app

# Install required system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Go binary from builder stage
COPY --from=builder /app/ml-prediction-service .

# Copy Python scripts and data directories
COPY scripts/ ./scripts/
COPY processor_data/ ./processor_data/

# Create model directory
RUN mkdir -p /app/models

# Create .env file from example if needed
COPY --from=builder /app/.env.example ./.env

# Expose the port
EXPOSE 8080

# Run the application
CMD ["./ml-prediction-service"]
