FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go.mod and go.sum files
COPY go.mod go.sum* ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o ml-service ./cmd

# Use a smaller image for the final container
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY scripts/ scripts/

# Create models directory
RUN mkdir -p models

# Copy the compiled Go binary
COPY --from=builder /app/ml-service .

# Expose the port
EXPOSE 8080

# Set environment variables
ENV RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
ENV POSTGRES_DSN=postgres://postgres:postgres@postgres:5432/ml_service?sslmode=disable

# Run the application
CMD ["./ml-service"]
