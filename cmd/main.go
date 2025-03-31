package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ml-service/internal/api"
	"ml-service/internal/config"
	"ml-service/internal/database"
	"ml-service/internal/ml"
	"ml-service/internal/rabbitmq"
)

const (
	minDataPoints = 3 // Minimum number of data points required for training
)

func main() {
	log.Println("Starting ML Service...")

	// Load configuration
	cfg := config.NewConfig()
	log.Printf("Configuration loaded: RabbitMQ=%s, PostgreSQL=%s", cfg.RabbitMQURL, cfg.PostgresDSN)

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize ML service
	mlSvc := ml.NewService(minDataPoints)
	log.Printf("ML service initialized with minimum %d data points", minDataPoints)

	// Initialize database
	db, err := database.New(cfg.PostgresDSN)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()
	log.Println("Connected to PostgreSQL")

	// Initialize RabbitMQ consumer
	consumer, err := rabbitmq.NewConsumer(cfg.RabbitMQURL, mlSvc)
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer consumer.Close()
	log.Println("Connected to RabbitMQ")

	// Start consuming messages
	if err := consumer.Start(ctx); err != nil {
		log.Fatalf("Failed to start RabbitMQ consumer: %v", err)
	}

	// Initialize HTTP server
	server := api.NewServer(mlSvc, db)
	
	// Start HTTP server in a goroutine
	go func() {
		log.Println("Starting HTTP server on :8080")
		if err := server.Start(":8080"); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start HTTP server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down...")

	// Cancel context to stop RabbitMQ consumer
	cancel()

	// Give some time for cleanup
	time.Sleep(time.Second)
	log.Println("Shutdown complete")
}
