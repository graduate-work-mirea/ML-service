package config

import (
	"os"
)

// Config holds all configuration for the application
type Config struct {
	RabbitMQURL string
	PostgresDSN string
}

// NewConfig creates a new Config from environment variables
func NewConfig() *Config {
	return &Config{
		RabbitMQURL: getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		PostgresDSN: getEnv("POSTGRES_DSN", "postgres://postgres:postgres@localhost:5432/ml_service?sslmode=disable"),
	}
}

// getEnv gets an environment variable or returns a default value
func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
