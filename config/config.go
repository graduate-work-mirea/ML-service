package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	DataPath          string
	ModelPath         string
	ProcessedDataPath string
	ServerPort        string
	SchedulerInterval time.Duration

	// PostgreSQL configuration
	PostgresHost     string
	PostgresPort     string
	PostgresUser     string
	PostgresPassword string
	PostgresDBName   string
	PostgresSSLMode  string
}

func New() (*Config, error) {
	// Data path
	dataPath := os.Getenv("DATA_PATH")
	if dataPath == "" {
		dataPath = "./data"
	}

	// Model path
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "./models"
	}

	// Processed data path
	processedDataPath := os.Getenv("PROCESSED_DATA_PATH")
	if processedDataPath == "" {
		processedDataPath = "./processor_data/processed"
	}

	// Server port
	serverPort := os.Getenv("SERVER_PORT")
	if serverPort == "" {
		serverPort = "8080"
	}

	// Scheduler interval (default: 24 hours)
	var schedulerInterval time.Duration
	intervalStr := os.Getenv("SCHEDULER_INTERVAL")
	if intervalStr == "" {
		schedulerInterval = 24 * time.Hour
	} else {
		intervalHours, err := strconv.Atoi(intervalStr)
		if err != nil {
			schedulerInterval = 24 * time.Hour
		} else {
			schedulerInterval = time.Duration(intervalHours) * time.Hour
		}
	}

	// PostgreSQL configuration
	postgresHost := os.Getenv("POSTGRES_HOST")
	if postgresHost == "" {
		postgresHost = "localhost"
	}

	postgresPort := os.Getenv("POSTGRES_PORT")
	if postgresPort == "" {
		postgresPort = "5432"
	}

	postgresUser := os.Getenv("POSTGRES_USER")
	if postgresUser == "" {
		postgresUser = "postgres"
	}

	postgresPassword := os.Getenv("POSTGRES_PASSWORD")
	if postgresPassword == "" {
		postgresPassword = "postgres"
	}

	postgresDBName := os.Getenv("POSTGRES_DB")
	if postgresDBName == "" {
		postgresDBName = "prediction_service"
	}

	postgresSSLMode := os.Getenv("POSTGRES_SSLMODE")
	if postgresSSLMode == "" {
		postgresSSLMode = "disable"
	}

	return &Config{
		DataPath:          dataPath,
		ModelPath:         modelPath,
		ProcessedDataPath: processedDataPath,
		ServerPort:        serverPort,
		SchedulerInterval: schedulerInterval,
		PostgresHost:      postgresHost,
		PostgresPort:      postgresPort,
		PostgresUser:      postgresUser,
		PostgresPassword:  postgresPassword,
		PostgresDBName:    postgresDBName,
		PostgresSSLMode:   postgresSSLMode,
	}, nil
}

// GetPostgresConnectionString returns the PostgreSQL connection string
func (c *Config) GetPostgresConnectionString() string {
	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		c.PostgresHost, c.PostgresPort, c.PostgresUser, c.PostgresPassword, c.PostgresDBName, c.PostgresSSLMode)
}
