package config

import (
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

	return &Config{
		DataPath:          dataPath,
		ModelPath:         modelPath,
		ProcessedDataPath: processedDataPath,
		ServerPort:        serverPort,
		SchedulerInterval: schedulerInterval,
	}, nil
}
