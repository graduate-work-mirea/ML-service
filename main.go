package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/graduate-work-mirea/data-processor-service/assembly"
	"github.com/graduate-work-mirea/data-processor-service/config"
	"github.com/joho/godotenv"
	"go.uber.org/zap"
)

// @title ML Prediction Service
// @version 1.0
// @description Predict product price and sales using LightGBM models
func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()
	sugar := logger.Sugar()

	if err := godotenv.Load(); err != nil {
		sugar.Warnf("Error loading .env file: %v", err)
	}

	cfg, err := config.New()
	if err != nil {
		sugar.Fatalf("Failed to load config: %v", err)
	}

	locator, err := assembly.NewServiceLocator(cfg, sugar)
	if err != nil {
		sugar.Fatalf("Failed to initialize service locator: %v", err)
	}
	defer locator.Close()

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Check if models exist, if not, train them
	if !locator.MLPredictionService.CheckModelsExist() {
		sugar.Info("Models not found, training new models...")
		result, err := locator.MLPredictionService.TrainModels()
		if err != nil {
			sugar.Warnf("Failed to train models: %v", err)
		} else {
			sugar.Infof("Models trained successfully: %v", result)
		}
	}

	// Start HTTP server
	go func() {
		sugar.Infof("Starting HTTP server on port %s", cfg.ServerPort)
		if err := locator.HTTPServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			sugar.Fatalf("Failed to start HTTP server: %v", err)
		}
	}()

	// Wait for termination signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigCh
	sugar.Infof("Received signal: %v, shutting down...", sig)

	// Create context with timeout for graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, 5*time.Second)
	defer shutdownCancel()

	// Shutdown HTTP server
	if err := locator.HTTPServer.Shutdown(shutdownCtx); err != nil {
		sugar.Errorf("HTTP server shutdown error: %v", err)
	} else {
		sugar.Info("HTTP server shutdown gracefully")
	}
}
