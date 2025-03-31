package ml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"sync"
)

// SalesData represents the input data for training
type SalesData struct {
	ProductID string    `json:"product_id"`
	Sales     float64   `json:"sales"`
	Date      string    `json:"date"`
}

// ModelInfo contains information about a trained model
type ModelInfo struct {
	ModelPath          string `json:"model_path"`
	ProductID          string `json:"product_id"`
	LastDate           string `json:"last_date"`
	LastDaysSinceFirst int    `json:"last_days_since_first"`
	FirstDate          string `json:"first_date"`
}

// PredictionResult contains the prediction result
type PredictionResult struct {
	ProductID       string  `json:"product_id"`
	ForecastedSales float64 `json:"forecasted_sales"`
	Date            string  `json:"date"`
}

// Service handles ML operations
type Service struct {
	dataStore     map[string][]SalesData
	modelInfos    map[string]ModelInfo
	minDataPoints int
	mu            sync.RWMutex
}

// NewService creates a new ML service
func NewService(minDataPoints int) *Service {
	return &Service{
		dataStore:     make(map[string][]SalesData),
		modelInfos:    make(map[string]ModelInfo),
		minDataPoints: minDataPoints,
		mu:            sync.RWMutex{},
	}
}

// ProcessData processes incoming sales data
func (s *Service) ProcessData(data SalesData) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Add data to store
	s.dataStore[data.ProductID] = append(s.dataStore[data.ProductID], data)

	// Check if we have enough data points to train a model
	if len(s.dataStore[data.ProductID]) >= s.minDataPoints {
		// Train model
		modelInfo, err := s.trainModel(data.ProductID)
		if err != nil {
			return false, fmt.Errorf("failed to train model: %w", err)
		}

		// Store model info
		s.modelInfos[data.ProductID] = modelInfo
		
		// Clear data store for this product
		s.dataStore[data.ProductID] = []SalesData{}
		
		return true, nil
	}

	return false, nil
}

// trainModel trains a model for the specified product
func (s *Service) trainModel(productID string) (ModelInfo, error) {
	// Get data for the product
	data := s.dataStore[productID]
	
	// Convert to JSON
	jsonData, err := json.Marshal(data)
	if err != nil {
		return ModelInfo{}, err
	}

	// Run Python script
	cmd := exec.Command("python", filepath.Join("scripts", "train_model.py"))
	cmd.Stdin = bytes.NewBuffer(jsonData)
	
	var out bytes.Buffer
	cmd.Stdout = &out
	
	if err := cmd.Run(); err != nil {
		return ModelInfo{}, err
	}

	// Parse result
	var result ModelInfo
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		return ModelInfo{}, err
	}

	log.Printf("Trained model for product %s: %+v", productID, result)
	return result, nil
}

// Predict makes a prediction for the specified product
func (s *Service) Predict(productID string) (PredictionResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check if we have a model for this product
	modelInfo, ok := s.modelInfos[productID]
	if !ok {
		return PredictionResult{}, fmt.Errorf("no model available for product %s", productID)
	}

	// Prepare input data
	input := struct {
		ProductID string    `json:"product_id"`
		ModelInfo ModelInfo `json:"model_info"`
	}{
		ProductID: productID,
		ModelInfo: modelInfo,
	}

	// Convert to JSON
	jsonData, err := json.Marshal(input)
	if err != nil {
		return PredictionResult{}, err
	}

	// Run Python script
	cmd := exec.Command("python", filepath.Join("scripts", "predict.py"))
	cmd.Stdin = bytes.NewBuffer(jsonData)
	
	var out bytes.Buffer
	cmd.Stdout = &out
	
	if err := cmd.Run(); err != nil {
		return PredictionResult{}, err
	}

	// Parse result
	var result PredictionResult
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		return PredictionResult{}, err
	}

	log.Printf("Prediction for product %s: %+v", productID, result)
	return result, nil
}
