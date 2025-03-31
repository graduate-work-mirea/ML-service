package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"ml-service/internal/database"
	"ml-service/internal/ml"
)

// PredictRequest represents a prediction request
type PredictRequest struct {
	ProductID string `json:"product_id" binding:"required"`
}

// Server handles HTTP requests
type Server struct {
	router *gin.Engine
	mlSvc  *ml.Service
	db     *database.Database
}

// NewServer creates a new HTTP server
func NewServer(mlSvc *ml.Service, db *database.Database) *Server {
	router := gin.Default()
	
	server := &Server{
		router: router,
		mlSvc:  mlSvc,
		db:     db,
	}
	
	// Set up routes
	router.POST("/predict", server.handlePredict)
	router.GET("/health", server.handleHealth)
	
	return server
}

// handlePredict handles prediction requests
func (s *Server) handlePredict(c *gin.Context) {
	var req PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Make prediction
	prediction, err := s.mlSvc.Predict(req.ProductID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Save prediction to database
	predictionJSON, err := json.Marshal(map[string]interface{}{
		"forecasted_sales": prediction.ForecastedSales,
		"date":             prediction.Date,
	})
	if err != nil {
		log.Printf("Failed to marshal prediction: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process prediction"})
		return
	}
	
	if err := s.db.SaveForecast(c.Request.Context(), req.ProductID, predictionJSON); err != nil {
		log.Printf("Failed to save prediction: %v", err)
		// Continue anyway to return the prediction to the client
	}
	
	// Return prediction
	c.JSON(http.StatusOK, prediction)
}

// handleHealth handles health check requests
func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// Start starts the HTTP server
func (s *Server) Start(addr string) error {
	return s.router.Run(addr)
}

// Close closes the server
func (s *Server) Close() {
	// Nothing to do here for now
}
