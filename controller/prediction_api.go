package controller

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/graduate-work-mirea/data-processor-service/service"
	"go.uber.org/zap"
)

// PredictionAPIController handles HTTP requests for ML predictions
type PredictionAPIController struct {
	mlService *service.MLPredictionService
	logger    *zap.SugaredLogger
}

// NewPredictionAPIController creates a new prediction API controller
func NewPredictionAPIController(mlService *service.MLPredictionService, logger *zap.SugaredLogger) *PredictionAPIController {
	return &PredictionAPIController{
		mlService: mlService,
		logger:    logger,
	}
}

// RegisterRoutes registers the HTTP routes for the prediction API
func (c *PredictionAPIController) RegisterRoutes(router *gin.Engine) {
	api := router.Group("/api/v1")
	{
		api.POST("/predict", c.HandlePredict)
		api.POST("/predict/minimal", c.HandlePredictMinimal)
		api.POST("/train", c.HandleTrain)
		api.GET("/status", c.HandleStatus)
	}
}

// HandlePredict handles prediction requests with full feature set
// @Summary Make a price and sales prediction with full feature set
// @Description Predict future price and sales for a product based on input features
// @Accept json
// @Produce json
// @Param request body service.PredictionRequest true "Product data for prediction"
// @Success 200 {object} service.PredictionResult
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /api/v1/predict [post]
func (c *PredictionAPIController) HandlePredict(ctx *gin.Context) {
	var request service.PredictionRequest

	// Parse request body
	if err := ctx.ShouldBindJSON(&request); err != nil {
		c.logger.Errorw("Invalid prediction request", "error", err)
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format: " + err.Error()})
		return
	}

	// Validate that required fields have reasonable values
	if request.Price <= 0 {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Price must be positive"})
		return
	}

	// Make prediction
	result, err := c.mlService.Predict(&request)
	if err != nil {
		c.logger.Errorw("Error making prediction", "error", err,
			"product", request.ProductName, "region", request.Region, "seller", request.Seller)

		// Check if this might be a problem with JSON parsing from Python script
		if err.Error() == "error extracting JSON from output" ||
			err.Error() == "error parsing prediction results" {
			ctx.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to parse prediction results from model. This might be due to issues with the Python script.",
				"details": err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to make prediction: " + err.Error()})
		return
	}

	// Return prediction result
	ctx.JSON(http.StatusOK, result)
}

// HandlePredictMinimal handles prediction requests with minimal input
// @Summary Make a price and sales prediction with minimal input
// @Description Predict future price and sales for a product using minimal input and auto-fetched historical data
// @Accept json
// @Produce json
// @Param request body service.PredictionRequestMinimal true "Minimal product data for prediction"
// @Success 200 {object} service.PredictionResult
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /api/v1/predict/minimal [post]
func (c *PredictionAPIController) HandlePredictMinimal(ctx *gin.Context) {
	var request service.PredictionRequestMinimal

	// Parse request body
	if err := ctx.ShouldBindJSON(&request); err != nil {
		c.logger.Errorw("Invalid minimal prediction request", "error", err)
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	// Make prediction with minimal data
	result, err := c.mlService.PredictMinimal(&request)
	if err != nil {
		c.logger.Errorw("Error making prediction with minimal data", "error", err)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to make prediction: " + err.Error()})
		return
	}

	// Return prediction result
	ctx.JSON(http.StatusOK, result)
}

// HandleTrain handles model training requests
// @Summary Train the prediction models
// @Description Train the price and sales prediction models using the processed data
// @Accept json
// @Produce json
// @Success 200 {object} service.TrainingResult
// @Failure 500 {object} map[string]string
// @Router /api/v1/train [post]
func (c *PredictionAPIController) HandleTrain(ctx *gin.Context) {
	// Train models
	result, err := c.mlService.TrainModels()
	if err != nil {
		errMsg := err.Error()

		// Check if this is Python output that we should log as info
		if len(errMsg) > 13 && errMsg[:13] == "python_output:" {
			// Extract and log the Python output as info
			pythonOutput := errMsg[13:]
			c.logger.Infow("Python training process", "python_logs", pythonOutput)

			// Try to still find valid metrics in the Python output
			// We'll treat this as a partial success if the models were trained
			if c.mlService.CheckModelsExist() {
				c.logger.Infow("Models were successfully created despite warnings")
				ctx.JSON(http.StatusOK, gin.H{
					"message":       "Training completed with warnings, models created",
					"python_output": pythonOutput,
				})
			} else {
				ctx.JSON(http.StatusInternalServerError, gin.H{
					"error":         "Training did not complete successfully",
					"python_output": pythonOutput,
				})
			}
			return
		}

		// For other errors, log as normal info
		c.logger.Infow("Training process completed with issues", "details", err)
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to train models"})
		return
	}

	// Always log the Python output if available
	if result.PythonOutput != "" {
		c.logger.Infow("Python training process completed successfully", "python_logs", result.PythonOutput)
	}

	// Return training result
	ctx.JSON(http.StatusOK, result)
}

// HandleStatus handles model status requests
// @Summary Check model status
// @Description Check if the prediction models are trained and available
// @Produce json
// @Success 200 {object} map[string]bool
// @Router /api/v1/status [get]
func (c *PredictionAPIController) HandleStatus(ctx *gin.Context) {
	// Check if models exist
	modelsExist := c.mlService.CheckModelsExist()

	// Return status
	ctx.JSON(http.StatusOK, gin.H{"models_trained": modelsExist})
}
