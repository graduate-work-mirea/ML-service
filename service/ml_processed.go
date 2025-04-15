package service

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"time"

	"github.com/graduate-work-mirea/data-processor-service/repository"
	"go.uber.org/zap"
)

// MLPredictionService provides functionality for training ML models and making predictions
type MLPredictionService struct {
	fileRepo      *repository.FileRepository
	postgresRepo  *repository.PostgresRepository
	scriptPath    string
	trainDataPath string
	testDataPath  string
	logger        *zap.SugaredLogger
}

// NewMLPredictionService creates a new ML prediction service
func NewMLPredictionService(fileRepo *repository.FileRepository, postgresRepo *repository.PostgresRepository, logger *zap.SugaredLogger) *MLPredictionService {
	return &MLPredictionService{
		fileRepo:      fileRepo,
		postgresRepo:  postgresRepo,
		scriptPath:    "scripts/lightGBM_model.py",
		trainDataPath: "train_data.csv",
		testDataPath:  "test_data.csv",
		logger:        logger,
	}
}

// PredictionRequest represents the input data for making a prediction
type PredictionRequest struct {
	ProductName               string  `json:"product_name"`
	Brand                     string  `json:"brand"`
	Category                  string  `json:"category"`
	Region                    string  `json:"region"`
	Seller                    string  `json:"seller"`
	Price                     float64 `json:"price"`
	OriginalPrice             float64 `json:"original_price"`
	DiscountPercentage        float64 `json:"discount_percentage"`
	StockLevel                float64 `json:"stock_level"`
	CustomerRating            float64 `json:"customer_rating"`
	ReviewCount               float64 `json:"review_count"`
	DeliveryDays              float64 `json:"delivery_days"`
	IsWeekend                 bool    `json:"is_weekend"`
	IsHoliday                 bool    `json:"is_holiday"`
	DayOfWeek                 int     `json:"day_of_week"`
	Month                     int     `json:"month"`
	Quarter                   int     `json:"quarter"`
	SalesQuantityLag1         float64 `json:"sales_quantity_lag_1"`
	PriceLag1                 float64 `json:"price_lag_1"`
	SalesQuantityLag3         float64 `json:"sales_quantity_lag_3"`
	PriceLag3                 float64 `json:"price_lag_3"`
	SalesQuantityLag7         float64 `json:"sales_quantity_lag_7"`
	PriceLag7                 float64 `json:"price_lag_7"`
	SalesQuantityRollingMean3 float64 `json:"sales_quantity_rolling_mean_3"`
	PriceRollingMean3         float64 `json:"price_rolling_mean_3"`
	SalesQuantityRollingMean7 float64 `json:"sales_quantity_rolling_mean_7"`
	PriceRollingMean7         float64 `json:"price_rolling_mean_7"`
}

// PredictionRequestMinimal represents the minimal input data for making a prediction
type PredictionRequestMinimal struct {
	ProductName    string     `json:"product_name" binding:"required"`
	Region         string     `json:"region" binding:"required"`
	Seller         string     `json:"seller" binding:"required"`
	PredictionDate *time.Time `json:"prediction_date,omitempty"`
	// Optional overrides for testing scenarios
	Price          *float64 `json:"price,omitempty"`
	OriginalPrice  *float64 `json:"original_price,omitempty"`
	StockLevel     *float64 `json:"stock_level,omitempty"`
	CustomerRating *float64 `json:"customer_rating,omitempty"`
	ReviewCount    *float64 `json:"review_count,omitempty"`
	DeliveryDays   *float64 `json:"delivery_days,omitempty"`
}

// PredictionResult represents the result of a prediction
type PredictionResult struct {
	PredictedPrice float64 `json:"predicted_price"`
	PredictedSales float64 `json:"predicted_sales"`
}

// TrainingResult represents the result of model training
type TrainingResult struct {
	PriceModel struct {
		BestIteration int     `json:"best_iteration"`
		BestScore     float64 `json:"best_score"`
	} `json:"price_model"`
	SalesModel struct {
		BestIteration int     `json:"best_iteration"`
		BestScore     float64 `json:"best_score"`
	} `json:"sales_model"`
	PythonOutput string `json:"-"`
}

// extractJSON extracts JSON from a string output
func extractJSON(output string) (string, error) {
	// Find a well-formed JSON object in the output
	// First try to extract using a stricter approach
	reStrict := regexp.MustCompile(`(?s)\{(?:[^{}]|(?:\{[^{}]*\}))*\}`)
	matches := reStrict.FindAllString(output, -1)

	// Try each match, starting from the last one (usually contains the result)
	if len(matches) > 0 {
		for i := len(matches) - 1; i >= 0; i-- {
			candidate := matches[i]
			// Verify that it's valid JSON
			var js json.RawMessage
			if err := json.Unmarshal([]byte(candidate), &js); err == nil {
				// Found valid JSON
				return candidate, nil
			}
		}
	}

	// Fallback to simpler regex
	re := regexp.MustCompile(`(?s)\{.*\}`)
	match := re.FindString(output)
	if match != "" {
		// Try to validate it
		var js json.RawMessage
		if err := json.Unmarshal([]byte(match), &js); err == nil {
			return match, nil
		}

		// If not valid, try to clean it up
		// Remove trailing commas which are invalid in JSON
		cleaned := regexp.MustCompile(`,\s*\}`).ReplaceAllString(match, "}")
		cleaned = regexp.MustCompile(`,\s*\]`).ReplaceAllString(cleaned, "]")

		if err := json.Unmarshal([]byte(cleaned), &js); err == nil {
			return cleaned, nil
		}
	}

	// No valid JSON found
	return "", fmt.Errorf("no valid JSON found in output: %s", output)
}

// TrainModels trains the price and sales prediction models
func (s *MLPredictionService) TrainModels() (*TrainingResult, error) {
	// Check if the script exists
	if !s.fileRepo.FileExists(s.scriptPath) {
		return nil, fmt.Errorf("python script not found: %s", s.scriptPath)
	}

	fullTrainPath := s.fileRepo.GetDataFilePath(s.trainDataPath)
	fullValPath := s.fileRepo.GetDataFilePath(s.testDataPath)

	if !s.fileRepo.FileExists(fullTrainPath) {
		return nil, fmt.Errorf("training data file not found: %s", fullTrainPath)
	}
	if !s.fileRepo.FileExists(fullValPath) {
		return nil, fmt.Errorf("validation data file not found: %s", fullValPath)
	}

	// Run Python script to train models
	output, err := s.fileRepo.RunPythonScript(s.scriptPath, "train", fullTrainPath, "--val-data", fullValPath)
	if err != nil {
		return nil, fmt.Errorf("error running training script: %v\n\nOutput: %s", err, output)
	}

	// Save the output for logging purposes
	pythonOutput := output

	// Extract JSON from the output
	jsonStr, err := extractJSON(output)
	if err != nil {
		// Return the full Python output as part of the error
		return nil, fmt.Errorf("python_output:%s", pythonOutput)
	}

	// Parse the output to get training metrics
	var result TrainingResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("error parsing training results JSON: %v\n\nOutput: %s", err, pythonOutput)
	}

	result.PythonOutput = pythonOutput

	return &result, nil
}

// Predict makes predictions for product price and sales using the full request
func (s *MLPredictionService) Predict(request *PredictionRequest) (*PredictionResult, error) {
	// Check if the script exists
	if !s.fileRepo.FileExists(s.scriptPath) {
		return nil, fmt.Errorf("python script not found: %s", s.scriptPath)
	}

	// Convert request to JSON
	requestJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("error marshaling prediction request: %v", err)
	}

	// Run Python script to make prediction
	output, err := s.fileRepo.RunPythonScript(s.scriptPath, "predict", string(requestJSON))
	if err != nil {
		return nil, fmt.Errorf("error making prediction: %v", err)
	}

	// Extract JSON from the output
	jsonStr, err := extractJSON(output)
	if err != nil {
		return nil, fmt.Errorf("error extracting JSON from output: %v", err)
	}

	// Parse the output to get prediction results
	var result PredictionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("error parsing prediction results: %v", err)
	}

	return &result, nil
}

// PredictMinimal makes predictions with minimal input by fetching historical data from PostgreSQL
func (s *MLPredictionService) PredictMinimal(minRequest *PredictionRequestMinimal) (*PredictionResult, error) {
	// Determine prediction date (default to today if not provided)
	predictionDate := time.Now()
	if minRequest.PredictionDate != nil {
		predictionDate = *minRequest.PredictionDate
	}

	// Fetch historical data from PostgreSQL
	historicalData, err := s.postgresRepo.GetProductHistoricalData(
		minRequest.ProductName,
		minRequest.Region,
		minRequest.Seller,
		predictionDate,
	)
	if err != nil {
		s.logger.Errorw("Error fetching historical data", "error", err,
			"product", minRequest.ProductName,
			"region", minRequest.Region,
			"seller", minRequest.Seller)
		// Continue with default values instead of returning error
		historicalData = &repository.ProductHistoricalData{
			Brand:     "Unknown Brand",
			Category:  "Unknown Category",
			IsWeekend: predictionDate.Weekday() == time.Saturday || predictionDate.Weekday() == time.Sunday,
			IsHoliday: false,
			DayOfWeek: int(predictionDate.Weekday()),
			Month:     int(predictionDate.Month()),
			Quarter:   (int(predictionDate.Month())-1)/3 + 1,
		}
	}

	// Create full prediction request from historical data
	fullRequest := &PredictionRequest{
		ProductName: minRequest.ProductName,
		Brand:       historicalData.Brand,
		Category:    historicalData.Category,
		Region:      minRequest.Region,
		Seller:      minRequest.Seller,
		IsWeekend:   historicalData.IsWeekend,
		IsHoliday:   historicalData.IsHoliday,
		DayOfWeek:   historicalData.DayOfWeek,
		Month:       historicalData.Month,
		Quarter:     historicalData.Quarter,
	}

	// Set values from historical data if available, or use defaults if not
	// Price
	if historicalData.Price.Valid {
		fullRequest.Price = historicalData.Price.Float64
	} else {
		fullRequest.Price = 1000.0 // Default price
	}

	// Original price
	if historicalData.OriginalPrice.Valid {
		fullRequest.OriginalPrice = historicalData.OriginalPrice.Float64
	} else {
		fullRequest.OriginalPrice = fullRequest.Price // Default to current price
	}

	// Discount percentage
	if historicalData.DiscountPerc.Valid {
		fullRequest.DiscountPercentage = historicalData.DiscountPerc.Float64
	} else {
		// Calculate discount if we have original price and it's different from current price
		if fullRequest.OriginalPrice > fullRequest.Price {
			fullRequest.DiscountPercentage = (fullRequest.OriginalPrice - fullRequest.Price) / fullRequest.OriginalPrice * 100
		} else {
			fullRequest.DiscountPercentage = 0.0
		}
	}

	// Stock level
	if historicalData.StockLevel.Valid {
		fullRequest.StockLevel = historicalData.StockLevel.Float64
	} else {
		fullRequest.StockLevel = 100.0 // Default stock level
	}

	// Customer rating
	if historicalData.CustomerRating.Valid {
		fullRequest.CustomerRating = historicalData.CustomerRating.Float64
	} else {
		fullRequest.CustomerRating = 4.0 // Default rating
	}

	// Review count
	if historicalData.ReviewCount.Valid {
		fullRequest.ReviewCount = historicalData.ReviewCount.Float64
	} else {
		fullRequest.ReviewCount = 10.0 // Default review count
	}

	// Delivery days
	if historicalData.DeliveryDays.Valid {
		fullRequest.DeliveryDays = historicalData.DeliveryDays.Float64
	} else {
		fullRequest.DeliveryDays = 3.0 // Default delivery days
	}

	// Historical data - lags and rolling means
	// Use defaults if not available
	if historicalData.SalesQuantityLag1.Valid {
		fullRequest.SalesQuantityLag1 = historicalData.SalesQuantityLag1.Float64
	} else {
		fullRequest.SalesQuantityLag1 = 10.0 // Default sales
	}

	if historicalData.PriceLag1.Valid {
		fullRequest.PriceLag1 = historicalData.PriceLag1.Float64
	} else {
		fullRequest.PriceLag1 = fullRequest.Price // Default to current price
	}

	if historicalData.SalesQuantityLag3.Valid {
		fullRequest.SalesQuantityLag3 = historicalData.SalesQuantityLag3.Float64
	} else {
		fullRequest.SalesQuantityLag3 = 9.0 // Slightly different from lag 1
	}

	if historicalData.PriceLag3.Valid {
		fullRequest.PriceLag3 = historicalData.PriceLag3.Float64
	} else {
		fullRequest.PriceLag3 = fullRequest.Price * 0.98 // Slight price change from current
	}

	if historicalData.SalesQuantityLag7.Valid {
		fullRequest.SalesQuantityLag7 = historicalData.SalesQuantityLag7.Float64
	} else {
		fullRequest.SalesQuantityLag7 = 8.0 // Default
	}

	if historicalData.PriceLag7.Valid {
		fullRequest.PriceLag7 = historicalData.PriceLag7.Float64
	} else {
		fullRequest.PriceLag7 = fullRequest.Price * 0.95 // Slight price change from current
	}

	if historicalData.SalesQuantityRollingMean3.Valid {
		fullRequest.SalesQuantityRollingMean3 = historicalData.SalesQuantityRollingMean3.Float64
	} else {
		// Average of lag1 and lag3 if available, otherwise default
		fullRequest.SalesQuantityRollingMean3 = (fullRequest.SalesQuantityLag1 + fullRequest.SalesQuantityLag3) / 2
	}

	if historicalData.PriceRollingMean3.Valid {
		fullRequest.PriceRollingMean3 = historicalData.PriceRollingMean3.Float64
	} else {
		// Average of current, lag1 and lag3 if available
		fullRequest.PriceRollingMean3 = (fullRequest.Price + fullRequest.PriceLag1 + fullRequest.PriceLag3) / 3
	}

	if historicalData.SalesQuantityRollingMean7.Valid {
		fullRequest.SalesQuantityRollingMean7 = historicalData.SalesQuantityRollingMean7.Float64
	} else {
		// Average of lag1, lag3 and lag7 if available, otherwise default
		fullRequest.SalesQuantityRollingMean7 = (fullRequest.SalesQuantityLag1 + fullRequest.SalesQuantityLag3 + fullRequest.SalesQuantityLag7) / 3
	}

	if historicalData.PriceRollingMean7.Valid {
		fullRequest.PriceRollingMean7 = historicalData.PriceRollingMean7.Float64
	} else {
		// Average of current, lag1, lag3 and lag7
		fullRequest.PriceRollingMean7 = (fullRequest.Price + fullRequest.PriceLag1 + fullRequest.PriceLag3 + fullRequest.PriceLag7) / 4
	}

	// Override values if provided in the minimal request
	if minRequest.Price != nil {
		fullRequest.Price = *minRequest.Price
	}
	if minRequest.OriginalPrice != nil {
		fullRequest.OriginalPrice = *minRequest.OriginalPrice
	}
	if minRequest.StockLevel != nil {
		fullRequest.StockLevel = *minRequest.StockLevel
	}
	if minRequest.CustomerRating != nil {
		fullRequest.CustomerRating = *minRequest.CustomerRating
	}
	if minRequest.ReviewCount != nil {
		fullRequest.ReviewCount = *minRequest.ReviewCount
	}
	if minRequest.DeliveryDays != nil {
		fullRequest.DeliveryDays = *minRequest.DeliveryDays
	}

	// Call the regular predict method with the full request
	return s.Predict(fullRequest)
}

// CheckModelsExist checks if trained models exist
func (s *MLPredictionService) CheckModelsExist() bool {
	modelDir := s.fileRepo.GetModelPath()
	priceModelPath := filepath.Join(modelDir, "price_model.pkl")
	salesModelPath := filepath.Join(modelDir, "sales_model.pkl")
	featuresPath := filepath.Join(modelDir, "feature_info.json")

	return s.fileRepo.FileExists(priceModelPath) &&
		s.fileRepo.FileExists(salesModelPath) &&
		s.fileRepo.FileExists(featuresPath)
}
