package service

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"

	"github.com/graduate-work-mirea/data-processor-service/repository"
)

// MLPredictionService provides functionality for training ML models and making predictions
type MLPredictionService struct {
	fileRepo      *repository.FileRepository
	scriptPath    string
	trainDataPath string
	testDataPath  string
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

// PredictionResult represents the output from the prediction model
type PredictionResult struct {
	PredictedPrice float64 `json:"predicted_price"`
	PredictedSales float64 `json:"predicted_sales"`
}

// TrainingResult represents the metrics from model training
type TrainingResult struct {
	PriceModel struct {
		BestIteration int     `json:"best_iteration"`
		BestScore     float64 `json:"best_score"`
	} `json:"price_model"`
	SalesModel struct {
		BestIteration int     `json:"best_iteration"`
		BestScore     float64 `json:"best_score"`
	} `json:"sales_model"`
	PythonOutput string `json:"python_output,omitempty"`
}

// NewMLPredictionService creates a new ML prediction service
func NewMLPredictionService(fileRepo *repository.FileRepository) *MLPredictionService {
	return &MLPredictionService{
		fileRepo:      fileRepo,
		scriptPath:    filepath.Join("scripts", "lightGBM_model.py"),
		trainDataPath: "train_data.csv",
		testDataPath:  "test_data.csv",
	}
}

// extractJSON извлекает JSON объект из вывода, который может содержать логи и другие тексты
func extractJSON(output string) (string, error) {
	// Более строгий подход - ищем строку, которая содержит полный JSON объект
	// Начинается с { и заканчивается } и между ними валидный JSON
	// Проверяем все потенциальные JSON объекты от наиболее полного/последнего к началу

	// Находим все возможные JSON объекты (строки в фигурных скобках)
	re := regexp.MustCompile(`\{[^{}]*(\{[^{}]*\}[^{}]*)*\}`)
	matches := re.FindAllString(output, -1)

	if len(matches) == 0 {
		return "", fmt.Errorf("no JSON-like patterns found in output")
	}

	// Проверяем каждое совпадение с конца (предполагая, что последний JSON объект наиболее вероятно содержит результат)
	for i := len(matches) - 1; i >= 0; i-- {
		candidate := matches[i]
		var js json.RawMessage
		if err := json.Unmarshal([]byte(candidate), &js); err == nil {
			// Нашли валидный JSON
			return candidate, nil
		}
	}

	// Не нашли ни одного валидного JSON
	return "", fmt.Errorf("no valid JSON objects found in output")
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

// Predict makes predictions for product price and sales
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
