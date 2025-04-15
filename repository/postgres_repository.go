package repository

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

// PostgresRepository handles database operations for product data
type PostgresRepository struct {
	db *sql.DB
}

// ProductHistoricalData represents historical data for a product
type ProductHistoricalData struct {
	SalesQuantityLag1         sql.NullFloat64
	SalesQuantityLag3         sql.NullFloat64
	SalesQuantityLag7         sql.NullFloat64
	PriceLag1                 sql.NullFloat64
	PriceLag3                 sql.NullFloat64
	PriceLag7                 sql.NullFloat64
	SalesQuantityRollingMean3 sql.NullFloat64
	SalesQuantityRollingMean7 sql.NullFloat64
	PriceRollingMean3         sql.NullFloat64
	PriceRollingMean7         sql.NullFloat64
	// Current values
	Price          sql.NullFloat64
	OriginalPrice  sql.NullFloat64
	DiscountPerc   sql.NullFloat64
	StockLevel     sql.NullFloat64
	CustomerRating sql.NullFloat64
	ReviewCount    sql.NullFloat64
	DeliveryDays   sql.NullFloat64
	Brand          string
	Category       string
	// Date related
	IsWeekend bool
	IsHoliday bool
	DayOfWeek int
	Month     int
	Quarter   int
}

// NewPostgresRepository creates a new PostgresRepository instance
func NewPostgresRepository(connStr string) (*PostgresRepository, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Test the connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &PostgresRepository{
		db: db,
	}, nil
}

// Close closes the database connection
func (r *PostgresRepository) Close() error {
	return r.db.Close()
}

// GetLatestProductData retrieves the latest product data from the database
func (r *PostgresRepository) GetLatestProductData(productName, region, seller string) (*ProductHistoricalData, error) {
	query := `
		SELECT 
			brand, category, price, original_price, discount_percentage, 
			stock_level, customer_rating, review_count, delivery_days,
			is_weekend, is_holiday, day_of_week, month, quarter
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3
		ORDER BY date DESC
		LIMIT 1
	`

	var data ProductHistoricalData
	err := r.db.QueryRow(query, productName, region, seller).Scan(
		&data.Brand, &data.Category, &data.Price, &data.OriginalPrice, &data.DiscountPerc,
		&data.StockLevel, &data.CustomerRating, &data.ReviewCount, &data.DeliveryDays,
		&data.IsWeekend, &data.IsHoliday, &data.DayOfWeek, &data.Month, &data.Quarter,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			// No data found for this product, use default values
			data.Brand = "Unknown Brand"
			data.Category = "Unknown Category"

			// Use zero values for numeric fields (they're already initialized as sql.NullFloat64)
			// Return a valid object with defaults instead of error
			return &data, nil
		}
		return nil, fmt.Errorf("failed to get latest product data: %w", err)
	}

	return &data, nil
}

// GetProductHistoricalData retrieves historical data for a product from the database
func (r *PostgresRepository) GetProductHistoricalData(productName, region, seller string, date time.Time) (*ProductHistoricalData, error) {
	// Get the date in YYYY-MM-DD format
	dateStr := date.Format("2006-01-02")

	// Calculate date features for next day (prediction date)
	predictionDate := date.AddDate(0, 0, 1)
	dayOfWeek := int(predictionDate.Weekday())
	month := int(predictionDate.Month())
	quarter := (month-1)/3 + 1
	isWeekend := predictionDate.Weekday() == time.Saturday || predictionDate.Weekday() == time.Sunday

	// Get basic data (brand, category) from the latest record
	latestData, err := r.GetLatestProductData(productName, region, seller)
	if err != nil {
		return nil, err
	}

	// Now get historical data for lags and rolling means
	data := &ProductHistoricalData{
		Brand:     latestData.Brand,
		Category:  latestData.Category,
		IsWeekend: isWeekend,
		IsHoliday: false, // Would need a holiday calendar to determine this properly
		DayOfWeek: dayOfWeek,
		Month:     month,
		Quarter:   quarter,
	}

	// Get price and sales quantity lag 1
	lag1Date := date.AddDate(0, 0, -1)
	lag1Query := `
		SELECT price, sales_quantity 
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3 AND date = $4
		LIMIT 1
	`
	err = r.db.QueryRow(lag1Query, productName, region, seller, lag1Date.Format("2006-01-02")).
		Scan(&data.PriceLag1, &data.SalesQuantityLag1)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to get lag 1 data: %w", err)
	}

	// Get price and sales quantity lag 3
	lag3Date := date.AddDate(0, 0, -3)
	lag3Query := `
		SELECT price, sales_quantity 
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3 AND date = $4
		LIMIT 1
	`
	err = r.db.QueryRow(lag3Query, productName, region, seller, lag3Date.Format("2006-01-02")).
		Scan(&data.PriceLag3, &data.SalesQuantityLag3)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to get lag 3 data: %w", err)
	}

	// Get price and sales quantity lag 7
	lag7Date := date.AddDate(0, 0, -7)
	lag7Query := `
		SELECT price, sales_quantity 
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3 AND date = $4
		LIMIT 1
	`
	err = r.db.QueryRow(lag7Query, productName, region, seller, lag7Date.Format("2006-01-02")).
		Scan(&data.PriceLag7, &data.SalesQuantityLag7)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to get lag 7 data: %w", err)
	}

	// Calculate rolling means - for sales quantity over last 3 days
	rollingMean3Query := `
		SELECT AVG(price), AVG(sales_quantity)
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3 
		AND date BETWEEN $4 AND $5
	`
	rolling3StartDate := date.AddDate(0, 0, -2) // Last 3 days including current
	err = r.db.QueryRow(rollingMean3Query, productName, region, seller,
		rolling3StartDate.Format("2006-01-02"), dateStr).
		Scan(&data.PriceRollingMean3, &data.SalesQuantityRollingMean3)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to get rolling mean 3 data: %w", err)
	}

	// Calculate rolling means - for sales quantity over last 7 days
	rollingMean7Query := `
		SELECT AVG(price), AVG(sales_quantity)
		FROM processed_data 
		WHERE product_name = $1 AND region = $2 AND seller = $3 
		AND date BETWEEN $4 AND $5
	`
	rolling7StartDate := date.AddDate(0, 0, -6) // Last 7 days including current
	err = r.db.QueryRow(rollingMean7Query, productName, region, seller,
		rolling7StartDate.Format("2006-01-02"), dateStr).
		Scan(&data.PriceRollingMean7, &data.SalesQuantityRollingMean7)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to get rolling mean 7 data: %w", err)
	}

	// Set current values from latest data
	data.Price = latestData.Price
	data.OriginalPrice = latestData.OriginalPrice
	data.DiscountPerc = latestData.DiscountPerc
	data.StockLevel = latestData.StockLevel
	data.CustomerRating = latestData.CustomerRating
	data.ReviewCount = latestData.ReviewCount
	data.DeliveryDays = latestData.DeliveryDays

	return data, nil
}
