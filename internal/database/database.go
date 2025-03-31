package database

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

// Forecast represents a forecast record in the database
type Forecast struct {
	ID            int64           `db:"id"`
	ProductID     string          `db:"product_id"`
	ForecastResult json.RawMessage `db:"forecast_result"`
	ForecastedAt  time.Time       `db:"forecasted_at"`
	CreatedAt     time.Time       `db:"created_at"`
}

// ForecastResult represents the forecast result JSON structure
type ForecastResult struct {
	ForecastedSales float64   `json:"forecasted_sales"`
	Date            time.Time `json:"date"`
}

// Database handles database operations
type Database struct {
	db *sqlx.DB
}

// New creates a new Database instance
func New(dsn string) (*Database, error) {
	db, err := sqlx.Connect("postgres", dsn)
	if err != nil {
		return nil, err
	}

	// Set connection pool settings
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(time.Minute * 5)

	return &Database{db: db}, nil
}

// SaveForecast saves a forecast to the database
func (d *Database) SaveForecast(ctx context.Context, productID string, forecastResult []byte) error {
	query := `
		INSERT INTO forecasts (product_id, forecast_result)
		VALUES ($1, $2)
		RETURNING id
	`

	var id int64
	err := d.db.QueryRowContext(ctx, query, productID, forecastResult).Scan(&id)
	if err != nil {
		return err
	}

	log.Printf("Saved forecast with ID: %d for product: %s", id, productID)
	return nil
}

// Close closes the database connection
func (d *Database) Close() error {
	return d.db.Close()
}
