package assembly

import (
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/graduate-work-mirea/data-processor-service/config"
	"github.com/graduate-work-mirea/data-processor-service/controller"
	"github.com/graduate-work-mirea/data-processor-service/repository"
	"github.com/graduate-work-mirea/data-processor-service/service"
	"go.uber.org/zap"
)

type ServiceLocator struct {
	Config               *config.Config
	Logger               *zap.SugaredLogger
	FileRepository       *repository.FileRepository
	PostgresRepository   *repository.PostgresRepository
	MLPredictionService  *service.MLPredictionService
	PredictionController *controller.PredictionAPIController
	HTTPServer           *http.Server
	Router               *gin.Engine
}

func NewServiceLocator(cfg *config.Config, logger *zap.SugaredLogger) (*ServiceLocator, error) {
	// Initialize repositories
	fileRepo := repository.NewFileRepository(cfg.ProcessedDataPath, cfg.ModelPath)

	// Initialize PostgreSQL repository
	postgresRepo, err := repository.NewPostgresRepository(cfg.GetPostgresConnectionString())
	if err != nil {
		logger.Errorw("Failed to initialize PostgreSQL repository", "error", err)
		return nil, err
	}

	// Initialize services
	mlService := service.NewMLPredictionService(fileRepo, postgresRepo, logger)

	// Initialize controllers
	predictionController := controller.NewPredictionAPIController(mlService, logger)

	// Initialize Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	// Configure CORS middleware
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowOrigins = []string{"http://localhost"}
	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	corsConfig.AllowHeaders = []string{"Origin", "Content-Type", "Authorization"}
	router.Use(cors.New(corsConfig))

	// Register routes
	predictionController.RegisterRoutes(router)

	// Create HTTP server
	httpServer := &http.Server{
		Addr:    ":" + cfg.ServerPort,
		Handler: router,
	}

	return &ServiceLocator{
		Config:               cfg,
		Logger:               logger,
		FileRepository:       fileRepo,
		PostgresRepository:   postgresRepo,
		MLPredictionService:  mlService,
		PredictionController: predictionController,
		HTTPServer:           httpServer,
		Router:               router,
	}, nil
}

// Close closes all resources
func (l *ServiceLocator) Close() {
	// Close PostgreSQL connection if it exists
	if l.PostgresRepository != nil {
		if err := l.PostgresRepository.Close(); err != nil {
			l.Logger.Errorw("Error closing PostgreSQL connection", "error", err)
		}
	}
}
