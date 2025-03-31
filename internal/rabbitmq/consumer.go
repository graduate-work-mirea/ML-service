package rabbitmq

import (
	"context"
	"encoding/json"
	"log"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"ml-service/internal/ml"
)

const (
	queueName = "processed_data_queue"
)

// Consumer handles RabbitMQ message consumption
type Consumer struct {
	conn    *amqp.Connection
	channel *amqp.Channel
	mlSvc   *ml.Service
}

// NewConsumer creates a new RabbitMQ consumer
func NewConsumer(url string, mlSvc *ml.Service) (*Consumer, error) {
	// Connect to RabbitMQ
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, err
	}

	// Create a channel
	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, err
	}

	// Declare the queue
	_, err = channel.QueueDeclare(
		queueName, // name
		true,      // durable
		false,     // delete when unused
		false,     // exclusive
		false,     // no-wait
		nil,       // arguments
	)
	if err != nil {
		channel.Close()
		conn.Close()
		return nil, err
	}

	return &Consumer{
		conn:    conn,
		channel: channel,
		mlSvc:   mlSvc,
	}, nil
}

// Start begins consuming messages from the queue
func (c *Consumer) Start(ctx context.Context) error {
	// Set up quality of service
	err := c.channel.Qos(
		1,     // prefetch count
		0,     // prefetch size
		false, // global
	)
	if err != nil {
		return err
	}

	// Start consuming
	msgs, err := c.channel.Consume(
		queueName, // queue
		"",        // consumer
		false,     // auto-ack
		false,     // exclusive
		false,     // no-local
		false,     // no-wait
		nil,       // args
	)
	if err != nil {
		return err
	}

	// Process messages
	go func() {
		for {
			select {
			case <-ctx.Done():
				log.Println("Stopping RabbitMQ consumer")
				return
			case msg, ok := <-msgs:
				if !ok {
					log.Println("RabbitMQ channel closed")
					return
				}
				c.processMessage(msg)
			}
		}
	}()

	log.Println("RabbitMQ consumer started")
	return nil
}

// processMessage processes a message from the queue
func (c *Consumer) processMessage(msg amqp.Delivery) {
	defer func() {
		if err := msg.Ack(false); err != nil {
			log.Printf("Failed to acknowledge message: %v", err)
		}
	}()

	// Parse message
	var data ml.SalesData
	if err := json.Unmarshal(msg.Body, &data); err != nil {
		log.Printf("Failed to parse message: %v", err)
		return
	}

	log.Printf("Received data for product %s: sales=%f, date=%s", data.ProductID, data.Sales, data.Date)

	// Process data
	modelTrained, err := c.mlSvc.ProcessData(data)
	if err != nil {
		log.Printf("Failed to process data: %v", err)
		return
	}

	if modelTrained {
		log.Printf("Model trained for product %s", data.ProductID)
	} else {
		log.Printf("Data stored for product %s, waiting for more data points", data.ProductID)
	}
}

// Close closes the RabbitMQ connection
func (c *Consumer) Close() error {
	if err := c.channel.Close(); err != nil {
		return err
	}
	return c.conn.Close()
}
