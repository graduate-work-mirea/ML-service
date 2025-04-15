package rabbitmq

import (
	"context"
	"fmt"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

type Client struct {
	conn    *amqp.Connection
	channel *amqp.Channel
	logger  *zap.SugaredLogger
}

func NewClient(rabbitMQURL string, logger *zap.SugaredLogger) (*Client, error) {
	conn, err := amqp.Dial(rabbitMQURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RabbitMQ: %w", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to open a channel: %w", err)
	}

	return &Client{
		conn:    conn,
		channel: ch,
		logger:  logger,
	}, nil
}

func (c *Client) DeclareQueue(queueName string) (amqp.Queue, error) {
	return c.channel.QueueDeclare(
		queueName, // name
		true,      // durable
		false,     // delete when unused
		false,     // exclusive
		false,     // no-wait
		nil,       // arguments
	)
}

func (c *Client) PublishMessage(ctx context.Context, queueName string, body []byte) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	err := c.channel.PublishWithContext(
		ctx,
		"",        // exchange
		queueName, // routing key
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        body,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to publish message: %w", err)
	}

	c.logger.Infof("Published message to queue: %s", queueName)
	return nil
}

func (c *Client) Close() {
	if c.channel != nil {
		c.channel.Close()
	}
	if c.conn != nil {
		c.conn.Close()
	}
}

// Channel returns the underlying AMQP channel
func (c *Client) Channel() *amqp.Channel {
	return c.channel
}
