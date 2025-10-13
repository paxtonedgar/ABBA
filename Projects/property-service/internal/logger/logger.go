package logger

import (
	"context"
	"log/slog"
	"os"
	"time"
)

var logger *slog.Logger

func init() {
	// Use JSON in production, text in development
	var handler slog.Handler
	if os.Getenv("ENV") == "production" {
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	} else {
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
	}
	logger = slog.New(handler)
}

// Info logs at info level
func Info(msg string, args ...any) {
	logger.Info(msg, args...)
}

// Error logs at error level
func Error(msg string, args ...any) {
	logger.Error(msg, args...)
}

// Debug logs at debug level
func Debug(msg string, args ...any) {
	logger.Debug(msg, args...)
}

// WithOperation returns a logger with operation context
func WithOperation(ctx context.Context, operation string) context.Context {
	return context.WithValue(ctx, "operation", operation)
}

// LogOperation logs the duration and result of an operation
func LogOperation(operation string, start time.Time, err error) {
	duration := time.Since(start).Milliseconds()
	if err != nil {
		Error("operation_failed",
			"operation", operation,
			"duration_ms", duration,
			"error", err.Error(),
		)
	} else {
		Info("operation_completed",
			"operation", operation,
			"duration_ms", duration,
		)
	}
}