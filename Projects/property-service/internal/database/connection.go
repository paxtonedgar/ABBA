// Package database provides database connection management and configuration for the ent client.
package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"entgo.io/ent/dialect"
	entsql "entgo.io/ent/dialect/sql"
	_ "github.com/lib/pq"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
)

// NewClient creates a new ent client with database connection pool and tests connectivity.
func NewClient(dsn string) (*ent.Client, error) {
	if dsn == "" {
		return nil, fmt.Errorf("database DSN cannot be empty")
	}

	// Create underlying database connection
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(20)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(55 * time.Minute)
	db.SetConnMaxIdleTime(10 * time.Minute)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		_ = db.Close() // Ignore close error since we're already returning ping error
		return nil, fmt.Errorf("ping database: %w", err)
	}

	// Create ent client using the database connection
	drv := entsql.OpenDB(dialect.Postgres, db)
	client := ent.NewClient(ent.Driver(drv))

	return client, nil
}
