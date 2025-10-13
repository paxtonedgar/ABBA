package integration

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/evolve-interviews/paxtonedgar-case-study/internal/database"
)

// TestDatabaseConnection tests basic database connectivity
func TestDatabaseConnection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test")
	}

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "postgres://evolve_user:evolve123@localhost:5432/evolve_db?sslmode=disable"
	}

	client, err := database.NewClient(dsn)
	if err != nil {
		t.Skipf("Database not available: %v", err)
	}
	defer func() {
		if err := client.Close(); err != nil {
			t.Logf("error closing database connection: %v", err)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Test basic ent client functionality
	// This will test both connection and ent client setup
	_, err = client.User.Query().Count(ctx)
	if err != nil {
		t.Fatalf("Ent client query failed: %v", err)
	}

	t.Logf("Ent client connection and query executed successfully")
}
