package unit

import (
	"testing"

	"github.com/evolve-interviews/paxtonedgar-case-study/internal/database"
)

// TestNewClientInvalidDSN tests error handling for invalid DSN
func TestNewClientInvalidDSN(t *testing.T) {
	invalidDSN := "invalid://dsn"

	_, err := database.NewClient(invalidDSN)
	if err == nil {
		t.Error("Expected error for invalid DSN, got nil")
	}

	t.Logf("Got expected error for invalid DSN: %v", err)
}

// TestNewClientEmptyDSN tests error handling for empty DSN
func TestNewClientEmptyDSN(t *testing.T) {
	_, err := database.NewClient("")
	if err == nil {
		t.Error("Expected error for empty DSN, got nil")
	}

	t.Logf("Got expected error for empty DSN: %v", err)
}
