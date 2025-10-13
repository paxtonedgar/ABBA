package service

import (
	"context"
	"fmt"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
)

// =====================================================================================
// Shared Service Layer Helpers
// =====================================================================================
// These helpers eliminate duplicate boilerplate across service methods.

// getPropertyWithAuthCheck fetches a property and verifies the caller has permission to modify it.
// This pattern was duplicated in 3 service files.
func getPropertyWithAuthCheck(
	ctx context.Context,
	client *ent.Client,
	authService *AuthService,
	propertyID domain.PropertyID,
) (*ent.Property, error) {
	// Get property
	prop, err := client.Property.Get(ctx, propertyID.UUID)
	if err != nil {
		if ent.IsNotFound(err) {
			return nil, fmt.Errorf("%w: property not found", ErrNotFound)
		}
		return nil, fmt.Errorf("get property: %w", err)
	}

	// Verify permission to modify
	ownerID := domain.UserID{UUID: prop.OwnerID}
	if err := authService.CanModifyProperty(ctx, ownerID); err != nil {
		return nil, err
	}

	return prop, nil
}

// withTransaction executes a function within a transaction.
// Automatically handles commit/rollback based on returned error.
func withTransaction[T any](
	ctx context.Context,
	client *ent.Client,
	fn func(tx *ent.Tx) (T, error),
) (T, error) {
	var zero T

	tx, err := client.Tx(ctx)
	if err != nil {
		return zero, fmt.Errorf("start transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	result, err := fn(tx)
	if err != nil {
		return zero, err
	}

	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit transaction: %w", err)
	}

	return result, nil
}
