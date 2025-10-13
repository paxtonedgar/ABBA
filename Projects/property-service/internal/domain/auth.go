package domain

import (
	"context"
	"errors"
)

// =====================================================================================
// Authentication Context (Domain Layer)
// =====================================================================================

var (
	// ErrUnauthorized is returned when no authenticated user is found in context.
	ErrUnauthorized = errors.New("unauthorized: no authenticated user")
)

type ctxKey string

const authenticatedUserKey ctxKey = "authenticated_user"

// AuthenticatedUser represents a validated, authenticated user.
// This is a domain type used for passing authentication info through the service layer.
type AuthenticatedUser struct {
	ID    UserID
	Email string
	Role  string
	Name  string
}

// WithAuthenticatedUser attaches a validated user to the context.
func WithAuthenticatedUser(ctx context.Context, userID UserID, email string, role string, name string) context.Context {
	user := &AuthenticatedUser{
		ID:    userID,
		Email: email,
		Role:  role,
		Name:  name,
	}
	return context.WithValue(ctx, authenticatedUserKey, user)
}

// GetAuthenticatedUser extracts the authenticated user from context.
func GetAuthenticatedUser(ctx context.Context) (*AuthenticatedUser, error) {
	user, ok := ctx.Value(authenticatedUserKey).(*AuthenticatedUser)
	if !ok || user == nil {
		return nil, ErrUnauthorized
	}
	return user, nil
}
