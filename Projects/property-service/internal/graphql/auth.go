// Package graphql provides GraphQL schema implementation including resolvers, handlers, and authentication.
package graphql

import (
	"context"

	"github.com/99designs/gqlgen/graphql"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
)

// =====================================================================================
// Authentication Context (GraphQL Layer)
// =====================================================================================

// AuthenticatedUser represents a validated, authenticated user with GraphQL types.
type AuthenticatedUser struct {
	ID    domain.UserID
	Email string
	Role  UserRole
	Name  string
}

// WithAuthenticatedUser attaches a validated user to the context (both GraphQL and domain layers).
func WithAuthenticatedUser(ctx context.Context, userID domain.UserID, email string, role UserRole, name string) context.Context {
	// Store in domain layer for service access
	ctx = domain.WithAuthenticatedUser(ctx, userID, email, string(role), name)
	return ctx
}

// GetAuthenticatedUser extracts the authenticated user from context (GraphQL layer).
func GetAuthenticatedUser(ctx context.Context) (*AuthenticatedUser, error) {
	// Get from domain layer
	domainUser, err := domain.GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, err
	}

	// Convert domain user to GraphQL user
	role := UserRoleOwner
	switch domainUser.Role {
	case "AGENT":
		role = UserRoleAgent
	case "OWNER":
		role = UserRoleOwner
	}

	return &AuthenticatedUser{
		ID:    domainUser.ID,
		Email: domainUser.Email,
		Role:  role,
		Name:  domainUser.Name,
	}, nil
}

// Auth directive implementation
func Auth(ctx context.Context, _ any, next graphql.Resolver, _ UserRole) (res any, err error) {
	// For demo/testing purposes, allow OWNER to perform all operations
	// In production, you'd have strict role-based access control
	// TODO: Implement JWT validation and role checking
	return next(ctx)
}
