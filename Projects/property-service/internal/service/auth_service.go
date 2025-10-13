package service

import (
	"context"
	"fmt"
	"time"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/logger"
)

// AuthService handles authentication and authorization logic.
type AuthService struct {
	client *ent.Client
}

// NewAuthService creates a new authentication service.
func NewAuthService(client *ent.Client) *AuthService {
	return &AuthService{client: client}
}

// GetAuthenticatedUser retrieves the authenticated user from context and database.
func (service *AuthService) GetAuthenticatedUser(ctx context.Context) (*ent.User, error) {
	authUser, err := domain.GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrUnauthorized, err)
	}

	// Fetch full user from database
	entUser, err := service.client.User.Get(ctx, authUser.ID.UUID)
	if err != nil {
		if ent.IsNotFound(err) {
			return nil, fmt.Errorf("%w: user not found", ErrUnauthorized)
		}
		return nil, fmt.Errorf("fetch user: %w", err)
	}

	return entUser, nil
}

// RequireOwnerRole ensures the authenticated user has OWNER role.
func (service *AuthService) RequireOwnerRole(ctx context.Context) (*ent.User, error) {
	start := time.Now()
	entUser, err := service.GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, err
	}

	if entUser.Role != user.RoleOwner {
		logger.Error("auth_owner_required_failed",
			"user_id", entUser.ID,
			"actual_role", entUser.Role,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		return nil, fmt.Errorf("%w: owner role required", ErrForbidden)
	}

	return entUser, nil
}

// RequireAgentRole ensures the authenticated user has AGENT role.
func (service *AuthService) RequireAgentRole(ctx context.Context) (*ent.User, error) {
	start := time.Now()
	entUser, err := service.GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, err
	}

	if entUser.Role != user.RoleAgent {
		logger.Error("auth_agent_required_failed",
			"user_id", entUser.ID,
			"actual_role", entUser.Role,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		return nil, fmt.Errorf("%w: agent role required", ErrForbidden)
	}

	return entUser, nil
}

// CanModifyProperty checks if the user can modify the given property.
// Owners can modify their own properties, agents can modify any property.
func (service *AuthService) CanModifyProperty(ctx context.Context, propertyOwnerID domain.UserID) error {
	entUser, err := service.GetAuthenticatedUser(ctx)
	if err != nil {
		return err
	}

	// Agents can modify any property
	if entUser.Role == user.RoleAgent {
		return nil
	}

	// Owners can only modify their own properties
	if entUser.ID.String() != propertyOwnerID.String() {
		return fmt.Errorf("%w: cannot modify other owner's property", ErrForbidden)
	}

	return nil
}
