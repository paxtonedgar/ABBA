package service_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/enttest"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"

	_ "github.com/mattn/go-sqlite3"
)

// setupTestDB creates an in-memory SQLite database for testing.
func setupTestDB(t *testing.T) *ent.Client {
	client := enttest.Open(t, "sqlite3", "file:ent?mode=memory&cache=shared&_fk=1")
	return client
}

// createTestUser creates a test user in the database with a unique email.
func createTestUser(t *testing.T, client *ent.Client, role user.Role) *ent.User {
	email := fmt.Sprintf("test-%s@example.com", uuid.New().String()[:8])
	u, err := client.User.Create().
		SetEmail(email).
		SetRole(role).
		SetName("Test User").
		Save(context.Background())
	require.NoError(t, err)
	return u
}

func TestAuthService_GetAuthenticatedUser(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)

	t.Run("returns user when authenticated", func(t *testing.T) {
		// Create test user
		testUser := createTestUser(t, client, user.RoleOwner)

		// Add user to context
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: testUser.ID},
			testUser.Email,
			string(user.RoleOwner),
			testUser.Name,
		)

		// Get authenticated user
		u, err := authSvc.GetAuthenticatedUser(ctx)

		// Assertions
		require.NoError(t, err)
		assert.Equal(t, testUser.ID, u.ID)
		assert.Equal(t, testUser.Email, u.Email)
		assert.Equal(t, testUser.Role, u.Role)
	})

	t.Run("returns error when no user in context", func(t *testing.T) {
		ctx := context.Background()

		u, err := authSvc.GetAuthenticatedUser(ctx)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrUnauthorized)
		assert.Nil(t, u)
	})

	t.Run("returns error when user not found in database", func(t *testing.T) {
		// Add non-existent user to context
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: uuid.New()},
			"nonexistent@example.com",
			string(user.RoleOwner),
			"Nonexistent User",
		)

		u, err := authSvc.GetAuthenticatedUser(ctx)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrUnauthorized)
		assert.Nil(t, u)
	})
}

func TestAuthService_RequireOwnerRole(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)

	t.Run("allows owner role", func(t *testing.T) {
		testUser := createTestUser(t, client, user.RoleOwner)

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: testUser.ID},
			testUser.Email,
			string(user.RoleOwner),
			testUser.Name,
		)

		u, err := authSvc.RequireOwnerRole(ctx)

		require.NoError(t, err)
		assert.Equal(t, user.RoleOwner, u.Role)
	})

	t.Run("rejects agent role", func(t *testing.T) {
		testUser := createTestUser(t, client, user.RoleAgent)

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: testUser.ID},
			testUser.Email,
			string(user.RoleAgent),
			testUser.Name,
		)

		u, err := authSvc.RequireOwnerRole(ctx)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, u)
	})

	t.Run("returns error when not authenticated", func(t *testing.T) {
		ctx := context.Background()

		u, err := authSvc.RequireOwnerRole(ctx)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrUnauthorized)
		assert.Nil(t, u)
	})
}

func TestAuthService_RequireAgentRole(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)

	t.Run("allows agent role", func(t *testing.T) {
		testUser := createTestUser(t, client, user.RoleAgent)

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: testUser.ID},
			testUser.Email,
			string(user.RoleAgent),
			testUser.Name,
		)

		u, err := authSvc.RequireAgentRole(ctx)

		require.NoError(t, err)
		assert.Equal(t, user.RoleAgent, u.Role)
	})

	t.Run("rejects owner role", func(t *testing.T) {
		testUser := createTestUser(t, client, user.RoleOwner)

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: testUser.ID},
			testUser.Email,
			string(user.RoleOwner),
			testUser.Name,
		)

		u, err := authSvc.RequireAgentRole(ctx)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, u)
	})
}

func TestAuthService_CanModifyProperty(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)

	ownerUser := createTestUser(t, client, user.RoleOwner)
	agentUser := createTestUser(t, client, user.RoleAgent)
	otherOwner, err := client.User.Create().
		SetEmail("other@example.com").
		SetRole(user.RoleOwner).
		SetName("Other Owner").
		Save(context.Background())
	require.NoError(t, err)

	t.Run("owner can modify their own property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		err := authSvc.CanModifyProperty(ctx, domain.UserID{UUID: ownerUser.ID})

		assert.NoError(t, err)
	})

	t.Run("owner cannot modify other owner's property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		err := authSvc.CanModifyProperty(ctx, domain.UserID{UUID: otherOwner.ID})

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
	})

	t.Run("agent can modify any property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		// Agent can modify owner's property
		err := authSvc.CanModifyProperty(ctx, domain.UserID{UUID: ownerUser.ID})
		assert.NoError(t, err)

		// Agent can modify other owner's property
		err = authSvc.CanModifyProperty(ctx, domain.UserID{UUID: otherOwner.ID})
		assert.NoError(t, err)
	})

	t.Run("returns error when not authenticated", func(t *testing.T) {
		ctx := context.Background()

		err := authSvc.CanModifyProperty(ctx, domain.UserID{UUID: ownerUser.ID})

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrUnauthorized)
	})
}
