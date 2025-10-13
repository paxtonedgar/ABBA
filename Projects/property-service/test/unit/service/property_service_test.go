package service_test

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"

	_ "github.com/mattn/go-sqlite3"
)

func TestPropertyService_CreateProperty(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)
	propertySvc := service.NewPropertyService(client, authSvc)

	ownerUser := createTestUser(t, client, user.RoleOwner)
	agentUser := createTestUser(t, client, user.RoleAgent)

	t.Run("owner can create property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		input := service.CreatePropertyInput{
			Title:        "Test Property",
			Type:         "house",
			AddressLine1: "123 Main St",
			City:         "San Francisco",
			Region:       "CA",
			PostalCode:   "94102",
			Country:      "US",
			PetsAllowed:  false,
			SmokingAllowed: false,
		}

		prop, err := propertySvc.CreateProperty(ctx, input)

		require.NoError(t, err)
		assert.Equal(t, "Test Property", prop.Title)
		assert.Equal(t, property.TypeHouse, prop.Type)
		assert.Equal(t, ownerUser.ID, prop.OwnerID)
		assert.Equal(t, property.StatusDraft, prop.Status)
	})

	t.Run("agent cannot create property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		input := service.CreatePropertyInput{
			Title:        "Test Property",
			Type:         "house",
			AddressLine1: "123 Main St",
			City:         "San Francisco",
			Region:       "CA",
			PostalCode:   "94102",
			Country:      "US",
		}

		prop, err := propertySvc.CreateProperty(ctx, input)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, prop)
	})

	t.Run("validates required fields", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		tests := []struct {
			name  string
			input service.CreatePropertyInput
		}{
			{
				name: "missing title",
				input: service.CreatePropertyInput{
					Type:         "house",
					AddressLine1: "123 Main St",
					City:         "San Francisco",
					Region:       "CA",
					PostalCode:   "94102",
					Country:      "US",
				},
			},
			{
				name: "missing address",
				input: service.CreatePropertyInput{
					Title:      "Test Property",
					Type:       "house",
					City:       "San Francisco",
					Region:     "CA",
					PostalCode: "94102",
					Country:    "US",
				},
			},
			{
				name: "missing city",
				input: service.CreatePropertyInput{
					Title:        "Test Property",
					Type:         "house",
					AddressLine1: "123 Main St",
					Region:       "CA",
					PostalCode:   "94102",
					Country:      "US",
				},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				prop, err := propertySvc.CreateProperty(ctx, tt.input)

				assert.Error(t, err)
				assert.ErrorIs(t, err, service.ErrInvalidInput)
				assert.Nil(t, prop)
			})
		}
	})

	t.Run("requires authentication", func(t *testing.T) {
		ctx := context.Background()

		input := service.CreatePropertyInput{
			Title:        "Test Property",
			Type:         "house",
			AddressLine1: "123 Main St",
			City:         "San Francisco",
			Region:       "CA",
			PostalCode:   "94102",
			Country:      "US",
		}

		prop, err := propertySvc.CreateProperty(ctx, input)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrUnauthorized)
		assert.Nil(t, prop)
	})
}

func TestPropertyService_SubmitProperty(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)
	propertySvc := service.NewPropertyService(client, authSvc)

	ownerUser := createTestUser(t, client, user.RoleOwner)
	otherOwner, _ := client.User.Create().
		SetEmail("other@example.com").
		SetRole(user.RoleOwner).
		SetName("Other Owner").
		Save(context.Background())

	t.Run("owner can submit their property", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("123 Main St").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94102").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		submitted, err := propertySvc.SubmitProperty(ctx, domain.PropertyID{UUID: prop.ID})

		require.NoError(t, err)
		assert.Equal(t, property.StatusSubmitted, submitted.Status)
		assert.NotNil(t, submitted.SubmittedAt)
	})

	t.Run("owner cannot submit other owner's property", func(t *testing.T) {
		// Create property owned by other owner
		prop, _ := client.Property.Create().
			SetOwnerID(otherOwner.ID).
			SetTitle("Other Property").
			SetType(property.TypeHouse).
			SetAddressLine1("456 Oak Ave").
			SetCity("Oakland").
			SetRegion("CA").
			SetPostalCode("94601").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		submitted, err := propertySvc.SubmitProperty(ctx, domain.PropertyID{UUID: prop.ID})

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, submitted)
	})

	t.Run("returns error for non-existent property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		submitted, err := propertySvc.SubmitProperty(ctx, domain.PropertyID{UUID: uuid.New()})

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrNotFound)
		assert.Nil(t, submitted)
	})
}

func TestPropertyService_ReviewProperty(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)
	propertySvc := service.NewPropertyService(client, authSvc)

	ownerUser := createTestUser(t, client, user.RoleOwner)

	t.Run("agent can approve property", func(t *testing.T) {
		// Create agent for this test
		agentUser := createTestUser(t, client, user.RoleAgent)

		// Create submitted property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("123 Main St").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94102").
			SetCountry("US").
			SetStatus(property.StatusSubmitted).
			SetSubmittedAt(time.Now()).
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		notes := "Looks good!"
		input := service.ReviewPropertyInput{
			PropertyID: domain.PropertyID{UUID: prop.ID},
			Decision:   "approve",
			Notes:      &notes,
		}

		reviewed, err := propertySvc.ReviewProperty(ctx, input)

		require.NoError(t, err)
		assert.Equal(t, property.StatusApproved, reviewed.Status)
		assert.Equal(t, agentUser.ID, reviewed.ReviewedBy)
		assert.NotNil(t, reviewed.ReviewedAt)
		assert.Equal(t, "Looks good!", reviewed.ReviewNotes)
	})

	t.Run("agent can reject property", func(t *testing.T) {
		// Create agent for this test
		agentUser := createTestUser(t, client, user.RoleAgent)

		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Bad Property").
			SetType(property.TypeHouse).
			SetAddressLine1("999 Bad St").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94102").
			SetCountry("US").
			SetStatus(property.StatusSubmitted).
			SetSubmittedAt(time.Now()).
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		notes := "Not acceptable"
		input := service.ReviewPropertyInput{
			PropertyID: domain.PropertyID{UUID: prop.ID},
			Decision:   "reject",
			Notes:      &notes,
		}

		reviewed, err := propertySvc.ReviewProperty(ctx, input)

		require.NoError(t, err)
		assert.Equal(t, property.StatusRejected, reviewed.Status)
	})

	t.Run("owner cannot review property", func(t *testing.T) {
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("123 Main St").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94102").
			SetCountry("US").
			SetStatus(property.StatusSubmitted).
			SetSubmittedAt(time.Now()).
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		input := service.ReviewPropertyInput{
			PropertyID: domain.PropertyID{UUID: prop.ID},
			Decision:   "approve",
		}

		reviewed, err := propertySvc.ReviewProperty(ctx, input)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, reviewed)
	})
}
