package service_test

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"

	_ "github.com/mattn/go-sqlite3"
)

func TestAmenityService_SetAmenities(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)
	amenitySvc := service.NewAmenityService(client, authSvc)

	ownerUser := createTestUser(t, client, user.RoleOwner)
	agentUser := createTestUser(t, client, user.RoleAgent)
	otherOwner, _ := client.User.Create().
		SetEmail("other@example.com").
		SetRole(user.RoleOwner).
		SetName("Other Owner").
		Save(context.Background())

	// Create test amenities
	wifiAmenity, _ := client.Amenity.Create().
		SetCode("wifi").
		SetName("WiFi").
		SetCategory("internet").
		Save(context.Background())

	poolAmenity, _ := client.Amenity.Create().
		SetCode("pool").
		SetName("Swimming Pool").
		SetCategory("recreation").
		Save(context.Background())

	_, _ = client.Amenity.Create().
		SetCode("gym").
		SetName("Fitness Center").
		SetCategory("recreation").
		Save(context.Background())

	t.Run("owner can set amenities for their property", func(t *testing.T) {
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

		details := `{"speed":"100mbps"}`
		amenities := []service.AmenityInput{
			{
				Code:    "wifi",
				Details: &details,
			},
			{
				Code: "pool",
			},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify amenities were created
		propAmenities, _ := client.PropertyAmenity.Query().All(context.Background())
		assert.Len(t, propAmenities, 2)

		// Find wifi amenity and check details
		var wifiPA *struct{ Details map[string]interface{} }
		for _, pa := range propAmenities {
			if pa.AmenityID == wifiAmenity.ID {
				wifiPA = &struct{ Details map[string]interface{} }{pa.Details}
				break
			}
		}
		require.NotNil(t, wifiPA)
		assert.Equal(t, "100mbps", wifiPA.Details["speed"])
	})

	t.Run("owner cannot set amenities for other owner's property", func(t *testing.T) {
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

		amenities := []service.AmenityInput{
			{Code: "wifi"},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, result)
	})

	t.Run("agent can set amenities for any property", func(t *testing.T) {
		// Create property owned by owner
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Agent Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("789 Pine St").
			SetCity("Berkeley").
			SetRegion("CA").
			SetPostalCode("94702").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		amenities := []service.AmenityInput{
			{Code: "gym"},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities)

		require.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("skips unknown amenity codes", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Skip Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("321 Elm St").
			SetCity("San Jose").
			SetRegion("CA").
			SetPostalCode("95101").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		amenities := []service.AmenityInput{
			{Code: "wifi"},
			{Code: "unknown_amenity"}, // Should be skipped
			{Code: "pool"},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify only valid amenities were added (wifi and pool, not unknown)
		propAmenities, _ := client.PropertyAmenity.Query().All(context.Background())
		var count int
		for _, pa := range propAmenities {
			if pa.PropertyID == prop.ID {
				count++
			}
		}
		assert.Equal(t, 2, count, "Only 2 valid amenities should be added")
	})

	t.Run("skips invalid JSON details", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("JSON Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("555 JSON St").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94103").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		badJSON := `{invalid json`
		amenities := []service.AmenityInput{
			{Code: "pool"},
			{
				Code:    "wifi",
				Details: &badJSON, // Should be skipped due to invalid JSON
			},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify pool was added (wifi should be skipped due to bad JSON)
		propAmenities, _ := client.PropertyAmenity.Query().All(context.Background())
		var poolFound bool
		for _, pa := range propAmenities {
			if pa.PropertyID == prop.ID && pa.AmenityID == poolAmenity.ID {
				poolFound = true
			}
		}
		assert.True(t, poolFound, "Pool amenity should be added despite wifi having bad JSON")
	})

	t.Run("returns error for non-existent property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		amenities := []service.AmenityInput{
			{Code: "wifi"},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: uuid.New()}, amenities)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrNotFound)
		assert.Nil(t, result)
	})

	t.Run("upserts amenity details on conflict", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Upsert Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("777 Upsert Ave").
			SetCity("San Francisco").
			SetRegion("CA").
			SetPostalCode("94104").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		// First set wifi with initial details
		details1 := `{"speed":"50mbps"}`
		amenities1 := []service.AmenityInput{
			{
				Code:    "wifi",
				Details: &details1,
			},
		}

		_, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities1)
		require.NoError(t, err)

		// Update wifi with new details
		details2 := `{"speed":"1000mbps"}`
		amenities2 := []service.AmenityInput{
			{
				Code:    "wifi",
				Details: &details2,
			},
		}

		result, err := amenitySvc.SetAmenities(ctx, domain.PropertyID{UUID: prop.ID}, amenities2)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify details were updated
		propAmenities, _ := client.PropertyAmenity.Query().All(context.Background())
		var wifiPA *struct{ Details map[string]interface{} }
		for _, pa := range propAmenities {
			if pa.PropertyID == prop.ID && pa.AmenityID == wifiAmenity.ID {
				wifiPA = &struct{ Details map[string]interface{} }{pa.Details}
				break
			}
		}
		require.NotNil(t, wifiPA)
		assert.Equal(t, "1000mbps", wifiPA.Details["speed"])
	})
}
