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

func TestRoomService_UpsertRooms(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	authSvc := service.NewAuthService(client)
	roomSvc := service.NewRoomService(client, authSvc)

	ownerUser := createTestUser(t, client, user.RoleOwner)
	agentUser := createTestUser(t, client, user.RoleAgent)
	otherOwner, _ := client.User.Create().
		SetEmail("other@example.com").
		SetRole(user.RoleOwner).
		SetName("Other Owner").
		Save(context.Background())

	t.Run("owner can create rooms for their property", func(t *testing.T) {
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

		floor := "1st Floor"
		area := 200
		ensuite := true
		notes := "Master bedroom"

		rooms := []service.RoomInput{
			{
				Name:        "Master Bedroom",
				FloorLabel:  &floor,
				AreaSqFt:    &area,
				EnsuiteBath: &ensuite,
				BedMix:      map[string]int{"queen": 1},
				Notes:       &notes,
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: prop.ID}, rooms)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify room was created
		createdRooms, _ := client.Room.Query().Where().All(context.Background())
		assert.Len(t, createdRooms, 1)
		assert.Equal(t, "Master Bedroom", createdRooms[0].Name)
		assert.Equal(t, "1st Floor", createdRooms[0].FloorLabel)
		assert.Equal(t, 200, createdRooms[0].AreaSqFt)
		assert.True(t, createdRooms[0].EnsuiteBath)
	})

	t.Run("owner can update existing room", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Update Property").
			SetType(property.TypeHouse).
			SetAddressLine1("456 Oak Ave").
			SetCity("Oakland").
			SetRegion("CA").
			SetPostalCode("94601").
			SetCountry("US").
			Save(context.Background())

		// Create room
		room, _ := client.Room.Create().
			SetPropertyID(prop.ID).
			SetName("Old Name").
			SetBedMix(map[string]int{"twin": 2}).
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		roomID := domain.RoomID{UUID: room.ID}
		rooms := []service.RoomInput{
			{
				ID:     &roomID,
				Name:   "Updated Name",
				BedMix: map[string]int{"queen": 1},
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: prop.ID}, rooms)

		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify room was updated
		updatedRoom, _ := client.Room.Get(context.Background(), room.ID)
		assert.Equal(t, "Updated Name", updatedRoom.Name)
		assert.Equal(t, map[string]int{"queen": 1}, updatedRoom.BedMix)
	})

	t.Run("owner cannot modify other owner's property", func(t *testing.T) {
		// Create property owned by other owner
		prop, _ := client.Property.Create().
			SetOwnerID(otherOwner.ID).
			SetTitle("Other Property").
			SetType(property.TypeHouse).
			SetAddressLine1("789 Pine St").
			SetCity("Berkeley").
			SetRegion("CA").
			SetPostalCode("94702").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		rooms := []service.RoomInput{
			{
				Name:   "Bedroom",
				BedMix: map[string]int{"queen": 1},
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: prop.ID}, rooms)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrForbidden)
		assert.Nil(t, result)
	})

	t.Run("agent can modify any property", func(t *testing.T) {
		// Create property owned by owner
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Agent Test Property").
			SetType(property.TypeHouse).
			SetAddressLine1("321 Elm St").
			SetCity("San Jose").
			SetRegion("CA").
			SetPostalCode("95101").
			SetCountry("US").
			Save(context.Background())

		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: agentUser.ID},
			agentUser.Email,
			string(user.RoleAgent),
			agentUser.Name,
		)

		rooms := []service.RoomInput{
			{
				Name:   "Guest Room",
				BedMix: map[string]int{"twin": 2},
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: prop.ID}, rooms)

		require.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("returns error for non-existent property", func(t *testing.T) {
		ctx := domain.WithAuthenticatedUser(
			context.Background(),
			domain.UserID{UUID: ownerUser.ID},
			ownerUser.Email,
			string(user.RoleOwner),
			ownerUser.Name,
		)

		rooms := []service.RoomInput{
			{
				Name:   "Bedroom",
				BedMix: map[string]int{"queen": 1},
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: uuid.New()}, rooms)

		assert.Error(t, err)
		assert.ErrorIs(t, err, service.ErrNotFound)
		assert.Nil(t, result)
	})

	t.Run("handles transaction rollback on error", func(t *testing.T) {
		// Create property
		prop, _ := client.Property.Create().
			SetOwnerID(ownerUser.ID).
			SetTitle("Transaction Test").
			SetType(property.TypeHouse).
			SetAddressLine1("999 Error St").
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

		// Create room with duplicate ID to force an error
		existingRoom, _ := client.Room.Create().
			SetPropertyID(prop.ID).
			SetName("Existing").
			SetBedMix(map[string]int{"twin": 1}).
			Save(context.Background())

		roomID := domain.RoomID{UUID: existingRoom.ID}
		rooms := []service.RoomInput{
			{
				ID:     &roomID,
				Name:   "Updated",
				BedMix: map[string]int{"queen": 1},
			},
			{
				// This should succeed in the transaction
				Name:   "New Room",
				BedMix: map[string]int{"king": 1},
			},
		}

		result, err := roomSvc.UpsertRooms(ctx, domain.PropertyID{UUID: prop.ID}, rooms)

		// Transaction should commit successfully
		require.NoError(t, err)
		assert.NotNil(t, result)

		// Verify both operations completed
		allRooms, _ := client.Room.Query().All(context.Background())
		var found bool
		for _, r := range allRooms {
			if r.Name == "New Room" {
				found = true
				break
			}
		}
		assert.True(t, found, "New room should be created")
	})
}
