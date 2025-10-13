package filters_test

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/enttest"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service/filters"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *ent.Client {
	client := enttest.Open(t, "sqlite3", "file:ent?mode=memory&cache=shared&_fk=1")
	return client
}

func createTestOwner(t *testing.T, client *ent.Client) uuid.UUID {
	u, err := client.User.Create().
		SetEmail("test-" + uuid.New().String()[:8] + "@example.com").
		SetRole("owner").
		SetName("Test Owner").
		Save(context.Background())
	require.NoError(t, err)
	return u.ID
}

func TestIDFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.IDFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	// Create test properties
	prop1, err := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Property 1").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetPetsAllowed(false).
		SetSmokingAllowed(false).
		Save(context.Background())
	require.NoError(t, err)

	prop2, err := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Property 2").
		SetType(property.TypeApartment).
		SetAddressLine1("456 Oak Ave").
		SetCity("Oakland").
		SetRegion("CA").
		SetPostalCode("94601").
		SetCountry("US").
		SetPetsAllowed(false).
		SetSmokingAllowed(false).
		Save(context.Background())
	require.NoError(t, err)

	t.Run("filters by single ID", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			Ids: []string{prop1.ID.String()},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, prop1.ID, results[0].ID)
	})

	t.Run("filters by multiple IDs", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			Ids: []string{prop1.ID.String(), prop2.ID.String()},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		assert.Len(t, results, 2)
	})

	t.Run("does not apply when filter is empty", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			Ids: []string{},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})

	t.Run("skips invalid IDs", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			Ids: []string{"invalid-id", prop1.ID.String()},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, prop1.ID, results[0].ID)
	})
}

func TestStatusFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.StatusFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	// Create test properties with different statuses
	draft, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Draft Property").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetStatus(property.StatusDraft).
		Save(context.Background())

	_, _ = client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Submitted Property").
		SetType(property.TypeHouse).
		SetAddressLine1("456 Oak Ave").
		SetCity("Oakland").
		SetRegion("CA").
		SetPostalCode("94601").
		SetCountry("US").
		SetStatus(property.StatusSubmitted).
		Save(context.Background())

	t.Run("filters by status", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			StatusIn: []graphql.PropertyStatus{graphql.PropertyStatusDraft},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, draft.ID, results[0].ID)
	})

	t.Run("filters by multiple statuses", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			StatusIn: []graphql.PropertyStatus{
				graphql.PropertyStatusDraft,
				graphql.PropertyStatusSubmitted,
			},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		assert.Len(t, results, 2)
	})

	t.Run("does not apply when empty", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{
			StatusIn: []graphql.PropertyStatus{},
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})
}

func TestLocationFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.LocationFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	// Create test properties in different locations
	sfProp, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("SF Property").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetPetsAllowed(false).
		SetSmokingAllowed(false).
		Save(context.Background())

	_, _ = client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Oakland Property").
		SetType(property.TypeHouse).
		SetAddressLine1("456 Oak Ave").
		SetCity("Oakland").
		SetRegion("CA").
		SetPostalCode("94601").
		SetCountry("US").
		SetPetsAllowed(false).
		SetSmokingAllowed(false).
		Save(context.Background())

	t.Run("filters by city", func(t *testing.T) {
		q := client.Property.Query()
		city := "San Francisco"
		gqlFilter := &graphql.PropertyFilter{
			CityEq: &city,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, sfProp.ID, results[0].ID)
	})

	t.Run("filters by region", func(t *testing.T) {
		q := client.Property.Query()
		region := "CA"
		gqlFilter := &graphql.PropertyFilter{
			RegionEq: &region,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		assert.Len(t, results, 2) // Both in CA
	})

	t.Run("does not apply when no location filters", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})
}

func TestFullTextFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.FullTextFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	// Create test properties
	beach, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Beach House").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetDescriptionShort("Beautiful beachfront property").
		Save(context.Background())

	mountain, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Mountain Cabin").
		SetType(property.TypeHouse).
		SetAddressLine1("456 Oak Ave").
		SetCity("Tahoe").
		SetRegion("CA").
		SetPostalCode("96150").
		SetCountry("US").
		SetDescriptionShort("Mountain retreat with lake view").
		Save(context.Background())

	t.Run("searches in title", func(t *testing.T) {
		q := client.Property.Query()
		text := "beach"
		gqlFilter := &graphql.PropertyFilter{
			FullText: &text,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, beach.ID, results[0].ID)
	})

	t.Run("searches in description", func(t *testing.T) {
		q := client.Property.Query()
		text := "beachfront"
		gqlFilter := &graphql.PropertyFilter{
			FullText: &text,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, beach.ID, results[0].ID)
	})

	t.Run("searches in city", func(t *testing.T) {
		q := client.Property.Query()
		text := "tahoe"
		gqlFilter := &graphql.PropertyFilter{
			FullText: &text,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, mountain.ID, results[0].ID)
	})

	t.Run("does not apply when empty", func(t *testing.T) {
		q := client.Property.Query()
		emptyText := ""
		gqlFilter := &graphql.PropertyFilter{
			FullText: &emptyText,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})

	t.Run("case insensitive search", func(t *testing.T) {
		q := client.Property.Query()
		text := "BEACH"
		gqlFilter := &graphql.PropertyFilter{
			FullText: &text,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, beach.ID, results[0].ID)
	})
}

func TestPolicyFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.PolicyFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	// Create test properties with different policies
	petFriendly, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Pet Friendly House").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetPetsAllowed(true).
		SetSmokingAllowed(false).
		Save(context.Background())

	noPets, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("No Pets House").
		SetType(property.TypeHouse).
		SetAddressLine1("456 Oak Ave").
		SetCity("Oakland").
		SetRegion("CA").
		SetPostalCode("94601").
		SetCountry("US").
		SetPetsAllowed(false).
		SetSmokingAllowed(true).
		Save(context.Background())

	t.Run("filters by pets allowed", func(t *testing.T) {
		q := client.Property.Query()
		petsAllowed := true
		gqlFilter := &graphql.PropertyFilter{
			PetsAllowed: &petsAllowed,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, petFriendly.ID, results[0].ID)
	})

	t.Run("filters by smoking allowed", func(t *testing.T) {
		q := client.Property.Query()
		smokingAllowed := true
		gqlFilter := &graphql.PropertyFilter{
			SmokingAllowed: &smokingAllowed,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, noPets.ID, results[0].ID)
	})

	t.Run("does not apply when no policy filters", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})
}

func TestCapacityFilter(t *testing.T) {
	client := setupTestDB(t)
	defer client.Close()

	filter := &filters.CapacityFilter{}

	// Create test owner
	ownerID := createTestOwner(t, client)

	guests4 := 4
	guests8 := 8
	baths2 := 2.0
	baths3 := 3.5

	// Create test properties with different capacities
	_, _ = client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Small House").
		SetType(property.TypeHouse).
		SetAddressLine1("123 Main St").
		SetCity("San Francisco").
		SetRegion("CA").
		SetPostalCode("94102").
		SetCountry("US").
		SetMaxGuests(guests4).
		SetBathroomsTotal(baths2).
		Save(context.Background())

	large, _ := client.Property.Create().
		SetOwnerID(ownerID).
		SetTitle("Large House").
		SetType(property.TypeHouse).
		SetAddressLine1("456 Oak Ave").
		SetCity("Oakland").
		SetRegion("CA").
		SetPostalCode("94601").
		SetCountry("US").
		SetMaxGuests(guests8).
		SetBathroomsTotal(baths3).
		Save(context.Background())

	t.Run("filters by min guests", func(t *testing.T) {
		q := client.Property.Query()
		minGuests := 6
		gqlFilter := &graphql.PropertyFilter{
			MinGuestsGte: &minGuests,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, large.ID, results[0].ID)
	})

	t.Run("filters by min bathrooms", func(t *testing.T) {
		q := client.Property.Query()
		minBaths := 3.0
		gqlFilter := &graphql.PropertyFilter{
			MinBathroomsGte: &minBaths,
		}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.True(t, applied)
		results, _ := q.All(context.Background())
		require.Len(t, results, 1)
		assert.Equal(t, large.ID, results[0].ID)
	})

	t.Run("does not apply when no capacity filters", func(t *testing.T) {
		q := client.Property.Query()
		gqlFilter := &graphql.PropertyFilter{}

		applied := filter.Apply(context.Background(), q, gqlFilter)

		assert.False(t, applied)
	})
}
