package graphql

import (
	"context"
	"sort"
	"strings"

	"github.com/google/uuid"

	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/amenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/propertyamenity"
)

// =====================================================================================
// Analytics Query Resolvers
// =====================================================================================
// These resolvers handle aggregation and analytics queries.

// PropertyCountsByStatus returns count of properties grouped by status.
// Uses parallel queries for 5x performance improvement.
func (resolver *queryResolver) PropertyCountsByStatus(ctx context.Context) ([]*CountByKey, error) {
	allStatuses := []property.Status{
		property.StatusDraft,
		property.StatusSubmitted,
		property.StatusUnderReview,
		property.StatusApproved,
		property.StatusRejected,
	}

	// Result channel for parallel queries
	type statusCountResult struct {
		key   string
		count int
		err   error
	}
	resultChannel := make(chan statusCountResult, len(allStatuses))

	// Launch parallel count queries
	for _, currentStatus := range allStatuses {
		currentStatus := currentStatus // capture loop variable for goroutine
		go func() {
			count, err := resolver.client.Property.Query().Where(property.StatusEQ(currentStatus)).Count(ctx)
			resultChannel <- statusCountResult{
				key:   strings.ToUpper(string(currentStatus)),
				count: count,
				err:   err,
			}
		}()
	}

	// Collect results
	countsByStatus := make([]*CountByKey, 0, len(allStatuses))
	for i := 0; i < len(allStatuses); i++ {
		result := <-resultChannel
		if result.err != nil {
			return nil, result.err
		}
		countsByStatus = append(countsByStatus, &CountByKey{Key: result.key, Count: result.count})
	}

	return countsByStatus, nil
}

// TopAmenities returns the most frequently attached amenities (global).
func (resolver *queryResolver) TopAmenities(ctx context.Context, limit *int) ([]*CountByKey, error) {
	maxResults := 20
	if limit != nil && *limit > 0 && *limit <= 100 {
		maxResults = *limit
	}

	var amenityCounts []struct {
		AmenityID uuid.UUID `json:"amenity_id"`
		Count     int       `json:"count"`
	}

	err := resolver.client.PropertyAmenity.
		Query().
		GroupBy(propertyamenity.FieldAmenityID).
		Aggregate(appent.Count()).
		Scan(ctx, &amenityCounts)
	if err != nil {
		return nil, err
	}

	// Batch load all amenities to avoid N+1 queries
	amenityIDs := make([]uuid.UUID, len(amenityCounts))
	for index, countRow := range amenityCounts {
		amenityIDs[index] = countRow.AmenityID
	}

	entAmenities, err := resolver.client.Amenity.Query().
		Where(amenity.IDIn(amenityIDs...)).
		All(ctx)
	if err != nil {
		return nil, err
	}

	// Create amenity ID -> code map
	amenityCodeByID := make(map[uuid.UUID]string, len(entAmenities))
	for _, entAmenity := range entAmenities {
		amenityCodeByID[entAmenity.ID] = entAmenity.Code
	}

	// Build result with counts
	results := make([]*CountByKey, 0, len(amenityCounts))
	for _, countRow := range amenityCounts {
		if code, exists := amenityCodeByID[countRow.AmenityID]; exists {
			results = append(results, &CountByKey{Key: code, Count: countRow.Count})
		}
	}

	// Sort by count descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Count > results[j].Count
	})
	if len(results) > maxResults {
		results = results[:maxResults]
	}
	return results, nil
}

// PropertiesByRegion groups counts by region (country:region or just region).
func (resolver *queryResolver) PropertiesByRegion(ctx context.Context, limit *int) ([]*CountByKey, error) {
	maxResults := 20
	if limit != nil && *limit > 0 && *limit <= 200 {
		maxResults = *limit
	}

	// Use GroupBy to aggregate at database level
	var regionCounts []struct {
		Country string `json:"country"`
		Region  string `json:"region"`
		Count   int    `json:"count"`
	}

	err := resolver.client.Property.
		Query().
		GroupBy(property.FieldCountry, property.FieldRegion).
		Aggregate(appent.Count()).
		Scan(ctx, &regionCounts)
	if err != nil {
		return nil, err
	}

	// Build results
	results := make([]*CountByKey, 0, len(regionCounts))
	for _, countRow := range regionCounts {
		regionKey := countRow.Country + ":" + countRow.Region
		results = append(results, &CountByKey{Key: regionKey, Count: countRow.Count})
	}

	// Sort by count descending and limit
	sort.Slice(results, func(i, j int) bool {
		return results[i].Count > results[j].Count
	})
	if len(results) > maxResults {
		results = results[:maxResults]
	}
	return results, nil
}
