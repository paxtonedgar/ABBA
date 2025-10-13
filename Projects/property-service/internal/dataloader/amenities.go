// Package dataloader provides request-scoped dataloaders for batching database queries.
package dataloader

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/propertyamenity"
	"github.com/google/uuid"

	"github.com/graph-gophers/dataloader/v7"
)

// AmenitiesByPropertyLoader batches amenities grouped by property ID.
type AmenitiesByPropertyLoader = dataloader.Loader[string, []*ent.PropertyAmenity]

// NewAmenitiesByPropertyLoader creates a new dataloader for batching amenity queries by property ID.
func NewAmenitiesByPropertyLoader(client *ent.Client) *AmenitiesByPropertyLoader {
	return newBatchLoaderByFK(
		func(ctx context.Context, propertyIDs []uuid.UUID) ([]*ent.PropertyAmenity, error) {
			return client.PropertyAmenity.Query().Where(propertyamenity.PropertyIDIn(propertyIDs...)).All(ctx)
		},
		func(pa *ent.PropertyAmenity) uuid.UUID { return pa.PropertyID },
	)
}
