package dataloader

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/google/uuid"

	"github.com/graph-gophers/dataloader/v7"
)

// PropertyLoader batches property lookups by ID.
type PropertyLoader = dataloader.Loader[string, *ent.Property]

// PropertiesByOwnerLoader batches properties grouped by owner ID.
type PropertiesByOwnerLoader = dataloader.Loader[string, []*ent.Property]

// NewPropertyLoader creates a new dataloader for batching property lookups by ID.
func NewPropertyLoader(client *ent.Client) *PropertyLoader {
	return newBatchLoaderByID(
		func(ctx context.Context, ids []uuid.UUID) ([]*ent.Property, error) {
			return client.Property.Query().Where(property.IDIn(ids...)).All(ctx)
		},
		func(p *ent.Property) uuid.UUID { return p.ID },
	)
}

// NewPropertiesByOwnerLoader creates a new dataloader for batching property queries by owner ID.
func NewPropertiesByOwnerLoader(client *ent.Client) *PropertiesByOwnerLoader {
	return newBatchLoaderByFK(
		func(ctx context.Context, ownerIDs []uuid.UUID) ([]*ent.Property, error) {
			return client.Property.Query().Where(property.OwnerIDIn(ownerIDs...)).All(ctx)
		},
		func(p *ent.Property) uuid.UUID { return p.OwnerID },
	)
}
