package dataloader

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
)

type ctxKey string

const key ctxKey = "dataloaders"

// Loaders holds all dataloaders for batching database queries in a request.
type Loaders struct {
	UserByID          *UserLoader
	PropertyByID      *PropertyLoader
	PropertiesByOwner *PropertiesByOwnerLoader
	RoomsByProperty   *RoomsByPropertyLoader
	AmenitiesByProp   *AmenitiesByPropertyLoader
}

// NewLoaders creates a new set of dataloaders for a request
func NewLoaders(client *ent.Client) *Loaders {
	return &Loaders{
		UserByID:          NewUserLoader(client),
		PropertyByID:      NewPropertyLoader(client),
		PropertiesByOwner: NewPropertiesByOwnerLoader(client),
		RoomsByProperty:   NewRoomsByPropertyLoader(client),
		AmenitiesByProp:   NewAmenitiesByPropertyLoader(client),
	}
}

// With attaches the dataloaders to the context.
func With(ctx context.Context, l *Loaders) context.Context {
	return context.WithValue(ctx, key, l)
}

// From extracts the dataloaders from the context.
func From(ctx context.Context) *Loaders {
	if v, ok := ctx.Value(key).(*Loaders); ok {
		return v
	}
	return nil
}
