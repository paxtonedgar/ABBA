package dataloader

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/room"
	"github.com/google/uuid"

	"github.com/graph-gophers/dataloader/v7"
)

// RoomsByPropertyLoader batches rooms grouped by property ID.
type RoomsByPropertyLoader = dataloader.Loader[string, []*ent.Room]

// NewRoomsByPropertyLoader creates a new dataloader for batching room queries by property ID.
func NewRoomsByPropertyLoader(client *ent.Client) *RoomsByPropertyLoader {
	return newBatchLoaderByFK(
		func(ctx context.Context, propertyIDs []uuid.UUID) ([]*ent.Room, error) {
			return client.Room.Query().Where(room.PropertyIDIn(propertyIDs...)).All(ctx)
		},
		func(r *ent.Room) uuid.UUID { return r.PropertyID },
	)
}
