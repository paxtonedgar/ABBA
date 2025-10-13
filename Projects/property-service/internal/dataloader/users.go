package dataloader

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/google/uuid"
	"github.com/graph-gophers/dataloader/v7"
)

// UserLoader batches User lookups by ID.
type UserLoader = dataloader.Loader[string, *ent.User]

// NewUserLoader creates a request-scoped dataloader for users.
func NewUserLoader(client *ent.Client) *UserLoader {
	return newBatchLoaderByID(
		func(ctx context.Context, ids []uuid.UUID) ([]*ent.User, error) {
			return client.User.Query().Where(user.IDIn(ids...)).All(ctx)
		},
		func(u *ent.User) uuid.UUID { return u.ID },
	)
}
