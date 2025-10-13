package dataloader

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/graph-gophers/dataloader/v7"
)

// =====================================================================================
// Generic Dataloader Helpers
// =====================================================================================
// These helpers eliminate boilerplate for common dataloader patterns.

// newBatchLoaderByID creates a dataloader for fetching entities by ID.
// This eliminates ~40 lines of duplicate code per loader.
func newBatchLoaderByID[T any](
	fetchFunc func(ctx context.Context, ids []uuid.UUID) ([]T, error),
	getID func(T) uuid.UUID,
) *dataloader.Loader[string, T] {
	batchLoadFunc := func(ctx context.Context, keys []string) []*dataloader.Result[T] {
		results := make([]*dataloader.Result[T], len(keys))

		// Convert string keys to UUIDs
		uuids := make([]uuid.UUID, len(keys))
		for index, idString := range keys {
			uuids[index] = uuid.MustParse(idString)
		}

		// Fetch entities
		entities, err := fetchFunc(ctx, uuids)
		if err != nil {
			for index := range keys {
				results[index] = &dataloader.Result[T]{Error: err}
			}
			return results
		}

		// Build lookup map
		entityByID := make(map[string]T, len(entities))
		for _, entity := range entities {
			entityByID[getID(entity).String()] = entity
		}

		// Map results back to keys
		for index, idString := range keys {
			results[index] = &dataloader.Result[T]{Data: entityByID[idString]}
		}
		return results
	}

	return dataloader.NewBatchedLoader(batchLoadFunc,
		dataloader.WithWait[string, T](2*time.Millisecond),
		dataloader.WithBatchCapacity[string, T](100),
	)
}

// newBatchLoaderByFK creates a dataloader for fetching entities grouped by foreign key.
// This eliminates ~40 lines of duplicate code per loader.
func newBatchLoaderByFK[T any](
	fetchFunc func(ctx context.Context, fkIDs []uuid.UUID) ([]T, error),
	getFK func(T) uuid.UUID,
) *dataloader.Loader[string, []T] {
	batchLoadFunc := func(ctx context.Context, keys []string) []*dataloader.Result[[]T] {
		results := make([]*dataloader.Result[[]T], len(keys))

		// Convert string keys to UUIDs
		fkUUIDs := make([]uuid.UUID, len(keys))
		for index, idString := range keys {
			fkUUIDs[index] = uuid.MustParse(idString)
		}

		// Fetch entities
		entities, err := fetchFunc(ctx, fkUUIDs)
		if err != nil {
			for index := range keys {
				results[index] = &dataloader.Result[[]T]{Error: err}
			}
			return results
		}

		// Build lookup map (one-to-many)
		entitiesByFK := make(map[string][]T)
		for _, entity := range entities {
			fkString := getFK(entity).String()
			entitiesByFK[fkString] = append(entitiesByFK[fkString], entity)
		}

		// Map results back to keys
		for index, idString := range keys {
			results[index] = &dataloader.Result[[]T]{Data: entitiesByFK[idString]}
		}
		return results
	}

	return dataloader.NewBatchedLoader(batchLoadFunc,
		dataloader.WithWait[string, []T](2*time.Millisecond),
		dataloader.WithBatchCapacity[string, []T](100),
	)
}
