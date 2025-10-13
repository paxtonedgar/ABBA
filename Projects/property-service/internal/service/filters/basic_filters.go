package filters

import (
	"context"
	"strings"

	"github.com/google/uuid"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// IDFilter filters properties by ID.
type IDFilter struct{}

func (f *IDFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	if len(filter.Ids) == 0 {
		return false
	}

	ids := make([]domain.PropertyID, 0, len(filter.Ids))
	for _, idStr := range filter.Ids {
		id, err := domain.ParsePropertyID(idStr)
		if err == nil {
			ids = append(ids, id)
		}
	}

	if len(ids) > 0 {
		uuids := make([]uuid.UUID, len(ids))
		for i, id := range ids {
			uuids[i] = id.UUID
		}
		q.Where(property.IDIn(uuids...))
		return true
	}

	return false
}

// OwnerFilter filters properties by owner ID.
type OwnerFilter struct{}

func (f *OwnerFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	if filter.OwnerID == nil {
		return false
	}

	ownerID, err := domain.ParseUserID(*filter.OwnerID)
	if err != nil {
		return false
	}

	q.Where(property.OwnerID(ownerID.UUID))
	return true
}

// StatusFilter filters properties by status.
type StatusFilter struct{}

func (f *StatusFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	if len(filter.StatusIn) == 0 {
		return false
	}

	statuses := make([]property.Status, 0, len(filter.StatusIn))
	for _, s := range filter.StatusIn {
		statuses = append(statuses, property.Status(strings.ToLower(string(s))))
	}

	q.Where(property.StatusIn(statuses...))
	return true
}

// TypeFilter filters properties by type.
type TypeFilter struct{}

func (f *TypeFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	if len(filter.TypeIn) == 0 {
		return false
	}

	types := make([]property.Type, 0, len(filter.TypeIn))
	for _, t := range filter.TypeIn {
		types = append(types, property.Type(strings.ToLower(string(t))))
	}

	q.Where(property.TypeIn(types...))
	return true
}
