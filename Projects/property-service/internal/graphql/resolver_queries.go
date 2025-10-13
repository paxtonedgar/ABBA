package graphql

import (
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"context"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/google/uuid"

	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/amenity"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/room"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/dataloader"
)

// =====================================================================================
// Basic Query Resolvers
// =====================================================================================
// These resolvers handle basic read operations.

// Me returns the currently authenticated user.
func (resolver *queryResolver) Me(ctx context.Context) (*User, error) {
	authUser, err := GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, err
	}

	// Fetch full user details using typed ID
	entUser, err := dataloader.From(ctx).UserByID.Load(ctx, authUser.ID.String())()
	if err != nil {
		return nil, err
	}
	if entUser != nil {
		return ConvertUserToGraphQL(entUser), nil
	}

	// Fallback to email lookup
	entUser, err = resolver.client.User.Query().Where(user.EmailEqualFold(authUser.Email)).First(ctx)
	if err != nil {
		return nil, fmt.Errorf("me: %w", err)
	}
	return ConvertUserToGraphQL(entUser), nil
}

// Property fetches a single property by ID (convenience for detail pages).
func (resolver *queryResolver) Property(ctx context.Context, id string) (*Property, error) {
	// Parse to typed ID at GraphQL boundary
	propertyID, err := domain.ParsePropertyID(id)
	if err != nil {
		return nil, err
	}

	entProperty, err := dataloader.From(ctx).PropertyByID.Load(ctx, propertyID.String())()
	if err != nil {
		return nil, err
	}
	return ConvertPropertyToGraphQL(entProperty), nil
}

// Properties implements a general property search with filters/sort.
func (resolver *queryResolver) Properties(
	ctx context.Context,
	filter *PropertyFilter, sort *PropertySortInput,
) ([]*Property, error) {
	propertyQuery := resolver.client.Property.Query()

	// Apply filters and sorting
	if err := applyPropertyFilter(ctx, propertyQuery, filter); err != nil {
		return nil, err
	}
	applyPropertySort(propertyQuery, sort)

	entProperties, err := propertyQuery.All(ctx)
	if err != nil {
		return nil, err
	}

	graphqlProperties := make([]*Property, len(entProperties))
	for index, entProperty := range entProperties {
		graphqlProperties[index] = ConvertPropertyToGraphQL(entProperty)
	}

	return graphqlProperties, nil
}

// MyProperties returns the current owner's properties (user story: owner dashboard).
func (resolver *queryResolver) MyProperties(
	ctx context.Context,
	filter *PropertyFilter, sort *PropertySortInput,
) ([]*Property, error) {
	authUser, err := GetAuthenticatedUser(ctx)
	if err != nil {
		return nil, err
	}
	if authUser.Role != UserRoleOwner {
		return nil, fmt.Errorf("forbidden: only owners can query myProperties")
	}
	if filter == nil {
		filter = &PropertyFilter{}
	}
	ownerIDStr := authUser.ID.String()
	filter.OwnerID = &ownerIDStr

	return resolver.Properties(ctx, filter, sort)
}

// AgentQueue surfaces listings in submitted/under_review with newest-first
// (user story: agent review worklist).
func (resolver *queryResolver) AgentQueue(
	ctx context.Context,
	filter *PropertyFilter, sort *PropertySortInput,
) ([]*Property, error) {
	// You can enforce role=AGENT here if you carry it in ctx.
	if filter == nil {
		filter = &PropertyFilter{}
	}
	sts := []PropertyStatus{PropertyStatusSubmitted, PropertyStatusUnderReview}
	filter.StatusIn = sts

	// Default sort: SUBMITTED_AT desc
	if sort == nil {
		sort = &PropertySortInput{By: PropertySortBySubmittedAt, Direction: SortDirectionDesc}
	}
	return resolver.Properties(ctx, filter, sort)
}

// Amenities lists amenity dictionary items (optionally filtered) with simple cursoring.
func (resolver *queryResolver) Amenities(ctx context.Context, filter *AmenityFilter) ([]*Amenity, error) {
	aq := resolver.client.Amenity.Query()
	if filter != nil {
		if len(filter.CodeIn) > 0 {
			aq = aq.Where(amenity.CodeIn(filter.CodeIn...))
		}
		if len(filter.CategoryIn) > 0 {
			aq = aq.Where(amenity.CategoryIn(filter.CategoryIn...))
		}
		if filter.Q != nil && *filter.Q != "" {
			aq = aq.Where(amenity.Or(amenity.CodeContainsFold(*filter.Q), amenity.NameContainsFold(*filter.Q), amenity.CategoryContainsFold(*filter.Q)))
		}
	}

	items, err := aq.Order(appent.Asc(amenity.FieldID)).All(ctx)
	if err != nil {
		return nil, err
	}

	amenities := make([]*Amenity, len(items))
	for i, a := range items {
		amenities[i] = ConvertAmenityToGraphQL(a)
	}

	return amenities, nil
}

// Rooms lists rooms with optional filtering.
func (resolver *queryResolver) Rooms(ctx context.Context, filter *RoomFilter) ([]*Room, error) {
	rq := resolver.client.Room.Query()
	if filter != nil {
		if filter.PropertyID != nil {
			if pid, err := uuid.Parse(*filter.PropertyID); err == nil {
				rq = rq.Where(room.PropertyIDEQ(pid))
			}
		}
		if filter.NameContains != nil && *filter.NameContains != "" {
			rq = rq.Where(room.NameContainsFold(*filter.NameContains))
		}
		if filter.EnsuiteBath != nil {
			rq = rq.Where(room.EnsuiteBathEQ(*filter.EnsuiteBath))
		}
		// TODO: Implement bed filtering once JSON predicates are available
		// For now, skip bed filtering as bed_mix JSON field predicates aren't generated
		// if len(filter.BedAnyOf) > 0 { ... }
	}

	items, err := rq.Order(appent.Asc(room.FieldID)).All(ctx)
	if err != nil {
		return nil, err
	}

	rooms := make([]*Room, len(items))
	for i, r := range items {
		rooms[i] = ConvertRoomToGraphQL(r)
	}

	return rooms, nil
}

// Node implements global ID resolution for User/Property/Room/Amenity.
// Global IDs are base64-encoded strings of form "Type:uuid".
func (resolver *queryResolver) Node(ctx context.Context, id string) (Node, error) {
	raw, err := base64.StdEncoding.DecodeString(id)
	if err != nil {
		return nil, fmt.Errorf("node: bad id")
	}
	parts := strings.SplitN(string(raw), ":", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("node: bad id format")
	}
	typ, sid := parts[0], parts[1]

	switch typ {
	case "User":
		userID, err := domain.ParseUserID(sid)
		if err != nil {
			return nil, err
		}
		u, err := dataloader.From(ctx).UserByID.Load(ctx, userID.String())()
		if err != nil || u == nil {
			return nil, err
		}
		return ConvertUserToGraphQL(u), nil
	case "Property":
		propertyID, err := domain.ParsePropertyID(sid)
		if err != nil {
			return nil, err
		}
		p, err := dataloader.From(ctx).PropertyByID.Load(ctx, propertyID.String())()
		if err != nil || p == nil {
			return nil, err
		}
		return ConvertPropertyToGraphQL(p), nil
	case "Room":
		roomID, err := domain.ParseRoomID(sid)
		if err != nil {
			return nil, err
		}
		r, err := resolver.client.Room.Get(ctx, roomID.UUID)
		if err != nil {
			return nil, err
		}
		return ConvertRoomToGraphQL(r), nil
	case "Amenity":
		amenityID, err := domain.ParseAmenityID(sid)
		if err != nil {
			return nil, err
		}
		a, err := resolver.client.Amenity.Get(ctx, amenityID.UUID)
		if err != nil {
			return nil, err
		}
		return ConvertAmenityToGraphQL(a), nil
	default:
		return nil, fmt.Errorf("node: unsupported type %q", typ)
	}
}
