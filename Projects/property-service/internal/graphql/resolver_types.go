package graphql

import (
	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"
)

// =====================================================================================
// Resolver Types
// =====================================================================================

// queryResolver holds the ent client and implements query resolution methods.
type queryResolver struct {
	client *appent.Client
}

// mutationResolver holds the ent client and service layer dependencies.
type mutationResolver struct {
	client          *appent.Client
	propertyService *service.PropertyService
	roomService     *service.RoomService
	amenityService  *service.AmenityService
}

// Query returns a reference to this queryResolver (helper for accessing Me, etc.).
func (r *mutationResolver) Query() *queryResolver {
	return &queryResolver{client: r.client}
}

// Ensure implementations satisfy the GraphQL interfaces.
var (
	_ QueryResolver    = (*queryResolver)(nil)
	_ MutationResolver = (*mutationResolver)(nil)
)
