package graphql

import (
	appent "github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/service"
)

// =====================================================================================
// Root Resolver
// =====================================================================================

// Resolver is the application root resolver.
// It holds the ent.Client for DB access and service layer dependencies.
type Resolver struct {
	client          *appent.Client
	authService     *service.AuthService
	propertyService *service.PropertyService
	roomService     *service.RoomService
	amenityService  *service.AmenityService
}

// NewResolver constructs a new root resolver with dependency injection.
func NewResolver(client *appent.Client) *Resolver {
	// Initialize services
	authService := service.NewAuthService(client)
	propertyService := service.NewPropertyService(client, authService)
	roomService := service.NewRoomService(client, authService)
	amenityService := service.NewAmenityService(client, authService)

	return &Resolver{
		client:          client,
		authService:     authService,
		propertyService: propertyService,
		roomService:     roomService,
		amenityService:  amenityService,
	}
}

// Query implements QueryResolver (root queries).
func (resolver *Resolver) Query() QueryResolver {
	return &queryResolver{
		client: resolver.client,
	}
}

// Mutation implements MutationResolver (root mutations).
func (resolver *Resolver) Mutation() MutationResolver {
	return &mutationResolver{
		client:          resolver.client,
		propertyService: resolver.propertyService,
		roomService:     resolver.roomService,
		amenityService:  resolver.amenityService,
	}
}
