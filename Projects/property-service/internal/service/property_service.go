package service

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/logger"
)

// PropertyService handles property business logic.
type PropertyService struct {
	client      *ent.Client
	authService *AuthService
}

// NewPropertyService creates a new property service.
func NewPropertyService(client *ent.Client, authService *AuthService) *PropertyService {
	return &PropertyService{
		client:      client,
		authService: authService,
	}
}

// CreatePropertyInput contains data for creating a property.
type CreatePropertyInput struct {
	Title            string
	Type             string
	DescriptionShort *string
	DescriptionLong  *string
	MaxGuests        *int
	BathroomsTotal   *float64
	AddressLine1     string
	AddressLine2     *string
	City             string
	Region           string
	PostalCode       string
	Country          string
	Latitude         *float64
	Longitude        *float64
	CheckinTime      *time.Time
	CheckoutTime     *time.Time
	MinNights        *int
	MaxNights        *int
	PetsAllowed      bool
	SmokingAllowed   bool
	HouseRules       *string
}

// CreateProperty creates a new property for the authenticated owner.
func (service *PropertyService) CreateProperty(ctx context.Context, input CreatePropertyInput) (*ent.Property, error) {
	start := time.Now()

	// Verify user is an owner
	owner, err := service.authService.RequireOwnerRole(ctx)
	if err != nil {
		return nil, err
	}

	// Validate input
	if err := service.validateCreateInput(input); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidInput, err)
	}

	// Create property
	prop, err := service.client.Property.
		Create().
		SetOwnerID(owner.ID).
		SetTitle(input.Title).
		SetType(property.Type(strings.ToLower(input.Type))).
		SetNillableDescriptionShort(input.DescriptionShort).
		SetNillableDescriptionLong(input.DescriptionLong).
		SetNillableMaxGuests(input.MaxGuests).
		SetNillableBathroomsTotal(input.BathroomsTotal).
		SetAddressLine1(input.AddressLine1).
		SetNillableAddressLine2(input.AddressLine2).
		SetCity(input.City).
		SetRegion(input.Region).
		SetPostalCode(input.PostalCode).
		SetCountry(input.Country).
		SetNillableLatitude(input.Latitude).
		SetNillableLongitude(input.Longitude).
		SetNillableCheckinTime(input.CheckinTime).
		SetNillableCheckoutTime(input.CheckoutTime).
		SetNillableMinNights(input.MinNights).
		SetNillableMaxNights(input.MaxNights).
		SetPetsAllowed(input.PetsAllowed).
		SetSmokingAllowed(input.SmokingAllowed).
		SetNillableHouseRules(input.HouseRules).
		Save(ctx)

	if err != nil {
		logger.Error("property_creation_failed",
			"owner_id", owner.ID,
			"title", input.Title,
			"error", err.Error(),
			"duration_ms", time.Since(start).Milliseconds(),
		)
		return nil, fmt.Errorf("create property: %w", err)
	}

	logger.Info("property_created",
		"property_id", prop.ID,
		"owner_id", owner.ID,
		"title", input.Title,
		"type", input.Type,
		"duration_ms", time.Since(start).Milliseconds(),
	)

	return prop, nil
}

// SubmitProperty submits a property for agent review.
func (service *PropertyService) SubmitProperty(ctx context.Context, propertyID domain.PropertyID) (*ent.Property, error) {
	start := time.Now()

	// Get property and verify permission
	prop, err := getPropertyWithAuthCheck(ctx, service.client, service.authService, propertyID)
	if err != nil {
		return nil, err
	}

	// Update status
	prop, err = service.client.Property.
		UpdateOneID(propertyID.UUID).
		SetStatus(property.StatusSubmitted).
		SetSubmittedAt(time.Now()).
		Save(ctx)

	if err != nil {
		logger.Error("property_submission_failed",
			"property_id", propertyID.String(),
			"error", err.Error(),
		)
		return nil, fmt.Errorf("submit property: %w", err)
	}

	logger.Info("property_submitted",
		"property_id", propertyID.String(),
		"duration_ms", time.Since(start).Milliseconds(),
	)

	return prop, nil
}

// ReviewPropertyInput contains data for reviewing a property.
type ReviewPropertyInput struct {
	PropertyID domain.PropertyID
	Decision   string // "approve", "reject", "return_for_edit"
	Notes      *string
}

// ReviewProperty allows an agent to approve/reject a property.
func (service *PropertyService) ReviewProperty(ctx context.Context, input ReviewPropertyInput) (*ent.Property, error) {
	start := time.Now()

	// Verify user is an agent
	reviewer, err := service.authService.RequireAgentRole(ctx)
	if err != nil {
		return nil, err
	}

	// Determine status from decision
	var status property.Status
	switch strings.ToLower(input.Decision) {
	case "approve":
		status = property.StatusApproved
	case "reject":
		status = property.StatusRejected
	default:
		status = property.StatusUnderReview
	}

	// Update property
	prop, err := service.client.Property.
		UpdateOneID(input.PropertyID.UUID).
		SetStatus(status).
		SetReviewedBy(reviewer.ID).
		SetReviewedAt(time.Now()).
		SetNillableReviewNotes(input.Notes).
		Save(ctx)

	if err != nil {
		if ent.IsNotFound(err) {
			return nil, fmt.Errorf("%w: property not found", ErrNotFound)
		}
		logger.Error("property_review_failed",
			"property_id", input.PropertyID.String(),
			"reviewer_id", reviewer.ID,
			"decision", input.Decision,
			"error", err.Error(),
		)
		return nil, fmt.Errorf("review property: %w", err)
	}

	logger.Info("property_reviewed",
		"property_id", input.PropertyID.String(),
		"reviewer_id", reviewer.ID,
		"decision", input.Decision,
		"duration_ms", time.Since(start).Milliseconds(),
	)

	return prop, nil
}

// validateCreateInput validates property creation input.
func (service *PropertyService) validateCreateInput(input CreatePropertyInput) error {
	if strings.TrimSpace(input.Title) == "" {
		return fmt.Errorf("title is required")
	}

	if strings.TrimSpace(input.AddressLine1) == "" {
		return fmt.Errorf("address line 1 is required")
	}

	if strings.TrimSpace(input.City) == "" {
		return fmt.Errorf("city is required")
	}

	if strings.TrimSpace(input.Region) == "" {
		return fmt.Errorf("region is required")
	}

	if strings.TrimSpace(input.PostalCode) == "" {
		return fmt.Errorf("postal code is required")
	}

	if strings.TrimSpace(input.Country) == "" {
		return fmt.Errorf("country is required")
	}

	return nil
}
