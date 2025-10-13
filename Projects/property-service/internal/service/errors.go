// Package service provides business logic services that are independent of the transport layer (GraphQL, REST, etc).
package service

import "errors"

// Domain errors that can be returned by service layer
var (
	// ErrUnauthorized indicates the user is not authenticated
	ErrUnauthorized = errors.New("unauthorized: authentication required")

	// ErrForbidden indicates the user lacks permission for the operation
	ErrForbidden = errors.New("forbidden: insufficient permissions")

	// ErrNotFound indicates the requested resource doesn't exist
	ErrNotFound = errors.New("not found")

	// ErrInvalidInput indicates the input data is invalid
	ErrInvalidInput = errors.New("invalid input")

	// ErrConflict indicates a resource conflict (e.g., duplicate)
	ErrConflict = errors.New("conflict: resource already exists")
)
