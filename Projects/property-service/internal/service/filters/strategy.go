// Package filters provides extensible property filtering using the Strategy pattern.
package filters

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// PropertyFilterStrategy defines the interface for property filter implementations.
// Each strategy is responsible for applying one aspect of filtering to a query.
type PropertyFilterStrategy interface {
	// Apply applies this filter strategy to the given query.
	// Returns true if the filter was applied, false if it was skipped (e.g., nil value).
	Apply(ctx context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool
}

// PropertyFilterChain applies multiple filter strategies in sequence.
type PropertyFilterChain struct {
	strategies []PropertyFilterStrategy
}

// NewPropertyFilterChain creates a new filter chain with the default set of strategies.
func NewPropertyFilterChain() *PropertyFilterChain {
	return &PropertyFilterChain{
		strategies: []PropertyFilterStrategy{
			&IDFilter{},
			&OwnerFilter{},
			&StatusFilter{},
			&TypeFilter{},
			&LocationFilter{},
			&PolicyFilter{},
			&CapacityFilter{},
			&GeoBoundsFilter{},
			&FullTextFilter{},
			&AmenityFilter{},
		},
	}
}

// Apply applies all filter strategies to the query.
func (c *PropertyFilterChain) Apply(ctx context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) {
	if filter == nil {
		return
	}

	for _, strategy := range c.strategies {
		strategy.Apply(ctx, q, filter)
	}
}

// AddStrategy adds a custom filter strategy to the chain.
// This allows for extension without modifying existing code (Open/Closed Principle).
func (c *PropertyFilterChain) AddStrategy(strategy PropertyFilterStrategy) {
	c.strategies = append(c.strategies, strategy)
}
