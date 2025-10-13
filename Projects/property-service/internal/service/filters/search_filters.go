package filters

import (
	"context"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/property"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/graphql"
)

// FullTextFilter performs multi-field text search.
type FullTextFilter struct{}

func (f *FullTextFilter) Apply(_ context.Context, q *ent.PropertyQuery, filter *graphql.PropertyFilter) bool {
	if filter.FullText == nil || *filter.FullText == "" {
		return false
	}

	text := *filter.FullText
	q.Where(
		property.Or(
			property.TitleContainsFold(text),
			property.DescriptionShortContainsFold(text),
			property.DescriptionLongContainsFold(text),
			property.CityContainsFold(text),
			property.RegionContainsFold(text),
		),
	)

	return true
}
