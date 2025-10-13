package graphql

import (
	"net/http"
	"time"

	"github.com/99designs/gqlgen/graphql/handler"
	"github.com/99designs/gqlgen/graphql/playground"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/dataloader"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/logger"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/middleware"
)

// NewHandler creates a new GraphQL handler with dataloaders and demo auth
func NewHandler(client *ent.Client) http.Handler {
	// Create the executable schema
	cfg := Config{
		Resolvers: NewResolver(client),
		Directives: DirectiveRoot{
			Auth: Auth,
		},
	}
	srv := handler.NewDefaultServer(NewExecutableSchema(cfg))

	// Wrap with dataloaders and demo auth
	withAuth := middleware.DemoAuthMiddleware(client)(srv)
	return withLoaders(client, withAuth)
}

// withLoaders attaches request-scoped dataloaders to the GraphQL handler
func withLoaders(client *ent.Client, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ld := dataloader.NewLoaders(client)
		ctx := dataloader.With(r.Context(), ld)

		// Log GraphQL requests
		logger.Debug("graphql_request",
			"method", r.Method,
			"path", r.URL.Path,
			"remote_addr", r.RemoteAddr,
		)

		next.ServeHTTP(w, r.WithContext(ctx))

		logger.Info("graphql_request_completed",
			"method", r.Method,
			"path", r.URL.Path,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}

// NewMux creates a mux with both GraphQL endpoint and playground
func NewMux(gql http.Handler) *http.ServeMux {
	mux := http.NewServeMux()

	// GraphQL endpoint
	mux.Handle("/graphql", gql)

	// Playground at /
	mux.Handle("/", playground.Handler("GraphQL Playground", "/graphql"))

	return mux
}
