package middleware

import (
	"net/http"
	"strings"

	"github.com/evolve-interviews/paxtonedgar-case-study/ent"
	"github.com/evolve-interviews/paxtonedgar-case-study/ent/user"
	"github.com/evolve-interviews/paxtonedgar-case-study/internal/domain"
)

// DemoAuthMiddleware adds a test user to context for demo/development purposes.
// In production, this would be replaced with JWT validation middleware.
//
// Usage: Clients can pass X-User-Email header to specify which user to authenticate as.
// If no header is provided, defaults to owner@example.com
func DemoAuthMiddleware(client *ent.Client) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Get user email from header (default to owner)
			email := r.Header.Get("X-User-Email")
			if email == "" {
				email = "owner@example.com"
			}

			// Look up user in database
			entUser, err := client.User.Query().
				Where(user.EmailEQ(email)).
				First(r.Context())

			if err != nil {
				// User not found - continue without auth (will fail at resolver level if needed)
				next.ServeHTTP(w, r)
				return
			}

			// Add user to context (store as domain-level auth, uppercase role for consistency)
			userID := domain.UserID{UUID: entUser.ID}
			ctx := domain.WithAuthenticatedUser(
				r.Context(),
				userID,
				entUser.Email,
				strings.ToUpper(string(entUser.Role)), // Convert to uppercase for GraphQL enum compatibility
				entUser.Name,
			)

			// Continue with authenticated context
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}
