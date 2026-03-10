# ABBA: Sports Decision Intelligence Engine — Architecture Plan

## Vision

ABBA evolves from an NHL prediction toolkit into a **multi-sport decision intelligence engine** that:
1. Serves as a **data pipeline and context layer** for LLMs (ChatGPT, Claude, custom agents)
2. Enables **historical canonization** — connecting players, contracts, GM decisions, peer comparisons, and coaching context into queryable decision narratives
3. Provides **realistic betting workflow tools** — not "pick winners," but CLV tracking, market timing, bankroll management across sportsbooks
4. Supports **NFL, CFB (with NIL), MLB, and NHL** with sport-specific analytics modules

---

## Part 1: Multi-Sport Data Architecture

### Data Source Map

| Sport | Stats API | Contracts/Cap | Odds | Historical |
|-------|-----------|---------------|------|------------|
| **NHL** | NHL API (free, PBP from 2010+), MoneyPuck (free xG CSVs from 2007+), NaturalStatTrick | PuckPedia API (paid, only option post-CapFriendly shutdown), Spotrac | The Odds API | Hockey-Reference, Elite Prospects (paid API, 1.2M players), ProSportsTransactions (trades/waivers) |
| **NFL** | ESPN API (free), Sportradar (paid), nfl_data_py | Spotrac API, Over The Cap (scrape) | The Odds API | Pro Football Reference, nflverse |
| **CFB** | CollegeFootballData.com API (free, excellent) | NIL: Opendorse/On3 (scrape/partner) | The Odds API | Sports-Reference CFB |
| **MLB** | MLB StatsAPI (free), baseballr/pybaseball | Spotrac API, Cot's Contracts (BP) | The Odds API | Baseball-Reference, Retrosheet, Lahman DB |

### Connector Architecture

```
src/abba/connectors/
├── base.py              # BaseConnector with _fetch_json, error capture, rate limiting
├── live.py              # NHLLiveConnector, OddsLiveConnector (existing)
├── nhl/
│   ├── stats.py         # NHL API — games, rosters, standings
│   ├── advanced.py      # NaturalStatTrick — Corsi, xG, zone entries
│   └── contracts.py     # PuckPedia API — cap, contracts, transactions
├── nfl/
│   ├── stats.py         # ESPN/Sportradar — games, box scores, play-by-play
│   ├── advanced.py      # nflverse — EPA, CPOE, DVOA-style metrics
│   └── contracts.py     # Spotrac — cap, contracts, dead money
├── cfb/
│   ├── stats.py         # CollegeFootballData API — games, team stats, recruiting
│   ├── advanced.py      # CFBD advanced — PPA, success rate, havoc
│   └── nil.py           # NIL tracker — Opendorse data, On3 valuations
├── mlb/
│   ├── stats.py         # MLB StatsAPI — games, player stats, standings
│   ├── advanced.py      # pybaseball — Statcast, FanGraphs, pitch-level
│   └── contracts.py     # Spotrac/Cot's — contracts, arbitration, service time
└── shared/
    ├── odds.py          # The Odds API — unified across all sports
    └── reference.py     # Sports-Reference family scraper (fallback)
```

### Storage Schema Evolution

```sql
-- Core tables (sport-agnostic)
games (game_id, sport, season, date, home_team, away_team, status, score_home, score_away, ...)
teams (team_id, sport, name, abbreviation, conference, division, ...)
players (player_id, sport, name, position, team_id, birth_date, draft_year, draft_pick, ...)

-- Contract & financial tables
contracts (contract_id, player_id, team_id, sport, season, years, total_value, aav, cap_hit,
           signing_date, structure JSON, no_trade_clause, source)
transactions (txn_id, sport, date, type ENUM('trade','signing','waiver','release','draft','nil_deal'),
              player_id, from_team, to_team, details JSON, reported_value)
cap_snapshots (team_id, sport, season, date, cap_ceiling, cap_floor, committed, projected_space,
               dead_cap, ltir_used)

-- NIL-specific (CFB)
nil_deals (deal_id, player_id, brand, category, reported_value, duration, platform,
           announced_date, school, conference)
nil_valuations (player_id, date, on3_valuation, followers JSON, sport)

-- Historical context
coaching_tenures (coach_id, team_id, role, start_date, end_date, record JSON)
gm_tenures (gm_id, team_id, start_date, end_date, notable_moves JSON)
draft_picks (year, round, pick, team_id, player_id, sport)

-- Analytics (sport-specific)
nhl_advanced_stats (...)     -- existing
nfl_advanced_stats (team_id, season, epa_pass, epa_rush, cpoe, dvoa_off, dvoa_def, ...)
cfb_advanced_stats (team_id, season, ppa_off, ppa_def, success_rate, havoc_rate, ...)
mlb_advanced_stats (team_id, season, war_total, run_diff, babip, fip_minus, ...)
```

---

## Part 2: Historical Canonization System

### The Core Question

> "Why was this player worth this much? Compare him to peers across the league.
> What was the GM thinking? What was the surrounding decision set?"

### Canonization Schema

```
PlayerCanon:
  player_id, name, sport
  career_timeline: [
    { season, team, role, stats_summary, contract_at_time, injuries }
  ]
  peer_cohort: [
    { player_id, similarity_score, comparison_axes: [position, era, production, draft_class] }
  ]
  contract_history: [
    { date, type, from_team, to_team, value, context: { cap_pct, market_rank, comparable_deals } }
  ]
  decision_context: [
    { date, decision_type, gm, coach, team_situation: { cap_space, record, window_status },
      alternatives_available: [free_agents, trade_targets],
      outcome_grade: { immediate, 3yr, career } }
  ]
```

### Peer Comparison Engine

```python
class PeerEngine:
    """Find comparable players across eras and leagues."""

    def find_peers(self, player_id, axes=None, top_n=10):
        """
        axes: ['production', 'contract', 'draft_position', 'age_curve', 'role']
        Returns ranked list of comparable players with similarity scores.
        """

    def contract_context(self, contract_id):
        """
        For a given contract, return:
        - Cap % at signing vs league average for position
        - Comparable deals signed within 12 months
        - Player's production rank at position at time of signing
        - How the deal aged (surplus value by year)
        """

    def gm_decision_tree(self, team_id, date):
        """
        For a given team at a point in time, reconstruct:
        - Cap situation and upcoming commitments
        - Available free agents / trade targets
        - What they actually did vs alternatives
        - Outcome comparison
        """
```

### Queryable Narratives

The canonization system enables LLM queries like:

- "Why did Colorado give Makar that deal?" → contract context + peer cohort + cap situation + alternative D-men available
- "Compare McDavid's contract to peers" → cap %, production rank, surplus value, era adjustment
- "Show me every GM decision that led to Chicago's rebuild" → transaction timeline + decision context + outcome grades
- "Which NIL deals in CFB have actually correlated with on-field production?" → nil_deals + cfb_advanced_stats + transfer portal data

---

## Part 3: Realistic Betting Workflow

### What Actually Makes Money (from research)

- **Edge size**: Realistic edge is 2-5% on game outcomes, higher on props/totals
- **Most inefficient markets**: Player props, live/in-game, totals, early lines
- **Gold standard**: Beating Pinnacle's closing line (CLV) — if you consistently get +EV relative to closing line, you're skilled
- **Sample size**: ~5,000 bets at 55% to prove skill at 95% confidence
- **What serious bettors need**: Not picks — they need data pipelines, market timing, bankroll tools

### CLV Tracking System

```python
class CLVTracker:
    """Track closing line value — the only metric that proves edge."""

    def record_bet(self, bet):
        """Record: game_id, sportsbook, odds_at_placement, timestamp, stake, side"""

    def record_closing_line(self, game_id, sportsbook):
        """Capture Pinnacle closing line for comparison"""

    def clv_report(self, period="30d"):
        """
        Returns:
        - Average CLV (should be positive if skilled)
        - CLV by sport, market type, sportsbook
        - Statistical significance (z-score)
        - Actual P&L vs expected P&L from CLV
        """
```

### Market Timing Tools

```python
class MarketTimer:
    """Identify optimal bet placement timing."""

    def line_movement(self, game_id):
        """Track odds movement from open to close across books"""

    def steam_moves(self, sport=None):
        """Detect sharp money moves (rapid line changes at Pinnacle/Circa)"""

    def stale_lines(self, sport=None):
        """Find sportsbooks slow to adjust after news/sharp action"""
```

### Multi-Book Arbitrage Scanner

```python
class ArbScanner:
    """Cross-sportsbook opportunity detection."""

    def scan_arbs(self, sport=None):
        """Pure arbitrage (rare, <1% margins)"""

    def scan_middles(self, sport=None):
        """Middle opportunities on spreads/totals"""

    def scan_best_available(self, game_id):
        """Best odds per side across all tracked books"""
```

---

## Part 4: Sport-Specific Analytics Modules

### Module Pattern

Each sport gets its own engine module following the NHL pattern:

```
src/abba/engine/
├── ensemble.py          # Sport-agnostic ensemble combiner (existing)
├── features.py          # Sport-agnostic feature engineering (existing)
├── confidence.py        # Sport-agnostic confidence metadata (existing)
├── elo.py               # Multi-sport Elo (existing, parameterize K/HFA per sport)
├── kelly.py             # Kelly criterion (existing, sport-agnostic)
├── value.py             # Value scanning (existing, sport-agnostic)
├── hockey.py            # NHL: Corsi, xG, goaltender, special teams (existing)
├── football.py          # NFL: EPA, DVOA-style, QB evaluation, pace
├── college_football.py  # CFB: recruiting, transfer portal, SP+, NIL ROI
└── baseball.py          # MLB: Statcast, pitching matchups, park factors, platoon splits
```

### NFL Engine Highlights

```python
class NFLAnalytics:
    """NFL-specific analytics — EPA-based with situational context."""

    def build_nfl_features(self, home_stats, away_stats, weather=None, injuries=None):
        """EPA/play, CPOE, pass rate over expected, defensive pressure rate, red zone efficiency"""

    def qb_matchup(self, home_qb, away_qb):
        """QB evaluation: EPA, CPOE, time to throw, pressure performance"""

    def predict_nfl_game(self, features, elo_prob=None):
        """Models: EPA composite, Pythagorean, Elo, turnover-adjusted, SOS-adjusted, market"""
```

### CFB Engine Highlights

```python
class CFBAnalytics:
    """College football analytics with recruiting and NIL context."""

    def build_cfb_features(self, home_stats, away_stats, recruiting=None):
        """PPA, success rate, havoc, explosiveness, recruiting composite, portal impact"""

    def nil_roi(self, player_id):
        """Correlate NIL valuation with on-field production metrics"""

    def transfer_portal_impact(self, team_id, season):
        """Quantify roster turnover effect on team performance"""

    def predict_cfb_game(self, features, elo_prob=None):
        """Models: PPA composite, talent composite, Elo, returning production, market"""
```

### MLB Engine Highlights

```python
class MLBAnalytics:
    """MLB analytics — Statcast-driven with pitching matchup focus."""

    def build_mlb_features(self, home_stats, away_stats, pitching_matchup=None, park=None):
        """wOBA, xwOBA, barrel rate, K-BB%, FIP, park-adjusted runs"""

    def pitching_matchup(self, pitcher_id, opposing_lineup):
        """Pitch mix vs lineup handedness splits, recent velocity trends"""

    def park_adjustment(self, park_id, stat):
        """Park factor adjustments for HR, runs, K, etc."""

    def predict_mlb_game(self, features, elo_prob=None):
        """Models: run line Pythagorean, FIP-based, Statcast composite, Elo, bullpen-adjusted, market"""
```

---

## Part 5: Toolkit Expansion

### New Tool Mixins

```
src/abba/server/tools/
├── __init__.py
├── data.py              # Existing — query_games, query_odds, etc.
├── analytics.py         # Existing — predict_game, explain_prediction
├── market.py            # Existing — find_value, compare_odds, kelly_sizing
├── nhl.py               # Existing — nhl_predict_game, season_review, etc.
├── session.py           # Existing — refresh_data, workflows
├── registry.py          # Existing — list_tools, call_tool
├── nfl.py               # NEW — nfl_predict_game, query_depth_chart, matchup_report
├── cfb.py               # NEW — cfb_predict_game, nil_tracker, portal_tracker, recruiting
├── mlb.py               # NEW — mlb_predict_game, pitching_matchup, park_factors
├── canonization.py      # NEW — player_canon, contract_context, peer_comparison, gm_decisions
├── betting.py           # NEW — clv_report, record_bet, line_movement, stale_lines
└── league.py            # NEW — cross_sport queries, unified player/team search
```

### ABBAToolkit Composition

```python
class ABBAToolkit(
    DataToolsMixin,
    AnalyticsToolsMixin,
    MarketToolsMixin,
    NHLToolsMixin,
    NFLToolsMixin,        # new
    CFBToolsMixin,        # new
    MLBToolsMixin,        # new
    CanonizationMixin,    # new
    BettingToolsMixin,    # new
    SessionToolsMixin,
    ToolRegistryMixin,
):
```

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Current → +2 weeks)
- [x] Toolkit mixin decomposition
- [x] Elo as Model 8, player impact, confidence metadata
- [ ] Fix `last_refresh_ts` to actually track data freshness
- [ ] Fix K-factor to K=4 (FiveThirtyEight NHL standard)
- [ ] Add CLV tracking table and `record_bet` / `clv_report` tools
- [ ] Add `line_movement` tool (already have odds snapshots)
- [ ] Parameterize Elo for multi-sport (K, HFA vary by sport)

### Phase 2: NFL Module (+2-4 weeks)
- [ ] NFL connector (ESPN API free tier + nfl_data_py for historical)
- [ ] NFL engine (EPA, CPOE, pressure rate, red zone)
- [ ] NFL tools mixin (predict, depth chart, matchup report)
- [ ] NFL Elo ratings (K=20, HFA=48 per 538)
- [ ] Spotrac connector for NFL contracts/cap

### Phase 3: Canonization System (+4-6 weeks)
- [ ] Player/contract/transaction schema in DuckDB
- [ ] Peer comparison engine (production similarity, contract context)
- [ ] GM decision tree reconstruction
- [ ] Canonization tools mixin
- [ ] Backfill: NHL historical contracts (PuckPedia), NFL (Spotrac)

### Phase 4: CFB + NIL (+6-8 weeks)
- [ ] CFB connector (CollegeFootballData.com — free, excellent API)
- [ ] CFB engine (PPA, recruiting composite, portal impact)
- [ ] NIL tracker connector (On3 scrape / Opendorse partnership)
- [ ] NIL ROI analysis tool
- [ ] CFB tools mixin

### Phase 5: MLB (+8-10 weeks)
- [ ] MLB connector (MLB StatsAPI free + pybaseball for Statcast)
- [ ] MLB engine (wOBA, FIP, park factors, platoon splits)
- [ ] Pitching matchup analysis
- [ ] MLB tools mixin
- [ ] MLB contract backfill (Spotrac + Cot's/Baseball Prospectus)

### Phase 6: Betting Workflow (+10-12 weeks)
- [ ] Multi-book odds tracking (expand beyond The Odds API)
- [ ] Market timing tools (steam moves, stale lines)
- [ ] Bankroll management with per-sport Kelly adjustments
- [ ] Betting journal with CLV tracking dashboard
- [ ] Performance attribution (was the model right, or just lucky?)

---

## Part 7: Key Design Principles

1. **Data pipeline first, predictions second.** Serious users have their own models. ABBA's value is structured data + context, not "who will win."

2. **Confidence is mandatory.** Every prediction response includes confidence grade, data freshness, model agreement, and caveats. Never present false precision.

3. **CLV is the only truth.** Track closing line value on every recorded bet. P&L lies (variance), CLV doesn't.

4. **Sport modules are independent.** NFL engine knows nothing about NHL. Shared infrastructure (Elo, ensemble, Kelly, confidence) is sport-agnostic.

5. **Historical context is the moat.** Anyone can build a prediction model. Nobody has built "why was this player worth this much, compare to peers, show me the GM's decision tree" as structured queryable data.

6. **Respect the ceiling.** NHL game prediction tops out at ~62%. NFL at ~67%. The real edge is in market selection (props, live, totals) and timing, not game outcomes.

---

## Data Source Costs

| Source | Cost | Sports | Notes |
|--------|------|--------|-------|
| NHL API | Free | NHL | Official, real-time |
| CollegeFootballData.com | Free | CFB | Excellent API, Patreon for extra |
| MLB StatsAPI | Free | MLB | Official, real-time |
| ESPN API | Free (unofficial) | NFL, NBA | Undocumented but stable |
| nfl_data_py / nflverse | Free | NFL | Historical play-by-play |
| pybaseball | Free | MLB | Statcast, FanGraphs |
| The Odds API | Free tier (500 req/mo) | All | $20/mo for serious use |
| Spotrac API | Paid (unknown) | All | Contracts, cap data |
| PuckPedia API | Paid (unknown) | NHL | Contracts, cap, transactions |
| MoneyPuck | Free (credit req) | NHL | Shot-level xG CSVs from 2007-08, 1.7M shots |
| Natural Stat Trick | Free (Patreon) | NHL | Corsi/Fenwick, zone entries, on-ice |
| Evolving Hockey | $5/mo | NHL | WAR/RAPM/GAR models |
| Elite Prospects | Paid API | NHL (all leagues) | 1.2M player DB, pre-NHL development paths |
| ProSportsTransactions | Free (scrape) | All | Trades, waivers, buyouts — needs Cloudflare bypass |
| Sportradar | Paid ($$$) | All | Enterprise-grade |
| SportsDataIO | $25-50/mo | All | Good mid-tier option |

---

## Appendix: NHL Data Source Deep Dive

### Play-by-Play Historical Coverage

| Era | Data Available | Best Source |
|-----|---------------|-------------|
| Pre-2007 | Box scores, goals, assists only | Hockey-Reference |
| 2007-08 to 2009-10 | Shot x/y coords exist but messy | MoneyPuck (cleaned xG CSVs) |
| 2010-11 to 2020-21 | Full PBP JSON: shots, faceoffs, hits, blocks, penalties | NHL API, hockeyR, Hockey-Scraper |
| 2021-22 onward | NHL EDGE puck/player tracking: skater speed, shot speed, granular TOI | api-web.nhle.com |

### Key NHL API Endpoints (free, no auth)

```
https://api-web.nhle.com/v1/player/{playerId}/landing          -- bio, career stats
https://api-web.nhle.com/v1/player/{playerId}/game-log/{season}/{gameType}  -- per-game
https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play    -- event-level
https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore        -- box scores
https://api-web.nhle.com/v1/standings/{date}                    -- historical standings
https://api-web.nhle.com/v1/draft-tracker/picks/now             -- draft picks
https://api-web.nhle.com/v1/schedule/{date}                     -- back to 1917-18
```

Season format: `YYYYYYYY` (e.g., `20072008`). No official rate limits — community consensus is 1-2 req/sec.

### The CapFriendly Gap

CapFriendly (the gold standard for NHL cap data) was acquired by the Washington Capitals in June 2024 and shut down for public access in July 2024. Replacements:

- **PuckPedia** (paid API, contact api@puckpedia.com) — only programmatic source for historical contracts
- **CapWages** (free website) — current cap data, retained salary, LTIR, buyouts
- **Spotrac** (has developer API, pricing unknown) — contract values, free agent tracker

### Python Libraries for NHL Data

- **[Hockey-Scraper](https://github.com/HarryShomer/Hockey-Scraper)** — PBP + shift data from NHL/ESPN APIs
- **[hockeyR](https://github.com/danmorse314/hockeyR)** — R package, clean PBP from 2010+
- **[nhl-api-py](https://pypi.org/project/nhl-api-py/)** — Python wrapper for NHL API
- **[pro_sports_transactions](https://github.com/rsforbes/pro_sports_transactions)** — trades/waivers (needs Cloudflare workaround)

### What's Novel About Our Canonization Layer

Nobody has connected the **decision layer** (GM trades, coaching changes, contract context, cap situation, available alternatives) to the **performance layer** (play-by-play, advanced stats, xG). The closest work is in soccer (SoccerNet GraphRAG) and basketball (MbgKG knowledge graph). An NHL/multi-sport version would be first-of-its-kind.
