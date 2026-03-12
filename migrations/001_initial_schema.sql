-- ABBA NHL Supabase schema
-- Run this in the Supabase SQL editor (or via psql) to create all tables.

-- ============================================================
-- Core tables (sport-agnostic)
-- ============================================================

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    status TEXT DEFAULT 'scheduled',
    metadata JSONB,
    source TEXT DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games (sport, date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games (status);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games (home_team, away_team);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    sportsbook TEXT NOT NULL,
    market_type TEXT NOT NULL,
    home_odds DOUBLE PRECISION,
    away_odds DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    total DOUBLE PRECISION,
    over_odds DOUBLE PRECISION,
    under_odds DOUBLE PRECISION,
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots (game_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_odds_sportsbook ON odds_snapshots (sportsbook);

CREATE TABLE IF NOT EXISTS player_stats (
    player_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    team TEXT,
    stats JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, sport, season)
);

CREATE TABLE IF NOT EXISTS team_stats (
    team_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    season TEXT NOT NULL,
    stats JSONB NOT NULL,
    source TEXT DEFAULT 'unknown',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (team_id, sport, season)
);

CREATE INDEX IF NOT EXISTS idx_team_stats_sport ON team_stats (sport);

CREATE TABLE IF NOT EXISTS weather (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction TEXT,
    precipitation_chance DOUBLE PRECISION,
    conditions TEXT,
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions_cache (
    prediction_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    model_version TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    prediction JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- ============================================================
-- Session / observability (optional — can skip if not needed remotely)
-- ============================================================

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    budget_remaining DOUBLE PRECISION DEFAULT 1000.0,
    budget_total DOUBLE PRECISION DEFAULT 1000.0,
    tool_calls INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_call_log (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input_params JSONB,
    output_summary JSONB,
    cost DOUBLE PRECISION DEFAULT 0.0,
    latency_ms DOUBLE PRECISION,
    cache_hit BOOLEAN DEFAULT FALSE,
    called_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reasoning_log (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    plan TEXT,
    uncertainty JSONB,
    data_trust JSONB,
    workflow_gaps JSONB,
    want_to_verify JSONB,
    raw_thought TEXT,
    context_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- NHL-specific tables
-- ============================================================

CREATE TABLE IF NOT EXISTS goaltender_stats (
    goaltender_id TEXT NOT NULL,
    team TEXT NOT NULL,
    season TEXT NOT NULL,
    stats JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (goaltender_id, season)
);

CREATE INDEX IF NOT EXISTS idx_goalie_team ON goaltender_stats (team);

CREATE TABLE IF NOT EXISTS nhl_advanced_stats (
    team_id TEXT NOT NULL,
    season TEXT NOT NULL,
    stats JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (team_id, season)
);

CREATE TABLE IF NOT EXISTS salary_cap (
    player_id TEXT NOT NULL,
    team TEXT NOT NULL,
    season TEXT NOT NULL,
    name TEXT NOT NULL,
    position TEXT,
    cap_hit DOUBLE PRECISION,
    aav DOUBLE PRECISION,
    contract_years_remaining INTEGER,
    status TEXT DEFAULT 'active',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS roster (
    player_id TEXT NOT NULL,
    team TEXT NOT NULL,
    season TEXT NOT NULL,
    name TEXT NOT NULL,
    position TEXT,
    line_number INTEGER,
    stats JSONB,
    injury_status TEXT DEFAULT 'healthy',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, team, season)
);

CREATE INDEX IF NOT EXISTS idx_roster_team ON roster (team, season);

-- ============================================================
-- Tracking / historical
-- ============================================================

CREATE TABLE IF NOT EXISTS data_freshness (
    table_name TEXT PRIMARY KEY,
    last_refresh_at TIMESTAMPTZ NOT NULL,
    source TEXT DEFAULT 'unknown',
    row_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS standings_snapshots (
    snapshot_date DATE NOT NULL,
    team_id TEXT NOT NULL,
    sport TEXT DEFAULT 'NHL',
    stats JSONB NOT NULL,
    PRIMARY KEY (snapshot_date, team_id)
);

CREATE INDEX IF NOT EXISTS idx_standings_date ON standings_snapshots (snapshot_date);

-- ============================================================
-- SportsRadar query budget tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS sr_query_log (
    id BIGSERIAL PRIMARY KEY,
    endpoint TEXT NOT NULL,
    params JSONB,
    status_code INTEGER,
    called_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sr_log_date ON sr_query_log (called_at DESC);
