# Value Scan

## When to use
User is looking for betting edges -- +EV opportunities across all games.

**Trigger phrases:**
- "Any good bets tonight?"
- "Where's the value?"
- "Find me +EV plays"
- "What are the edges tonight?"
- "Any mispriced games?"

## Workflow

```python
tk.run_workflow("value_scan", sport="NHL", bankroll=10000, min_ev=0.02)
```

## What it does
1. Scans all scheduled games against sportsbook odds
2. Compares 6-model probability vs implied probability
3. Filters by minimum EV threshold
4. Sizes each bet with Kelly Criterion
5. Reports total bankroll exposure

## Output
- `sized_bets`: each opportunity with team, odds, edge, EV, recommended stake
- `total_recommended_stake` / `bankroll_pct_at_risk`
