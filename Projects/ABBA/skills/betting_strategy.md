# Betting Strategy Distillation

## When to use
User wants a disciplined, structured betting approach -- not just picks, but a methodology with risk management.

**Trigger phrases:**
- "What's my betting strategy for tonight?"
- "Build me a betting plan"
- "Give me a disciplined approach to NHL betting"
- "How should I allocate my bankroll?"
- "What's the optimal betting system?"
- "I have $5000, how should I bet tonight?"

## Workflow

```python
tk.run_workflow("betting_strategy", bankroll=5000, risk_tolerance="moderate")
```

## What it does
1. **Calibrates risk parameters** based on tolerance:
   - Conservative: quarter-Kelly, 5%+ EV threshold, 3% daily risk cap
   - Moderate: half-Kelly, 3%+ EV threshold, 5% daily risk cap
   - Aggressive: 3/4 Kelly, 2%+ EV threshold, 10% daily risk cap
2. **Scans all games** for value using 6-model predictions vs sportsbook odds
3. **Sizes each bet** using Kelly Criterion (adjusted for risk tolerance)
4. **Enforces daily risk cap** -- won't recommend more than max % at risk
5. **Generates discipline rules** the agent presents to the user
6. **Calculates expected P&L** for the day

## Output structure
- `strategy`: risk parameters, Kelly multiplier, EV thresholds
- `discipline_rules`: list of rules to follow (max risk, when to pause, etc.)
- `todays_plays`: each play with team, odds, edge, EV, stake, potential win
- `total_stake` / `total_to_win` / `expected_profit`
- `games_scanned`: transparency on coverage

## The discipline rules matter
The rules aren't decoration. They're what separates profitable bettors from degenerate gamblers:
- Never exceed daily risk cap
- Stick to the model, don't chase
- Track every bet for ROI validation
- If model accuracy drops below 55%, pause and investigate

## Parameters
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `sport` | string | no | NHL (default) or MLB |
| `bankroll` | number | no | Default $10,000 |
| `risk_tolerance` | string | no | conservative, moderate (default), aggressive |
| `timeframe` | string | no | tonight (default) |
