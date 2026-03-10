# Game Prediction

## When to use
User asks about a specific game outcome, matchup, or wants to know who will win.

**Trigger phrases:**
- "Who wins tonight?"
- "Predict the Rangers game"
- "NYR vs BOS -- who takes it?"
- "What are the chances the Avs win?"
- "Should I bet on the Panthers tonight?"
- "Break down tonight's matchup"

## Workflow

```python
from abba import ABBAToolkit
from abba.workflows import run_workflow

tk = ABBAToolkit()
result = tk.run_workflow("game_prediction", team="NYR")
```

## What it does
1. **Refreshes live data** from NHL Stats API (standings, schedule, roster)
2. **Identifies the game** from team name or game ID
3. **Pulls team profiles** -- record, advanced stats (Corsi, xG), goaltender stats
4. **Computes rest/B2B** from recent schedule
5. **Runs 6-model prediction** -- log5, pythagorean, Corsi, xG, goaltender matchup, combined
6. **Compares odds** across sportsbooks
7. **Calculates expected value** and Kelly sizing if +EV
8. **Returns narrative** with key factors, matchup context, and betting recommendation

## Output structure
- `headline`: "NYR (24-30-8) vs BOS (35-20-7)"
- `home_team` / `away_team`: record, points, recent form, starter goaltender
- `prediction`: home win probability with confidence interval
- `features`: all 14 features that drove the prediction
- `rest`: B2B status, rest days for each team
- `odds`: best available odds across books
- `ev`: expected value if betting
- `sizing`: recommended stake (Kelly Criterion)
- `key_factors`: narrative bullet points

## Parameters
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `team` | string | yes* | Team abbreviation (NYR, BOS, etc.) |
| `game_id` | string | yes* | Specific game ID |

*One of `team` or `game_id` required.
