# Playoff Race

## When to use
User asks about playoff standings, chances, or the conference race.

**Trigger phrases:**
- "Will the Rangers make the playoffs?"
- "How tight is the Eastern Conference?"
- "What does the playoff picture look like?"
- "Who's on the bubble?"
- "Are the Sabres eliminated?"

## Workflow

```python
tk.run_workflow("playoff_race", team="NYR")
tk.run_workflow("playoff_race", conference="Eastern")
```

## What it does
1. Fetches all standings
2. Runs Monte Carlo playoff simulation for each team
3. Identifies bubble teams
4. Reports projected points and clinch/elimination status

## Output
- `standings`: sorted by points with playoff probabilities
- `bubble_teams`: teams within 10 pts of cutline
- `team_focus`: detail for specific team if requested
