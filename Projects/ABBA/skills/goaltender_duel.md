# Goaltender Duel

## When to use
User wants to compare two goaltenders.

**Trigger phrases:**
- "Compare Shesterkin vs Bobrovsky"
- "Who's the better goalie?"
- "Swayman or Oettinger?"
- "Goalie comparison"

## Workflow

```python
tk.run_workflow("goaltender_duel", goalie1_team="NYR", goalie2_team="FLA")
```

## What it does
1. Pulls both goaltenders' full stat profiles
2. Compares Sv%, GAA, GSAA, xGSAA, quality starts
3. Calculates matchup edge factor
4. Counts category wins
5. Delivers a verdict

## Output
- `goalie1` / `goalie2`: full stat profiles
- `matchup_edge`: quantified edge (-1 to 1)
- `categories_won`: who wins each metric
- `verdict`: "Shesterkin has the edge" / "Too close to call"
