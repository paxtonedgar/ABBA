# Tonight's Slate

## When to use
User wants the full picture of what's happening tonight across the league.

**Trigger phrases:**
- "What NHL games are on tonight?"
- "Give me the full slate"
- "What's the schedule?"
- "Break down tonight's card"
- "Any good games tonight?"

## Workflow

```python
tk.run_workflow("tonights_slate", sport="NHL")
```

## What it does
1. Refreshes live schedule data
2. For each game: runs prediction, compares odds, checks for value
3. Sorts by confidence (highest conviction picks first)
4. Flags any +EV value plays
5. Identifies the "best bet" (highest edge)

## Output
- `games`: array of matchups with predictions, odds, and picks
- `value_picks`: only the +EV opportunities
- `best_bet`: single strongest play if one exists
