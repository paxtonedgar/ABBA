# Team Comparison

## When to use
User wants to compare two teams across all dimensions.

**Trigger phrases:**
- "Compare the Rangers and Bruins"
- "NYR vs BOS -- who's better?"
- "Which team is stronger?"
- "Head to head comparison"

## Workflow

```python
tk.run_workflow("team_comparison", team1="NYR", team2="BOS")
```

## What it does
1. Runs full season review for both teams
2. Compares across 6 categories: record, goal diff, Corsi, xGF%, PP%, PK%
3. Counts category wins
4. Returns side-by-side data

## Output
- `team1` / `team2`: records and points
- `categories`: each metric with both teams' values
- `category_wins`: which team wins more categories
- `review1` / `review2`: full season reviews for context
