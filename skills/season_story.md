# Season Story

## When to use
User wants a comprehensive narrative about a team's season -- not just numbers but context, trends, and meaning.

**Trigger phrases:**
- "Tell me the story of the Rangers' season"
- "How are the Bruins doing?"
- "Give me the full picture on Colorado"
- "What's going on with the Panthers?"
- "Break down the Leafs' season for me"
- "How's the season going for Winnipeg?"

## Workflow

```python
tk.run_workflow("season_story", team="NYR")
```

## What it does
1. **Refreshes live data** (standings + roster)
2. **Builds 9 chapters** of a season narrative:
   - **Record summary**: W-L-OTL, points, goals for/against per game
   - **Luck or skill?**: Pythagorean wins vs actual (are they overperforming?)
   - **Process (analytics)**: Corsi, xGF%, PDO, shooting %, with narrative interpretation
   - **Goaltending**: Starter and backup Sv%, GAA, GSAA with grade
   - **Special teams**: PP% and PK% with grade
   - **Roster composition**: Forward/D/G counts, injured players
   - **Cap situation**: Space, dead cap, trade flexibility
   - **Playoff outlook**: Projected points, probability, status
   - **Recent results**: Last 5-10 games with scores

## This is storytelling
The output isn't a data dump. Each chapter includes narrative interpretation:
- "Overperforming their underlying numbers -- regression risk"
- "Dominant possession team generating high-quality chances"
- "Running hot (high PDO -- potential regression)"
- "Limited flexibility, need to move money out"
- "Bubble team fighting for wildcard spot"

The agent can present this as a coherent story, not a spreadsheet.

## Parameters
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `team` | string | yes | Team abbreviation |
| `season` | string | no | Season (default: current) |
