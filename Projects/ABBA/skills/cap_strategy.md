# Cap Strategy

## When to use
User asks about salary cap, trade deadline, or roster construction.

**Trigger phrases:**
- "Can the Rangers afford a trade?"
- "What's the Panthers' cap situation?"
- "Who should they trade at the deadline?"
- "How cap-strapped are the Leafs?"
- "What contracts are expiring?"

## Workflow

```python
tk.run_workflow("cap_strategy", team="NYR")
```

## What it does
1. Pulls cap data and roster
2. Determines buyer vs seller mode (based on playoff probability)
3. Identifies trade chips (expiring contracts, moveable pieces)
4. Reports top contracts, dead cap, LTIR relief
5. Assesses trade flexibility

## Output
- `mode`: buyer or seller
- `cap_space` / `effective_space`
- `trade_chips`: moveable contracts
- `top_contracts`: biggest cap hits
- `strategy_summary`: one-sentence assessment
