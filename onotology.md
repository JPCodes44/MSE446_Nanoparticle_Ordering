# Function Ontology

This file is the agent-facing index of reusable code in this repository. The pre-hook prints this file at the start of an agent run so existing functions can be reused before new code is written.

## How To Use

- Check this file before creating new helpers, feature extractors, parsers, model utilities, or plotting code.
- Reuse listed functions when they match the current task.
- Run `python scripts/hooks/post_agent.py` after code changes to refresh the generated function index.

## Function Index

No Python functions are currently indexed. Add Python modules under `src/`, `tests/`, or another repository code directory, then run the post-hook to populate this section.
