---
name: my-skill-name
description: One line, written so the agent can tell from the failing checks and previews whether this applies
stages: [geometry]
version: 0.1.0
---
# Title

## When this applies
Observable symptoms: which checks fail, what the previews show, which objects it typically hits.
Also say when it does NOT apply.

## Procedure
Numbered steps that name concrete tools and parameters. Reference files in this directory by their
relative path (for example `reference.py` for run_python).

## Verification
What must hold before submit (checks, metrics that must not get worse, what to look at in previews).

## Reporting
What to state in the submit summary, especially anything synthesized.

## Pitfalls
Wrong fixes that look right.
