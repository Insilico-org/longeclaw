# Skills

User-defined, repeatable procedures for LongevityClaw. Each skill is a directory
containing a `SKILL.md` manifest:

```
skills/
  <slug>/
    SKILL.md
```

`SKILL.md` has a small frontmatter block followed by free-text instructions:

```markdown
---
name: Methylation triage
description: Run all clocks on a methylation file, then flag accelerated organs.
keywords: methylation, triage, acceleration
created: 2026-06-08T12:00:00
---

1. Call predict_age_from_file on the user's CSV.
2. For any clock with an age gap > 3 years, call interpret_individual.
3. Summarize which organs/systems are most accelerated.
```

## Creating skills

Ask the agent to save a procedure after you have run it:

> "save this as a skill called methylation triage"

The agent writes a generalized recipe (referring to the tools and inputs each
step needs), not a transcript of one specific run, so the skill works on future
inputs. The bundled **`/skill-builder`** skill walks the agent through this:
generalizing the steps, naming the skill, and saving it.

## Running skills

Type `/<slug>` in the CLI (e.g. `/methylation-triage`), optionally followed by
inputs. List saved skills with `/skills`. Skills are *prompt recipes*: replaying
one re-runs the steps against the current inputs rather than replaying a frozen
sequence of tool calls.

Saving skills requires the read/write filesystem policy (`LONGEVITYCLAW_FS=readwrite`),
so the hosted web app cannot create them.
