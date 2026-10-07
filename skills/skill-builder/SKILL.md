---
name: Skill builder
description: Turn a procedure (from the recent conversation or a user description) into a new saved skill.
keywords: skill, meta, build skill, save procedure, automate, repeatable
created: 2026-06-08T00:00:00
---

Help the user capture a repeatable procedure as a new skill via the save_skill
tool. Work from whatever the user gives you: a description of what they want, or
the steps you just performed earlier in this conversation.

1. **Identify the procedure.** Decide what the skill should do. If the user is
   pointing at something already done in this chat ("save what you just did"),
   review the recent tool calls and results and reconstruct the sequence. If
   they describe a new workflow, use that. If the scope is unclear, ask one
   short clarifying question before saving.

2. **Generalize it.** Write the steps as a reusable recipe, not a transcript of
   one run. Replace specific inputs (a file path, an age, a gene, a hallmark)
   with placeholders the future user will supply, and name the tools each step
   calls and what inputs they need. Keep numbered steps short and concrete.

3. **Name and describe it.** Choose a short, human-readable name (it becomes the
   /slug used to invoke it) and a one-line description. Suggest a few trigger
   keywords.

4. **Confirm, then save.** Briefly show the user the proposed name, description,
   and steps. Once they agree (or if they already said "just save it"), call
   save_skill with name, description, instructions, and keywords.

5. **Tell them how to use it.** Report the saved slug and that they can run it by
   typing /<slug>, optionally with inputs (e.g. "/my-skill data/sample.csv").

Notes:
- Good skills are self-contained and input-driven. Avoid baking in one-off
  values; pass them as arguments at run time instead.
- If saving is refused (read-only or disabled filesystem policy), tell the user
  rather than retrying.
- This skill builds *other* skills — don't try to save a copy of itself.
