---
name: Find targets
description: Discover novel druggable aging targets for a hallmark, validate against OpenTargets, run safety deep-dive via PubMed, revisit L-LLM scores with external evidence, assess commercial viability, and report a safety- and commercially-adjusted ranked shortlist.
keywords: targets, drug discovery, novel targets, druggable, commercial, sarcopenia, aging targets, hallmarks
created: 2026-06-08T20:07:30
---

Run an end-to-end novel-target-discovery pass for the aging hallmark(s) the user
names (default: all 14). Carry out the steps in order, reusing the conversation's
context for any inputs the user already supplied.

1. **Scope the run.** Determine which hallmark(s) to analyze from the user's
   request (e.g. "inflammation", "cellular_senescence"). If the user specifies
   a disease/condition instead (e.g. "sarcopenia", "Alzheimer's"), map it to
   the most relevant Lopez-Otin hallmarks (2-4 hallmarks). If none is given,
   run all 14. Note how many consensus runs the user wants (default 1).

2. **Generate candidates.** Call `discover_targets` with the chosen
   `hallmarks` and `num_runs`. This returns genes scored on 6 dimensions
   (Novelty, Druggability, Confidence, Safety, Commercial, Mechanism, each 0-100).
   Record the original L-LLM-derived 6D scores for every candidate — these will
   be revisited in Step 6.

3. **Filter to novel + druggable.** Keep targets meeting the standard bar:
   Novelty >= 76 AND Druggability >= 51. Take the top ~10-12 by combined
   Novelty + Druggability for the next step.

4. **Validate novelty externally.** Call `validate_targets` with that gene list
   to cross-check against OpenTargets: novelty_tier, novelty_score, known_drugs
   count, and max clinical phase. Flag any candidate that is actually already
   well-developed (max clinical phase >= 2 or tier "established"/"well_known") —
   those are less novel than the 6D score suggested.

5. **Safety deep-dive (MANDATORY).** For EVERY candidate that passes steps 3-4,
   run targeted PubMed searches to assess safety risks. For each gene, search
   for at minimum these query patterns (run in parallel where possible):
   
   a. `"<GENE> tumor suppressor oncogene cancer predisposition syndrome"`
   b. `"<GENE> knockout mouse toxicity lethality phenotype"`
   c. `"<GENE> haploinsufficiency dosage sensitivity"`  (optional if time permits)
   d. `"<GENE> gain of function overexpression adverse effects"`  (optional if time permits)
   
   From the results, classify each target into a safety tier:
   
   - 🟢 **Clean** (Safety 60-100): No known tumor suppressor/oncogene role,
     no familial syndromes, no embryonic lethality in KO models.
   - 🟡 **Caution** (Safety 40-59): Some cancer associations or context-dependent
     roles, but not a primary tumor suppressor. May need tissue-specific delivery.
   - 🔴 **High risk** (Safety <40): Known tumor suppressor or oncogene,
     haploinsufficient, familial cancer syndrome, or embryonic lethal KO.
     Demote in final ranking and flag prominently.

6. **Revisit L-LLM scores (MANDATORY).** After gathering external evidence from
   Steps 4 and 5, re-evaluate the original 6D scores from Step 2. For each
   surviving candidate, compare what the L-LLM predicted vs what PubMed and
   OpenTargets actually show. Adjust scores where evidence contradicts the
   original assessment:
   
   - **Safety score:** Increase if PubMed found no adverse evidence and gene
     has clean KO phenotype. Decrease if oncogene/tumor-suppressor/lethal-KO
     was found. Use the safety tier from Step 5 as ground truth.
   - **Novelty score:** Decrease if OpenTargets revealed existing drugs in
     clinical development (phase >= 2) or "established"/"well_known" tier.
     Increase if OpenTargets confirms 0 drugs and high novelty score.
   - **Druggability score:** Adjust based on PubMed evidence of existing tool
     compounds, solved crystal structures, or druggable protein family membership.
   - **Commercial score:** Will be fully assessed in Step 7, but pre-adjust
     if OpenTargets reveals a crowded competitive landscape.
   - **Confidence score:** Increase if multiple PubMed sources corroborate the
     mechanism. Decrease if mechanism is speculative with zero publications.
   - **Mechanism score:** Adjust if PubMed reveals the gene's actual biological
     role differs from what the L-LLM assumed.
   
   Present a brief "Score Revision Table" showing: Gene | Dimension | Original
   Score | Revised Score | Reason. Only show rows where scores changed.
   Eliminate any candidate whose revised scores no longer pass the filter bar
   (Novelty >= 76 AND Druggability >= 51) or that fell to 🔴 safety tier.

7. **Commercial viability assessment (MANDATORY).** For ALL candidates that
   survived Step 6, evaluate commercial potential. Run PubMed searches as needed
   for market intelligence (e.g. disease burden, competing therapies, existing
   drug platforms for the target's protein family). Assess each target across
   these commercial dimensions:
   
   a. **Market expansion** — Can one drug serve multiple indications beyond the
      primary? More indications = higher peak revenue. Score 1-5 stars.
   b. **Existing validation/investor readiness** — Is the biological pathway
      well-understood enough that investors/pharma will fund it without years
      of basic research? Score 1-5 stars.
   c. **Pharma interest / first-mover advantage** — Are there 0 drugs despite
      high disease association? That's a massive opportunity. Score 1-5 stars.
   d. **Platform potential** — Can this target anchor a platform company
      (multiple drugs, companion diagnostics)? Score 1-5 stars.
   e. **Regulatory path** — Is there a clear FDA indication to enter with,
      and can sarcopenia/aging be a label extension? Score 1-5 stars.
   f. **IP landscape** — Is the space clean (no blocking patents)? Score 1-5 stars.
   g. **Druggability chemistry** — Is it easier to make a drug for this target
      (inhibitor preferred over activator, existing chemical matter)? Score 1-5 stars.
   h. **Development risk** — Overall risk of failure. Lower = better. Score 1-5 stars.
   
   For each target, provide a 💡 **Recommended commercial strategy** — one paragraph
   outlining the optimal development path (which indication first, partnership
   strategy, peak revenue estimate if possible).
   
   Rank all targets by overall commercial viability. This ranking may differ from
   the science ranking — call that out explicitly.

8. **Final report.** Present two outputs:
   
   A. **Safety-adjusted ranked table** (science-first): Gene | Hallmark |
      Novelty (revised) | Druggability (revised) | Safety (revised) | Safety Tier |
      OT novelty | one-line rationale. Sort by: Safety tier first (🟢 > 🟡 > 🔴),
      then by combined Novelty + Druggability within each tier.
   
   B. **Commercial ranking table**: Gene | Market Expansion | Validation |
      First-Mover | Platform | Regulatory | IP | Chemistry | Risk | Overall
      Commercial Score | Recommended strategy (one-liner).
   
   C. **Score Revision Summary**: The table from Step 6 showing which L-LLM
      scores were adjusted and why.
   
   D. **Bottom-line recommendation**: Which target(s) to pursue, noting whether
      the best scientific target differs from the best commercial target, and
      suggesting whether a portfolio approach (multiple targets) is warranted.
   
   Clearly flag any demoted or eliminated targets and explain why.

9. **Offer follow-ups**, e.g.: deep-dive a specific target's safety further,
   search for druggability approaches (crystal structures, tool compounds),
   explore combination strategies, check aging clock evidence for top targets,
   write a formal report with `write_file`, or analyze interventions with
   `analyze_control_laws`.
