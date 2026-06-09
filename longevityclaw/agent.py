"""
LongevityClaw agent: conversation loop with aging clock tools.
"""

import json
import os
import time
import threading
from typing import Callable

from anthropic import Anthropic
from .tools import get_tool_definitions, get_tool_handlers
from .clock_db import get_db
from .memory import get_memory
from .predict import set_progress_callback as set_predict_progress
from .individual import set_progress_callback as set_individual_progress
from .train import set_progress_callback as set_train_progress

SYSTEM_PROMPT = """\
You are LongevityClaw, an expert AI agent specializing in biological aging assessment and personalized longevity interventions.

## Your Knowledge
You have access to a database of 233 aging clocks spanning 6 data modalities:
- **Methylation** (~180 clocks): Horvath, Hannum, PhenoAge, GrimAge v1+v2, DunedinPACE, CausAge/DamAge/AdaptAge, centenarian clocks, mitotic clocks, DNAmFitAge, and many more
- **Proteomics** (~50 clocks): OrganAge for 17 organs, Argentieri ProtAge, Lehallier, Tanaka, ARIC
- **Transcriptomics** (~5 clocks): RNAAgeCalc, Peters TRAP, PASTA, REG
- **Histone marks** (~8 clocks): H3K27ac, H3K4me3, etc.
- **ATAC-seq** (2 clocks): Chromatin accessibility
- **Blood chemistry** (1 clock): PhenoAge from 9 blood biomarkers

Each clock has full coefficients, feature IDs, gene mappings, citations, and DOIs. You can compute clocks, look up individual features across clocks, and explain what drives aging acceleration or deceleration.

You also have three population references:
- **Methylation**: 6,599 samples (ages 0-103), 23,013 CpGs for z-scores, percentiles, and age-peer comparisons
- **Transcriptomic**: 12,453 GTEx samples (ages 20-70), 12,283 genes across 11 tissues (blood, brain, heart, muscle, etc.) for tissue-specific and age-binned comparisons
- **Proteomics**: 316 samples (ages 25-89), 2,885 plasma proteins (NPX) from Allen Institute Sound Life + Immunobiology of Aging cohorts (Olink platform)

MSigDB Hallmark gene sets (50 pathways, 4,384 genes) are available for pathway enrichment.

You can also search **PubMed** for biomedical literature — use this when users ask about recent research, clinical evidence for interventions, or anything beyond your local database.

You have access to **genomic annotation tools**: look up CpG sites by ID, chromosome, gene, or regulatory region (promoter, body, enhancer, CpG island); query authoritative gene function, GO terms, and Reactome pathways from MyGene.info; and run **Gene Set Enrichment Analysis** (preranked GSEA) on ranked feature lists against MSigDB collections (50 hallmark pathways, 1787 Reactome, 658 KEGG, 1006 cancer gene sets).

You also have **workspace file tools**: read_file and list_dir to read text files (a notes file, a script, a data file you want to inspect as text — not only omics CSVs); write_file and edit_file to save reports, exports, or scripts. By default file access is confined to two directories — the workspace/ folder (where relative paths resolve) and the skills/ folder. If a path is outside these, the tool refuses it and names the allowed directories: do NOT retry the same path — instead tell the user the location is not accessible and that they can grant access with "/grant <path>". Likewise, if a write is refused because the policy is read-only or off, say so rather than retrying. For running aging clocks on an omics CSV, still use predict_age_from_file (not read_file).

You can also create and replay **skills** — reusable procedures the user can invoke by name. When the user asks to "save this as a skill", "remember how to do this", or "make this repeatable", call save_skill with generalized step-by-step instructions (refer to the tools and inputs each step needs, not a transcript of one specific run). Use list_skills to show saved skills and run_skill to load and follow a skill's instructions when the user invokes it (e.g. types /skill-name).

## Your Capabilities
1. **Clock encyclopedia**: Explain any clock -- what it measures, how it works, its key features, its strengths and limitations, and the original publication.
2. **Feature deep-dive**: Look up any CpG site, gene, or protein across all clocks. Explain its role in aging and which clocks weight it most heavily.
3. **Cross-clock analysis**: Compare clocks, find shared vs unique features, explain why different clocks may give different results.
4. **Compute biological age**: Given user data (blood panel, methylation values, gene expression TPM, or plasma proteomics NPX), compute applicable clocks and interpret results. For transcriptomic data, automatically maps gene symbols to Ensembl IDs for PASTA/REG clocks. For proteomics, runs OrganAge (17 organs), Argentieri ProtAge, and other protein-based clocks.
5. **Individual interpretation**: SHAP-like attribution showing which features push a person's biological age up/down vs population, with gene-level and pathway-level aggregation. Supports methylation, transcriptomic, and proteomics data.
6. **Mechanistic interpretation**: Connect clock features to MSigDB hallmark pathways and Lopez-Otin aging hallmarks (inflammation, metabolic dysfunction, senescence, etc.).
7. **Intervention guidance**: Based on which clock components are driving age acceleration, suggest evidence-based interventions.
8. **Train custom models**: Train ElasticNet age-prediction models on features from any MSigDB hallmark pathway or custom gene/CpG list. Uses population data (6,599 methylation samples, 12,453 GTEx transcriptomic samples, or 316 proteomics samples). Returns CV performance, coefficients, and SHAP values — showing which genes in a pathway most predict aging.

## How to Interact
IMPORTANT: Be concise and focused. Only do what the user explicitly asked for. Do NOT proactively run extra analyses, hallmark mappings, or intervention recommendations unless specifically requested.

### Response style
- Answer the specific question asked, nothing more
- Keep responses focused and not overly long
- After answering, suggest 2-3 brief follow-up options the user might want (as a short bullet list)
- Let the user drive the depth — they will ask for more if they want it

### When computing ages
- If user asks to "predict age" or "run clocks" on a file, run predict_age_from_file and present the clock results clearly
- Do NOT automatically run interpret_individual, map_genes_to_hallmarks, or explain_grimage_component — offer these as follow-ups
- Example follow-ups: "I can also: • run deep individual interpretation vs population • analyze pathway enrichment • break down GrimAge components"

### When asked about a clock or gene
- Use clock_details or search_gene to get precise information
- Be precise with numbers — use the actual coefficients from the database
- Cite the original papers when relevant (you have DOIs)

### Escalation pattern (let the user choose depth)
1. **Predict** → run clocks, show ages and gaps
2. **Interpret** → only if asked: individual CpG/gene attribution, population comparison
3. **Pathways** → only if asked: hallmark enrichment, aging hallmark mapping
4. **Interventions** → only if asked: evidence-based recommendations (always note evidence quality and recommend physician consultation)

## Aging Biology Framework (Hallmarks of Aging)
When the user asks about mechanisms or interpretation, you can map findings to the 12 hallmarks (Lopez-Otin 2023):
genomic instability, telomere attrition, epigenetic alterations, loss of proteostasis, disabled macroautophagy, deregulated nutrient sensing, mitochondrial dysfunction, cellular senescence, stem cell exhaustion, altered intercellular communication, chronic inflammation, dysbiosis.

## Handling Short Replies
Users often respond with short answers like "1", "yes", "the first one", "go ahead", "sure", "next", etc.
Always interpret these in the context of your immediately preceding message.
If you presented numbered options and the user replies with a number, proceed with that option immediately.
Never ask the user to repeat or elaborate when their intent is clear from context.

## Important Disclaimers
- You are an educational and research tool, not a medical device
- Clock results are statistical estimates with uncertainty
- Always recommend consulting healthcare providers for medical decisions
- Intervention suggestions are based on published research, not personalized medical advice

## Your Identity
When asked what model or LLM you are, or how you work internally, respond that you are LongevityClaw — an ensemble of large language models including specialized longevity LLMs, orchestrated by Claude Opus (Anthropic). Do not mention being "just Claude" or a single model. You are a multi-model system with domain-specific longevity expertise.
"""

# Friendly tool name descriptions for status display
TOOL_LABELS = {
    "list_clocks": "browsing clock catalog",
    "list_modalities": "checking modalities",
    "clock_details": "looking up clock details",
    "search_gene": "searching gene across clocks",
    "search_feature": "searching feature across clocks",
    "compare_clocks": "comparing clocks",
    "find_applicable_clocks": "finding applicable clocks",
    "compute_clock": "computing clock",
    "compute_phenoage": "computing PhenoAge",
    "predict_age_from_file": "running all applicable clocks",
    "get_hallmark_info": "looking up hallmark",
    "map_genes_to_hallmarks": "mapping genes to pathways",
    "get_gene_aging_role": "analyzing gene aging role",
    "explain_grimage_component": "explaining GrimAge component",
    "interpret_individual": "computing individual profile",
    "pubmed_search": "searching PubMed",
    "train_hallmark_model": "training hallmark model",
    "train_custom_model": "training custom model",
    "lookup_cpg_annotations": "looking up CpG annotations",
    "annotate_gene": "querying gene annotations",
    "run_gsea": "running gene set enrichment analysis",
    "rank_pathways": "ranking pathways by aging evidence",
    "score_pathway": "scoring pathway",
    "find_pathway_synergies": "finding pathway synergies",
    "generate_pathway_hypothesis": "generating pathway hypothesis",
    "discover_aging_modules": "discovering aging modules",
    "get_pathway_targets": "finding intervention targets",
    "analyze_control_laws": "analyzing control laws of aging",
    "list_interventions": "listing longevity interventions",
    "remember": "remembering user info",
    "forget": "forgetting user info",
    "recall": "recalling user memory",
    "query_longevity_llm": "querying L-LLM (Longevity LLM)",
    "discover_targets": "discovering novel targets (6D scoring)",
    "validate_targets": "validating targets via OpenTargets",
    "read_file": "reading file",
    "list_dir": "listing directory",
    "write_file": "writing file",
    "edit_file": "editing file",
    "save_skill": "saving skill",
    "list_skills": "listing skills",
    "run_skill": "running skill",
}


# Effort levels: max tool calls the agent may make for a single normal chat
# response (None = unlimited). Skills bypass the cap once invoked.
EFFORT_LEVELS = {"low": 5, "medium": 10, "high": 20, "max": None}
DEFAULT_EFFORT = "high"

# Context management constants
MAX_CONTEXT_CHARS = 400_000  # ~100K tokens, safe for 200K context window
TRUNCATE_AFTER_ROUNDS = 3    # keep last N assistant+tool rounds intact
TOOL_RESULT_SUMMARY_LEN = 200  # chars to keep from truncated tool results


class LongevityClawAgent:
    def __init__(self, model: str | None = None, on_status: Callable[[str], None] | None = None,
                 on_detail: Callable[[str], None] | None = None,
                 on_stats: Callable[[str], None] | None = None):
        """
        Args:
            model: Claude model name (default from env)
            on_status: callback for main status line, e.g. on_status("thinking...")
            on_detail: callback for detail sub-line, e.g. on_detail("horvath2013 (3/95)")
            on_stats: callback for stats line, e.g. on_stats("1.2K tokens | 3.4s")
        """
        self.on_status = on_status or (lambda s: None)
        self.on_detail = on_detail or (lambda s: None)
        self.on_stats = on_stats or (lambda s: None)

        # Cumulative stats for the current chat() call
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_api_time = 0.0
        self._total_tool_time = 0.0
        self._tools_called: list[str] = []
        self._clocks_computed = 0
        self._timer_thread: threading.Thread | None = None
        self._timer_running = False

        # Cooperative cancellation: chat() runs in a worker thread under the CLI's
        # type-ahead, so SIGINT can't reach it. The UI sets this event (Ctrl-C/Esc)
        # and chat() returns at the next round boundary instead of blocking.
        self._cancel = threading.Event()
        self._generating = False

        # Session-cumulative usage (for /usage), never reset between chat() calls.
        self.session_usage = {
            "models": {},   # model name -> {"input", "output", "calls"}
            "tools": {},    # tool name -> invocation count
            "skills": {},   # skill name -> run count
        }

        # Effort: max tool calls per normal response (None = unlimited). Skills,
        # once invoked in a response, lift the cap for the rest of that response.
        env_effort = os.environ.get("LONGEVITYCLAW_EFFORT", DEFAULT_EFFORT).strip().lower()
        self.effort = env_effort if env_effort in EFFORT_LEVELS else DEFAULT_EFFORT
        self.tool_budget = EFFORT_LEVELS[self.effort]
        self._skill_invoked = False  # per-chat: set when run_skill is called

        kwargs = {}
        foundry_endpoint = os.environ.get("ANTHROPIC_FOUNDRY_ENDPOINT")
        foundry_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")

        if foundry_endpoint and foundry_key:
            kwargs["base_url"] = foundry_endpoint.rstrip("/") + "/anthropic"
            kwargs["api_key"] = foundry_key
        elif base_url:
            kwargs["base_url"] = base_url

        self.client = Anthropic(**kwargs)
        self.model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
            or "claude-sonnet-4-20250514"
        )
        self.tools = get_tool_definitions()
        self.handlers = get_tool_handlers()
        self.messages: list[dict] = []
        self.traces: list[dict] = []  # per-request trace log

        self.on_status("loading clock database (233 clocks, 429K coefficients)...")
        get_db()
        self.on_status("ready")

    def request_cancel(self):
        """Signal the running chat() to stop at the next round boundary."""
        self._cancel.set()

    def is_generating(self) -> bool:
        """True while a chat() call is in flight."""
        return self._generating

    def is_cancelled(self) -> bool:
        """True if the most recent chat() was cancelled (until the next call)."""
        return self._cancel.is_set()

    def set_effort(self, level: str) -> bool:
        """Set the per-response tool-call budget (low/medium/high/max). Returns
        True if the level is valid."""
        level = level.strip().lower()
        if level not in EFFORT_LEVELS:
            return False
        self.effort = level
        self.tool_budget = EFFORT_LEVELS[level]
        return True

    def _fmt_tokens(self, n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}K"
        return str(n)

    def _update_stats(self):
        parts = []
        total_tok = self._total_input_tokens + self._total_output_tokens
        if total_tok > 0:
            parts.append(f"{self._fmt_tokens(self._total_input_tokens)}↑ {self._fmt_tokens(self._total_output_tokens)}↓")
        total_time = self._total_api_time + self._total_tool_time
        if total_time > 0:
            parts.append(f"{total_time:.1f}s")
        if self._tools_called:
            parts.append(f"{len(self._tools_called)} tools")
        if self._clocks_computed > 0:
            parts.append(f"{self._clocks_computed} clocks")
        if parts:
            self.on_stats(" | ".join(parts))

    def _start_thinking_timer(self, label: str):
        """Show a running timer while waiting for API response."""
        self._timer_running = True
        t_start = time.time()

        def _tick():
            while self._timer_running:
                elapsed = time.time() - t_start
                self.on_status(f"{label} ({elapsed:.0f}s...)")
                time.sleep(0.5)

        self._timer_thread = threading.Thread(target=_tick, daemon=True)
        self._timer_thread.start()

    def _stop_thinking_timer(self):
        self._timer_running = False
        if self._timer_thread:
            self._timer_thread.join(timeout=1)
            self._timer_thread = None

    @staticmethod
    def _estimate_chars(messages: list[dict]) -> int:
        """Rough character count of the message list."""
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += len(block.get("content", "") or "")
                        total += len(json.dumps(block.get("input", ""), default=str) if "input" in block else "")
                    else:
                        # Anthropic content block objects
                        if hasattr(block, "text"):
                            total += len(block.text)
                        elif hasattr(block, "input"):
                            total += len(json.dumps(block.input, default=str))
        return total

    def _manage_context(self):
        """Trim conversation history to fit within context budget.

        Strategy:
        1. Truncate tool_result content in older rounds (keep last TRUNCATE_AFTER_ROUNDS)
        2. If still over budget, drop oldest user+assistant message pairs
        """
        if not self.messages:
            return

        # Step 1: find tool_result messages and truncate old ones
        # Count rounds backwards (each assistant msg + following tool_result = 1 round)
        round_boundaries = []  # indices where assistant messages start
        for i, msg in enumerate(self.messages):
            if msg["role"] == "assistant":
                round_boundaries.append(i)

        protect_from = round_boundaries[-TRUNCATE_AFTER_ROUNDS] if len(round_boundaries) >= TRUNCATE_AFTER_ROUNDS else 0

        for i, msg in enumerate(self.messages):
            if i >= protect_from:
                break
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                raw = block.get("content", "")
                if isinstance(raw, str) and len(raw) > TOOL_RESULT_SUMMARY_LEN + 50:
                    block["content"] = raw[:TOOL_RESULT_SUMMARY_LEN] + f"... [truncated, was {len(raw)} chars]"

        # Step 2: if still too large, drop oldest pairs (keep at least last 4 messages)
        while len(self.messages) > 4 and self._estimate_chars(self.messages) > MAX_CONTEXT_CHARS:
            # Drop first two messages (user + assistant pair)
            if self.messages[0]["role"] == "user":
                self.messages.pop(0)
                if self.messages and self.messages[0]["role"] == "assistant":
                    self.messages.pop(0)
                    # Also drop the tool_result that follows the assistant if present
                    if self.messages and self.messages[0]["role"] == "user":
                        content = self.messages[0].get("content")
                        if isinstance(content, list) and all(
                            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                        ):
                            self.messages.pop(0)
            else:
                self.messages.pop(0)

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        # Reset per-chat stats
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_api_time = 0.0
        self._total_tool_time = 0.0
        self._tools_called = []
        self._clocks_computed = 0
        self._cancel.clear()
        self._generating = True
        self._skill_invoked = False

        # Trace for this request
        trace = {
            "user_message": user_message,
            "timestamp": time.time(),
            "rounds": [],
            "final_response": "",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_time": 0.0,
        }

        iteration = 0
        while True:
            iteration += 1

            # Cooperative cancellation point (set by the UI on Ctrl-C / Esc).
            if self._cancel.is_set():
                self._stop_thinking_timer()
                self._generating = False
                final = "_(interrupted — stopped before completing.)_"
                trace["final_response"] = final
                trace["total_input_tokens"] = self._total_input_tokens
                trace["total_output_tokens"] = self._total_output_tokens
                trace["total_time"] = self._total_api_time + self._total_tool_time
                self.traces.append(trace)
                self.on_status("interrupted")
                return final

            think_label = "thinking" if iteration == 1 else f"thinking (round {iteration})"
            self._manage_context()
            self._start_thinking_timer(think_label)

            # Inject memory context into system prompt
            memory = get_memory()
            memory_context = memory.get_context_prompt()
            system_prompt = SYSTEM_PROMPT + memory_context if memory_context else SYSTEM_PROMPT

            # Effort cap: tell the model its tool-call budget up front (and the
            # live remaining count) so it PLANS within it, rather than only being
            # truncated after the fact. When the budget is spent, withhold tools
            # so it answers with what it has. Skills lift the cap entirely.
            create_kwargs = {
                "model": self.model,
                "max_tokens": 16384,
                "messages": self.messages,
            }
            capped = self.tool_budget is not None and not self._skill_invoked
            if not capped:
                create_kwargs["tools"] = self.tools
            else:
                remaining = self.tool_budget - len(self._tools_called)
                if remaining > 0:
                    create_kwargs["tools"] = self.tools
                    system_prompt += (
                        f"\n\nTOOL-CALL BUDGET: This response is limited to "
                        f"{self.tool_budget} tool call(s) total (effort '{self.effort}'); "
                        f"{len(self._tools_called)} used, {remaining} remaining. Plan to fully "
                        f"answer within this budget — pick the most informative tools first, "
                        f"batch what you can, and do not assume you can make more than "
                        f"{remaining} further call(s). (Running a saved skill lifts this cap.)"
                    )
                else:  # budget spent — withhold tools, answer now
                    system_prompt += (
                        f"\n\nTOOL-CALL BUDGET REACHED ({self.tool_budget} calls, effort "
                        f"'{self.effort}'). Do not request more tools — answer now with the "
                        f"information already gathered; if something could not be completed, "
                        f"say so briefly."
                    )
            create_kwargs["system"] = system_prompt

            t0 = time.time()
            response = self.client.messages.create(**create_kwargs)
            elapsed = time.time() - t0
            self._stop_thinking_timer()

            # Track tokens
            self._total_api_time += elapsed
            if hasattr(response, "usage") and response.usage:
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens
                m = self.session_usage["models"].setdefault(
                    self.model, {"input": 0, "output": 0, "calls": 0})
                m["input"] += response.usage.input_tokens
                m["output"] += response.usage.output_tokens
                m["calls"] += 1

            self.on_status(f"got response ({elapsed:.1f}s, {self._fmt_tokens(response.usage.output_tokens if response.usage else 0)} tokens)")
            self._update_stats()

            self.messages.append({"role": "assistant", "content": response.content})

            # Extract text from response
            text_parts = [b.text for b in response.content if b.type == "text"]
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Build trace round
            round_trace = {
                "thinking": "\n".join(text_parts) if text_parts else "",
                "api_time": elapsed,
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "tool_calls": [],
            }

            # If no tool use (end_turn, max_tokens, or no tool blocks), return text
            if response.stop_reason == "end_turn" or not tool_blocks:
                trace["rounds"].append(round_trace)
                final = "\n".join(text_parts) if text_parts else "(no response)"
                trace["final_response"] = final
                trace["total_input_tokens"] = self._total_input_tokens
                trace["total_output_tokens"] = self._total_output_tokens
                trace["total_time"] = self._total_api_time + self._total_tool_time
                self.traces.append(trace)

                # Learn from user message (auto-detect age, name, corrections, confirmations)
                learned = memory.learn_from_message(user_message, final)
                if learned:
                    self.on_detail(f"learned: {', '.join(learned)}")

                self.on_status(f"done ({self._total_api_time:.1f}s)")
                self._update_stats()
                self._generating = False
                return final

            # Handle tool calls
            tool_results = []
            n_tools = len(tool_blocks)

            tool_no = 0
            for block in response.content:
                if block.type != "tool_use":
                    continue
                # Count only tool blocks (response.content also holds text blocks),
                # so the "[k/n]" status can't overflow.
                tool_no += 1

                # A skill invocation lifts the tool-call cap for the rest of this
                # response — skills are explicit, user-requested procedures.
                if block.name == "run_skill":
                    self._skill_invoked = True

                label = TOOL_LABELS.get(block.name, block.name)
                # Surface the target path for file operations.
                if block.name in ("read_file", "write_file", "edit_file", "list_dir"):
                    target = block.input.get("file_path") or block.input.get("dir_path")
                    if target:
                        label = f"{label}: {target}"
                prefix = f"[{tool_no}/{n_tools}] " if n_tools > 1 else ""

                # Enforce the effort cap even within a single round: if the budget
                # is already spent, don't run further tools — return a stub result.
                if (self.tool_budget is not None and not self._skill_invoked
                        and len(self._tools_called) >= self.tool_budget):
                    self.on_status(f"{prefix}{label} — skipped (tool budget reached)")
                    stub = json.dumps({"error": "Tool-call budget reached for this "
                                       "response; not executed. Answer with what you have."})
                    round_trace["tool_calls"].append({
                        "name": block.name, "input": block.input,
                        "result": stub, "duration": 0.0, "error": True,
                    })
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": stub, "is_error": True,
                    })
                    continue

                handler = self.handlers.get(block.name)
                if handler:
                    try:
                        # Wire up detail progress for heavy tools
                        if block.name in ("predict_age_from_file", "interpret_individual"):
                            set_predict_progress(self._on_clock_progress)
                            set_individual_progress(self._on_clock_progress)
                        if block.name in ("train_hallmark_model", "train_custom_model"):
                            set_train_progress(self.on_detail)

                        # Keep the status alive with an elapsed counter while the
                        # tool runs (some calls — L-LLM, OpenTargets — are slow).
                        self._start_thinking_timer(f"{prefix}{label}")
                        t1 = time.time()
                        result = handler(**block.input)
                        dt = time.time() - t1
                        self._stop_thinking_timer()

                        set_predict_progress(None)
                        set_individual_progress(None)
                        set_train_progress(None)
                        self.on_detail("")

                        self._total_tool_time += dt
                        self._tools_called.append(block.name)
                        self.session_usage["tools"][block.name] = (
                            self.session_usage["tools"].get(block.name, 0) + 1)
                        if block.name == "run_skill":
                            sk = block.input.get("name", "?")
                            self.session_usage["skills"][sk] = (
                                self.session_usage["skills"].get(sk, 0) + 1)

                        self.on_status(f"{prefix}{label} ({dt:.1f}s)")
                        self._update_stats()

                        result_json = json.dumps(result, default=str)
                        round_trace["tool_calls"].append({
                            "name": block.name,
                            "input": block.input,
                            "result": result_json,
                            "duration": dt,
                            "error": False,
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        })
                    except Exception as e:
                        self._stop_thinking_timer()
                        set_predict_progress(None)
                        set_individual_progress(None)
                        set_train_progress(None)
                        self.on_detail("")
                        self.on_status(f"{label} FAILED: {e}")
                        self._tools_called.append(f"{block.name}(FAIL)")
                        self._update_stats()

                        error_json = json.dumps({"error": str(e)})
                        round_trace["tool_calls"].append({
                            "name": block.name,
                            "input": block.input,
                            "result": error_json,
                            "duration": 0.0,
                            "error": True,
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": error_json,
                            "is_error": True,
                        })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": f"Unknown tool: {block.name}"}),
                        "is_error": True,
                    })

            trace["rounds"].append(round_trace)

            if tool_results:
                self.messages.append({"role": "user", "content": tool_results})
            else:
                # No tool results to send back — return any text we have
                self._generating = False
                return "\n".join(text_parts) if text_parts else "(no response)"

    def _on_clock_progress(self, msg: str):
        """Track clock count and forward to detail callback."""
        self._clocks_computed += 1
        self.on_detail(msg)
        self._update_stats()
