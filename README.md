# LongevityClaw

An AI agent that predicts your biological age across 233 clocks, explains why you're aging faster or slower, and helps you investigate the mechanisms.

![LongevityClaw Demo](demo.gif)

Today's aging clocks exist in silos. A person might get an epigenetic age from a methylation array, a blood-test-based age from routine labs, and a proteomic age from a plasma panel, but nobody connects the dots. LongevityClaw accepts any data the user has, runs every applicable clock, synthesizes results across modalities, explains the mechanisms driving age acceleration or deceleration, and recommends personalized interventions grounded in evidence.

## Quick Start

```bash
uv sync
cp .env.example .env   # edit with your API key
uv run longevityclaw
```

## Clock Database

| Modality | Clocks | Features | Examples |
|----------|--------|----------|---------|
| Methylation | ~180 | 379K CpGs | Horvath, Hannum, PhenoAge, GrimAge v1+v2, DunedinPACE, CausAge/DamAge/AdaptAge, centenarian, mitotic, DNAmFitAge |
| Proteomics | ~97 | 11K proteins | OrganAge (17 organs), Argentieri ProtAge, Lehallier, Tanaka, ARIC |
| Transcriptomics | ~5 | 29K genes | RNAAgeCalc, Peters TRAP, PASTA, REG |
| Histone marks | ~8 | 9K peaks | H3K27ac, H3K4me3, H3K27me3, H3K36me3 |
| ATAC-seq | 2 | 610 regions | Chromatin accessibility clocks |
| Blood chemistry | 1 | 9 biomarkers | PhenoAge from standard CBC + metabolic panel |

Total: 233 clocks, 429,165 coefficients, 6 modalities. All clocks have full coefficients, feature IDs, gene mappings, citations, and DOIs.

## Population References

Three population-scale datasets for individual interpretation (z-scores, percentiles, age-peer comparison) and on-the-fly model training:

| Reference | Samples | Features | Age Range | Source |
|-----------|---------|----------|-----------|--------|
| Methylation | 6,599 | 23,013 CpGs | 0--103 | GEO methylation arrays |
| Transcriptomics | 12,453 | 12,283 genes | 20--70 | GTEx v8 (11 tissues) |
| Proteomics | 316 | 2,885 proteins | 25--89 | Allen Institute (Olink) |

## What It Does

**Multi-clock age prediction.** Give it a file (methylation betas, gene expression TPMs, or plasma protein NPX values) and LongevityClaw runs every clock that has sufficient feature coverage, returning predicted ages with age gaps, clock types, and per-clock feature contributors.

**Individual interpretation.** SHAP-like attribution showing which features push a person's biological age up or down versus the population, aggregated at gene level and hallmark pathway level.

**Clock encyclopedia.** Look up any of the 233 clocks -- what it measures, how it works, its top features, the original paper. Search genes and features across all clocks.

**Pathway enrichment.** Map genes to the 12 Hallmarks of Aging (Lopez-Otin 2023) and 50 MSigDB hallmark pathways via Fisher's exact test. Preranked GSEA against MSigDB v2025.1 collections (50 hallmark, 1787 Reactome, 658 KEGG, 1006 cancer gene sets) with automatic deprecated symbol modernization.

**CpG genomic annotations.** Look up CpG sites by ID, chromosome, gene, or regulatory region. Returns genomic coordinates (GRCh38), CpG island context, ENCODE cCRE classification, nearest genes with region annotations (TSS200, Body, etc.), and transcription factor binding sites.

**Gene annotations.** Query authoritative gene function, GO terms, and Reactome pathways from MyGene.info with a pre-populated 211 MB SQLite cache for offline use.

**PhenoAge from blood panel.** Compute biological age from 9 standard blood biomarkers -- no omics data needed.

**On-the-fly model training.** Train ElasticNet age-prediction models on any MSigDB hallmark pathway or custom gene list, with 5-fold cross-validated performance and SHAP feature importance.

**PubMed search.** Search biomedical literature directly from the conversation via NCBI E-utilities.

**Workspace files.** Read text files, and -- in the local CLI -- write and edit files to save reports or exports. By default access is confined to two folders: `workspace/` (where relative paths resolve) and `skills/`. Grant the agent access to other locations on demand with `/grant <path>` (or set `LONGEVITYCLAW_ALLOWED`). The policy is configurable (`readwrite` / `readonly` / `off`); the hosted web app runs with filesystem tools disabled.

**Skills.** Save a procedure you just ran as a named, reusable skill, then replay it later by typing `/<skill-name>`. Skills are stored as `skills/<name>/SKILL.md` and are prompt recipes -- replaying one re-runs the steps against fresh inputs.

## Scientific Analysis Modules

Three new research modules for longevity biology analysis:

### Pathway Generator (6 tools)

Analyze aging pathways using cross-clock coefficient correlations:

```
you> rank aging pathways
you> score HALLMARK_MTORC1_SIGNALING pathway in detail
you> find pathway synergies between MTORC1 and OXIDATIVE_PHOSPHORYLATION
you> generate hypothesis from seed genes TP53, CDKN2A, SIRT1
you> discover aging modules from clock co-occurrence
you> get intervention targets in DNA_REPAIR pathway
```

**Scientific basis:** Ranks 50 MSigDB hallmark pathways by correlation with 233 aging clock coefficients. Identifies synergistic pathway pairs sharing regulatory genes.

### DrugAge Compound Scoring (5 tools)

Score ~1000 compounds from DrugAge database by longevity evidence:

```
you> top drugage compounds
you> score rapamycin longevity evidence
you> search drugage for senolytics
you> show ITP-validated compounds only
you> link metformin to aging pathways
```

**Scientific basis:** Multi-species lifespan extension data with ITP (Interventions Testing Program) gold-standard validation. Scoring formula weighs species diversity, effect consistency, and ITP replication.

### Control Theory of Aging (2 tools)

Model interventions as vector fields in 12-dimensional hallmark space (Lopez-Otin 2023):

```
you> analyze control laws for age 50 with rapamycin and metformin
you> list longevity interventions
you> what interventions target cellular senescence?
you> find synergy between senolytics and NAD precursors
```

**Scientific basis:** Based on arxiv 2605.16781. Models biological age as control cost, interventions as vector fields acting on Lopez-Otin 12 hallmarks. Lie bracket analysis reveals synergy/antagonism between intervention pairs.

**8 interventions in knowledge base:**
| Intervention | Mechanism | Primary Hallmarks |
|--------------|-----------|-------------------|
| Rapamycin | mTOR inhibition | Nutrient sensing, Autophagy |
| Metformin | AMPK activation | Nutrient sensing, Mitochondria |
| Senolytics (D+Q) | Senescent cell clearance | Cellular senescence, Inflammation |
| NAD+ precursors | NAD+ restoration | Mitochondria, Epigenetics |
| Spermidine | Autophagy induction | Proteostasis, Autophagy |
| Caloric restriction | Metabolic reprogramming | Nutrient sensing, Stem cells |
| Exercise | Systemic adaptation | Multiple hallmarks |
| Fisetin | Senolytic flavonoid | Cellular senescence |

### Novel Target Discovery with 6D Scoring (1 tool)

Discover novel aging targets using 6-dimensional L-LLM scoring across 14 aging hallmarks:

```
you> discover novel aging targets                           # all 14 hallmarks
you> discover targets for inflammation and senescence       # specific hallmarks
you> find novel druggable targets in mitochondrial dysfunction
```

**Tool:** `discover_targets` with parameters:
- `hallmarks` — list of hallmarks to analyze (default: all 14)
- `num_runs` — generation runs for consensus (default: 1)
- `parallel` — enable parallel processing (default: true)
- `max_workers` — thread pool size (default: 8)

**6D Scoring Dimensions (0-100 each):**
| Dimension | Description |
|-----------|-------------|
| Novelty | How underexplored/novel the target is |
| Druggability | Feasibility as drug target (enzymes/kinases/GPCRs score higher) |
| Confidence | Strength of evidence for aging role |
| Safety | Expected safety profile |
| Commercial | Commercial viability |
| Mechanism | Depth of mechanistic understanding |

**Filter Criteria:** `Novelty >= 76 AND Druggability >= 51` = "novel druggable" target

**14 Aging Hallmarks:**
`inflammation`, `genomic_instability`, `altered_intercellular_communication`, `mitochondrial_dysfunction`, `impaired_proteostasis`, `ecm_stiffness`, `cellular_senescence`, `deregulated_nutrient_signaling`, `epigenetic_shift`, `stem_cell_exhaustion`, `telomere_attrition`, `retrotranspositions`, `overall_longevity`, `druggable_aging`

**Output:** Returns top novel druggable targets with all 6 scores, rationale, and hallmark classification.

### L-LLM Integration (1 tool)

Query Longevity LLM — a fine-tuned model for aging biology tasks:

```
you> query L-LLM about mTOR role in aging
you> predict lifespan effect of spermidine in mice
you> score autophagy pathway relevance to aging
```

**Tool:** `query_longevity_llm` with 3 modes:
- `general` — aging biology questions
- `lifespan_prediction` — compound lifespan effect prediction
- `pathway_scoring` — pathway aging relevance via L-LLM

**Configuration:** `HF_TOKEN` environment variable for HuggingFace Inference Endpoints.

**Environment Setup:**

```bash
cp .env.example .env
# Edit .env with your keys
```

`.env.example` contains all available configuration options:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key for main agent |
| `ANTHROPIC_MODEL` | No | Model override (default: claude-sonnet-4-5) |
| `ANTHROPIC_FOUNDRY_ENDPOINT` | Yes* | Azure Foundry endpoint (alternative to direct API) |
| `ANTHROPIC_FOUNDRY_API_KEY` | Yes* | Azure Foundry API key |
| `HF_TOKEN` | No | HuggingFace token for L-LLM inference endpoints |
| `LLM_BACKEND` | No | Set to `vllm` for local vLLM server |
| `VLLM_ENDPOINT` | No | Local vLLM server URL |
| `VLLM_MODEL` | No | Model name served by vLLM |
| `VLLM_API_KEY` | No | Optional auth for vLLM |

*Either `ANTHROPIC_API_KEY` or `ANTHROPIC_FOUNDRY_*` required.

**Local vLLM Backend:**

For running with a local vLLM server instead of HuggingFace endpoints:

```bash
# Configure in .env
LLM_BACKEND=vllm
VLLM_ENDPOINT=http://your-server:8770
VLLM_MODEL=your-model-name
VLLM_API_KEY=your-optional-api-key
```

**Batch Target Discovery Scripts:**

Two scripts for running novel target discovery in batch mode:

```bash
# Single 5-run with consensus (faster, standard output)
uv run python run_local_vllm.py

# Full 5-run saving ALL genes with ALL 6D scores (comprehensive)
uv run python run_5_multirun.py
```

| Script | Output | Use Case |
|--------|--------|----------|
| `run_local_vllm.py` | `data/5run_LOCAL_VLLM.json`, `data/5run_LOCAL_VLLM.csv` | Quick consensus targets |
| `run_5_multirun.py` | `data/5run_FULL_*.json` | Full gene scores across all runs |

**Direct API Endpoints (Web UI):**

Purple "L-LLM" buttons in the web interface call these endpoints directly, guaranteeing L-LLM invocation without agent decision-making:

| Endpoint | Button Location | Function |
|----------|-----------------|----------|
| `POST /api/llm/pathway-score` | Pathway detail modal | Score pathway aging relevance (0-10 scale) |
| `POST /api/llm/lifespan-predict` | Drug detail modal | Predict compound lifespan effect (% change + confidence) |
| `POST /api/llm/mechanism` | Control Laws modal | Analyze intervention mechanism |

**L-LLM Response Features:**
- Lifespan prediction shows percentage change (e.g., +18.5%) with confidence level (30%/60%/90%)
- "Send to Chat" button in all L-LLM modals to continue discussion with the main agent
- Graceful fallback to "Unknown" when model can't extract percentage (rare edge cases)

### User Memory & Learning (3 tools)

Persistent memory system that learns from conversations:

```
you> remember that I prefer detailed explanations
you> forget my age
you> recall all memory
```

**Auto-learning capabilities:**
- Detects user age and name from natural conversation
- Tracks discussed topics (clocks, pathways, genes)
- Learns language preference (Russian/English)
- Adapts response style based on user message patterns
- Remembers corrections to avoid repeating mistakes
- Detects expertise level from vocabulary (cpg, methylation → expert)

**Storage:**
- CLI mode: `data/user_memory.json` (single file for local use)
- Web mode: `data/memories/{session_id}.json` (isolated per browser session)

---

## Web Interface

```bash
uv run longevityclaw-web                        # open http://localhost:8765
uv run longevityclaw-web --password secret      # with authentication
uv run longevityclaw-web --port 9000            # custom port
```

**Features:**
- Full terminal emulator (xterm.js) in browser with all CLI capabilities
- Socket.IO transport with automatic fallback for Cloudflare Tunnels
- Password authentication with 24h session tokens (httponly cookies)
- Session-isolated user memory (each browser session gets separate memory file)
- File upload via drag & drop to `data/uploads/`

**Pathway Explorer UI (left panel):**

| Tab | Description |
|-----|-------------|
| **Ranked** | Top 50 aging pathways sorted by clock correlation score |
| **Synergies** | Shift+click to select multiple pathways, analyze synergies |
| **Drugs** | Top DrugAge compounds with longevity scores and ITP badges |

**Control Laws Modal:**
- Click any intervention in Drugs tab → detailed modal
- Age-specific analysis buttons (30, 50, 70 years)
- PubMed evidence search, mechanism details, side effects
- Compare interventions, view targeted hallmarks

**Cloudflare Tunnel deployment:**
```bash
uv run longevityclaw-web &
cloudflared tunnel --url http://localhost:8765
```

## CLI Interface

```
you> analyze @data/example_plasma_proteome_age55.csv for a 55 year old
you> tell me about cl@GrimAge
you> what role does g@FOXO3 play in aging?
you> train a model on inflammatory response genes in blood
```

Autocomplete: `@` for file paths, `g@` for gene names, `cl@` for clock names, `/` for commands and saved skills.

Commands: `/help`, `/clocks`, `/skills`, `/allowed`, `/grant <path>`, `/loop [freq] <task>`, `/save`, `/showwhy` (agent reasoning trace), `/clear`, `/model`, `/quit`. Invoke a saved skill with `/<skill-name>`. `/loop` repeats a request on a timer (e.g. `/loop 30m check PubMed for new GrimAge papers`) until Ctrl-C.

**Type-ahead:** while the agent is working on your question, you can start typing the next one -- it is held and sent automatically once the current answer is rendered, so you never have to wait on the model to compose a follow-up. Press **Esc** (or Ctrl-C) to interrupt a running request; it stops at the next step and shows "Interrupted by user".

## Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

For direct Anthropic API:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5
```

For Azure Foundry:
```
ANTHROPIC_FOUNDRY_ENDPOINT=https://your-resource.services.ai.azure.com/
ANTHROPIC_FOUNDRY_API_KEY=your-azure-api-key
ANTHROPIC_MODEL=claude-sonnet-4-5
```

## Dependencies

Python 3.11+, managed via `uv`. Core: anthropic, pandas, numpy, rich, prompt-toolkit, pyarrow, scipy, scikit-learn, shap, biothings-client, gseapy.

## References

Horvath (2013) Genome Biology; Hannum et al. (2013) Molecular Cell; Levine et al. (2018) Aging; Lu et al. (2019) Aging; Belsky et al. (2022) eLife; Argentieri et al. (2024) Nature Aging; Oh et al. (2023) Nature; Lopez-Otin et al. (2023) Cell; Liberzon et al. (2015) Cell Systems.
