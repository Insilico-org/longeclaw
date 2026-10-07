"""
Agent tools: functions that the LLM agent can call via tool_use.
Each function returns a dict that gets serialized as tool result.
"""

import json
from .clock_db import get_db
from .annotations import get_gene_hallmarks, get_hallmark_info, map_genes_to_hallmarks, GRIMAGE_COMPONENTS, run_preranked_gsea
from .predict import load_beta_values, run_all_applicable_clocks, compute_cross_clock_feature_importance, interpret_clock_results
from .individual import compute_individual_profile, get_population_reference, get_transcriptomic_population_reference, get_proteomics_population_reference, PopulationReference
from .pubmed import search_and_fetch as pubmed_search_and_fetch
from .train import train_hallmark_model, train_custom_model
from .pathway_generator import (
    rank_all_pathways,
    score_pathway,
    find_pathway_synergies,
    generate_pathway_hypothesis,
    discover_aging_modules,
    get_pathway_intervention_targets,
)
from .control_laws import (
    control_law_analysis,
    KNOWN_INTERVENTIONS,
    HALLMARKS,
)
from .memory import get_memory
from . import fs_access, skills
from .llm_client import query_llm, predict_lifespan_effect, analyze_aging_mechanism, score_pathway_with_llm, get_current_backend, is_local_backend
from .novel_target_generator import (
    run_novel_target_generation,
    analyze_generation_results,
    GeneScore,
    HallmarkResult,
    HALLMARK_CATEGORIES,
)
from .opentargets_client import assess_novelty, batch_assess_novelty


def tool_list_clocks(modality: str | None = None) -> dict:
    """List available aging clocks, optionally filtered by modality."""
    db = get_db()
    if modality == "all" or modality is None:
        clocks = db.list_clocks()
    else:
        clocks = db.list_clocks(modality)

    return {
        "total": len(clocks),
        "clocks": [
            {
                "name": c.name,
                "description": c.description,
                "modality": c.modality,
                "year": c.year,
                "n_features": c.n_features,
            }
            for c in clocks
        ],
    }


def tool_list_modalities() -> dict:
    """List all available data modalities and clock counts."""
    db = get_db()
    mods = db.list_modalities()
    return {"modalities": [{"name": k, "n_clocks": v} for k, v in mods.items()]}


def tool_clock_details(clock_name: str) -> dict:
    """Get full details about a specific clock: description, citation, top features."""
    db = get_db()
    info = db.get_clock_info(clock_name)
    if not info:
        return {"error": f"Clock '{clock_name}' not found. Use list_clocks to see available clocks."}

    top = db.get_top_features(clock_name, n=20)
    model = db.get_clock_model(clock_name)

    result = {
        "name": info.name,
        "description": info.description,
        "modality": info.modality,
        "year": info.year,
        "n_features": info.n_features,
        "citation": info.citation,
        "doi": info.doi,
        "intercept": model.intercept if model else None,
        "top_features": top,
    }
    return result


def tool_search_gene(gene: str) -> dict:
    """Find all clocks that use a specific gene as a feature."""
    db = get_db()
    clock_names = db.find_clocks_by_gene(gene)
    if not clock_names:
        return {"gene": gene, "found_in_clocks": 0, "clocks": [],
                "note": f"Gene '{gene}' not found in any clock. Try the official gene symbol."}

    results = []
    for cn in clock_names:
        model = db.get_clock_model(cn)
        info = db.get_clock_info(cn)
        if not model or not info:
            continue
        # Find the coefficient(s) for this gene
        gene_feats = [
            {"feature_id": fid, "coefficient": coeff}
            for fid, coeff in model.coefficients.items()
            if model.gene_map.get(fid, "").upper() == gene.upper()
        ]
        results.append({
            "clock": cn,
            "modality": info.modality,
            "description": info.description,
            "gene_features": gene_feats,
        })

    return {"gene": gene, "found_in_clocks": len(results), "clocks": results}


def tool_search_feature(feature_id: str) -> dict:
    """Find all clocks that use a specific feature (CpG site, protein, gene ID)."""
    db = get_db()
    clock_names = db.find_clocks_by_feature(feature_id)
    if not clock_names:
        return {"feature_id": feature_id, "found_in_clocks": 0, "clocks": []}

    results = []
    for cn in clock_names:
        model = db.get_clock_model(cn)
        coeff = model.coefficients.get(feature_id, 0)
        gene = model.gene_map.get(feature_id, "")
        results.append({
            "clock": cn,
            "coefficient": coeff,
            "gene": gene,
            "direction": "accelerates aging" if coeff > 0 else "decelerates aging",
        })
    results.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return {"feature_id": feature_id, "found_in_clocks": len(results), "clocks": results}


def tool_compare_clocks(clock_names: list[str]) -> dict:
    """Compare multiple clocks: shared features, unique features, modalities."""
    db = get_db()
    clock_data = {}
    for cn in clock_names:
        model = db.get_clock_model(cn)
        info = db.get_clock_info(cn)
        if model and info:
            clock_data[cn] = {
                "features": set(model.coefficients.keys()),
                "genes": set(model.gene_map.values()),
                "info": info,
            }

    if len(clock_data) < 2:
        return {"error": "Need at least 2 valid clock names to compare."}

    names = list(clock_data.keys())
    all_features = set()
    for cd in clock_data.values():
        all_features |= cd["features"]

    # Pairwise overlap
    comparisons = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            f1 = clock_data[n1]["features"]
            f2 = clock_data[n2]["features"]
            shared = f1 & f2
            comparisons.append({
                "clock_a": n1,
                "clock_b": n2,
                "shared_features": len(shared),
                "unique_to_a": len(f1 - f2),
                "unique_to_b": len(f2 - f1),
                "jaccard_similarity": round(len(shared) / len(f1 | f2), 4) if f1 | f2 else 0,
            })

    return {
        "clocks": [
            {
                "name": cn,
                "modality": clock_data[cn]["info"].modality,
                "n_features": clock_data[cn]["info"].n_features,
                "year": clock_data[cn]["info"].year,
            }
            for cn in names
        ],
        "total_unique_features": len(all_features),
        "pairwise_comparisons": comparisons,
    }


def tool_find_applicable_clocks(feature_ids: list[str]) -> dict:
    """Given a list of available feature IDs, find which clocks can be computed."""
    db = get_db()
    results = db.find_applicable_clocks(set(feature_ids))
    return {
        "n_features_provided": len(feature_ids),
        "applicable_clocks": results[:30],  # top 30
        "total_applicable": len(results),
    }


def tool_compute_clock(clock_name: str, feature_values: dict[str, float]) -> dict:
    """Compute a specific clock from feature values."""
    db = get_db()
    if clock_name == "horvath2013":
        return db.compute_horvath_age(feature_values)
    return db.compute_clock(clock_name, feature_values)


def tool_compute_phenoage(blood_markers: dict[str, float]) -> dict:
    """Compute PhenoAge from blood biomarkers.

    Required markers (with units):
      albumin_g_l, creatinine_umol_l, glucose_mg_dl, crp_mg_l,
      lymphocyte_pct, mcv_fl, rdw_pct, alkaline_phosphatase_u_l,
      wbc_10e3_ul, age
    """
    db = get_db()
    return db.compute_phenoage_blood(blood_markers)


def tool_lookup_cpg_annotations(
    cg_ids: list[str] | None = None,
    chromosome: str | None = None,
    gene: str | None = None,
    region: str | None = None,
    island_relation: str | None = None,
    encode_type: str | None = None,
) -> dict:
    """Look up CpG genomic annotations: coordinates, genes, regulatory context, TFBS."""
    from .cpg_db import get_cpg_db
    db = get_cpg_db()

    if cg_ids:
        results = db.lookup(cg_ids)
        return {"mode": "lookup", "n_queried": len(cg_ids), "results": results}

    results = db.search(
        chromosome=chromosome, gene=gene, region=region,
        island_relation=island_relation, encode_type=encode_type, limit=100,
    )
    meta = db.get_metadata()
    return {
        "mode": "search",
        "genome_build": meta.get("genome_build", "GRCh38"),
        "n_results": len(results),
        "results": results,
        "filters": {k: v for k, v in {
            "chromosome": chromosome, "gene": gene, "region": region,
            "island_relation": island_relation, "encode_type": encode_type,
        }.items() if v},
    }


def tool_annotate_gene(gene_symbol: str | None = None, gene_symbols: list[str] | None = None) -> dict:
    """Get authoritative gene annotations from MyGene.info."""
    from .gene_lookup import get_gene_client
    client = get_gene_client()

    if gene_symbols:
        symbols = gene_symbols[:20]
        results = client.lookup_batch(symbols)
        return {"n_queried": len(symbols), "annotations": results}

    if gene_symbol:
        result = client.lookup(gene_symbol)
        return {"annotation": result}

    return {"error": "Provide gene_symbol or gene_symbols"}


def tool_run_gsea(
    ranked_genes: dict[str, float],
    pathway_database: str = "hallmarks",
) -> dict:
    """Run preranked GSEA on a ranked gene list."""
    if len(ranked_genes) < 10:
        return {"error": "Need at least 10 ranked genes for meaningful GSEA. "
                "Consider using all clock features, not just the top N."}

    from .annotations import load_msigdb_json
    gene_sets = load_msigdb_json(pathway_database)
    all_gs_genes = {g for genes in gene_sets.values() for g in genes}
    input_genes = set(ranked_genes.keys())
    overlap = input_genes & all_gs_genes
    unmatched = sorted(input_genes - all_gs_genes)

    if len(overlap) < 5:
        return {
            "error": f"Only {len(overlap)}/{len(input_genes)} input genes have standard HGNC symbols "
                     f"matching {pathway_database} gene sets. GSEA requires gene symbols like TP53, FOXO3. "
                     f"CpG-based clock features (cg...) must be mapped to gene symbols first.",
            "n_matched": len(overlap),
            "n_unmatched": len(unmatched),
            "unmatched_examples": unmatched[:10],
            "matched_examples": sorted(overlap)[:10],
            "hint": "For CpG clocks: use clock_details to get gene mappings, then pass the gene symbols. "
                    "For small gene lists (<50): consider Fisher's exact test (map_genes_to_hallmarks) instead.",
        }

    try:
        results = run_preranked_gsea(ranked_genes, database=pathway_database)
    except Exception as e:
        return {"error": f"GSEA failed: {e}"}

    sig = [r for r in results if r["fdr"] < 0.25]
    return {
        "database": pathway_database,
        "n_gene_sets_tested": len(results),
        "n_significant_fdr25": len(sig),
        "n_input_genes": len(ranked_genes),
        "n_matched_to_gene_sets": len(overlap),
        "n_unmatched": len(unmatched),
        "pathways": results[:50],
    }


def tool_query_longevity_llm(
    query: str,
    mode: str = "general",
    compound: str | None = None,
    species: str | None = None,
    pathway_name: str | None = None,
    genes: list[str] | None = None,
) -> dict:
    """Query the L-LLM (Longevity LLM) for aging biology analysis."""
    import os
    backend = get_current_backend()
    if is_local_backend(backend):
        if not (os.environ.get("VLLM_ENDPOINT") or os.environ.get("LOCAL_ENDPOINT")):
            return {"error": "VLLM_ENDPOINT (or LOCAL_ENDPOINT) not set. The local L-LLM backend requires an endpoint."}
    elif not os.environ.get("HF_TOKEN"):
        return {"error": "HF_TOKEN environment variable not set. HuggingFace backend requires authentication."}

    try:
        if mode == "lifespan_prediction":
            if not compound or not species:
                return {"error": "lifespan_prediction mode requires 'compound' and 'species' parameters"}
            result = predict_lifespan_effect(compound, species, pubmed_abstract=query if query else None)
            return {"mode": mode, "result": result}

        elif mode == "pathway_scoring":
            if not pathway_name or not genes:
                return {"error": "pathway_scoring mode requires 'pathway_name' and 'genes' parameters"}
            result = score_pathway_with_llm(pathway_name, genes, context=query if query else None)
            return {"mode": mode, "result": result}

        else:  # general mode
            response = analyze_aging_mechanism(query)
            return {
                "mode": "general",
                "response": response.content,
                "model": response.model,
                "usage": response.usage,
            }
    except Exception as e:
        return {"error": f"L-LLM query failed: {e}"}


def tool_discover_novel_targets(
    hallmarks: list[str] | None = None,
    num_runs: int = 1,
    parallel: bool = True,
    max_workers: int = 8,
) -> dict:
    """
    Run novel target discovery using 6-dimension scoring (Novelty, Druggability,
    Confidence, Safety, Commercial, Mechanism). Generates targets across aging hallmarks.

    6D Scoring (0-100 each):
    - Novelty: How underexplored/novel the target is (76+ = High)
    - Druggability: Feasibility as drug target (enzymes/kinases/GPCRs score higher)
    - Confidence: Strength of evidence for aging role
    - Safety: Expected safety profile
    - Commercial: Commercial viability
    - Mechanism: Depth of mechanistic understanding

    Filter: Novelty >= 76 AND Druggability >= 51 = "novel druggable" target

    Args:
        hallmarks: List of hallmarks to analyze (default: all 14 hallmarks)
        num_runs: Number of generation runs for consensus (default: 1)
        parallel: Run hallmarks in parallel (default: True)
        max_workers: Max parallel workers (default: 8)
    """
    try:
        # Run generation across hallmarks
        results = run_novel_target_generation(
            hallmarks=hallmarks,
            parallel=parallel,
            max_workers=max_workers,
        )

        # Analyze and aggregate results
        analysis = analyze_generation_results(results)

        # Get top novel+druggable targets (novelty >= 76 AND druggability >= 51)
        top_novel_druggable = analysis.get("top_50_novel_druggable", [])[:20]

        # Categorize by novelty score bands
        all_genes = analysis.get("all_genes_ranked", [])
        high_novelty = [g for g in all_genes if g.get("novelty", 0) >= 76]
        moderate_novelty = [g for g in all_genes if 51 <= g.get("novelty", 0) < 76]
        emerging_novelty = [g for g in all_genes if 26 <= g.get("novelty", 0) < 51]

        return {
            "method": "6D_scoring",
            "scoring_dimensions": ["novelty", "druggability", "confidence", "safety", "commercial", "mechanism"],
            "filter_criteria": "novelty >= 76 AND druggability >= 51",
            "total_genes_generated": analysis.get("total_genes", 0),
            "novel_druggable_count": analysis.get("novel_druggable_count", 0),
            "summary": {
                "high_novelty_76plus": len(high_novelty),
                "moderate_novelty_51_75": len(moderate_novelty),
                "emerging_novelty_26_50": len(emerging_novelty),
            },
            "hallmarks_analyzed": list(analysis.get("hallmark_counts", {}).keys()),
            "top_novel_druggable_targets": [
                {
                    "gene": g["gene"],
                    "novelty": g["novelty"],
                    "druggability": g["druggability"],
                    "confidence": g["confidence"],
                    "safety": g["safety"],
                    "commercial": g["commercial"],
                    "mechanism": g["mechanism"],
                    "total_score": g["total"],
                    "hallmark": g["hallmark"],
                    "rationale": g.get("rationale", "")[:200],
                }
                for g in top_novel_druggable
            ],
            "paper_targets_overlap": {
                "reference": "PMC9004567",
                "targets": analysis.get("paper_targets", []),
                "overlap": analysis.get("paper_overlap", []),
                "overlap_count": analysis.get("paper_overlap_count", 0),
            },
            "genes_per_hallmark": analysis.get("hallmark_counts", {}),
        }
    except Exception as e:
        return {"error": f"6D target discovery failed: {e}"}


def tool_validate_targets(genes: list[str]) -> dict:
    """Validate targets against OpenTargets to check pharma pipeline status."""
    try:
        results = batch_assess_novelty(genes)
        return {
            "total_genes": len(genes),
            "results": [
                {
                    "gene": r.gene,
                    "ensembl_id": r.ensembl_id,
                    "approved_name": r.approved_name,
                    "known_drugs_count": r.known_drugs_count,
                    "max_clinical_phase": r.max_clinical_phase,
                    "associated_diseases_count": r.associated_diseases_count,
                    "novelty_tier": r.novelty_tier,
                    "novelty_score": round(r.novelty_score, 2),
                }
                for r in results
            ],
            "summary": {
                "novel": sum(1 for r in results if r.novelty_tier == "novel"),
                "emerging": sum(1 for r in results if r.novelty_tier == "emerging"),
                "established": sum(1 for r in results if r.novelty_tier == "established"),
                "well_known": sum(1 for r in results if r.novelty_tier == "well_known"),
            },
        }
    except Exception as e:
        return {"error": f"OpenTargets validation failed: {e}"}


# ── Filesystem tools (workspace-confined, policy-gated) ────────────────

def tool_read_file(file_path: str, max_bytes: int = fs_access.MAX_READ_BYTES) -> dict:
    """Read a UTF-8 text file from within the workspace."""
    if not fs_access.can_read():
        return {"error": "Filesystem access is disabled (LONGEVITYCLAW_FS=off)."}
    try:
        resolved = fs_access.resolve_in_workspace(file_path)
    except fs_access.FsAccessError as e:
        return {"error": str(e)}
    if not resolved.exists() or not resolved.is_file():
        return {"error": f"No such file: {file_path}"}

    cap = min(int(max_bytes), fs_access.MAX_READ_BYTES)
    raw = resolved.read_bytes()
    truncated = len(raw) > cap
    chunk = raw[:cap]
    if b"\x00" in chunk[:4096]:
        return {"path": fs_access.relative_to_workspace(resolved),
                "error": "File appears to be binary; cannot display as text.",
                "bytes": len(raw)}
    return {
        "path": fs_access.relative_to_workspace(resolved),
        "bytes": len(raw),
        "truncated": truncated,
        "content": chunk.decode("utf-8", errors="replace"),
    }


def tool_list_dir(dir_path: str = ".") -> dict:
    """List the contents of a directory within the workspace."""
    if not fs_access.can_read():
        return {"error": "Filesystem access is disabled (LONGEVITYCLAW_FS=off)."}
    try:
        resolved = fs_access.resolve_in_workspace(dir_path)
    except fs_access.FsAccessError as e:
        return {"error": str(e)}
    if not resolved.is_dir():
        return {"error": f"Not a directory: {dir_path}"}

    entries = []
    for entry in sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name)):
        is_dir = entry.is_dir()
        try:
            size = entry.stat().st_size if not is_dir else None
        except OSError:
            size = None
        entries.append({"name": entry.name + ("/" if is_dir else ""),
                        "type": "dir" if is_dir else "file", "size": size})
        if len(entries) >= fs_access.MAX_LIST_ENTRIES:
            break
    return {"path": fs_access.relative_to_workspace(resolved),
            "n_entries": len(entries), "entries": entries}


def tool_write_file(file_path: str, content: str, overwrite: bool = False) -> dict:
    """Write a text file within the workspace (read/write policy only)."""
    if not fs_access.can_write():
        return {"error": "Writing is disabled by policy "
                         f"(LONGEVITYCLAW_FS={fs_access.get_policy()})."}
    if len(content.encode("utf-8")) > fs_access.MAX_WRITE_BYTES:
        return {"error": f"Content exceeds the {fs_access.MAX_WRITE_BYTES}-byte write limit."}
    try:
        resolved = fs_access.resolve_in_workspace(file_path)
    except fs_access.FsAccessError as e:
        return {"error": str(e)}
    existed = resolved.exists()
    if existed and not overwrite:
        return {"error": f"File exists: {file_path}. Pass overwrite=true to replace it."}
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return {"path": fs_access.relative_to_workspace(resolved),
            "bytes_written": len(content.encode("utf-8")),
            "action": "overwritten" if existed else "created"}


def tool_edit_file(file_path: str, old_string: str, new_string: str,
                   replace_all: bool = False) -> dict:
    """Replace an exact substring in a workspace file (read/write policy only)."""
    if not fs_access.can_write():
        return {"error": "Editing is disabled by policy "
                         f"(LONGEVITYCLAW_FS={fs_access.get_policy()})."}
    try:
        resolved = fs_access.resolve_in_workspace(file_path)
    except fs_access.FsAccessError as e:
        return {"error": str(e)}
    if not resolved.is_file():
        return {"error": f"No such file: {file_path}"}

    text = resolved.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return {"error": "old_string not found in file."}
    if count > 1 and not replace_all:
        return {"error": f"old_string occurs {count} times; pass replace_all=true "
                         "or supply a longer unique string."}
    updated = text.replace(old_string, new_string)
    resolved.write_text(updated, encoding="utf-8")
    return {"path": fs_access.relative_to_workspace(resolved),
            "replacements": count if replace_all else 1}


# ── Skill tools (save / list / run repeatable procedures) ──────────────

def tool_save_skill(name: str, description: str, instructions: str,
                    keywords: list[str] | None = None) -> dict:
    """Save a reusable procedure as a named skill."""
    if not fs_access.can_write():
        return {"error": "Saving skills is disabled by policy "
                         f"(LONGEVITYCLAW_FS={fs_access.get_policy()})."}
    try:
        saved = skills.save_skill(name, description, instructions, keywords)
    except ValueError as e:
        return {"error": str(e)}
    return {"saved": True, **saved,
            "note": f"Skill saved. The user can replay it by typing {saved['invoke_with']}."}


def tool_list_skills() -> dict:
    """List all saved skills."""
    catalog = [
        {"name": s["name"], "slug": s["slug"], "description": s["description"],
         "keywords": s["keywords"], "invoke_with": f"/{s['slug']}"}
        for s in skills.list_skills()
    ]
    return {"n_skills": len(catalog), "skills": catalog}


def tool_run_skill(name: str, arguments: str | None = None) -> dict:
    """Load a saved skill's instructions so they can be followed now."""
    skill = skills.get_skill(name)
    if not skill:
        available = [s["slug"] for s in skills.list_skills()]
        return {"error": f"No skill named '{name}'.", "available_skills": available}
    return {
        "name": skill["name"],
        "slug": skill["slug"],
        "description": skill["description"],
        "instructions": skill["instructions"],
        "arguments": arguments or "",
        "note": "Follow these instructions now, using the provided arguments and the "
                "current conversation context. Call whatever tools the steps require.",
    }


# ── Tool registry for the agent ────────────────────────────────────────

TOOLS = [
    {
        "name": "list_clocks",
        "description": "List all available aging clocks in the database, optionally filtered by modality (methylation, proteomics, transcriptomics, histone_mark, atac, blood_chemistry). Returns clock name, description, modality, year, and number of features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": "Filter by modality. Options: methylation, proteomics, transcriptomics, histone_mark, atac, blood_chemistry. Omit for all.",
                    "enum": ["methylation", "proteomics", "transcriptomics", "histone_mark", "atac", "blood_chemistry"],
                },
            },
            "required": [],
        },
        "handler": tool_list_clocks,
    },
    {
        "name": "list_modalities",
        "description": "List all data modalities and how many clocks are available for each.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "handler": tool_list_modalities,
    },
    {
        "name": "clock_details",
        "description": "Get detailed information about a specific aging clock: full description, citation, DOI, intercept, and top 20 features ranked by absolute coefficient weight. Use this to explain what a clock measures and which genes/CpGs drive it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clock_name": {"type": "string", "description": "Exact clock name from the database"},
            },
            "required": ["clock_name"],
        },
        "handler": tool_clock_details,
    },
    {
        "name": "search_gene",
        "description": "Find all aging clocks that include a specific gene as a feature. Returns the clocks and the gene's coefficient in each, showing whether higher expression/methylation of that gene accelerates or decelerates aging across different clocks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gene": {"type": "string", "description": "Gene symbol (e.g. EDARADD, FOXO3, TP53)"},
            },
            "required": ["gene"],
        },
        "handler": tool_search_gene,
    },
    {
        "name": "search_feature",
        "description": "Find all aging clocks that use a specific feature ID (CpG site like cg00000029, protein ID, Ensembl gene ID, etc). Returns the coefficient and direction (accelerates/decelerates aging) in each clock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_id": {"type": "string", "description": "Feature identifier (e.g. cg14424579, ENSG00000000003)"},
            },
            "required": ["feature_id"],
        },
        "handler": tool_search_feature,
    },
    {
        "name": "compare_clocks",
        "description": "Compare two or more clocks: find shared and unique features, Jaccard similarity of feature sets, and basic metadata comparison.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clock_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 2+ clock names to compare",
                },
            },
            "required": ["clock_names"],
        },
        "handler": tool_compare_clocks,
    },
    {
        "name": "find_applicable_clocks",
        "description": "Given a list of available feature IDs from user data, determine which clocks can be computed (>=90% feature coverage).",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature IDs available in user's data",
                },
            },
            "required": ["feature_ids"],
        },
        "handler": tool_find_applicable_clocks,
    },
    {
        "name": "compute_clock",
        "description": "Compute a specific aging clock score from feature values. Returns raw score, top contributors, and missing features. For horvath2013, also returns the transformed Horvath age.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clock_name": {"type": "string", "description": "Clock name to compute"},
                "feature_values": {
                    "type": "object",
                    "description": "Dict of feature_id -> numeric value",
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["clock_name", "feature_values"],
        },
        "handler": tool_compute_clock,
    },
    {
        "name": "compute_phenoage",
        "description": "Compute Levine PhenoAge from standard blood panel. Returns biological age, age gap, and per-biomarker contribution breakdown with clinical interpretation. Units: albumin g/dL, creatinine umol/L, glucose mmol/L, CRP mg/dL, lymphocyte %, MCV fL, RDW %, ALP U/L, WBC 1000/uL, age years.",
        "input_schema": {
            "type": "object",
            "properties": {
                "blood_markers": {
                    "type": "object",
                    "description": "Blood biomarker values with specific unit keys. The agent should help users convert units if needed (e.g. glucose mg/dL -> mmol/L: divide by 18.016, CRP mg/L -> mg/dL: divide by 10, creatinine mg/dL -> umol/L: multiply by 88.4).",
                    "properties": {
                        "albumin_g_dl": {"type": "number", "description": "Albumin in g/dL (typical 3.5-5.5)"},
                        "creatinine_umol_l": {"type": "number", "description": "Creatinine in umol/L (typical 60-110)"},
                        "glucose_mmol_l": {"type": "number", "description": "Glucose in mmol/L (typical 3.9-6.1)"},
                        "crp_mg_dl": {"type": "number", "description": "CRP in mg/dL (typical 0.01-0.5)"},
                        "lymphocyte_pct": {"type": "number", "description": "Lymphocyte % (typical 20-40)"},
                        "mcv_fl": {"type": "number", "description": "MCV in fL (typical 80-100)"},
                        "rdw_pct": {"type": "number", "description": "RDW % (typical 11.5-14.5)"},
                        "alkaline_phosphatase_u_l": {"type": "number", "description": "ALP in U/L (typical 44-147)"},
                        "wbc_10e3_ul": {"type": "number", "description": "WBC in 1000/uL (typical 4.5-11.0)"},
                        "age": {"type": "number", "description": "Chronological age in years"},
                    },
                    "required": ["albumin_g_dl", "creatinine_umol_l", "glucose_mmol_l",
                                 "crp_mg_dl", "lymphocyte_pct", "mcv_fl", "rdw_pct",
                                 "alkaline_phosphatase_u_l", "wbc_10e3_ul", "age"],
                },
            },
            "required": ["blood_markers"],
        },
        "handler": tool_compute_phenoage,
    },
    {
        "name": "predict_age_from_file",
        "description": "Load a CSV/TSV file containing CpG beta values, gene expression values (TPM), or plasma protein NPX values for a single person, run ALL applicable aging clocks (>=95% feature coverage), and return multi-clock age predictions with per-clock feature importance. Supports methylation (CpG IDs like cg00000029), transcriptomic data (gene symbols like FOXO3), and proteomics (protein gene symbols with NPX values). For transcriptomic clocks using Ensembl IDs, gene symbols are mapped automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to CSV/TSV file with CpG beta values"},
                "chronological_age": {"type": "number", "description": "Person's chronological age (optional, for computing age gaps)"},
                "min_coverage": {"type": "number", "description": "Minimum feature coverage to run a clock (default 0.95)", "default": 0.95},
            },
            "required": ["file_path"],
        },
        "handler": lambda file_path, chronological_age=None, min_coverage=0.95: _tool_predict_age_from_file(file_path, chronological_age, min_coverage),
    },
    {
        "name": "get_hallmark_info",
        "description": "Get detailed information about a specific hallmark of aging, including associated genes and keywords. Hallmark IDs: genomic_instability, telomere_attrition, epigenetic_alterations, loss_of_proteostasis, disabled_macroautophagy, deregulated_nutrient_sensing, mitochondrial_dysfunction, cellular_senescence, stem_cell_exhaustion, altered_intercellular_communication, chronic_inflammation, dysbiosis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hallmark_id": {"type": "string", "description": "Hallmark identifier"},
            },
            "required": ["hallmark_id"],
        },
        "handler": lambda hallmark_id: get_hallmark_info(hallmark_id) or {"error": f"Unknown hallmark: {hallmark_id}"},
    },
    {
        "name": "map_genes_to_hallmarks",
        "description": "Map a list of genes to the 12 hallmarks of aging. Returns which hallmarks are enriched and which genes map to each. Useful for interpreting which aging mechanisms are driving clock results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols to map",
                },
            },
            "required": ["genes"],
        },
        "handler": lambda genes: map_genes_to_hallmarks(genes),
    },
    {
        "name": "get_gene_aging_role",
        "description": "Look up a gene's role in aging: which hallmarks it's associated with, which clocks use it, and its coefficient direction across clocks. Combines hallmark annotation with cross-clock evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gene": {"type": "string", "description": "Gene symbol"},
            },
            "required": ["gene"],
        },
        "handler": lambda gene: _tool_gene_aging_role(gene),
    },
    {
        "name": "explain_grimage_component",
        "description": "Get detailed explanation of a GrimAge sub-model component: what it measures biologically, which aging hallmarks it relates to, and evidence-based interventions. Components: DNAmPACKYRS, DNAmADM, DNAmB2M, DNAmCystatinC, DNAmGDF15, DNAmLeptin, DNAmPAI1, DNAmTIMP1, DNAmlogCRP, DNAmlogA1C.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "description": "GrimAge component name"},
            },
            "required": ["component"],
        },
        "handler": lambda component: GRIMAGE_COMPONENTS.get(component, {"error": f"Unknown component: {component}. Available: {list(GRIMAGE_COMPONENTS.keys())}"}),
    },
    {
        "name": "interpret_individual",
        "description": "Deep individual interpretation of a person's data against a population reference. Supports methylation (23K CpGs, 6,599 samples), transcriptomic (12K genes, 12,453 GTEx samples across 11 tissues), and proteomics (2.9K proteins, 316 Allen Institute samples). For each feature used by applicable clocks, computes: (1) z-score and percentile vs global population, (2) z-score vs age-matched peers, (3) clock-weighted personal impact = coefficient * (individual_value - population_mean). This is SHAP-like attribution showing exactly which features push this person's biological age up or down. Results are aggregated at feature, gene, and hallmark levels. Use this after predict_age_from_file to explain WHY someone is aging faster/slower.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to CSV/TSV file with CpG beta values"},
                "chronological_age": {"type": "number", "description": "Person's chronological age in years"},
                "top_n": {"type": "integer", "description": "Number of top CpGs to return (default 30)", "default": 30},
            },
            "required": ["file_path", "chronological_age"],
        },
        "handler": lambda file_path, chronological_age, top_n=30: _tool_interpret_individual(file_path, chronological_age, top_n),
    },
    {
        "name": "pubmed_search",
        "description": "Search PubMed for biomedical research articles. Returns titles, authors, journal, year, DOI, and abstracts. Use this when the user asks about recent research, specific interventions, clinical evidence, or anything beyond the local clock database. Construct focused queries using MeSH terms or keywords (e.g. 'GrimAge epigenetic clock interventions', 'metformin aging clinical trial', 'DunedinPACE longitudinal study').",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search query (supports boolean operators AND/OR/NOT and MeSH terms)"},
                "max_results": {"type": "integer", "description": "Number of articles to return (default 5, max 20)", "default": 5},
            },
            "required": ["query"],
        },
        "handler": lambda query, max_results=5: pubmed_search_and_fetch(query, min(max_results, 20)),
    },
    {
        "name": "train_hallmark_model",
        "description": "Train an ElasticNet regression model predicting age from features in a specific MSigDB hallmark pathway. Uses population data (methylation: 6,599 samples; transcriptomics: 12,453 GTEx samples; proteomics: 316 Allen Institute samples). Returns cross-validated MAE/R², feature coefficients, and SHAP values showing which genes in that pathway most predict aging. Use this when the user asks to analyze a pathway's aging signature, build a custom clock, or understand which genes in a hallmark drive age prediction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hallmark_id": {
                    "type": "string",
                    "description": "MSigDB hallmark pathway ID (e.g. inflammatory_response, p53_pathway, oxidative_phosphorylation). Use get_hallmark_info to browse available pathways.",
                },
                "modality": {
                    "type": "string",
                    "enum": ["auto", "methylation", "transcriptomics", "proteomics"],
                    "description": "Data modality to train on. 'auto' trains on all available.",
                    "default": "auto",
                },
                "tissue": {
                    "type": "string",
                    "description": "For transcriptomics: filter to specific tissue (Blood, Brain, Heart, Muscle, Skin, etc.). Default: all tissues.",
                },
                "compute_shap": {
                    "type": "boolean",
                    "description": "Compute SHAP values for feature importance (default true, slightly slower).",
                    "default": True,
                },
            },
            "required": ["hallmark_id"],
        },
        "handler": lambda hallmark_id, modality="auto", tissue=None, compute_shap=True: train_hallmark_model(
            hallmark_id, modality=modality, tissue=tissue, compute_shap=compute_shap,
        ),
    },
    {
        "name": "train_custom_model",
        "description": "Train an ElasticNet age-prediction model on an arbitrary list of genes or CpG sites. Like train_hallmark_model but for user-specified feature sets. Returns cross-validated performance, coefficients, and SHAP values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols or CpG IDs to use as features",
                },
                "modality": {
                    "type": "string",
                    "enum": ["auto", "methylation", "transcriptomics", "proteomics"],
                    "description": "Data modality. 'auto' tries all available.",
                    "default": "auto",
                },
                "tissue": {
                    "type": "string",
                    "description": "For transcriptomics: filter to specific tissue.",
                },
                "compute_shap": {
                    "type": "boolean",
                    "description": "Compute SHAP values (default true).",
                    "default": True,
                },
            },
            "required": ["feature_list"],
        },
        "handler": lambda feature_list, modality="auto", tissue=None, compute_shap=True: train_custom_model(
            feature_list, modality=modality, tissue=tissue, compute_shap=compute_shap,
        ),
    },
    {
        "name": "lookup_cpg_annotations",
        "description": "Look up genomic annotations for CpG sites. Two modes: (1) Direct lookup by CpG ID — returns coordinates, gene relations (promoter/body/UTR), CpG island context, ENCODE cCRE type, and transcription factor binding sites. (2) Reverse search — find CpGs by chromosome, gene, genomic region (TSS200, TSS1500, 5UTR, 1stExon, Body, 3UTR), island relation, or ENCODE element type. Use this when the user asks about genomic location, regulatory context, CpG island status, or wants to filter CpG features by chromosome, gene, or region type. GRCh38 coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cg_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CpG IDs to look up directly (e.g. ['cg00000029', 'cg14424579'])",
                },
                "chromosome": {
                    "type": "string",
                    "description": "Filter by chromosome (e.g. 'chr14', 'chr1')",
                },
                "gene": {
                    "type": "string",
                    "description": "Filter by gene symbol (e.g. 'FOXO3', 'TP53')",
                },
                "region": {
                    "type": "string",
                    "description": "Filter by genomic region relative to gene",
                    "enum": ["TSS200", "TSS1500", "5UTR", "1stExon", "Body", "3UTR"],
                },
                "island_relation": {
                    "type": "string",
                    "description": "Filter by CpG island relation",
                    "enum": ["Island", "N_Shore", "S_Shore", "N_Shelf", "S_Shelf", "OpenSea"],
                },
                "encode_type": {
                    "type": "string",
                    "description": "Filter by ENCODE cCRE type",
                    "enum": ["PLS", "pELS", "dELS", "CTCF-only", "DNase-H3K4me3"],
                },
            },
            "required": [],
        },
        "handler": tool_lookup_cpg_annotations,
    },
    {
        "name": "annotate_gene",
        "description": "Get authoritative gene annotations from MyGene.info including official name, functional summary, Gene Ontology terms (molecular function, biological process, cellular component), Reactome pathways, and aliases. Use this instead of your training knowledge when you need precise gene function, pathway membership, or GO terms. Supports single gene or batch queries (up to 20).",
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "Single gene symbol to annotate (e.g. 'FOXO3', 'TP53')",
                },
                "gene_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols to annotate in batch (max 20)",
                    "maxItems": 20,
                },
            },
            "required": [],
        },
        "handler": tool_annotate_gene,
    },
    {
        "name": "run_gsea",
        "description": "Run Gene Set Enrichment Analysis on a ranked gene list. Unlike Fisher's exact enrichment, GSEA considers the magnitude of gene-level statistics (fold changes, z-scores, coefficients) to find pathways that are coordinately up- or down-regulated. Use this when the user has expression/methylation data with effect sizes, not just a gene list. Available databases: hallmarks (50 pathways, fast), reactome (1787, comprehensive), kegg (658, disease-oriented), cancer (1006, senescence/oncogene), all (~3500, thorough but slower).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ranked_genes": {
                    "type": "object",
                    "description": "Dict of gene_symbol -> numeric statistic (e.g. fold change, z-score, coefficient). The sign and magnitude both matter.",
                    "additionalProperties": {"type": "number"},
                },
                "pathway_database": {
                    "type": "string",
                    "description": "MSigDB gene set collection to test against",
                    "enum": ["hallmarks", "reactome", "kegg", "cancer", "all"],
                    "default": "hallmarks",
                },
            },
            "required": ["ranked_genes"],
        },
        "handler": tool_run_gsea,
    },
    {
        "name": "rank_pathways",
        "description": "Rank all 50 MSigDB hallmark pathways by aging clock evidence. For each pathway, computes: (1) coverage — fraction of pathway genes present in aging clocks, (2) mean absolute coefficient across all clocks, (3) coefficient consistency — do clocks agree on direction?, (4) Lopez-Otin hallmark mapping. Returns pathways sorted by composite aging score. Use this to discover which biological processes have the strongest aging clock evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_genes": {
                    "type": "integer",
                    "description": "Minimum genes required in pathway (default 5)",
                    "default": 5,
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top pathways to return (default 20)",
                    "default": 20,
                },
            },
            "required": [],
        },
        "handler": lambda min_genes=5, top_n=20: {
            "pathways": [
                {
                    "pathway_id": p.pathway_id,
                    "name": p.pathway_name,
                    "aging_score": round(p.aging_score, 4),
                    "n_genes": p.n_genes,
                    "n_genes_in_clocks": p.n_genes_in_clocks,
                    "coverage": round(p.coverage, 4),
                    "mean_coefficient": round(p.mean_abs_coefficient, 6),
                    "consistency": round(p.coefficient_consistency, 4),
                    "clock_count": p.clock_count,
                    "lopez_otin_hallmarks": list(p.lopez_otin_hallmarks),
                    "top_genes": [(g, round(c, 6)) for g, c in p.top_genes[:5]],
                }
                for p in rank_all_pathways(min_genes)[:top_n]
            ]
        },
    },
    {
        "name": "score_pathway",
        "description": "Get detailed aging clock evidence for a specific MSigDB hallmark pathway. Returns coverage, coefficient statistics, top contributing genes, and Lopez-Otin hallmark mappings. Use this to drill into a specific pathway's aging relevance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathway_id": {
                    "type": "string",
                    "description": "MSigDB hallmark pathway ID (e.g. inflammatory_response, p53_pathway)",
                },
            },
            "required": ["pathway_id"],
        },
        "handler": lambda pathway_id: (
            lambda p: {
                "pathway_id": p.pathway_id,
                "name": p.pathway_name,
                "aging_score": round(p.aging_score, 4),
                "n_genes": p.n_genes,
                "n_genes_in_clocks": p.n_genes_in_clocks,
                "coverage": round(p.coverage, 4),
                "mean_coefficient": round(p.mean_abs_coefficient, 6),
                "consistency": round(p.coefficient_consistency, 4),
                "clock_count": p.clock_count,
                "lopez_otin_hallmarks": list(p.lopez_otin_hallmarks),
                "top_genes": [(g, round(c, 6)) for g, c in p.top_genes],
            } if p else {"error": f"Unknown pathway: {pathway_id}"}
        )(score_pathway(pathway_id)),
    },
    {
        "name": "find_pathway_synergies",
        "description": "Find synergistic combinations among pathways based on shared genes and combined clock coverage. Pathways that share genes and together cover more aging clocks than expected may represent interacting aging mechanisms. Use after rank_pathways to explore how top pathways relate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathway_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pathway IDs to analyze for synergies",
                },
                "min_shared": {
                    "type": "integer",
                    "description": "Minimum shared genes to consider a synergy (default 3)",
                    "default": 3,
                },
            },
            "required": ["pathway_ids"],
        },
        "handler": lambda pathway_ids, min_shared=3: {
            "synergies": [
                {
                    "pathways": list(s.pathway_ids),
                    "shared_genes": sorted(s.shared_genes)[:20],
                    "n_shared": len(s.shared_genes),
                    "combined_coverage": round(s.combined_coverage, 4),
                    "synergy_score": round(s.synergy_score, 4),
                }
                for s in find_pathway_synergies(pathway_ids, min_shared)[:15]
            ]
        },
    },
    {
        "name": "generate_pathway_hypothesis",
        "description": "Generate a novel aging pathway hypothesis from seed genes. Expands seeds by finding genes that co-occur in the same clocks, then characterizes the resulting module: direction (pro/anti-aging), enriched hallmarks, Lopez-Otin mapping, clock coverage. Use this to explore custom gene sets or build hypotheses from interesting genes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seed_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols to seed the pathway",
                },
                "expansion_depth": {
                    "type": "integer",
                    "description": "How many rounds of expansion (default 1, max 3)",
                    "default": 1,
                },
            },
            "required": ["seed_genes"],
        },
        "handler": lambda seed_genes, expansion_depth=1: generate_pathway_hypothesis(
            seed_genes, min(expansion_depth, 3)
        ),
    },
    {
        "name": "discover_aging_modules",
        "description": "Discover coherent aging-related gene modules from clock data. Clusters genes by clock co-occurrence to find functionally related groups. Returns modules with their direction, coverage, and hallmark enrichments. Use this for unsupervised discovery of aging mechanisms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_genes": {
                    "type": "integer",
                    "description": "Minimum genes per module (default 10)",
                    "default": 10,
                },
                "min_coverage": {
                    "type": "number",
                    "description": "Minimum clock coverage fraction (default 0.1)",
                    "default": 0.1,
                },
            },
            "required": [],
        },
        "handler": lambda min_genes=10, min_coverage=0.1: {
            "modules": discover_aging_modules(min_genes, min_coverage)
        },
    },
    {
        "name": "get_pathway_targets",
        "description": "Get potential intervention targets for a pathway based on clock coefficients. Returns genes with consistent coefficients across clocks, ranked by confidence. Shows whether to inhibit (pro-aging) or activate (anti-aging) each target. Use after identifying an interesting pathway to find druggable targets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pathway_id": {
                    "type": "string",
                    "description": "MSigDB hallmark pathway ID",
                },
            },
            "required": ["pathway_id"],
        },
        "handler": lambda pathway_id: {
            "pathway_id": pathway_id,
            "targets": get_pathway_intervention_targets(pathway_id),
        },
    },
    {
        "name": "analyze_control_laws",
        "description": "Analyze aging interventions using Control Theory of Aging framework (arxiv 2605.16781). Models biological state as a point in 12-dimensional Lopez-Otin hallmark space, interventions as vector fields, and biological age as control cost. Returns: (1) intervention rankings by biological age reduction, (2) synergy analysis via Lie brackets for multi-intervention combinations, (3) optimal intervention sequence. Known interventions: rapamycin, metformin, senolytics_dq, nad_precursors, spermidine, caloric_restriction, exercise, fisetin.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chronological_age": {
                    "type": "number",
                    "description": "Person's chronological age in years (default 50)",
                    "default": 50,
                },
                "intervention_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of interventions to analyze (default: all known interventions). Use list_interventions to see available options.",
                },
            },
            "required": [],
        },
        "handler": lambda chronological_age=50, intervention_names=None: control_law_analysis(
            chronological_age, intervention_names
        ),
    },
    {
        "name": "list_interventions",
        "description": "List all known longevity interventions with their effects on the 12 hallmarks of aging. Each intervention shows: name, description, effects on hallmarks (negative = improvement), confidence level, and evidence level (preclinical/clinical/meta-analysis). Use this to understand available interventions before running analyze_control_laws.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": lambda: {
            "interventions": [
                {
                    "id": k,
                    "name": v.name,
                    "description": v.description,
                    "effects": {h: round(e, 3) for h, e in v.effects.items()},
                    "confidence": v.confidence,
                    "evidence_level": v.evidence_level,
                    "drug_class": v.drug_class,
                }
                for k, v in KNOWN_INTERVENTIONS.items()
            ],
            "hallmarks": HALLMARKS,
        },
    },
    # L-LLM (Longevity LLM) for advanced aging biology analysis
    {
        "name": "query_longevity_llm",
        "description": "Query L-LLM (Longevity LLM), a specialized model for aging biology. Modes: 'general' for aging questions, 'lifespan_prediction' to predict compound effects on lifespan (requires compound, species), 'pathway_scoring' to score pathway relevance to aging (requires pathway_name, genes list). Requires HF_TOKEN environment variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or context for L-LLM. For general mode: your aging biology question. For lifespan_prediction: optional PubMed abstract context. For pathway_scoring: optional additional context.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["general", "lifespan_prediction", "pathway_scoring"],
                    "description": "Query mode. Default: general.",
                },
                "compound": {
                    "type": "string",
                    "description": "For lifespan_prediction: name of the compound/drug to analyze.",
                },
                "species": {
                    "type": "string",
                    "description": "For lifespan_prediction: target species (e.g., 'mouse', 'C. elegans').",
                },
                "pathway_name": {
                    "type": "string",
                    "description": "For pathway_scoring: name of the biological pathway.",
                },
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For pathway_scoring: list of genes in the pathway.",
                },
            },
            "required": ["query"],
        },
        "handler": tool_query_longevity_llm,
    },
    # Memory tools for learning and personalization
    {
        "name": "remember",
        "description": "Remember a piece of information about the user for future conversations. Keys: 'age' (user's chronological age), 'name' (user's name), 'style' (response style: concise/detailed/balanced), 'focus' (health focus area), 'preferred' (preferred intervention), 'avoid' (avoided intervention/contraindication), 'modality' (preferred data modality). Or use any custom key for notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "What to remember (age, name, style, focus, preferred, avoid, modality, or custom)"},
                "value": {"type": "string", "description": "The value to remember"},
            },
            "required": ["key", "value"],
        },
        "handler": lambda key, value: {"result": get_memory().remember(key, value)},
    },
    {
        "name": "forget",
        "description": "Forget something about the user. Use 'all' to clear all memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "What to forget (age, name, or 'all' for everything)"},
            },
            "required": ["key"],
        },
        "handler": lambda key: {"result": get_memory().forget(key)},
    },
    {
        "name": "recall",
        "description": "Recall what we know about the user. Returns all stored preferences, corrections, and notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Specific key to recall, or omit for all memory"},
            },
            "required": [],
        },
        "handler": lambda key=None: {"memory": get_memory().recall(key)},
    },
    {
        "name": "discover_targets",
        "description": "Run novel target discovery using 6-dimension scoring. Generates NOVEL + DRUGGABLE aging targets across 14 hallmarks with scores: Novelty (0-100), Druggability (0-100), Confidence (0-100), Safety (0-100), Commercial (0-100), Mechanism (0-100). Filter: Novelty >= 76 AND Druggability >= 51 = 'novel druggable' target. Returns ranked targets emphasizing underexplored genes with therapeutic potential.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hallmarks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of hallmarks to analyze. Options: inflammation, genomic_instability, altered_intercellular_communication, mitochondrial_dysfunction, impaired_proteostasis, ecm_stiffness, cellular_senescence, deregulated_nutrient_signaling, epigenetic_shift, stem_cell_exhaustion, telomere_attrition, retrotranspositions, overall_longevity, druggable_aging. Omit for all 14.",
                },
                "num_runs": {
                    "type": "integer",
                    "description": "Number of generation runs for consensus (default: 1). Higher = more consistent results.",
                },
                "parallel": {
                    "type": "boolean",
                    "description": "Run hallmarks in parallel (default: true). Faster but uses more resources.",
                },
                "max_workers": {
                    "type": "integer",
                    "description": "Max parallel workers when parallel=true (default: 8).",
                },
            },
            "required": [],
        },
        "handler": tool_discover_novel_targets,
    },
    {
        "name": "validate_targets",
        "description": "Validate gene targets against OpenTargets database to check their pharma pipeline status and novelty. Returns: novelty_tier (novel/emerging/established/well_known), novelty_score (0-10, higher=more novel), known_drugs_count, max_clinical_phase (0-4), associated_diseases_count. Use this to verify if discovered targets are truly novel or already in drug development.",
        "input_schema": {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of gene symbols to validate (e.g. ['SIRT1', 'FOXO3', 'TERT'])",
                },
            },
            "required": ["genes"],
        },
        "handler": tool_validate_targets,
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace (e.g. CLAUDE.md, a README, a notes file, a script, or a data file you want to inspect as text). Paths are relative to the workspace root or absolute within it; access outside the workspace is refused. Use this for general file reading — for running aging clocks on an omics CSV use predict_age_from_file instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (workspace-relative or absolute within the workspace)"},
                "max_bytes": {"type": "integer", "description": "Max bytes to read (default 1,000,000)"},
            },
            "required": ["file_path"],
        },
        "handler": lambda file_path, max_bytes=fs_access.MAX_READ_BYTES: tool_read_file(file_path, max_bytes),
    },
    {
        "name": "list_dir",
        "description": "List the files and subdirectories of a directory within the workspace. Use this to discover what files are available before reading or analyzing them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Directory path (default: workspace root '.')"},
            },
            "required": [],
        },
        "handler": lambda dir_path=".": tool_list_dir(dir_path),
    },
    {
        "name": "write_file",
        "description": "Write a text file within the workspace (e.g. save a report, an exported result, a script, or notes). Refuses to overwrite an existing file unless overwrite=true. Only available when the filesystem policy is read/write (the local CLI); disabled in the hosted web app.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Destination path (workspace-relative or absolute within the workspace)"},
                "content": {"type": "string", "description": "Full text content to write"},
                "overwrite": {"type": "boolean", "description": "Allow replacing an existing file (default false)"},
            },
            "required": ["file_path", "content"],
        },
        "handler": lambda file_path, content, overwrite=False: tool_write_file(file_path, content, overwrite),
    },
    {
        "name": "edit_file",
        "description": "Replace an exact substring in an existing workspace text file. The old_string must match exactly and be unique unless replace_all=true. Use this for small, targeted edits instead of rewriting a whole file. Read/write policy only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "Exact text to find"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        "handler": lambda file_path, old_string, new_string, replace_all=False: tool_edit_file(file_path, old_string, new_string, replace_all),
    },
    {
        "name": "save_skill",
        "description": "Save a reusable procedure as a named skill so the user can replay it later by typing /<skill-name>. Use this when the user asks to 'save this as a skill', 'remember how to do this', or 'make this repeatable'. Write the instructions as a generalized, step-by-step recipe (referring to the tools you used and what inputs they need), not a transcript of this specific run — so it works on future inputs. Skills are prompt recipes: replaying one re-runs the steps with fresh inputs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short human-readable skill name (becomes the /slug)"},
                "description": {"type": "string", "description": "One-line summary of what the skill does"},
                "instructions": {"type": "string", "description": "Generalized step-by-step instructions for performing the procedure, including which tools to call and what inputs they need"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "Optional trigger keywords for discovery"},
            },
            "required": ["name", "description", "instructions"],
        },
        "handler": lambda name, description, instructions, keywords=None: tool_save_skill(name, description, instructions, keywords),
    },
    {
        "name": "list_skills",
        "description": "List the user's saved skills with their names, descriptions, and how to invoke them. Use this when the user asks what skills exist or to find a relevant saved procedure.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "handler": lambda: tool_list_skills(),
    },
    {
        "name": "run_skill",
        "description": "Load a saved skill's recorded instructions so you can follow them now with the current inputs. Use this when the user invokes a skill (e.g. types /skill-name) or asks to run a saved procedure. After calling it, carry out the returned step-by-step instructions, calling whatever tools they require.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name or slug to run"},
                "arguments": {"type": "string", "description": "Optional inputs the user supplied for this run (e.g. a file path, an age, a gene)"},
            },
            "required": ["name"],
        },
        "handler": lambda name, arguments=None: tool_run_skill(name, arguments),
    },
]


def _tool_predict_age_from_file(file_path: str, chronological_age: float | None = None, min_coverage: float = 0.95) -> dict:
    """Load beta values and run all applicable clocks."""
    try:
        beta_values = load_beta_values(file_path)
    except Exception as e:
        return {"error": f"Failed to load file: {e}"}

    clock_results = run_all_applicable_clocks(beta_values, min_coverage=min_coverage)
    if not clock_results:
        return {
            "error": "No clocks could be computed with the provided data.",
            "n_features_loaded": len(beta_values),
            "hint": "Ensure the file contains CpG IDs (e.g. cg00000029) with beta values, or gene symbols with expression values (TPM/counts).",
        }

    interpretation = interpret_clock_results(clock_results, chronological_age)

    return {
        "n_features_loaded": len(beta_values),
        "n_clocks_computed": len(clock_results),
        "chronological_age": chronological_age,
        "age_predictions": interpretation["age_summary"]["clocks"],
        "hallmark_profile": interpretation["hallmark_profile"],
        "cross_clock_features": interpretation["cross_clock_features"][:15],
        "grimage_components": interpretation["grimage_components"],
        "all_clock_results": [
            {
                "clock": cr["clock_name"],
                "type": cr.get("clock_type", ""),
                "raw_score": cr["raw_score"],
                "predicted_age": cr.get("predicted_age"),
                "pace": cr.get("pace_of_aging"),
                "coverage": cr["coverage"],
                "n_features": cr["n_features_used"],
            }
            for cr in clock_results
        ],
    }


def _detect_data_modality(feature_values: dict[str, float]) -> str:
    """Detect whether data is methylation (CpG IDs), transcriptomic (gene symbols), or proteomics."""
    sample_keys = list(feature_values.keys())[:100]
    cpg_count = sum(1 for k in sample_keys if k.startswith("cg"))
    if cpg_count > len(sample_keys) / 2:
        return "methylation"

    # Check if features match proteomics clock proteins
    db = get_db()
    prot_clocks = db._modality_index.get("proteomics", [])
    prot_features = set()
    for cn in prot_clocks:
        prot_features |= set(db.models[cn].coefficients.keys())
    match_count = sum(1 for k in sample_keys if k in prot_features)

    # Also check transcriptomic clock features
    trans_clocks = db._modality_index.get("transcriptomics", [])
    trans_features = set()
    for cn in trans_clocks:
        trans_features |= set(db.models[cn].coefficients.keys())
        trans_features |= set(db.models[cn].gene_map.values())
    trans_match = sum(1 for k in sample_keys if k in trans_features)

    if match_count > trans_match and match_count > 5:
        return "proteomics"
    return "transcriptomics"


def _tool_interpret_individual(file_path: str, chronological_age: float, top_n: int = 30) -> dict:
    """Individual interpretation with population comparison."""
    try:
        beta_values = load_beta_values(file_path)
    except Exception as e:
        return {"error": f"Failed to load file: {e}"}

    # Auto-detect modality and pick the right population reference
    modality = _detect_data_modality(beta_values)

    try:
        if modality == "proteomics":
            pop = get_proteomics_population_reference()
        elif modality == "transcriptomics":
            pop = get_transcriptomic_population_reference()
        else:
            pop = get_population_reference()
    except Exception as e:
        return {"error": f"Failed to load population reference: {e}"}

    # Map to CpG IDs if needed (methylation population ref uses CpG IDs)
    if modality == "methylation":
        sample_key = next(iter(beta_values))
        if not sample_key.startswith("cg"):
            from pathlib import Path
            map_path = Path(__file__).resolve().parent.parent / "data" / "prepared_353" / "cpg_mapping.tsv"
            if not map_path.exists():
                map_path = Path(__file__).resolve().parent.parent.parent / "llmregress" / "data" / "prepared_353" / "cpg_mapping.tsv"
            if map_path.exists():
                import pandas as pd
                cpg_map = pd.read_csv(map_path, sep="\t")
                short_to_cpg = dict(zip(cpg_map["short_name"], cpg_map["cpg_id"]))
                beta_values = {short_to_cpg.get(k, k): v for k, v in beta_values.items()}

    profile = compute_individual_profile(
        beta_values=beta_values,
        chronological_age=chronological_age,
        population=pop,
        top_n=top_n,
    )

    # Summarize for the agent (full profile can be large)
    summary = {
        "modality": modality,
        "chronological_age": chronological_age,
        "age_bin": profile["age_bin"],
        "n_clocks_analyzed": profile["n_clocks_analyzed"],
        "population_size": profile["population_size"],
        "top_aging_features": [],
        "top_protective_features": [],
        "gene_level_impacts": profile["gene_level_impacts"][:15],
        "hallmark_level_impacts": profile["hallmark_level_impacts"],
        "clock_summaries": profile["clock_summaries"],
        "interpretation_notes": profile["interpretation_notes"],
    }

    # Split top features into aging vs protective
    for feat in profile["per_cpg_impacts"][:top_n]:
        net_impact = sum(c["impact"] for c in feat["clocks"].values())
        entry = {
            "feature_id": feat["cpg_id"],
            "gene": feat["gene"],
            "value": feat["beta_value"],
            "pop_mean": feat["pop_mean"],
            "zscore": feat["zscore"],
            "percentile": feat["percentile"],
            "peer_zscore": feat["peer_zscore"],
            "net_impact": round(net_impact, 6),
            "n_clocks": feat["n_clocks"],
            "total_abs_impact": feat["total_impact"],
        }
        if net_impact > 0:
            summary["top_aging_features"].append(entry)
        else:
            summary["top_protective_features"].append(entry)

    return summary


def _tool_gene_aging_role(gene: str) -> dict:
    """Combined gene lookup: hallmarks + cross-clock presence."""
    db = get_db()
    hallmarks = get_gene_hallmarks(gene)
    clock_names = db.find_clocks_by_gene(gene)

    clock_details = []
    for cn in clock_names[:20]:  # limit to 20
        model = db.get_clock_model(cn)
        info = db.get_clock_info(cn)
        if not model or not info:
            continue
        gene_feats = [
            {"feature_id": fid, "coefficient": round(coeff, 6),
             "direction": "accelerates aging" if coeff > 0 else "decelerates aging"}
            for fid, coeff in model.coefficients.items()
            if model.gene_map.get(fid, "").upper() == gene.upper()
        ]
        clock_details.append({
            "clock": cn,
            "modality": info.modality,
            "gene_features": gene_feats,
        })

    return {
        "gene": gene,
        "hallmarks": hallmarks,
        "n_clocks": len(clock_names),
        "clock_details": clock_details,
    }


def get_tool_definitions() -> list[dict]:
    """Return tool definitions in Anthropic API format (without handlers)."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in TOOLS
    ]


def get_tool_handlers() -> dict:
    """Return name -> handler mapping."""
    return {t["name"]: t["handler"] for t in TOOLS}
