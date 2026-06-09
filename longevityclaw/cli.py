"""
LongevityClaw CLI: interactive chat with the aging clock agent.
Rich terminal rendering, inline status, @ file autocomplete, / commands.
"""

import json
import os
import sys
import glob
import time
import random
import threading
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.style import Style

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.keys import Keys

import re

console = Console()

PASTE_COLLAPSE_THRESHOLD = 300
PASTE_MARKER_RE = re.compile(r"\x00PASTE:(\d+):(\d+)\x00")


class PasteCollapseProcessor(Processor):
    """Render paste markers as collapsed [Pasted text #N — M chars] indicators."""

    def apply_transformation(self, transformation_input):
        ti = transformation_input
        line_text = "".join(frag[1] for frag in ti.fragments)

        markers = list(PASTE_MARKER_RE.finditer(line_text))
        if not markers:
            return Transformation(ti.fragments)

        new_frags = []
        markers_info = []
        pos = 0

        for m in markers:
            if m.start() > pos:
                new_frags.append(("", line_text[pos:m.start()]))
            num, chars = m.group(1), m.group(2)
            label = f"[Pasted text #{num} — {chars} chars]"
            new_frags.append(("class:paste-collapsed", label))
            markers_info.append((m.start(), m.end(), len(label)))
            pos = m.end()

        if pos < len(line_text):
            new_frags.append(("", line_text[pos:]))

        def source_to_display(i, _mi=markers_info):
            offset = 0
            for ms, me, ll in _mi:
                ml = me - ms
                if i <= ms:
                    return i + offset
                elif i >= me:
                    offset += ll - ml
                else:
                    return ms + offset
            return i + offset

        def display_to_source(i, _mi=markers_info):
            offset = 0
            for ms, me, ll in _mi:
                ml = me - ms
                ds = ms + offset
                de = ds + ll
                if i <= ds:
                    return i - offset
                elif i >= de:
                    offset += ll - ml
                else:
                    return me
            return i - offset

        return Transformation(new_frags,
                              source_to_display=source_to_display,
                              display_to_source=display_to_source)

# Palette: deep teal -> bright mint
C_DEEP    = "#005461"
C_DARK    = "#0C7779"
C_MID     = "#249E94"
C_BRIGHT  = "#3BC1A8"

STATUS_STYLE = Style(color=C_DARK)
DETAIL_STYLE = Style(color=C_MID, italic=True)

BANNER = f"""
[bold {C_DEEP}] _                                 _ _          ____ _[/bold {C_DEEP}]
[bold {C_DEEP}]| |    ___  _ __   __ _  _____   _(_) |_ _   _ / ___| | __ ___      __[/bold {C_DEEP}]
[bold {C_DARK}]| |   / _ \\| '_ \\ / _` |/ _ \\ \\ / / | __| | | | |   | |/ _` \\ \\ /\\ / /[/bold {C_DARK}]
[bold {C_MID}]| |__| (_) | | | | (_| |  __/\\ V /| | |_| |_| | |___| | (_| |\\ V  V /[/bold {C_MID}]
[bold {C_BRIGHT}]|_____\\___/|_| |_|\\__, |\\___| \\_/ |_|\\__|\\__, |\\____|_|\\__,_| \\_/\\_/[/bold {C_BRIGHT}]
[bold {C_BRIGHT}]                  |___/                  |___/[/bold {C_BRIGHT}]

  [{C_DARK}]233 clocks[/{C_DARK}] | [{C_DARK}]429K coefficients[/{C_DARK}] | [{C_DARK}]6 modalities[/{C_DARK}] | [{C_DARK}]23K CpG + 12K gene + 2.9K protein pop refs[/{C_DARK}]

  [{C_MID}]@[/{C_MID}] attach files  •  [{C_MID}]g@[/{C_MID}] gene lookup  •  [{C_MID}]cl@[/{C_MID}] clock lookup  •  [{C_MID}]/[/{C_MID}] commands

  [{C_DARK} italic]"tell me about cl@GrimAge"  •  "what role does g@FOXO3 play?"[/{C_DARK} italic]
  [{C_DARK} italic]"train a model on inflammatory response genes in blood"[/{C_DARK} italic]
  [{C_DARK} italic]"which p53 pathway CpGs best predict age?"[/{C_DARK} italic]
  [{C_DARK} italic]"build a custom clock from FOXO3, SIRT1, MTOR, TP53"[/{C_DARK} italic]
"""

# ── Slash commands ─────────────────────────────────────────────────────

COMMANDS = {
    "/help": "Show available commands and usage tips",
    "/clocks": "List all available clock modalities and counts",
    "/skills": "List saved skills (invoke one with /<skill-name>)",
    "/allowed": "Show directories the agent may read/write",
    "/grant": "Grant the agent access to a file or folder (/grant <path>)",
    "/loop": "Repeat a request on a timer (/loop [freq] <task>, Ctrl-C to stop)",
    "/empty_queue": "Clear any follow-up message you've queued",
    "/effort": "Cap tool calls per response (/effort low|medium|high|max)",
    "/usage": "Show session token/model/tool/skill usage (Esc to dismiss)",
    "/save": "Save conversation to markdown file",
    "/showwhy": "Show agent reasoning trace for a recent request",
    "/clear": "Clear conversation history (start fresh)",
    "/model": "Show current model name",
    "/exit": "Exit LongevityClaw (or /quit)",
    "/quit": "Exit LongevityClaw",
}

COMMAND_HELP = f"""
[bold]Commands:[/bold]
  [{C_MID}]/help[/{C_MID}]      Show this help message
  [{C_MID}]/clocks[/{C_MID}]    List clock modalities and counts
  [{C_MID}]/skills[/{C_MID}]    List saved skills — invoke one by typing /<skill-name>
  [{C_MID}]/allowed[/{C_MID}]   Show directories the agent may read/write
  [{C_MID}]/grant[/{C_MID}]     Grant access to a file or folder — "/grant ~/data/project"
  [{C_MID}]/loop[/{C_MID}]      Repeat a request on a timer — "/loop 30m check PubMed for new GrimAge papers" (Ctrl-C stops)
  [{C_MID}]/empty_queue[/{C_MID}]  Clear a follow-up you queued while the model was working
  [{C_MID}]/effort[/{C_MID}]    Cap tool calls per response — low (5) · medium (10) · high (20) · max (unlimited); skills bypass it
  [{C_MID}]/usage[/{C_MID}]     Show session usage — tokens by model, L-LLM calls, tools & skills (Esc to dismiss)
  [{C_MID}]/save[/{C_MID}]      Save conversation to markdown (optional: /save filename.md)
  [{C_MID}]/showwhy[/{C_MID}]   Show agent reasoning trace (tool calls, inputs, results)
  [{C_MID}]/clear[/{C_MID}]     Clear conversation history
  [{C_MID}]/model[/{C_MID}]     Show current model
  [{C_MID}]/exit[/{C_MID}]      Exit (or /quit)

[bold]Special tokens:[/bold]
  [{C_MID}]@[/{C_MID}]          Attach file path — "analyze @data/sample.csv for a 60 year old"
  [{C_MID}]g@[/{C_MID}]         Gene lookup — "what role does g@FOXO3 play in aging?"
  [{C_MID}]cl@[/{C_MID}]        Clock lookup — "tell me about cl@horvath2013"

[bold]Tips:[/bold]
  • Compare clocks: "compare cl@horvath2013 and cl@hannum"
  • Gene deep-dive: "g@TP53 across all clocks"
  • Compute PhenoAge: share your blood panel values
  • Analyze methylation: "analyze @data/sample.csv for a 60 year old"
  • Analyze transcriptome: "analyze @data/example_blood_transcriptome_age51.csv"
  • Analyze proteome: "analyze @data/example_plasma_proteome_age55.csv for a 55 year old"
"""


# ── Autocomplete ───────────────────────────────────────────────────────

class LongevityClawCompleter(Completer):
    """
    Dropdown-style autocomplete:
    - '/' at line start: slash commands dropdown
    - 'g@' anywhere: gene name dropdown (from clock database)
    - 'cl@' anywhere: clock name dropdown (from clock database)
    - '@' anywhere (not preceded by g or cl): file/directory path dropdown
    Triggers as-you-type so the dropdown appears immediately.
    """

    def __init__(self):
        self._genes = None      # lazy-loaded: {GENE: n_clocks}
        self._clocks = None     # lazy-loaded: {clock_name: description}

    def _load_db_indices(self):
        if self._genes is not None:
            return
        try:
            from .clock_db import get_db
            db = get_db()
            self._genes = {g: len(cs) for g, cs in db._gene_to_clocks.items()}
            self._clocks = {
                n: (c.description[:60] + "..." if len(c.description) > 60 else c.description)
                for n, c in db.clocks.items()
            }
        except Exception:
            self._genes = {}
            self._clocks = {}

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # ── Slash commands: only at start of line ──────────────────
        if text.startswith("/"):
            cmd_text = text.lower()
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(cmd_text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )
            # Saved skills are invokable as /<slug>
            try:
                from .skills import list_skills
                for skill in list_skills():
                    slug_cmd = f"/{skill['slug']}"
                    if slug_cmd.startswith(cmd_text):
                        yield Completion(
                            slug_cmd,
                            start_position=-len(text),
                            display=slug_cmd,
                            display_meta=f"skill · {skill['description'][:50]}",
                        )
            except Exception:
                pass
            return

        # ── g@ gene completion ────────────────────────────────────
        g_pos = text.rfind("g@")
        cl_pos = text.rfind("cl@")

        if g_pos != -1 and (cl_pos == -1 or g_pos > cl_pos):
            self._load_db_indices()
            prefix = text[g_pos + 2:]
            if " " in prefix:
                pass  # fall through to file completion
            else:
                prefix_upper = prefix.upper()
                count = 0
                for gene, n_clocks in sorted(self._genes.items()):
                    if prefix_upper and not gene.startswith(prefix_upper):
                        continue
                    yield Completion(
                        gene,
                        start_position=-len(prefix),
                        display=gene,
                        display_meta=f"{n_clocks} clocks",
                    )
                    count += 1
                    if count >= 50:
                        break
                return

        # ── cl@ clock completion ──────────────────────────────────
        if cl_pos != -1 and (g_pos == -1 or cl_pos > g_pos):
            self._load_db_indices()
            prefix = text[cl_pos + 3:]
            if " " in prefix:
                pass  # fall through to file completion
            else:
                prefix_lower = prefix.lower()
                count = 0
                for clock_name, desc in sorted(self._clocks.items()):
                    if prefix_lower and not clock_name.lower().startswith(prefix_lower):
                        continue
                    yield Completion(
                        clock_name,
                        start_position=-len(prefix),
                        display=clock_name,
                        display_meta=desc,
                    )
                    count += 1
                    if count >= 50:
                        break
                return

        # ── @ file path completion ─────────────────────────────────
        at_pos = text.rfind("@")
        if at_pos == -1:
            return

        # Skip if this @ is part of g@ or cl@
        if at_pos > 0 and text[at_pos - 1] in ("g",):
            return
        if at_pos > 1 and text[at_pos - 2:at_pos] == "cl":
            return

        path_text = text[at_pos + 1:]

        # Stop completing after user added a space past the path
        if " " in path_text and not path_text.endswith("/"):
            return

        # Expand ~
        expanded = os.path.expanduser(path_text) if path_text.startswith("~") else path_text

        # Split into dir + prefix
        if expanded == "" or expanded.endswith("/"):
            search_dir = expanded or "."
            prefix = ""
        elif os.path.isdir(expanded):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        if not os.path.isdir(search_dir):
            return

        try:
            entries = sorted(os.listdir(search_dir))
        except PermissionError:
            return

        for entry in entries:
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue

            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)
            suffix = "/" if is_dir else ""

            # Size hint for files
            if is_dir:
                meta = "dir"
            else:
                try:
                    size = os.path.getsize(full_path)
                    if size > 1024 * 1024:
                        meta = f"{size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        meta = f"{size / 1024:.0f} KB"
                    else:
                        meta = f"{size} B"
                except OSError:
                    meta = ""

                # Add type tag for data files
                if entry.endswith((".csv", ".tsv")):
                    meta = f"data · {meta}"
                elif entry.endswith((".npz", ".npy")):
                    meta = f"numpy · {meta}"
                elif entry.endswith((".gz", ".zip")):
                    meta = f"archive · {meta}"

            yield Completion(
                entry + suffix,
                start_position=-len(prefix),
                display=entry + suffix,
                display_meta=meta,
            )


# ── Status line ────────────────────────────────────────────────────────

STATS_STYLE = Style(color=C_DARK, dim=True)


class StatusLine:
    """Three-line status: main action + detail + cumulative stats."""

    def __init__(self):
        self._live = None
        self._main = ""
        self._detail = ""
        self._stats = ""

    def update(self, msg: str):
        self._main = msg
        self._detail = ""
        self._refresh()

    def detail(self, msg: str):
        self._detail = msg
        self._refresh()

    def stats(self, msg: str):
        self._stats = msg
        self._refresh()

    def _refresh(self):
        parts = [Text(f"  ⟳ {self._main}", style=STATUS_STYLE)]
        if self._detail:
            parts.append(Text(f"    ↳ {self._detail}", style=DETAIL_STYLE))
        if self._stats:
            parts.append(Text(f"    {self._stats}", style=STATS_STYLE))
        if self._live:
            self._live.update(Group(*parts))

    def start(self):
        self._live = Live(
            Text("", style=STATUS_STYLE),
            console=console, refresh_per_second=12, transient=True,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
        self._detail = ""
        self._stats = ""


# ── .env loader ────────────────────────────────────────────────────────

def load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


# ── Save conversation ─────────────────────────────────────────────────

def _save_conversation(messages: list[dict], filename: str | None = None):
    """Export conversation to a markdown file inside the workspace."""
    from datetime import datetime
    from . import fs_access

    if not fs_access.can_write():
        console.print(f"[{C_DARK}]  saving is disabled by policy "
                      f"(LONGEVITYCLAW_FS={fs_access.get_policy()})[/{C_DARK}]")
        return

    if not filename:
        filename = f"longevityclaw_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Resolve into the workspace (or a granted location) like the agent's tools,
    # so /save can't write to arbitrary paths and lands somewhere predictable.
    try:
        target = fs_access.resolve_in_workspace(filename)
    except fs_access.FsAccessError as e:
        console.print(f"[{C_DARK}]  {e}[/{C_DARK}]")
        return

    lines = [f"# LongevityClaw Session\n",
             f"*Saved {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                lines.append(f"---\n\n## You\n\n{content}\n\n")
        elif role == "assistant":
            text_parts = []
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
            elif isinstance(content, str):
                text_parts.append(content)
            if text_parts:
                lines.append(f"## LongevityClaw\n\n{''.join(text_parts)}\n\n")

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[{C_DARK}]  saved to {fs_access.relative_to_workspace(target)}[/{C_DARK}]")


# ── Streaming response for demo mode ──────────────────────────────────

def _make_panel(content):
    return Panel(
        content,
        title=f"[bold {C_BRIGHT}]longevityclaw[/bold {C_BRIGHT}]",
        border_style=C_DEEP,
        padding=(1, 2),
    )


def _print_tool_hint(agent, request_count: int):
    """Print the 'N tools used' footer for the most recent turn."""
    if not agent.traces:
        return
    last_trace = agent.traces[-1]
    n_tools = sum(len(r["tool_calls"]) for r in last_trace["rounds"])
    if n_tools > 0:
        hint = "  /showwhy for details" if request_count <= 3 else ""
        console.print(f"  [dim]{n_tools} tool{'s' if n_tools != 1 else ''} used ·{hint}[/dim]")


# ── Prompt chrome: bright status bar + reserved tip row above the input ─

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Shown below the status bar the first time the user queues a message this
# session, so they learn how to cancel a queued follow-up exactly when relevant.
_QUEUE_TIP = "/empty_queue clears any follow-up you've queued"
_queue_tip_shown = False

TIPS = [
    'attach a file with @ — "analyze @data/sample.csv for a 60 year old"',
    "type cl@ or g@ to autocomplete clock and gene names",
    "/loop [freq] <task> repeats a request on a timer (Ctrl-C stops)",
    'save a procedure as a skill, then run it with /<skill-name>',
    "/grant <path> opens a folder outside the workspace to the agent",
    "press Esc while the model is working to interrupt it",
    "/showwhy reveals the agent's tool calls and reasoning",
    "/skill-builder turns what you just did into a reusable skill",
    "/effort low|medium|high|max caps tool calls per response (lower = faster)",
    _QUEUE_TIP,
]


def _pick_tip(prob: float = 0.4) -> str:
    """Occasionally surface a tip; empty string the rest of the time."""
    return random.choice(TIPS) if random.random() < prob else ""


def _make_chrome(bar_text, tip: str):
    """Build a multi-line prompt message: a full-width bright status/separator
    bar, a reserved tip row, then the 'you> ' input line. ``bar_text`` is a
    callable returning the bar's inline text (re-evaluated on each redraw)."""
    def _message():
        try:
            width = get_app().output.get_size().columns
        except Exception:
            width = 80
        text = bar_text().strip()
        prefix = f"── {text} " if text else "── "
        bar = (prefix + "─" * max(0, width - len(prefix)))[:width]
        frags = [("class:statusbar", bar + "\n")]
        # Reserve the tip row whether or not a tip is shown, so the input
        # position never jumps.
        frags.append(("class:tip", f"  {tip}\n" if tip else "\n"))
        frags.append(("class:you", "you> "))
        return frags
    return _message


def _idle_bar_text(effort: str = "") -> str:
    head = "LongevityClaw" + (f"  ·  effort: {effort}" if effort else "")
    return f"{head}    Esc clear · Ctrl-C interrupt · Tab complete"


# ── /usage: transient session-usage table (erased on Esc) ──────────────

def _usage_renderable(agent):
    """Build a Rich renderable summarizing this session's model/tool/skill use."""
    from rich.table import Table
    from .llm_client import get_llm_stats

    su = agent.session_usage
    llm = get_llm_stats()

    models = Table(title="Models & tokens", title_style=f"bold {C_BRIGHT}",
                   border_style=C_DEEP, header_style=C_MID, expand=False)
    for col, j in (("model", "left"), ("calls", "right"), ("in", "right"),
                   ("cached", "right"), ("out", "right"), ("total", "right")):
        models.add_column(col, justify=j)
    tot_in = tot_out = tot_cached = 0
    for name, m in su["models"].items():
        # input_tokens is the uncached remainder — true input adds the cached part
        true_in = m["input"] + m.get("cache_read", 0) + m.get("cache_write", 0)
        cached = m.get("cache_read", 0)
        models.add_row(name, str(m["calls"]), f"{true_in:,}",
                       f"{cached:,}" if cached else "—",
                       f"{m['output']:,}", f"{true_in + m['output']:,}")
        tot_in += true_in; tot_out += m["output"]; tot_cached += cached
    claude_in = tot_in  # cache only applies to the Claude model(s)
    if llm["calls"]:
        lname = "L-LLM: " + (", ".join(llm["by_model"]) or "longevity")
        models.add_row(lname, str(llm["calls"]), f"{llm['input']:,}", "—",
                       f"{llm['output']:,}", f"{llm['input'] + llm['output']:,}")
        tot_in += llm["input"]; tot_out += llm["output"]
    if not su["models"] and not llm["calls"]:
        models.add_row("(nothing yet)", "", "", "", "", "")
    else:
        models.add_section()
        models.add_row("[bold]total[/bold]", "", f"[bold]{tot_in:,}[/bold]",
                       f"[bold]{tot_cached:,}[/bold]" if tot_cached else "—",
                       f"[bold]{tot_out:,}[/bold]", f"[bold]{tot_in + tot_out:,}[/bold]")

    extra = Table(title="Tools & skills", title_style=f"bold {C_BRIGHT}",
                  border_style=C_DEEP, header_style=C_MID, expand=False)
    for col, j in (("kind", "left"), ("distinct", "right"),
                   ("calls", "right"), ("most used", "left")):
        extra.add_column(col, justify=j)
    tools, skills = su["tools"], su["skills"]
    top_tools = sorted(tools.items(), key=lambda x: -x[1])[:5]
    extra.add_row("tools", str(len(tools)), str(sum(tools.values())),
                  ", ".join(f"{k}×{v}" for k, v in top_tools) or "—")
    extra.add_row("skills", str(len(skills)), str(sum(skills.values())),
                  ", ".join(f"{k}×{v}" for k, v in skills.items()) or "—")

    cap = "unlimited" if agent.tool_budget is None else f"max {agent.tool_budget} tools/response"
    c = su.get("cache", {"read": 0, "write": 0})
    hit = (c["read"] / claude_in * 100) if claude_in else 0
    # Net tok-equiv vs no caching: each read costs 0.1× instead of 1× (save 0.9×);
    # each write costs the TTL's premium over 1× (5m → 0.25×, 1h → 1.0×). Premium
    # is read from agent so this number can never drift from the TTL we request.
    from .agent import CACHE_WRITE_PREMIUM, CACHE_WRITE_MULTIPLIER, _CACHE_TTL
    saved = int(c["read"] * 0.9 - c["write"] * CACHE_WRITE_PREMIUM)
    cache_line = (f"[{C_DARK}]prompt cache:[/{C_DARK}] "
                  f"[{C_MID}]{c['read']:,}[/{C_MID}] read (~0.1×) · "
                  f"{c['write']:,} written (~{CACHE_WRITE_MULTIPLIER:g}×, {_CACHE_TTL} TTL) · "
                  f"[{C_MID}]{hit:.0f}%[/{C_MID}] of Claude input cached "
                  f"[{C_DARK}](≈{saved:,} tok-equiv {'saved' if saved >= 0 else 'lost'})[/{C_DARK}]"
                  + ("   [dim](0 = caching inactive / silent invalidator)[/dim]"
                     if not c["read"] else ""))
    return Group(f"[bold {C_BRIGHT}]Session usage[/bold {C_BRIGHT}]   "
                 f"[{C_DARK}]effort: {agent.effort} ({cap})[/{C_DARK}]", "",
                 models, "", extra, "", cache_line)


def _render_ansi(renderable) -> str:
    """Render a Rich renderable to an ANSI string for embedding in prompt_toolkit."""
    from io import StringIO
    buf = StringIO()
    tmp = Console(file=buf, force_terminal=True, color_system="truecolor",
                  width=min(console.size.width, 100))
    tmp.print(renderable)
    return buf.getvalue()


def _show_usage(agent):
    """Display the usage table in a transient overlay that erases when Esc is
    pressed (the table never persists in the chat — only the /usage command)."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings

    body = _render_ansi(_usage_renderable(agent))

    def _content():
        return ANSI(body + f"\n  Esc to return to chat")

    kb = KeyBindings()

    @kb.add("escape")
    @kb.add("enter")
    @kb.add("q")
    @kb.add("c-c")
    def _(event):
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(_content),
                                     wrap_lines=True)])),
        key_bindings=kb,
        erase_when_done=True,
        full_screen=False,
    )
    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        pass


# ── /loop: repeat a request on a timer ─────────────────────────────────

_DURATION_RE = re.compile(r"^(\d+)\s*([smh]?)$", re.IGNORECASE)


def _parse_loop_args(rest: str) -> tuple[int, str]:
    """Parse '[frequency] task'. Frequency is 30s / 5m / 2h, or a bare integer
    (minutes). With no leading frequency, defaults to 10 minutes. Returns
    (interval_seconds, task); interval is floored at 5s to avoid hammering."""
    parts = rest.split(maxsplit=1)
    if parts and (m := _DURATION_RE.match(parts[0])):
        unit = (m.group(2) or "m").lower()
        secs = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[unit]
        task = parts[1] if len(parts) > 1 else ""
    else:
        secs, task = 600, rest
    return max(secs, 5), task.strip()


def _fmt_interval(secs: int) -> str:
    if secs >= 3600 and secs % 3600 == 0:
        return f"{secs // 3600}h"
    if secs >= 60 and secs % 60 == 0:
        return f"{secs // 60}m"
    return f"{secs}s"


def _loop_wait(interval: int, status):
    """Sleep `interval` seconds with a live countdown; Ctrl-C breaks out."""
    status.start()
    try:
        for remaining in range(interval, 0, -1):
            status.update(f"next run in {_fmt_interval(remaining)} (Ctrl-C to stop)")
            time.sleep(1)
    finally:
        status.stop()


def _run_loop(agent, task: str, interval: int, status, stream_enabled: bool):
    """Send `task` to the agent repeatedly every `interval` seconds until the
    user presses Ctrl-C. Each iteration renders like a normal turn."""
    console.print(f"  [dim]looping every {_fmt_interval(interval)} — Ctrl-C to stop[/dim]")
    n = 0
    try:
        while True:
            n += 1
            console.print(f"\n  [{C_MID}]loop #{n}[/{C_MID}]  [dim]{task}[/dim]")
            try:
                status.start()
                response = agent.chat(task)
                status.stop()
            except Exception as e:
                status.stop()
                console.print(f"  [bold {C_DEEP}]error>[/bold {C_DEEP}] {e}")
                response = None
            if response is not None:
                console.print()
                if stream_enabled:
                    _stream_response(response)
                else:
                    console.print(_make_panel(Markdown(response)))
                _print_tool_hint(agent, 99)  # past the /showwhy intro hint
            _loop_wait(interval, status)
    except KeyboardInterrupt:
        status.stop()
        console.print(f"\n  [dim italic]loop stopped after {n} run{'s' if n != 1 else ''}[/dim italic]")


def _stream_response(response: str):
    """Pseudo-stream the first part of the answer, then print the whole thing
    at once.

    Short answers stream word-by-word in full. Long answers stream only a brief
    first window — kept small and clear of the screen edge — then complete
    instantly with the full panel. Keeping the streamed region small avoids the
    flicker that tall Live regions cause near the bottom of the screen, and the
    line/time caps stop a long answer from streaming forever.
    """
    # console.size can mis-report right after a prompt_toolkit app closes; clamp
    # it so a bad reading can't disable the long-answer cutoff.
    term_h = console.size.height
    if not (8 <= term_h <= 400):
        term_h = 30
    usable_w = max(console.size.width - 8, 40)

    def _wrapped(text: str) -> int:
        return sum(1 + len(ln) // usable_w for ln in text.split("\n"))

    # Stream at most a small window: never near the screen edge, never huge.
    stream_lines = max(6, min(term_h - 8, 16))
    is_long = _wrapped(response) > stream_lines

    words = response.split(" ")
    shown = ""
    chunk_size = 1
    t0 = time.time()

    with Live(
        _make_panel(Markdown("")),
        console=console, refresh_per_second=20,
        transient=is_long,
    ) as live:
        i = 0
        while i < len(words):
            shown += " ".join(words[i:i + chunk_size]) + " "
            if is_long and (_wrapped(shown) >= stream_lines or time.time() - t0 > 1.5):
                break
            live.update(_make_panel(Markdown(shown)))
            i += chunk_size
            chunk_size = random.randint(1, 4)
            time.sleep(random.uniform(0.03, 0.07))

    if is_long:
        console.print(_make_panel(Markdown(response)))


# ── Type-ahead chat ────────────────────────────────────────────────────

def _chat_with_typeahead(agent, user_input, session, status_text):
    """Run ``agent.chat(user_input)`` in a worker thread while the user composes
    their next message at a live prompt.

    Returns ``(response, error, next_text, submitted, interrupted)``:
        response    - the agent's reply, or None if chat() raised
        error       - the exception if chat() raised, else None
        next_text   - whatever the user typed while waiting (may be "")
        submitted   - True if the user pressed Enter (next_text is ready to send),
                      False if it is an unfinished draft (or nothing was sent)
        interrupted - True if the user cancelled this turn (Ctrl-C / Esc)
    """
    global _queue_tip_shown
    holder: dict = {}
    by_worker = {"flag": False}

    def _worker():
        try:
            holder["resp"] = agent.chat(user_input)
        except Exception as e:  # surfaced to the caller; never crash the thread
            holder["err"] = e
        # When generation finishes, close whichever live prompt is currently open
        # so its answer can render. The prompt may be momentarily between renders
        # (e.g. just after the user pressed Enter, before it reopens), so retry
        # briefly until we catch it running.
        app = session.app

        def _close():
            if app.is_running:
                by_worker["flag"] = True
                app.exit(result=app.current_buffer.text)

        for _ in range(100):
            try:
                if app.is_running and app.loop is not None:
                    app.loop.call_soon_threadsafe(_close)
                    return
            except (RuntimeError, AttributeError):
                pass
            time.sleep(0.05)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    queued = None    # an Enter-submitted follow-up, kept separate from the live
                     # buffer so it can't be lost by later edits/clears
    tail = ""        # an unsent draft in the buffer when the answer arrives
    notified = False
    tip = _pick_tip()

    def _live_bar():
        frame = _SPINNER[int(time.time() * 5) % len(_SPINNER)]
        parts = [status_text[k] for k in ("main", "detail", "stats") if status_text.get(k)]
        line = "  ·  ".join(parts) if parts else "working…"
        suffix = "next queued ✓ · Esc to interrupt" if queued else "Esc to interrupt"
        return f"{frame} {line}    {suffix}"

    # The prompts leave no echo (the session uses erase_when_done): a queued
    # message is shown explicitly right before its answer instead, keeping the
    # chat ordered. Stay interactive the WHOLE time the model works (animated
    # status bar + Esc-to-interrupt); a submitted follow-up is stored in `queued`
    # and the prompt reopens empty, so it can't be lost by editing/clearing.
    while worker.is_alive():
        try:
            entered = session.prompt(
                _make_chrome(_live_bar, tip),
                refresh_interval=0.5,
            )
        except (KeyboardInterrupt, EOFError):
            agent.request_cancel()   # cooperative cancel; stay interactive
            continue

        if by_worker["flag"]:
            tail = entered           # worker finished; this is an unsent draft
            break

        if entered.strip().lower() in ("/empty_queue", "/empty-queue"):
            console.print(f"  [{C_DARK}]queue cleared[/{C_DARK}]" if queued
                          else f"  [{C_DARK}]nothing queued[/{C_DARK}]")
            queued = None
            continue

        if entered.strip():          # Enter while busy → queue it, reopen empty
            queued = entered.strip()
            if not notified:
                console.print(f"  [{C_DARK}]↩ queued — sent once the current answer "
                              f"is ready[/{C_DARK}]")
                notified = True
            # First queue this session: surface how to cancel it, in the tip row.
            if not _queue_tip_shown:
                tip = _QUEUE_TIP
                _queue_tip_shown = True

    worker.join()  # near-instant: worker has finished or is closing the prompt
    if agent.is_cancelled():
        return holder.get("resp"), holder.get("err"), "", False, True
    if queued:
        return holder.get("resp"), holder.get("err"), queued, True, False
    return holder.get("resp"), holder.get("err"), tail, False, False


# ── Trace display ──────────────────────────────────────────────────────

MAX_RESULT_PREVIEW = 500  # chars to show from tool results


def _render_trace(trace: dict):
    """Render a full reasoning trace with Rich."""
    from rich.tree import Tree
    from rich.syntax import Syntax

    tree = Tree(f"[bold {C_BRIGHT}]Request:[/bold {C_BRIGHT}] {trace['user_message'][:80]}")

    stats = (
        f"{trace['total_input_tokens']}↑ {trace['total_output_tokens']}↓ tokens  •  "
        f"{trace['total_time']:.1f}s total  •  "
        f"{sum(len(r['tool_calls']) for r in trace['rounds'])} tool calls  •  "
        f"{len(trace['rounds'])} rounds"
    )
    tree.add(f"[{C_DARK}]{stats}[/{C_DARK}]")

    for ri, round_data in enumerate(trace["rounds"], 1):
        round_node = tree.add(f"[bold {C_MID}]Round {ri}[/bold {C_MID}]  [{C_DARK}]({round_data['api_time']:.1f}s, {round_data['input_tokens']}↑ {round_data['output_tokens']}↓)[/{C_DARK}]")

        # Show thinking text if any
        thinking = round_data.get("thinking", "").strip()
        if thinking:
            preview = thinking[:200] + ("..." if len(thinking) > 200 else "")
            round_node.add(f"[italic {C_DARK}]💭 {preview}[/italic {C_DARK}]")

        # Show tool calls
        for tc in round_data["tool_calls"]:
            status_icon = "❌" if tc["error"] else "✓"
            tool_node = round_node.add(
                f"[bold {C_MID}]{status_icon} {tc['name']}[/bold {C_MID}]  [{C_DARK}]({tc['duration']:.1f}s)[/{C_DARK}]"
            )

            # Input params
            input_str = json.dumps(tc["input"], indent=2, default=str)
            if len(input_str) > MAX_RESULT_PREVIEW:
                input_str = input_str[:MAX_RESULT_PREVIEW] + f"\n... ({len(input_str)} chars total)"
            tool_node.add(Panel(
                Syntax(input_str, "json", theme="monokai", word_wrap=True),
                title="input", border_style=C_DEEP, padding=(0, 1),
            ))

            # Result preview
            result_str = tc["result"]
            if len(result_str) > MAX_RESULT_PREVIEW:
                result_str = result_str[:MAX_RESULT_PREVIEW] + f"\n... ({len(tc['result'])} chars total)"
            tool_node.add(Panel(
                Syntax(result_str, "json", theme="monokai", word_wrap=True),
                title="result", border_style=C_DEEP, padding=(0, 1),
            ))

    console.print()
    console.print(Panel(tree, title=f"[bold {C_BRIGHT}]/showwhy[/bold {C_BRIGHT}]", border_style=C_DEEP, padding=(1, 2)))


def _pick_trace(traces: list[dict], session) -> dict | None:
    """Show recent requests and let user pick one."""
    if not traces:
        console.print(f"  [{C_DARK}]no traces yet — ask a question first[/{C_DARK}]")
        return None

    recent = traces[-10:]  # last 10
    console.print()
    for i, t in enumerate(recent, 1):
        tool_names = []
        for r in t["rounds"]:
            for tc in r["tool_calls"]:
                tool_names.append(tc["name"])
        msg_preview = t["user_message"][:60] + ("..." if len(t["user_message"]) > 60 else "")
        if tool_names:
            tools_str = ", ".join(tool_names)
            console.print(f"  [{C_MID}]{i}[/{C_MID}]  {msg_preview}  [{C_DARK}]({tools_str} · {t['total_time']:.1f}s)[/{C_DARK}]")
        else:
            console.print(f"  [{C_MID}]{i}[/{C_MID}]  {msg_preview}  [{C_DARK}](no tools · {t['total_time']:.1f}s)[/{C_DARK}]")

    console.print(f"  [{C_DARK}]enter number to expand, or press Enter to go back[/{C_DARK}]")
    try:
        choice = session.prompt(
            HTML(f"<style fg='{C_BRIGHT}'>/showwhy #> </style>"),
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(recent):
            return recent[idx]
    except ValueError:
        pass
    console.print(f"  [{C_DARK}]invalid choice[/{C_DARK}]")
    return None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    _stream_enabled = os.environ.get("LONGEVITYCLAW_STREAM", "1") != "0"

    import argparse
    parser = argparse.ArgumentParser(description="LongevityClaw - AI aging clock agent")
    parser.add_argument("--model", default=None,
                        help="Claude model (default: from env)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query mode (non-interactive)")
    parser.add_argument("-d", action="store_true", default=False,
                        help="Demo mode")
    args = parser.parse_args()

    status = StatusLine()
    # Shared with the type-ahead toolbar: StatusLine.update() no-ops when the
    # Rich Live region isn't started, so these callbacks are safe in both modes.
    status_text = {"main": "", "detail": "", "stats": ""}

    def on_status(msg: str):
        status_text["main"] = msg
        status.update(msg)

    def on_detail(msg: str):
        status_text["detail"] = msg
        status.detail(msg)

    def on_stats(msg: str):
        status_text["stats"] = msg
        status.stats(msg)

    from .agent import LongevityClawAgent

    status.start()
    agent = LongevityClawAgent(model=args.model, on_status=on_status, on_detail=on_detail, on_stats=on_stats)
    status.stop()

    # Single query mode
    if args.query:
        status.start()
        response = agent.chat(args.query)
        status.stop()
        console.print()
        console.print(Markdown(response))
        return

    console.print(BANNER)

    demo_mode = args.d
    demo_queue = [
        "hello, who are you?",
        "If you were to develop a drug that inhibits a protein that is derived from both "
        "methylation and protein aging clocks (scoring high) and is likely to reduce biological "
        "age of the system when later measured by both of these clocks, what would this protein "
        "be? Name one and make a list of top-10 like this",
    ] if demo_mode else []

    _request_count = 0
    _paste_store = {}
    _paste_counter = [0]

    kb = KeyBindings()

    @kb.add("escape")
    def _(event):
        buf = event.current_buffer
        if not buf.text:
            # Empty input while the model is working → interrupt the request.
            if agent.is_generating():
                agent.request_cancel()
            return
        _paste_store.clear()
        _paste_counter[0] = 0
        buf.reset()

    @kb.add("backspace")
    def _(event):
        buf = event.current_buffer
        before = buf.text[:buf.cursor_position]
        m = re.search(r"\x00PASTE:(\d+):(\d+)\x00$", before)
        if m:
            _paste_store.pop(int(m.group(1)), None)
            buf.text = buf.text[:m.start()] + buf.text[buf.cursor_position:]
            buf.cursor_position = m.start()
        elif buf.cursor_position > 0:
            buf.delete_before_cursor(1)

    @kb.add("delete")
    def _(event):
        buf = event.current_buffer
        after = buf.text[buf.cursor_position:]
        m = re.match(r"\x00PASTE:(\d+):(\d+)\x00", after)
        if m:
            _paste_store.pop(int(m.group(1)), None)
            buf.text = buf.text[:buf.cursor_position] + buf.text[buf.cursor_position + m.end():]
        else:
            buf.delete(1)

    @kb.add("left")
    def _(event):
        buf = event.current_buffer
        if buf.cursor_position > 0:
            before = buf.text[:buf.cursor_position]
            m = re.search(r"\x00PASTE:\d+:\d+\x00$", before)
            buf.cursor_position = m.start() if m else buf.cursor_position - 1

    @kb.add("right")
    def _(event):
        buf = event.current_buffer
        if buf.cursor_position < len(buf.text):
            after = buf.text[buf.cursor_position:]
            m = re.match(r"\x00PASTE:\d+:\d+\x00", after)
            buf.cursor_position += m.end() if m else 1

    @kb.add(Keys.BracketedPaste)
    def _(event):
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        if len(data) > PASTE_COLLAPSE_THRESHOLD:
            _paste_counter[0] += 1
            n = _paste_counter[0]
            _paste_store[n] = data
            marker = f"\x00PASTE:{n}:{len(data)}\x00"
            event.current_buffer.insert_text(marker)
        else:
            event.current_buffer.insert_text(data)

    # Set up prompt_toolkit session with history and autocomplete
    history_path = Path(__file__).resolve().parent.parent / ".longevityclaw_history"
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=LongevityClawCompleter(),
        complete_while_typing=True,
        key_bindings=kb,
        input_processors=[PasteCollapseProcessor()],
        # Keep the input area "locked": every prompt erases its rendered chrome
        # (bar + tip + input) on submit, so the separator never piles up in the
        # scrollback. The submitted message is echoed as a plain "you> …" line.
        erase_when_done=True,
        style=PTStyle.from_dict({
            "prompt": f"bold {C_MID}",
            "completion-menu": f"bg:#0a1a1e {C_MID}",
            "completion-menu.completion": f"bg:#0a1a1e {C_MID}",
            "completion-menu.completion.current": f"bg:{C_BRIGHT} #0a1a1e bold",
            "completion-menu.meta.completion": f"bg:#0a1a1e {C_DARK} italic",
            "completion-menu.meta.completion.current": f"bg:{C_BRIGHT} {C_DEEP} italic",
            "paste-collapsed": f"{C_DARK} italic",
            # Prompt chrome: bright separator/status bar, dim tip row, input label
            "statusbar": f"{C_BRIGHT} bold",
            "tip": f"{C_DARK} italic",
            "you": f"bold {C_MID}",
        }),
    )

    pending = None          # a ready-to-send message captured during type-ahead
    pending_default = ""    # an unfinished draft to pre-fill the next prompt

    while True:
        try:
            console.print()
            demo_echoed = False
            if pending is not None:
                user_input = pending
                pending = None
            elif demo_queue:
                demo_text = demo_queue.pop(0)
                sys.stdout.write(f"\033[1;38;2;59;193;168myou>\033[0m ")
                sys.stdout.flush()
                for ch in demo_text:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    time.sleep(0.04)
                time.sleep(0.5)
                sys.stdout.write("\n")
                sys.stdout.flush()
                user_input = demo_text
                demo_echoed = True
            else:
                user_input = session.prompt(
                    _make_chrome(lambda: _idle_bar_text(agent.effort), _pick_tip()),
                    default=pending_default,
                ).strip()
                user_input = PASTE_MARKER_RE.sub(
                    lambda m: _paste_store.pop(int(m.group(1)), ""),
                    user_input,
                )
                _paste_counter[0] = 0
                _paste_store.clear()
            pending_default = ""
        except KeyboardInterrupt:
            _paste_store.clear()
            _paste_counter[0] = 0
            pending_default = ""
            continue
        except EOFError:
            console.print(f"\n[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
            break

        if not user_input:
            continue

        # The prompt erased its own chrome — echo the message into the scrollback
        # as a plain "you> …" line so the history stays clean and ordered. (Demo
        # mode already printed it via the typing animation.)
        if not demo_echoed:
            console.print(f"[bold {C_MID}]you>[/bold {C_MID}] {user_input}")

        # ── Handle slash commands ──────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit"):
                console.print(f"[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
                break

            elif cmd == "/help":
                console.print(COMMAND_HELP)
                continue

            elif cmd == "/clear":
                agent.messages = []
                console.print(f"[{C_DARK}]  conversation cleared[/{C_DARK}]")
                continue

            elif cmd == "/model":
                console.print(f"[{C_DARK}]  model: {agent.model}[/{C_DARK}]")
                continue

            elif cmd == "/clocks":
                from .clock_db import get_db
                db = get_db()
                mods = db.list_modalities()
                total = sum(mods.values())
                console.print(f"[bold]  {total} clocks across {len(mods)} modalities:[/bold]")
                for mod, n in sorted(mods.items(), key=lambda x: -x[1]):
                    bar = "█" * (n // 3)
                    console.print(f"    [{C_MID}]{mod:20s}[/{C_MID}] {n:>4d}  [{C_BRIGHT}]{bar}[/{C_BRIGHT}]")
                continue

            elif cmd == "/skills":
                from .skills import list_skills
                saved = list_skills()
                if not saved:
                    console.print(f"[{C_DARK}]  no skills saved yet — finish a task, then say "
                                  f"'save this as a skill'[/{C_DARK}]")
                else:
                    console.print(f"[bold]  {len(saved)} saved skill{'s' if len(saved) != 1 else ''}:[/bold]")
                    for s in saved:
                        console.print(f"    [{C_MID}]/{s['slug']}[/{C_MID}]  [{C_DARK}]{s['description']}[/{C_DARK}]")
                continue

            elif cmd == "/allowed":
                from . import fs_access
                console.print(f"[bold]  filesystem policy:[/bold] [{C_MID}]{fs_access.get_policy()}[/{C_MID}]")
                console.print(f"[{C_DARK}]  the agent may access:[/{C_DARK}]")
                for root in fs_access.get_allowed_roots():
                    console.print(f"    [{C_MID}]{root}[/{C_MID}]")
                continue

            elif cmd == "/grant":
                from . import fs_access
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    console.print(f"[{C_DARK}]  usage: /grant <path>[/{C_DARK}]")
                    continue
                target = Path(parts[1].strip().strip('"\'')).expanduser()
                if not target.exists():
                    console.print(f"[{C_DARK}]  no such path: {target}[/{C_DARK}]")
                    continue
                granted = fs_access.grant_path(target)
                console.print(f"[{C_MID}]  granted access:[/{C_MID}] {granted}")
                continue

            elif cmd == "/loop":
                rest = user_input[len("/loop"):].strip()
                interval, task = _parse_loop_args(rest) if rest else (0, "")
                if not task:
                    console.print(f"[{C_DARK}]  usage: /loop [freq] <task> — e.g. "
                                  f"/loop 30m check PubMed for new GrimAge papers[/{C_DARK}]")
                    continue
                _run_loop(agent, task, interval, status, _stream_enabled)
                continue

            elif cmd == "/empty_queue":
                console.print(f"[{C_DARK}]  queue cleared[/{C_DARK}]" if pending
                              else f"[{C_DARK}]  nothing queued[/{C_DARK}]")
                pending = None
                continue

            elif cmd == "/effort":
                from .agent import EFFORT_LEVELS

                def _cap_str(lvl):
                    cap = EFFORT_LEVELS[lvl]
                    return "unlimited" if cap is None else f"{cap} tool calls/response"

                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    lvl = parts[1].strip().lower()
                    if agent.set_effort(lvl):
                        console.print(f"[{C_MID}]  effort: {lvl}[/{C_MID}] [{C_DARK}]({_cap_str(lvl)})[/{C_DARK}]")
                    else:
                        console.print(f"[{C_DARK}]  usage: /effort low|medium|high|max[/{C_DARK}]")
                else:
                    console.print(f"[{C_MID}]  effort: {agent.effort}[/{C_MID}] [{C_DARK}]({_cap_str(agent.effort)})[/{C_DARK}]")
                    console.print(f"[{C_DARK}]  set with: /effort low (5) · medium (10) · high (20) · max (unlimited)"
                                  f" — skills bypass the cap[/{C_DARK}]")
                continue

            elif cmd == "/usage":
                _show_usage(agent)
                continue

            elif cmd == "/save":
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else None
                _save_conversation(agent.messages, filename)
                continue

            elif cmd == "/showwhy":
                trace = _pick_trace(agent.traces, session)
                if trace:
                    _render_trace(trace)
                continue

            else:
                # Not a built-in command — maybe it's a saved skill (/<slug>)
                from .skills import get_skill
                skill = get_skill(cmd[1:])
                if skill:
                    rest = user_input[len(cmd):].strip()
                    user_input = (f"Run my saved skill '{skill['name']}'."
                                  + (f" Inputs: {rest}" if rest else ""))
                    # fall through to chat — the agent will call run_skill
                else:
                    console.print(f"[{C_DARK}]  unknown command: {cmd} (try /help or /skills)[/{C_DARK}]")
                    continue

        # ── Handle quit without slash ──────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            console.print(f"[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
            break

        # ── Chat with agent ────────────────────────────────────────
        _request_count += 1

        # Type-ahead: while the model works, let the user compose the next
        # message. Disabled in demo mode and when streaming is turned off.
        if _stream_enabled and not demo_queue:
            response, err, next_text, submitted, interrupted = _chat_with_typeahead(
                agent, user_input, session, status_text)
            if interrupted:
                console.print(f"\n  [dim italic]Interrupted by user[/dim italic]")
            elif err is not None:
                console.print(f"\n[bold {C_DEEP}]error>[/bold {C_DEEP}] {err}")
            elif response is not None:
                console.print()
                _stream_response(response)
                _print_tool_hint(agent, _request_count)

            clean = PASTE_MARKER_RE.sub(
                lambda m: _paste_store.pop(int(m.group(1)), ""), next_text or "")
            _paste_counter[0] = 0
            _paste_store.clear()
            if clean.strip():
                if submitted:
                    pending = clean.strip()          # send it next, in order
                else:
                    pending_default = clean          # restore the draft
            continue

        try:
            status.start()
            response = agent.chat(user_input)
            status.stop()

            console.print()
            if _stream_enabled:
                _stream_response(response)
            else:
                console.print(Panel(
                    Markdown(response),
                    title=f"[bold {C_BRIGHT}]longevityclaw[/bold {C_BRIGHT}]",
                    border_style=C_DEEP,
                    padding=(1, 2),
                ))
            _print_tool_hint(agent, _request_count)
        except KeyboardInterrupt:
            status.stop()
            console.print(f"\n  [dim italic]Interrupted by user[/dim italic]")
        except Exception as e:
            status.stop()
            console.print(f"\n[bold {C_DEEP}]error>[/bold {C_DEEP}] {e}")


if __name__ == "__main__":
    main()
