"""
LongevityClaw web server: serves xterm.js terminal over Socket.IO + file upload.

Spawns the LongevityClaw CLI in a PTY and bridges it to the browser via Socket.IO,
preserving Rich rendering, prompt_toolkit autocomplete, and live status updates.

Socket.IO provides automatic fallback from WebSocket to HTTP long-polling,
which is essential for Cloudflare tunnels that strip WebSocket Upgrade headers.

Security: when --password is set, all routes require a session token obtained
by posting the correct password to /api/auth. The token is stored in a cookie.
"""

import asyncio
import fcntl
import hashlib
import hmac
import os
import pty
import secrets
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import socketio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"

# Auth state (set at startup if --password is used)
_password_hash: str | None = None
_auth_tokens: dict[str, float] = {}  # token -> expiry timestamp
TOKEN_LIFETIME = 24 * 3600  # 24 hours


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _check_auth(request: Request) -> bool:
    """Return True if request is authenticated (or no password set)."""
    if _password_hash is None:
        return True
    token = request.cookies.get("longevityclaw_token")
    if not token:
        return False
    expiry = _auth_tokens.get(token)
    if not expiry or time.time() > expiry:
        _auth_tokens.pop(token, None)
        return False
    return True


def _check_ws_auth(token: str | None) -> bool:
    """Check WebSocket auth via query param."""
    if _password_hash is None:
        return True
    if not token:
        return False
    expiry = _auth_tokens.get(token)
    if not expiry or time.time() > expiry:
        _auth_tokens.pop(token, None)
        return False
    return True


app = FastAPI(title="LongevityClaw")

# CORS - allow all origins for tunnel access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO with CORS and long-polling fallback
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)
socket_app = socketio.ASGIApp(sio, app)

# Store active PTY sessions: sid -> {pid, master_fd, reader_task}
_pty_sessions: dict[str, dict] = {}

# Debug middleware to log ALL requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import sys
    print(f"[REQ] {request.method} {request.url.path} | Headers: upgrade={request.headers.get('upgrade', 'none')}, connection={request.headers.get('connection', 'none')}", file=sys.stderr, flush=True)
    response = await call_next(request)
    print(f"[RES] {request.url.path} -> {response.status_code}", file=sys.stderr, flush=True)
    return response


# ── Socket.IO Terminal Events ────────────────────────────────────────────
@sio.event
async def connect(sid, environ):
    """Handle Socket.IO connection - auth check happens in 'auth' event."""
    print(f"[SIO] Client {sid} connected (transport: {environ.get('HTTP_UPGRADE', 'polling')})", file=sys.stderr, flush=True)
    return True


@sio.event
async def disconnect(sid):
    """Clean up PTY session on disconnect."""
    print(f"[SIO] Client {sid} disconnected", file=sys.stderr, flush=True)
    session = _pty_sessions.pop(sid, None)
    if session:
        try:
            session["reader_task"].cancel()
        except Exception:
            pass
        try:
            os.kill(session["pid"], signal.SIGTERM)
            os.waitpid(session["pid"], 0)
        except (OSError, ChildProcessError):
            pass
        try:
            os.close(session["master_fd"])
        except OSError:
            pass


@sio.event
async def auth(sid, data):
    """Authenticate and start terminal session."""
    token = data.get("token") if isinstance(data, dict) else None
    print(f"[SIO] Auth request from {sid}, token={'yes' if token else 'no'}", file=sys.stderr, flush=True)

    if not _check_ws_auth(token):
        print(f"[SIO] Auth failed for {sid}", file=sys.stderr, flush=True)
        await sio.emit("auth_error", {"error": "Unauthorized"}, to=sid)
        await sio.disconnect(sid)
        return

    print(f"[SIO] Auth OK for {sid}, starting PTY", file=sys.stderr, flush=True)

    # Spawn PTY with longevityclaw CLI
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    env["COLUMNS"] = "120"
    env["LINES"] = "40"
    # Hosted, multi-user context: disable agent filesystem tools entirely.
    env["LONGEVITYCLAW_FS"] = "off"
    if token:
        env["LONGEVITYCLAW_SESSION_ID"] = token

    project_root = Path(__file__).resolve().parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", 40, 120, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    pid = os.fork()
    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        os.chdir(str(project_root))
        os.execve(
            str(venv_python),
            [str(venv_python), "-m", "longevityclaw.cli"],
            env,
        )

    # Parent process
    os.close(slave_fd)

    loop = asyncio.get_event_loop()

    # Read from PTY -> send to Socket.IO client
    async def pty_reader():
        try:
            while True:
                data = await loop.run_in_executor(None, os.read, master_fd, 4096)
                if not data:
                    break
                await sio.emit("output", data, to=sid)
        except (OSError, Exception) as e:
            print(f"[SIO] PTY reader error for {sid}: {e}", file=sys.stderr, flush=True)

    reader_task = asyncio.create_task(pty_reader())

    _pty_sessions[sid] = {
        "pid": pid,
        "master_fd": master_fd,
        "reader_task": reader_task,
    }

    await sio.emit("auth_ok", {}, to=sid)


@sio.event
async def input(sid, data):
    """Handle terminal input from client."""
    session = _pty_sessions.get(sid)
    if not session:
        return

    try:
        if isinstance(data, bytes):
            os.write(session["master_fd"], data)
        elif isinstance(data, str):
            os.write(session["master_fd"], data.encode("utf-8"))
    except OSError as e:
        print(f"[SIO] Input write error for {sid}: {e}", file=sys.stderr, flush=True)


@sio.event
async def resize(sid, data):
    """Handle terminal resize from client."""
    session = _pty_sessions.get(sid)
    if not session:
        return

    try:
        rows = int(data.get("rows", 40))
        cols = int(data.get("cols", 120))
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(session["master_fd"], termios.TIOCSWINSZ, winsize)
        os.kill(session["pid"], signal.SIGWINCH)
    except (ValueError, OSError) as e:
        print(f"[SIO] Resize error for {sid}: {e}", file=sys.stderr, flush=True)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index(request: Request):
    if _password_hash is not None and not _check_auth(request):
        return HTMLResponse((STATIC_DIR / "login.html").read_text())
    # Inject the auth token into the page so JS can use it for WebSocket
    token = request.cookies.get("longevityclaw_token", "")
    html = (STATIC_DIR / "index.html").read_text()
    html = html.replace("__AUTH_TOKEN__", token)
    return HTMLResponse(html)


@app.post("/api/auth")
async def authenticate(request: Request):
    """Authenticate with password, returns a session token cookie."""
    if _password_hash is None:
        return JSONResponse({"ok": True})
    body = await request.json()
    pw = body.get("password", "")
    if not hmac.compare_digest(_hash_password(pw), _password_hash):
        return JSONResponse({"error": "Wrong password"}, status_code=403)
    token = secrets.token_urlsafe(32)
    _auth_tokens[token] = time.time() + TOKEN_LIFETIME
    resp = JSONResponse({"ok": True, "token": token})
    # Use samesite="lax" to allow cookie through Cloudflare tunnel
    # "strict" blocks cross-site requests which breaks through tunnels
    resp.set_cookie("longevityclaw_token", token, max_age=TOKEN_LIFETIME,
                     httponly=True, samesite="lax")
    return resp


@app.get("/api/token")
async def get_token(request: Request):
    """Return auth token for WebSocket - not cached by CDN."""
    if _password_hash is None:
        return JSONResponse({"token": ""})
    token = request.cookies.get("longevityclaw_token", "")
    if not token or token not in _auth_tokens:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    resp = JSONResponse({"token": token})
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


@app.get("/api/uploads")
async def list_uploads(request: Request):
    """List uploaded files."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            size = f.stat().st_size
            files.append({"name": f.name, "size": size})
    return JSONResponse({"files": files})


# ── Pathway API endpoints ────────────────────────────────────────────

@app.get("/api/pathways")
async def get_pathways(request: Request, min_genes: int = 5, top_n: int = 50):
    """Get ranked pathways by aging clock evidence."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.pathway_generator import rank_all_pathways
    results = rank_all_pathways(min_genes)[:top_n]
    return JSONResponse({
        "pathways": [
            {
                "id": p.pathway_id,
                "name": p.pathway_name,
                "n_genes": p.n_genes,
                "n_genes_in_clocks": p.n_genes_in_clocks,
                "coverage": round(p.coverage, 3),
                "mean_coefficient": round(p.mean_abs_coefficient, 4),
                "consistency": round(p.coefficient_consistency, 3),
                "clock_count": p.clock_count,
                "aging_score": round(p.aging_score, 3),
                "lopez_otin": list(p.lopez_otin_hallmarks),
                "top_genes": [{"gene": g, "score": round(s, 4)} for g, s in p.top_genes[:5]],
            }
            for p in results
        ]
    })


@app.get("/api/pathways/{pathway_id}")
async def get_pathway_detail(request: Request, pathway_id: str):
    """Get detailed pathway evidence."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.pathway_generator import score_pathway, get_pathway_intervention_targets
    p = score_pathway(pathway_id)
    if p is None:
        return JSONResponse({"error": f"Unknown pathway: {pathway_id}"}, status_code=404)
    targets = get_pathway_intervention_targets(pathway_id)
    return JSONResponse({
        "pathway": {
            "id": p.pathway_id,
            "name": p.pathway_name,
            "n_genes": p.n_genes,
            "n_genes_in_clocks": p.n_genes_in_clocks,
            "coverage": round(p.coverage, 3),
            "mean_coefficient": round(p.mean_abs_coefficient, 4),
            "consistency": round(p.coefficient_consistency, 3),
            "clock_count": p.clock_count,
            "aging_score": round(p.aging_score, 3),
            "lopez_otin": list(p.lopez_otin_hallmarks),
            "top_genes": [{"gene": g, "score": round(s, 4)} for g, s in p.top_genes],
        },
        "targets": targets[:10],
    })


@app.post("/api/pathways/synergies")
async def find_synergies(request: Request):
    """Find synergistic pathway combinations."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    pathway_ids = body.get("pathway_ids", [])
    min_shared = body.get("min_shared", 2)
    from longevityclaw.pathway_generator import find_pathway_synergies
    synergies = find_pathway_synergies(pathway_ids, min_shared)[:15]
    return JSONResponse({
        "synergies": [
            {
                "pathways": list(s.pathway_ids),
                "shared_genes": list(s.shared_genes)[:20],
                "combined_coverage": round(s.combined_coverage, 3),
                "synergy_score": round(s.synergy_score, 4),
            }
            for s in synergies
        ]
    })


@app.post("/api/pathways/hypothesis")
async def generate_hypothesis(request: Request):
    """Generate pathway hypothesis from seed genes."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    seed_genes = body.get("seed_genes", [])
    expansion_depth = body.get("expansion_depth", 1)
    if not seed_genes:
        return JSONResponse({"error": "seed_genes required"}, status_code=400)
    from longevityclaw.pathway_generator import generate_pathway_hypothesis
    result = generate_pathway_hypothesis(seed_genes, expansion_depth)
    return JSONResponse(result)


# ── DrugAge API endpoints ───────────────────────────────────────────────

@app.get("/api/drugage")
async def get_drugage_compounds(request: Request, min_studies: int = 2, top_n: int = 50):
    """Get ranked longevity compounds from DrugAge."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.drugage import rank_all_compounds
    results = rank_all_compounds(min_studies)[:top_n]
    return JSONResponse({
        "compounds": [
            {
                "compound": c.compound,
                "n_studies": c.n_studies,
                "n_species": c.n_species,
                "species": c.species_list,
                "mean_lifespan_change": c.mean_lifespan_change,
                "max_lifespan_change": c.max_lifespan_change,
                "consistency": c.consistency,
                "itp_validated": c.itp_validated,
                "longevity_score": c.longevity_score,
            }
            for c in results
        ]
    })


@app.get("/api/drugage/itp")
async def get_itp_compounds(request: Request):
    """Get ITP-validated compounds (gold standard interventions)."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.drugage import get_itp_compounds
    results = get_itp_compounds()
    return JSONResponse({
        "compounds": [
            {
                "compound": c.compound,
                "n_studies": c.n_studies,
                "n_species": c.n_species,
                "mean_lifespan_change": c.mean_lifespan_change,
                "max_lifespan_change": c.max_lifespan_change,
                "consistency": c.consistency,
                "longevity_score": c.longevity_score,
            }
            for c in results
        ]
    })


@app.get("/api/drugage/search")
async def search_drugage(request: Request, q: str = "", limit: int = 20):
    """Search compounds by name."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not q:
        return JSONResponse({"compounds": []})
    from longevityclaw.drugage import search_compounds
    results = search_compounds(q, limit)
    return JSONResponse({
        "compounds": [
            {
                "compound": c.compound,
                "n_studies": c.n_studies,
                "n_species": c.n_species,
                "mean_lifespan_change": c.mean_lifespan_change,
                "itp_validated": c.itp_validated,
                "longevity_score": c.longevity_score,
            }
            for c in results
        ]
    })


@app.get("/api/drugage/species")
async def get_species_list(request: Request):
    """Get list of species with study counts."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.drugage import get_species_list
    species = get_species_list()
    return JSONResponse({
        "species": [{"name": name, "count": count} for name, count in species]
    })


# NOTE: This route MUST be last among /api/drugage/* because it catches all patterns
@app.get("/api/drugage/{compound}")
async def get_compound_detail(request: Request, compound: str):
    """Get detailed compound information with pathway links."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    from longevityclaw.drugage import score_compound, link_compound_to_pathways
    score = score_compound(compound)
    if score is None:
        return JSONResponse({"error": f"Compound not found: {compound}"}, status_code=404)
    links = link_compound_to_pathways(compound)
    return JSONResponse({
        "compound": {
            "name": score.compound,
            "n_studies": score.n_studies,
            "n_species": score.n_species,
            "species": score.species_list,
            "mean_lifespan_change": score.mean_lifespan_change,
            "max_lifespan_change": score.max_lifespan_change,
            "consistency": score.consistency,
            "itp_validated": score.itp_validated,
            "longevity_score": score.longevity_score,
            "studies": [
                {
                    "species": s.species,
                    "strain": s.strain,
                    "dosage": s.dosage,
                    "avg_change": s.avg_lifespan_change,
                    "max_change": s.max_lifespan_change,
                    "gender": s.gender,
                    "pubmed": s.pubmed_id,
                }
                for s in score.top_studies
            ],
        },
        "pathway_links": links,
    })


# ── L-LLM Direct API endpoints ───────────────────────────────────────────

@app.post("/api/llm/pathway-score")
async def llm_pathway_score(request: Request):
    """Score pathway relevance to aging using L-LLM directly."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    import os
    from longevityclaw.llm_client import get_current_backend
    backend = get_current_backend()
    if backend == "local":
        if not os.environ.get("VLLM_ENDPOINT"):
            return JSONResponse({"error": "VLLM_ENDPOINT not configured for local backend"}, status_code=503)
    elif not os.environ.get("HF_TOKEN"):
        return JSONResponse({"error": "HF_TOKEN not configured for HuggingFace backend"}, status_code=503)

    body = await request.json()
    pathway_name = body.get("pathway_name")
    genes = body.get("genes", [])
    context = body.get("context")

    if not pathway_name or not genes:
        return JSONResponse({"error": "pathway_name and genes required"}, status_code=400)

    try:
        from longevityclaw.llm_client import score_pathway_with_llm
        result = score_pathway_with_llm(pathway_name, genes, context)
        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        return JSONResponse({"error": f"L-LLM query failed: {e}"}, status_code=500)


@app.post("/api/llm/lifespan-predict")
async def llm_lifespan_predict(request: Request):
    """Predict compound lifespan effect using L-LLM directly."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    import os
    from longevityclaw.llm_client import get_current_backend
    backend = get_current_backend()
    if backend == "local":
        if not os.environ.get("VLLM_ENDPOINT"):
            return JSONResponse({"error": "VLLM_ENDPOINT not configured for local backend"}, status_code=503)
    elif not os.environ.get("HF_TOKEN"):
        return JSONResponse({"error": "HF_TOKEN not configured for HuggingFace backend"}, status_code=503)

    body = await request.json()
    compound = body.get("compound")
    species = body.get("species", "mice")
    pubmed_abstract = body.get("pubmed_abstract")

    if not compound:
        return JSONResponse({"error": "compound required"}, status_code=400)

    try:
        from longevityclaw.llm_client import predict_lifespan_effect
        result = predict_lifespan_effect(compound, species, pubmed_abstract)

        # Transform to frontend-expected format
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        transformed = {
            "predicted_effect": result.get("lifespan_change_percent"),
            "confidence": confidence_map.get(str(result.get("confidence", "")).lower(), 0.5),
            "reasoning": result.get("reasoning"),
            "mechanisms": result.get("mechanisms"),
            "raw_response": result.get("raw_response"),
        }
        return JSONResponse({"success": True, "result": transformed})
    except Exception as e:
        return JSONResponse({"error": f"L-LLM query failed: {e}"}, status_code=500)


@app.post("/api/llm/mechanism")
async def llm_mechanism(request: Request):
    """Analyze aging mechanism using L-LLM directly."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    import os
    from longevityclaw.llm_client import get_current_backend
    backend = get_current_backend()
    if backend == "local":
        if not os.environ.get("VLLM_ENDPOINT"):
            return JSONResponse({"error": "VLLM_ENDPOINT not configured for local backend"}, status_code=503)
    elif not os.environ.get("HF_TOKEN"):
        return JSONResponse({"error": "HF_TOKEN not configured for HuggingFace backend"}, status_code=503)

    body = await request.json()
    query = body.get("query")

    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    try:
        from longevityclaw.llm_client import analyze_aging_mechanism
        response = analyze_aging_mechanism(query)
        return JSONResponse({
            "success": True,
            "result": {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
            }
        })
    except Exception as e:
        return JSONResponse({"error": f"L-LLM query failed: {e}"}, status_code=500)


@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a data file (CSV/TSV) to be accessible by the agent."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = Path(file.filename).name
    if not safe_name:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    dest = UPLOAD_DIR / safe_name
    content = await file.read()
    dest.write_bytes(content)

    return JSONResponse({
        "ok": True,
        "name": safe_name,
        "size": len(content),
        "path": f"data/uploads/{safe_name}",
    })


@app.delete("/api/upload/{filename}")
async def delete_upload(request: Request, filename: str):
    """Delete an uploaded file."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    safe_name = Path(filename).name
    dest = UPLOAD_DIR / safe_name
    if dest.exists():
        dest.unlink()
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.websocket("/ws/echo")
async def echo_ws(ws: WebSocket):
    """Simple WebSocket echo for debugging - no auth required."""
    import sys
    print(f"[WS-ECHO] Connection attempt!", file=sys.stderr, flush=True)
    await ws.accept()
    print(f"[WS-ECHO] Accepted!", file=sys.stderr, flush=True)
    await ws.send_text("Hello from server!")
    try:
        while True:
            data = await ws.receive_text()
            print(f"[WS-ECHO] Got: {data}", file=sys.stderr, flush=True)
            await ws.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"[WS-ECHO] Closed: {e}", file=sys.stderr, flush=True)


@app.websocket("/ws/terminal")
async def terminal_ws(ws: WebSocket, token: str | None = Query(default=None)):
    """WebSocket endpoint bridging xterm.js to AgingClaw CLI via PTY."""
    import sys

    # Log connection details for debugging Cloudflare issues
    client = ws.client
    headers = dict(ws.headers) if hasattr(ws, 'headers') else {}
    print(f"[WS] Connection attempt from {client}", file=sys.stderr, flush=True)
    print(f"[WS] Token: {'yes' if token else 'no'}", file=sys.stderr, flush=True)
    print(f"[WS] Headers: CF-Connecting-IP={headers.get('cf-connecting-ip', 'N/A')}, "
          f"X-Forwarded-For={headers.get('x-forwarded-for', 'N/A')}", file=sys.stderr, flush=True)

    if not _check_ws_auth(token):
        print(f"[WS] Auth failed", file=sys.stderr, flush=True)
        await ws.close(code=4001, reason="Unauthorized")
        return

    print(f"[WS] Auth OK, accepting connection", file=sys.stderr, flush=True)
    try:
        await ws.accept()
        print(f"[WS] Connection accepted successfully", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WS] Failed to accept: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return

    # Spawn PTY with agingclaw CLI
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["COLORTERM"] = "truecolor"
    env["COLUMNS"] = "120"
    env["LINES"] = "40"
    # Pass session token for per-session memory isolation
    if token:
        env["LONGEVITYCLAW_SESSION_ID"] = token

    # Find the agingclaw entry point
    project_root = Path(__file__).resolve().parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    master_fd, slave_fd = pty.openpty()

    # Set initial terminal size
    winsize = struct.pack("HHHH", 40, 120, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    pid = os.fork()
    if pid == 0:
        # Child process
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        os.chdir(str(project_root))
        os.execve(
            str(venv_python),
            [str(venv_python), "-m", "longevityclaw.cli"],
            env,
        )

    # Parent process
    os.close(slave_fd)

    loop = asyncio.get_event_loop()

    # Read from PTY -> send to WebSocket
    async def pty_reader():
        try:
            while True:
                data = await loop.run_in_executor(None, os.read, master_fd, 4096)
                if not data:
                    break
                await ws.send_bytes(data)
        except (OSError, WebSocketDisconnect):
            pass

    reader_task = asyncio.create_task(pty_reader())

    # Heartbeat to keep connection alive through Cloudflare (10s interval)
    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(10)
                await ws.send_bytes(b"\x00")  # Null byte as heartbeat (binary)
        except Exception:
            pass

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "bytes" in msg and msg["bytes"]:
                    os.write(master_fd, msg["bytes"])
                elif "text" in msg and msg["text"]:
                    text = msg["text"]
                    # Handle resize messages
                    if text.startswith("\x1b[8;"):
                        # Parse resize: ESC[8;rows;colst
                        try:
                            parts = text[4:-1].split(";")
                            rows, cols = int(parts[0]), int(parts[1])
                            winsize = struct.pack("HHHH", rows, cols, 0, 0)
                            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                            os.kill(pid, signal.SIGWINCH)
                        except (ValueError, IndexError, OSError):
                            pass
                    else:
                        os.write(master_fd, text.encode("utf-8"))
            elif msg["type"] == "websocket.disconnect":
                print(f"[WS] Client disconnected (websocket.disconnect)", file=sys.stderr, flush=True)
                break
    except WebSocketDisconnect as e:
        print(f"[WS] WebSocketDisconnect exception: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WS] Unexpected exception: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        print(f"[WS] Cleaning up connection", file=sys.stderr, flush=True)
        reader_task.cancel()
        heartbeat_task.cancel()
        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        except (OSError, ChildProcessError):
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass


def main():
    """Run the LongevityClaw web server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LongevityClaw Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--password", type=str, default=None,
                        help="Require password to access (recommended for public sharing)")
    args = parser.parse_args()

    # Load .env before starting
    from longevityclaw.cli import load_dotenv
    load_dotenv()

    # Set up auth
    global _password_hash
    password = args.password or os.environ.get("LONGEVITYCLAW_PASSWORD")
    if password:
        _password_hash = _hash_password(password)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  LongevityClaw Web Interface")
    print(f"  Open in browser: http://localhost:{args.port}")
    if _password_hash:
        print(f"  Password protection: enabled")
    else:
        print(f"  Password protection: disabled (use --password to enable)")
    print()

    # Use socket_app which wraps FastAPI with Socket.IO for terminal
    uvicorn.run(socket_app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
