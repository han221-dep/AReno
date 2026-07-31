"""Optional real-time UI for the elevator-dispatch agentic example.

Runs beside ``areno train`` on the same host (e.g. a Kaggle notebook), tails the
newest ``rollout_samples.*.jsonl`` areno writes, and beams a replay of each new
sample to the browser as Server-Sent Events. The browser animates the model's
dispatch tick by tick using the ``record=True`` frames from :func:`game.play`.

Zero third-party deps on the server side -- only the stdlib -- so it runs in a
locked-down Kaggle cell without pip. Serve the bundled UI at ``/`` and the event
stream at ``/events``; mirror ngrok like the areno dashboard if remote.

Honesty notes
-------------
- ``record_rollout_sample`` does NOT write the original ``building`` (it stores
  the rendered ``prompt`` string only -- see memory/areno-rollout-sample-fields).
  We recover the building by indexing the training-data JSONL with
  ``prompt_idx``. # UNVERIFIED: prompt_idx <-> dataset row index correspondence
  is assumed here, not confirmed in areno source; set ``--strict-buildings`` to
  reject samples whose building cannot be resolved instead of skipping silently.
- ``rollout_samples`` carries NO reward scalar (rewards land in TensorBoard).
  This UI shows episodes + metrics only; a reward-time chart would read
  ``events.out.tfevents.*`` separately -- not wired here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import game  # noqa: E402

DEFAULT_SAMPLES_GLOB = "/tmp/**/rollout_samples*.jsonl"
DEFAULT_DATASET = "/tmp/areno-elevator.jsonl"
DEFAULT_PORT = 8000
DEFAULT_POLL_SECONDS = 1.5


def _actions_from_sample(sample: dict[str, Any]) -> str:
    """Pull the dispatch action string out of a rollout sample.

    Agentic samples store ``tool_calls`` (parsed) and/or ``final_answer`` (raw
    text). Try both so the UI keeps working whether areno wrote the tool call or
    fell back to a ``<dispatch>`` tag in the answer.
    """

    for call in sample.get("tool_calls") or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "dispatch":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, dict):
            return _clean_actions(arguments.get("actions"))
    final = sample.get("final_answer") or ""
    return game.parse_action_sequence(final) if final else ""


def _clean_actions(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).upper()
    return "".join(ch for ch in text if ch in game.ACTIONS)


class DatasetIndex:
    """Lazily read training-data buildings by row, memoized by path+mtime.

    The training-data JSONL written by ``dataset_generator`` stores one raw
    building per line (``floors``/``capacity``/``arrivals``/``car``, plus an
    ``id``), not wrapped under a ``building`` key -- the loader re-wraps it only
    in memory. We accept both shapes: if a row has a ``building`` key we take
    that, otherwise the row itself is the building; trailing fields like ``id``
    are ignored by :func:`game.normalize_building`, which pins to the keys it
    needs.
    """

    def __init__(self, path: str, *, strict: bool) -> None:
        self.path = Path(path).expanduser()
        self.strict = strict
        self._buildings: list[dict[str, Any]] | None = None
        self._mtime: float | None = None

    def _load(self) -> list[dict[str, Any]] | None:
        if not self.path.exists():
            return None
        mtime = self.path.stat().st_mtime
        if self._buildings is not None and mtime == self._mtime:
            return self._buildings
        buildings: list[dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    buildings.append(row.get("building", row))
        except (OSError, json.JSONDecodeError):
            return self._buildings
        self._buildings = buildings
        self._mtime = mtime
        return buildings

    def get(self, prompt_idx: Any) -> dict[str, Any] | None:
        buildings = self._load()
        if buildings is None:
            return None
        try:
            idx = int(prompt_idx)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(buildings):
            return buildings[idx]
        return None


def _newest_samples_glob(pattern: str) -> str | None:
    files = [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _episode_payload(sample: dict, building: dict | None) -> dict[str, Any] | None:
    """Build one SSE payload for a finished sample, or None to skip it."""

    actions = _actions_from_sample(sample)
    meta = {
        "step": sample.get("step"),
        "epoch": sample.get("epoch"),
        "prompt_idx": sample.get("prompt_idx"),
        "sample_idx": sample.get("sample_idx"),
        "actions": actions,
        "has_tool_calls": bool(sample.get("tool_calls")),
    }
    if building is None or not actions:
        return {**meta, "skipped": True, "reason": ("no-building" if building is None else "no-actions")}
    replay = game.play(building, actions, max_steps=int(building.get("max_steps", game.DEFAULT_MAX_STEPS)), record=True)
    return {
        **meta,
        "skipped": False,
        "metrics": {
            "delivered": replay["delivered_passengers"],
            "mean_wait": replay["mean_wait"],
            "n_attempts": replay["n_attempts"],
            "n_invalid": replay["n_invalid"],
            "invalid_rate": replay["invalid_rate"],
            "remaining": replay["remaining_passengers"],
            "terminal": replay["terminal"],
        },
        # Only ship the per-frame deltas a browser needs; see game.play frames.
        "frames": [game.clone_state(f["state"]) for f in replay["frames"]],
        "frame_actions": [f["action"] for f in replay["frames"]],
    }


def _css_html() -> str:
    # Inlined so the whole UI ships from one endpoint with no static files.
    return PAGE_HTML


class _Handler(BaseHTTPRequestHandler):
    server_version = "ElevatorDashboard/1"

    def log_message(self, fmt, *args):  # silence default access log noise
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._serve_html()
        elif self.path.startswith("/events"):
            self._serve_events()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = _css_html().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        server: _Server = self.server  # type: ignore[assignment]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        server.stream_to(self.wfile)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, *, samples_glob: str, dataset_path: str, strict: bool, poll_seconds: float) -> None:
        super().__init__(addr, _Handler)
        self.samples_glob = samples_glob
        self.dataset = DatasetIndex(dataset_path, strict=strict)
        self.strict = strict
        self.poll_seconds = poll_seconds
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def stream_to(self, wfile) -> None:
        """Tail the newest samples file, emit one SSE line per finished sample.

        Re-resolves the newest matching file each time the seen file vanishes,
        so a fresh training run (new pid) is picked up without a restart.
        """

        current_file: str | None = None
        offset = 0
        self._write(wfile, "hello", {"message": "elevator dashboard connected"})
        while not self._stop:
            file_now = _newest_samples_glob(self.samples_glob)
            if file_now is None:
                self._sleep()
                continue
            if file_now != current_file:
                current_file, offset = file_now, 0  # new pid -> start of file
            try:
                size = os.path.getsize(current_file)
            except OSError:
                current_file, offset = None, 0
                self._sleep()
                continue
            if size > offset:
                try:
                    with open(current_file, "rb") as handle:
                        handle.seek(offset)
                        chunk = handle.read(size - offset)
                except OSError:
                    self._sleep()
                    continue
                offset = size
                for line in chunk.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        sample = json.loads(stripped.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    if sample.get("kind") != "agentic":
                        continue
                    self._emit_sample(wfile, sample)
            else:
                # Keep the SSE connection warm between writes.
                self._write(wfile, "ping", {"t": 0})
            self._sleep()

    def _emit_sample(self, wfile, sample: dict) -> None:
        prompt_idx = sample.get("prompt_idx")
        building = self.dataset.get(prompt_idx)
        if building is None and self.strict:
            self._write(wfile, "skip", {"prompt_idx": prompt_idx, "reason": "no-building (strict)"})
            return
        payload = _episode_payload(sample, building)
        if payload is None:
            return
        self._write(wfile, "episode", payload)

    def _write(self, wfile, event: str, data: dict) -> None:
        blob = json.dumps({"event": event, **data}, ensure_ascii=False)
        try:
            wfile.write(f"data: {blob}\n\n".encode("utf-8"))
            wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self._stop = True

    def _sleep(self) -> None:
        time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Elevator agentic training dashboard (real-time replay UI).")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES_GLOB, help="glob for rollout_samples.*.jsonl (newest wins)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="training-data JSONL used to resolve buildings by prompt_idx")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS, help="seconds between tail polls")
    parser.add_argument("--strict-buildings", action="store_true", help="emit a skip-line instead of dropping samples with no resolvable building")
    args = parser.parse_args()

    print(f"[dashboard] samples glob: {args.samples}", flush=True)
    print(f"[dashboard] dataset for buildings: {args.dataset}", flush=True)
    print(f"[dashboard] serving UI at http://{args.host}:{args.port}/  (events at /events)", flush=True)
    print("[dashboard] remember: opening this via ngrok needs a VPN (same as areno dashboard).", flush=True)
    server = _Server(
        (args.host, args.port),
        samples_glob=args.samples,
        dataset_path=args.dataset,
        strict=args.strict_buildings,
        poll_seconds=args.poll,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.server_close()


# --- bundled UI --------------------------------------------------------------
# Keep it framework-free so the page renders from a single / and the JS just
# consumes the /events stream. The animation draws floors, the car, passengers,
# and highlights each action letter as the replay steps one frame at a time.

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AReno elevator dispatch — live replay</title>
<style>
:root{color-scheme:light dark}
body{font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
#wrap{max-width:900px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}
#status{color:#7fd1ff;margin-bottom:12px;font-size:12px}
#stage{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
#shaft{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:10px;min-width:260px}
.floor{display:flex;align-items:center;gap:8px;height:54px;border-bottom:1px dashed #2a2f3a}
.floor:last-child{border-bottom:none}
.fnum{width:28px;color:#8a93a6;font-variant-numeric:tabular-nums}
.hall{flex:1;display:flex;gap:4px;flex-wrap:wrap}
.dot{width:18px;height:18px;border-radius:50%;display:grid;place-items:center;font-size:10px}
.waiting{background:#3a4a2a;color:#cfe6b0}
.car{width:70px;border:1px solid #4a5266;border-radius:4px;background:#222834;min-height:40px;padding:2px;display:flex;gap:3px;flex-wrap:wrap}
.car.open{border-color:#7fd1ff;box-shadow:0 0 8px #2a5c7a}
.aboard{background:#2a4a6a;color:#bfe3ff}
.gap{width:70px}
.fsel{background:#1c3a2a;border-radius:4px}
aside{flex:1;min-width:240px}
.metr{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:10px;margin-bottom:10px}
.metr b{color:#7fd1ff}
#log{max-height:180px;overflow:auto;background:#11141b;border:1px solid #2a2f3a;border-radius:8px;padding:8px;font-size:12px;font-family:ui-monospace,Menlo,monospace}
.row{border-bottom:1px solid #1e2530;padding:3px 0}
.row.skip{color:#caa}
button{background:#2a3a55;color:#e6e6e6;border:1px solid #3a4a66;border-radius:4px;padding:4px 10px;cursor:pointer}
</style></head><body><div id="wrap">
<h1>AReno · elevator dispatch — live replay</h1>
<div id="status">connecting…</div>
<div id="stage">
  <div id="shaft"></div>
  <aside>
    <div class="metr" id="meta"><b>step</b> — · <b>prompt_idx</b> —</div>
    <div class="metr" id="metrics">waiting for first episode…</div>
    <div style="margin-bottom:10px"><button id="play">▶ auto</button> <button id="step">⏭ step</button> <span id="speed">1x · 480ms</span></div>
    <div id="actions" style="font-family:ui-monospace,Menlo,monospace;letter-spacing:2px;margin-bottom:10px;min-height:20px"></div>
    <div id="log"></div>
  </aside>
</div></div>
<script>
const COLORS=["#7fd1ff","#ffb27f","#b0ff8f","#ff9fe0","#c9a7ff","#fff066","#7ffff0","#ff7f7f"];
let floors=[], frames=[], actions=[], cursor=0, auto=true, timer=null, SPEED=480, lastMet=null;
const $=id=>document.getElementById(id);

function buildShaft(n){
  const sha=$("shaft"); sha.innerHTML="";
  for(let f=n-1;f>=0;f--){const d=document.createElement("div");d.className="floor";
    d.innerHTML=`<div class="fnum">F${f}</div><div class="hall" id="hall${f}"></div><div class="gap" id="cargap${f}"></div>`;sha.appendChild(d);}
}
function plate(to){const c=COLORS[to%COLORS.length];
  return `<span class="dot waiting" style="background:${c};color:#111">→${to}</span>`;}
function aboardPlate(to){const c=COLORS[to%COLORS.length];
  return `<span class="dot aboard" style="background:${c};color:#111">${to}</span>`;}

function render(idx){
  if(!frames.length) return;
  const f=frames[idx]; const st=f.state; const car=st.car;
  if(st.floors!==floors.length){floors=Array(st.floors).fill(0).map((_,i)=>i); buildShaft(st.floors);}
  for(let fl=0;fl<st.floors;fl++){
    const q=st.hall_queues[fl]||[]; $("hall"+fl).innerHTML=q.map(p=>plate(p.to_floor)).join("");
    const gap=$("cargap"+fl);
    if(car.floor===fl){
      const cls="car"+(car.door_open?" open":""); const ps=(car.passengers||[]).map(p=>aboardPlate(p.to_floor)).join("");
      gap.innerHTML=`<div class="${cls}">${ps}</div>`;
      gap.classList.add("fsel");
    }else{gap.innerHTML=""; gap.classList.remove("fsel");}
  }
  // action highlight
  let a=""; for(let i=0;i<actions.length;i++){const at=actions[i];
    a+= at==null? "" : (i===idx-1? `<span style="color:#7fd1ff;text-decoration:underline">${at}</span>` : at);}
  $("actions").innerHTML=a||"(no dispatch actions)";
}

function tick(){ render(cursor); cursor++; if(cursor>=frames.length){stop();} }
function stop(){auto=false; if(timer){clearInterval(timer);timer=null;} $("play").textContent="▶ auto";}
function play(){ if(cursor>=frames.length) cursor=0; auto=true; $("play").textContent="⏸ pause";
  if(timer) clearInterval(timer); timer=setInterval(tick,SPEED); }
$("play").onclick=()=>{ if(auto){stop();}else{play();} };
$("step").onclick=()=>{ stop(); if(cursor>=frames.length) cursor=0; tick(); };

function episode(d){
  if(d.skipped){logRow(d,"skipped: "+d.reason); return;}
  frames=d.frames; actions=d.frame_actions; cursor=0; lastMet=d.metrics;
  $("meta").innerHTML=`<b>step</b> ${d.step} · <b>epoch</b> ${d.epoch} · <b>prompt_idx</b> ${d.prompt_idx} · <b>sample_idx</b> ${d.sample_idx}`;
  const m=d.metrics;
  $("metrics").innerHTML=`<b>delivered</b> ${m.delivered} · <b>mean_wait</b> ${m.mean_wait.toFixed(1)} · <b>invalid</b> ${m.n_invalid}/${m.n_attempts} (${(m.invalid_rate*100).toFixed(0)}%) · <b>remaining</b> ${m.remaining} · <b>terminal</b> ${m.terminal}`;
  logRow(d,"delivered="+m.delivered+" wait="+m.mean_wait.toFixed(1)+" actions="+d.actions);
  render(0); cursor=1;
  if(timer) clearInterval(timer); timer=setInterval(tick,SPEED); auto=true; $("play").textContent="⏸ pause";
}
function logRow(d,msg){const el=$("log");const r=document.createElement("div");r.className="row"+(d.skipped?" skip":"");
  r.textContent=`step=${d.step} pidx=${d.prompt_idx} ${msg}`;el.prepend(r); while(el.children.length>40) el.removeChild(el.lastChild);}
function status(s){$("status").textContent=s;}

const es=new EventSource("/events");
es.onopen=()=>status("connected · waiting for episodes");
es.onerror=()=>status("connection lost — retrying…");
es.onmessage=e=>{let d;try{d=JSON.parse(e.data);}catch(){return;}
  if(d.event==="ping")return;
  if(d.event==="hello"){status("connected · waiting for episodes");return;}
  if(d.event==="episode")episode(d);
  if(d.event==="skip"||d.event==="episode")status("step "+(d.step??"?")+" · prompt_idx "+(d.prompt_idx??"?"));
};
</script></body></html>"""


if __name__ == "__main__":
    main()
