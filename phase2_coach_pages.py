#!/usr/bin/env python3
"""
Phase 2: offline pages for the coaches, written into a blind pack.

  <coach>_pairs.html   two clips side by side, slow motion, click the better drive
                       (or press L / R), confidence, notes; "Download results"
                       saves <coach>_pairs.csv in the format phase2_pairs.py fit reads.
  <coach>_scoring.html one clip at a time with the rubric: C01-C12 as 2/1/0/X,
                       four pillars, overall, priority fault, confidence, minutes
                       (timed automatically); "Download results" saves
                       <coach>_score_sheet.csv in the format phase2_agreement.py reads.

The pages are plain HTML files with no network calls. They play the clips from
the pack's clips/ folder, so the clips never leave the coach's computer.
Progress is kept in the browser between sessions where the browser allows it.

  python phase2_coach_pages.py --pack <pack folder>
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUBRIC = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0.json"

STYLE = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#d9d9d4;--accent:#1f6f4a;--accent-fg:#fff;--pick:#e5f4e4}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#a3a39c;--card:#22221f;--line:#3a3a36;--accent:#4fb889;--accent-fg:#0d1a13;--pick:#1f3a2b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 -apple-system,system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 2px}.muted{color:var(--muted);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.row{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1 1 320px;min-width:0}
video{width:100%;max-height:62vh;background:#000;border-radius:8px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:8px 14px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}button.on{background:var(--pick);border-color:var(--accent)}
.big{font-size:18px;padding:14px 20px;width:100%}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--accent)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 12px;align-items:center}.seg{display:flex;gap:4px}.seg button{padding:6px 10px;min-width:40px}
input,select,textarea{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
input[type=number]{width:80px}textarea{width:100%}label{font-size:14px}.tools{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
"""

STORE_JS = """
const KEY = "battingiq_" + PAGE.kind + "_" + PAGE.coach;
function load(){ try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ return {}; } }
function save(s){ try { localStorage.setItem(KEY, JSON.stringify(s)); } catch(e){} }
function csvCell(v){ v = (v===undefined||v===null) ? "" : String(v); return /[",\\n]/.test(v) ? '"' + v.replace(/"/g,'""') + '"' : v; }
function download(name, header, rows){
  const text = [header.join(",")].concat(rows.map(r => header.map(h => csvCell(r[h])).join(","))).join("\\n") + "\\n";
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type:"text/csv"}));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
function speed(r){ document.querySelectorAll("video").forEach(v => { v.playbackRate = r; v.play(); }); }
function restart(){ document.querySelectorAll("video").forEach(v => { v.currentTime = 0; v.play(); }); }
"""

PAIRS_BODY = """
<main>
<h1>Which drive is better? <span class="muted" id="who"></span></h1>
<p class="muted">Judge the drive as a whole, as you would in the nets. Click the better one, or press L or R. About 25 minutes.</p>
<div class="bar"><i id="prog"></i></div><p class="muted" id="count"></p>
<div class="tools"><button onclick="restart()">Restart both</button><button onclick="speed(1)">1×</button><button onclick="speed(0.5)">0.5×</button><button onclick="speed(0.25)">0.25×</button><button onclick="go(-1)">← Back</button></div>
<div class="row">
 <div class="col card"><video id="vl" muted loop playsinline autoplay controls></video><button class="big primary" onclick="pick('L')">Left is better (L)</button></div>
 <div class="col card"><video id="vr" muted loop playsinline autoplay controls></video><button class="big primary" onclick="pick('R')">Right is better (R)</button></div>
</div>
<div class="card"><label>Confidence</label> <span class="seg" id="conf"><button>low</button><button>medium</button><button>high</button></span>
 <p><label for="notes">Notes (optional)</label><textarea id="notes" rows="2"></textarea></p></div>
<div class="card" id="done" hidden><b>All pairs done.</b> Click below and send the file to Alex. <button class="primary" onclick="dl()">Download results</button></div>
<p><button onclick="dl()">Download results so far</button></p>
</main>
<script>
const s = load(); s.answers = s.answers || {}; let i = s.i || 0;
document.getElementById("who").textContent = "— " + PAGE.coach;
function show(){
  const done = Object.keys(s.answers).length;
  document.getElementById("prog").style.width = (100*done/PAGE.items.length) + "%";
  document.getElementById("count").textContent = done + " of " + PAGE.items.length + " answered";
  if (i >= PAGE.items.length){ document.getElementById("done").hidden = false; i = PAGE.items.length - 1; }
  const p = PAGE.items[i], a = s.answers[p.pair_no] || {};
  document.getElementById("vl").src = "clips/" + p.left_id + ".mp4";
  document.getElementById("vr").src = "clips/" + p.right_id + ".mp4";
  document.getElementById("notes").value = a.notes || "";
  document.querySelectorAll("#conf button").forEach(b => b.classList.toggle("on", b.textContent === (a.confidence || "")));
  document.getElementById("count").textContent += " · pair " + (i+1);
}
document.querySelectorAll("#conf button").forEach(b => b.onclick = () => {
  const p = PAGE.items[i]; s.answers[p.pair_no] = Object.assign(s.answers[p.pair_no] || {}, {confidence: b.textContent}); save(s); show(); });
function pick(side){
  const p = PAGE.items[i];
  s.answers[p.pair_no] = Object.assign(s.answers[p.pair_no] || {}, {better: side, notes: document.getElementById("notes").value});
  i = Math.min(i + 1, PAGE.items.length); s.i = i; save(s); show();
}
function go(d){ i = Math.max(0, Math.min(PAGE.items.length - 1, i + d)); s.i = i; save(s); show(); }
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "l" || e.key === "L") pick("L"); if (e.key === "r" || e.key === "R") pick("R"); });
function dl(){ download(PAGE.coach + "_pairs.csv", ["pair_no","left_id","right_id","better","confidence","notes"],
  PAGE.items.map(p => Object.assign({}, p, s.answers[p.pair_no] || {}))); }
show();
</script>
"""

SCORING_BODY = """
<main>
<h1>Score each drive <span class="muted" id="who"></span></h1>
<p class="muted">Rate the checkpoints, then the pillars, then the overall score on its own. Do not add up the pillars. About 4 minutes a clip.</p>
<div class="bar"><i id="prog"></i></div><p class="muted" id="count"></p>
<div class="row">
 <div class="col card"><video id="v" muted loop playsinline autoplay controls></video>
  <div class="tools"><button onclick="restart()">Restart</button><button onclick="speed(1)">1×</button><button onclick="speed(0.5)">0.5×</button><button onclick="speed(0.25)">0.25×</button></div>
  <p class="muted">Not visible from this camera; do not mark down: <span id="nj"></span></p></div>
 <div class="col card"><div class="grid" id="cps"></div></div>
</div>
<div class="card"><div class="row">
 <div class="col"><div class="grid" id="pillars"></div></div>
 <div class="col"><label>Priority fault</label><br><select id="pf"></select>
  <p><label>Confidence</label> <span class="seg" id="conf"><button>low</button><button>medium</button><button>high</button></span></p>
  <p><label for="cj">Cannot judge because (if any X)</label><input id="cj" style="width:100%"></p>
  <p><label for="notes">Notes</label><textarea id="notes" rows="2"></textarea></p></div>
</div>
<div class="tools"><button onclick="go(-1)">← Previous</button><button class="primary" onclick="next()">Save and next →</button></div></div>
<div class="card" id="done" hidden><b>All clips scored.</b> Click below and send the file to Alex. <button class="primary" onclick="dl()">Download results</button></div>
<p><button onclick="dl()">Download results so far</button></p>
</main>
<script>
const s = load(); s.answers = s.answers || {}; let i = s.i || 0, started = Date.now();
document.getElementById("who").textContent = "— " + PAGE.coach;
document.getElementById("nj").textContent = PAGE.rubric.not_judged_from_this_camera.join("; ");
const cps = document.getElementById("cps");
PAGE.rubric.checkpoints.forEach(c => {
  const l = document.createElement("div"); l.innerHTML = "<b>" + c.id + "</b> " + c.text + " <span class='muted'>(" + c.pillar + ")</span>";
  const seg = document.createElement("div"); seg.className = "seg"; seg.dataset.cp = c.id;
  ["2","1","0","X"].forEach(v => { const b = document.createElement("button"); b.textContent = v; b.onclick = () => { cur()[c.id] = v; paint(); }; seg.appendChild(b); });
  cps.appendChild(l); cps.appendChild(seg); });
const pl = document.getElementById("pillars");
["access","tracking","stability","flow","overall"].forEach(p => {
  const l = document.createElement("label"); l.textContent = p === "overall" ? "Overall (0–100, on its own)" : p[0].toUpperCase() + p.slice(1) + " (0–25)";
  const inp = document.createElement("input"); inp.type = "number"; inp.min = 0; inp.max = p === "overall" ? 100 : 25; inp.id = "in_" + p;
  inp.oninput = () => { cur()[p] = inp.value; }; pl.appendChild(l); pl.appendChild(inp); });
const pf = document.getElementById("pf"); pf.innerHTML = "<option value=''>— pick one —</option>" +
  PAGE.rubric.faults_in_chain_order.map(f => "<option value='" + f.id + "'>" + f.id + " " + f.text + "</option>").join("");
pf.onchange = () => { cur().priority_fault = pf.value; };
document.querySelectorAll("#conf button").forEach(b => b.onclick = () => { cur().confidence = b.textContent; paint(); });
function cur(){ const id = PAGE.items[i].clip_id; return s.answers[id] = s.answers[id] || {}; }
function paint(){
  const a = cur();
  document.querySelectorAll("#cps .seg").forEach(seg => seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", a[seg.dataset.cp] === b.textContent)));
  document.querySelectorAll("#conf button").forEach(b => b.classList.toggle("on", a.confidence === b.textContent));
}
function show(){
  const done = Object.values(s.answers).filter(a => a.overall !== undefined && a.overall !== "").length;
  document.getElementById("prog").style.width = (100*done/PAGE.items.length) + "%";
  document.getElementById("count").textContent = done + " of " + PAGE.items.length + " scored · clip " + (i+1) + " (" + PAGE.items[i].clip_id + ")";
  if (done >= PAGE.items.length) document.getElementById("done").hidden = false;
  const a = cur();
  document.getElementById("v").src = "clips/" + PAGE.items[i].clip_id + ".mp4";
  ["access","tracking","stability","flow","overall"].forEach(p => document.getElementById("in_" + p).value = a[p] || "");
  pf.value = a.priority_fault || ""; document.getElementById("cj").value = a.cannot_judge_reason || "";
  document.getElementById("notes").value = a.notes || ""; paint(); started = Date.now();
}
function keepText(){ const a = cur(); a.cannot_judge_reason = document.getElementById("cj").value; a.notes = document.getElementById("notes").value;
  a.minutes_taken = ((parseFloat(a.minutes_taken) || 0) + (Date.now() - started) / 60000).toFixed(1); }
function next(){ keepText(); i = Math.min(i + 1, PAGE.items.length - 1); s.i = i; save(s); show(); }
function go(d){ keepText(); i = Math.max(0, i + d); s.i = i; save(s); show(); }
function dl(){ keepText(); save(s);
  download(PAGE.coach + "_score_sheet.csv", PAGE.fields, PAGE.items.map(it => Object.assign({order: it.order, clip_id: it.clip_id}, s.answers[it.clip_id] || {}))); }
show();
</script>
"""


def page(title: str, data: dict, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{STYLE}</style></head><body><script>const PAGE = {json.dumps(data)};{STORE_JS}</script>{body}</body></html>")


def build(pack: Path) -> list[Path]:
    rubric = json.loads(RUBRIC.read_text())
    written = []
    for sheet in sorted((pack / "sheets").glob("*_score_sheet.csv")):
        coach = sheet.name.replace("_score_sheet.csv", "")
        with sheet.open(encoding="utf-8-sig", newline="") as h:
            reader = csv.DictReader(h)
            fields = reader.fieldnames or []
            items = [{"order": r["order"], "clip_id": r["clip_id"]} for r in reader]
        out = pack / f"{coach}_scoring.html"
        out.write_text(page(f"Batting IQ scoring — {coach}", {"kind": "scoring", "coach": coach, "items": items,
                                                              "fields": fields, "rubric": rubric}, SCORING_BODY), encoding="utf-8")
        written.append(out)
        pairs = pack / "pairs" / f"{coach}_pairs.csv"
        if pairs.exists():
            with pairs.open(encoding="utf-8-sig", newline="") as h:
                pitems = [{"pair_no": r["pair_no"], "left_id": r["left_id"], "right_id": r["right_id"]} for r in csv.DictReader(h)]
            out = pack / f"{coach}_pairs.html"
            out.write_text(page(f"Batting IQ pairs — {coach}", {"kind": "pairs", "coach": coach, "items": pitems}, PAIRS_BODY),
                           encoding="utf-8")
            written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", required=True)
    args = parser.parse_args(argv)
    for p in build(Path(args.pack).expanduser()):
        print("Wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
