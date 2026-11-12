import { Chessground } from "https://cdn.jsdelivr.net/npm/chessground@9.2.1/dist/chessground.min.js";

import { nameOf } from "/static/names.js";

const $ = (s) => document.querySelector(s);
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------- tooltip ----------
const tip = $("#tip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, window.innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.max(8, y - r.height - 10)}px`;
}
const hideTip = () => { tip.hidden = true; };
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t) showTip(t.dataset.tip, e.clientX, e.clientY);
});
document.addEventListener("mousemove", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t) showTip(t.dataset.tip, e.clientX, e.clientY);
});
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });

// ---------- pieces ----------
function pairBars(me, opp, max) {
  const row = (cls, label, v) => `
    <div class="bar-row" data-tip="${esc(`${label}: ${pct(v, 2)} of winning chances lost per move`)}">
      <span class="bar-label">${label}</span>
      <span class="bar-track"><span class="bar ${cls}" style="width:${Math.max(1, (100 * v) / max)}%"></span></span>
      <span class="bar-val">${pct(v, 2)}</span>
    </div>`;
  return `<div class="bars">${row("me", "You", me)}${row("opp", "Opponents", opp)}</div>`;
}

function exampleBoards(container, examples) {
  for (const ex of examples) {
    const fig = document.createElement("figure");
    fig.className = "ex";
    const board = document.createElement("div");
    board.className = "ex-board";
    fig.appendChild(board);
    const cap = document.createElement("figcaption");
    const link = ex.link ? ` · <a href="${esc(ex.link)}" target="_blank" rel="noopener">game</a>` : "";
    cap.innerHTML = `<b>${esc(ex.played)}</b> −${Math.round(ex.loss * 100)}%<br>` +
      `<span class="hint">better ${esc(ex.better || "?")} · ${esc(ex.date)}${link}</span>`;
    fig.appendChild(cap);
    container.appendChild(fig);
    const shapes = [{ orig: ex.move.slice(0, 2), dest: ex.move.slice(2, 4), brush: "red" }];
    if (ex.best && ex.best !== ex.move) shapes.push({ orig: ex.best.slice(0, 2), dest: ex.best.slice(2, 4), brush: "green" });
    Chessground(board, {
      fen: ex.fen, orientation: ex.orientation, viewOnly: true, coordinates: false,
      drawable: { enabled: false, visible: true, autoShapes: shapes },
    });
  }
}

function renderTiles(r) {
  const me = r.me, p = r.peers;
  const tile = (label, value, sub) =>
    `<div class="tile"><div class="tile-label">${label}</div><div class="tile-value">${value}</div><div class="hint">${sub}</div></div>`;
  $("#tiles").innerHTML =
    tile("Winning chances lost per move", pct(me.loss_per_move, 2), `opponents ${pct(p.loss_per_move, 2)}`) +
    tile("Blunders per 100 moves", me.blunders_per_100.toFixed(1), `opponents ${p.blunders_per_100.toFixed(1)}`) +
    tile("Mistakes per 100 moves", me.mistakes_per_100.toFixed(1), `opponents ${p.mistakes_per_100.toFixed(1)}`);
}

function renderSpots(el, rows, withExamples) {
  el.innerHTML = "";
  if (!rows.length) {
    el.innerHTML = `<p class="hint">Nothing clearly different from your opponents.</p>`;
    return;
  }
  const max = Math.max(...rows.flatMap((r) => [r.loss_me, r.loss_peer]));
  rows.forEach((r, i) => {
    const card = document.createElement("article");
    card.className = "card";
    const cost = r.extra_per_game >= 0
      ? `costs you about <b>${pct(r.extra_per_game)}</b> winning chances per game`
      : `saves you about <b>${pct(-r.extra_per_game)}</b> winning chances per game`;
    const what = r.dimension === "mistake type"
      ? `${r.moves.toLocaleString()} such mistakes`
      : `${r.moves.toLocaleString()} of your moves (${pct(r.share_of_my_loss, 0)} of everything you lose)`;
    card.innerHTML = `
      <h4>${withExamples ? `<span class="rank">${i + 1}</span>` : ""}${esc(nameOf(r.tag))}</h4>
      <p class="hint">${cost} · ${what} · z ${r.z.toFixed(1)}</p>
      ${pairBars(r.loss_me, r.loss_peer, max)}`;
    if (withExamples && r.examples?.length) {
      const exs = document.createElement("div");
      exs.className = "examples";
      card.appendChild(exs);
      el.appendChild(card);
      exampleBoards(exs, r.examples);
    } else {
      el.appendChild(card);
    }
  });
}

function renderTrend(rows) {
  const el = $("#trend");
  if (rows.length < 2) { el.innerHTML = `<p class="hint">Not enough months yet.</p>`; return; }
  const W = 720, H = 220, L = 44, R = 70, T = 12, B = 28;
  const max = Math.max(...rows.flatMap((r) => [r.loss_me, r.loss_peer])) * 1.15;
  const x = (i) => L + (i * (W - L - R)) / (rows.length - 1);
  const y = (v) => T + (H - T - B) * (1 - v / max);
  const line = (key) => rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join("");
  const ticks = [0, max / 3, (2 * max) / 3].map((v) =>
    `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="axis" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${(v * 100).toFixed(1)}%</text>`).join("");
  const every = Math.ceil(rows.length / 8);
  const xl = rows.map((r, i) => (i % every === 0 || i === rows.length - 1)
    ? `<text class="axis" x="${x(i)}" y="${H - 8}" text-anchor="middle">${r.month}</text>` : "").join("");
  const last = rows.length - 1;
  // End labels: keep them at least 14px apart so they never overlap.
  const labelY = { me: y(rows[last].loss_me), opp: y(rows[last].loss_peer) };
  const gap = 14 - Math.abs(labelY.me - labelY.opp);
  if (gap > 0) {
    const meAbove = labelY.me <= labelY.opp;
    labelY.me += (meAbove ? -gap : gap) / 2;
    labelY.opp += (meAbove ? gap : -gap) / 2;
  }
  const hits = rows.map((r, i) => {
    const w = (W - L - R) / Math.max(1, rows.length - 1);
    const t = `<b>${r.month}</b><br>${r.games} games${r.rating ? ` · rating ${r.rating}` : ""}<br>` +
      `You ${pct(r.loss_me, 2)} · Opponents ${pct(r.loss_peer, 2)}<br>Your blunders: ${r.blunders_me_per_100.toFixed(1)} per 100 moves`;
    return `<rect class="hit" data-i="${i}" data-tip="${esc(t)}" x="${x(i) - w / 2}" y="${T}" width="${w}" height="${H - T - B}"/>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Winning chances lost per move by month, you versus opponents">
    ${ticks}${xl}
    <line class="cross" id="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/>
    <path class="ln opp" d="${line("loss_peer")}"/>
    <path class="ln me" d="${line("loss_me")}"/>
    <circle class="dot me" id="dot-me" r="4" visibility="hidden"/><circle class="dot opp" id="dot-opp" r="4" visibility="hidden"/>
    <text class="dlabel" x="${x(last) + 8}" y="${labelY.me + 4}">You</text>
    <text class="dlabel" x="${x(last) + 8}" y="${labelY.opp + 4}">Opponents</text>
    ${hits}</svg>`;
  const svg = el.querySelector("svg");
  svg.addEventListener("mouseover", (e) => {
    const i = e.target.dataset?.i;
    if (i == null) return;
    const r = rows[+i];
    const set = (id, attrs) => Object.entries(attrs).forEach(([k, v]) => svg.getElementById(id).setAttribute(k, v));
    set("cross", { x1: x(+i), x2: x(+i), visibility: "visible" });
    set("dot-me", { cx: x(+i), cy: y(r.loss_me), visibility: "visible" });
    set("dot-opp", { cx: x(+i), cy: y(r.loss_peer), visibility: "visible" });
  });
  svg.addEventListener("mouseleave", () => {
    ["cross", "dot-me", "dot-opp"].forEach((id) => svg.getElementById(id).setAttribute("visibility", "hidden"));
  });
}

function renderCauses(rows) {
  $("#causes").innerHTML = `<thead><tr><th>Mistake</th><th class="num">You /100 moves</th><th class="num">Opponents</th><th class="num">Share of your mistakes' cost</th></tr></thead><tbody>` +
    rows.map((c) => `<tr><td>${esc(c.cause)}</td><td class="num">${c.me_per_100.toFixed(2)}</td><td class="num">${c.peer_per_100.toFixed(2)}</td><td class="num">${pct(c.share_of_mistake_loss, 0)}</td></tr>`).join("") +
    "</tbody>";
}

function renderOpenings(rows) {
  $("#openings").innerHTML = rows.length
    ? `<thead><tr><th>First moves</th><th class="num">Games</th><th class="num">Your score</th><th class="num">Lost/move, moves 1–10</th></tr></thead><tbody>` +
      rows.slice(0, 12).map((o) => `<tr><td>${esc(o.opening)}</td><td class="num">${o.games}</td><td class="num">${pct(o.score, 0)}</td><td class="num">${pct(o.loss_me, 2)} <span class="hint">vs ${pct(o.loss_peer, 2)}</span></td></tr>`).join("") +
      "</tbody>"
    : `<tbody><tr><td class="hint">Openings need 15+ games each.</td></tr></tbody>`;
}

function renderAll(rows) {
  let dim = null;
  $("#all").innerHTML = `<thead><tr><th>Category</th><th class="num">Your moves</th><th class="num">You</th><th class="num">Opponents</th><th class="num">Difference</th><th class="num">z</th></tr></thead><tbody>` +
    rows.map((r) => {
      const head = r.dimension !== dim ? `<tr class="group"><td colspan="6">${esc(r.dimension)}</td></tr>` : "";
      dim = r.dimension;
      const flag = r.z >= 2 ? "worse" : r.z <= -2 ? "better" : "";
      return `${head}<tr><td>${esc(nameOf(r.tag))}</td><td class="num">${r.moves.toLocaleString()}</td><td class="num">${pct(r.loss_me, 2)}</td><td class="num">${pct(r.loss_peer, 2)}</td><td class="num ${flag}">${r.excess >= 0 ? "+" : "−"}${pct(Math.abs(r.excess), 2)}${flag ? ` (${flag})` : ""}</td><td class="num">${r.z.toFixed(1)}</td></tr>`;
    }).join("") + "</tbody>";
}

async function load() {
  const recent = $("#recent").value;
  $("#wk-scope").textContent = "Crunching your games (first time can take ~20 seconds)…";
  $("#wk-scope").classList.add("thinking");
  const res = await fetch(`/api/weakness?recent=${recent}`);
  const r = await res.json();
  $("#wk-scope").classList.remove("thinking");
  if (!r.me?.moves) {
    $("#wk-scope").textContent = "No analyzed games yet. Click “Analyze new games”.";
    return;
  }
  $("#wk-scope").textContent =
    `${r.games.toLocaleString()} games, ${r.period[0].replaceAll(".", "-")} to ${r.period[1].replaceAll(".", "-")} · ` +
    `your rating ~${r.my_rating}, opponents ~${r.opp_rating} · ${r.analyzed_total.toLocaleString()} games analyzed in total`;
  renderTiles(r);
  renderSpots($("#weak"), r.weak_spots || [], true);
  renderSpots($("#strong"), r.strengths || [], false);
  renderTrend(r.trend || []);
  renderCauses(r.causes || []);
  renderOpenings(r.openings || []);
  renderAll(r.tags || []);
}

async function pollStatus() {
  const s = await (await fetch("/api/weakness/status")).json();
  $("#update").disabled = s.running;
  $("#update-status").textContent = s.running
    ? "Stockfish is analyzing new games… (about a second per game)"
    : s.error ? `Analysis failed: ${s.error}` : s.last_added != null ? `Added ${s.last_added} new games.` : "";
  $("#update-status").classList.toggle("thinking", s.running);
  if (s.running) setTimeout(pollStatus, 4000);
  return s;
}

$("#recent").addEventListener("change", load);
$("#update").addEventListener("click", async () => {
  await fetch("/api/weakness/update", { method: "POST" });
  const wasRunning = (await pollStatus()).running;
  const wait = setInterval(async () => {
    const s = await (await fetch("/api/weakness/status")).json();
    if (!s.running) { clearInterval(wait); pollStatus(); if (wasRunning) load(); }
  }, 4000);
});

pollStatus();
load();
