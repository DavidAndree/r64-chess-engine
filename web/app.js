import { Chessground } from "https://cdn.jsdelivr.net/npm/chessground@9.2.1/dist/chessground.min.js";
import { nameOf } from "/static/names.js";

const $ = (sel) => document.querySelector(sel);

const els = {
  board: $("#board"),
  promotion: $("#promotion"),
  form: $("#new-game"),
  eloOut: $("#elo-out"),
  status: $("#status"),
  moves: $("#move-list"),
  resign: $("#resign"),
  pgn: $("#pgn-link"),
  games: $("#games"),
  eloLabel: $("#elo-label"),
  eloFixed: $("#elo-fixed"),
  personal: $("#personal"),
  personalSummary: $("#personal-summary"),
  personalVersions: $("#personal-versions"),
  personalTrain: $("#personal-train"),
  focusLabel: $("#focus-label"),
  drills: $("#drills"),
  drillList: $("#drill-list"),
  banner: $("#banner"),
  top: $("#top-bar"),
  bottom: $("#bottom-bar"),
};

let game = null;     // latest game state from the server
let busy = false;    // waiting for the server / opponent
let opponents = {};  // name -> {label, min_elo, max_elo}
let sparring = null; // /api/sparring: adaptive rating, targets, drills
let focusInfo = null; // focus summary of the finished sparring game on screen
const DEFAULT_ELO = { bot: 1200, stockfish: 1500 };

const cg = Chessground(els.board, {
  movable: { free: false, showDests: true, events: { after: onUserMove } },
  premovable: { enabled: false },
  draggable: { showGhost: true },
  animation: { duration: 200 },
  highlight: { lastMove: true, check: true },
});
window.addEventListener("resize", () => cg.redrawAll());

// ---------- API ----------

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : res.statusText);
  }
  return res.json();
}

// ---------- Rendering ----------

function render() {
  if (!game) {
    els.status.classList.toggle("thinking", busy);
    els.status.textContent = busy ? "Starting game" : "Start a new game.";
    return;
  }
  const myTurn = game.status === "ongoing" && game.turn === game.player_color && !busy;
  const config = {
    orientation: game.player_color,
    movable: {
      color: myTurn ? game.player_color : undefined,
      dests: myTurn ? new Map(Object.entries(game.dests)) : new Map(),
    },
  };
  // While the opponent is thinking, `game` is still the position from before my move: only lock
  // the board. Redrawing it from that position would make my piece jump back until the reply.
  if (!busy) {
    Object.assign(config, {
      fen: game.fen,
      turnColor: game.turn,
      lastMove: game.last_move || undefined,
      check: game.check,
    });
  }
  cg.set(config);

  const oppName = oppLabel(game.opponent) + (game.opponent_elo ? ` (${game.opponent_elo})` : "");
  els.top.textContent = oppName;
  els.bottom.textContent = "You";

  renderMoves();
  renderStatus();
  renderBanner();

  els.resign.disabled = game.status !== "ongoing" || busy;
  if (game.status === "finished") {
    els.pgn.hidden = false;
    els.pgn.href = `/api/games/${game.id}/pgn`;
    els.pgn.download = `r64-game-${game.id}.pgn`;
  } else {
    els.pgn.hidden = true;
  }
}

function renderMoves() {
  els.moves.innerHTML = "";
  // Drills can start mid-game, with Black to move: pad so moves line up in White/Black columns.
  const san = game.first_turn === "black" ? ["…", ...game.san] : game.san;
  const first = game.first_move_number || 1;
  for (let i = 0; i < san.length; i += 2) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="num">${first + i / 2}.</span><span></span><span></span>`;
    li.children[1].textContent = san[i];
    li.children[2].textContent = san[i + 1] || "";
    els.moves.appendChild(li);
  }
  els.moves.scrollTop = els.moves.scrollHeight;
}

function renderStatus() {
  els.status.classList.toggle("thinking", busy);
  if (busy) {
    els.status.textContent = "Opponent is thinking";
    return;
  }
  if (game.status === "finished") {
    els.status.textContent = `${resultText(game)} — ${game.termination.replaceAll("_", " ")}`;
    return;
  }
  els.status.textContent = game.turn === game.player_color
    ? (game.check ? "Your move — you're in check" : "Your move")
    : "Opponent to move";
}

function renderBanner() {
  const b = els.banner;
  if (game.mode === "sparring" && game.drill_tag) {
    let html = `<b>Sparring</b> · focus: ${esc(nameOf(game.drill_tag))}`;
    if (game.status === "finished" && focusInfo?.id === game.id && focusInfo.share != null) {
      const base = focusInfo.baseline != null ? ` (in your chess.com games: ${pct(focusInfo.baseline)})` : "";
      html += `<br><span class="hint">This game put you in that kind of position on ${pct(focusInfo.share)} of your moves${base}.</span>`;
    }
    b.innerHTML = html;
    b.hidden = false;
  } else if (game.mode === "drill" && game.drill_tag) {
    const wp = game.start_wp != null ? ` Stockfish gave you ${pct(game.start_wp)} winning chances here.` : "";
    b.innerHTML = `<b>Drill</b> · ${esc(nameOf(game.drill_tag))}<br><span class="hint">A position from your games where you went wrong.${wp}</span>`;
    b.hidden = false;
  } else {
    b.hidden = true;
  }
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function loadFocus() {
  if (game?.mode !== "sparring" || game.status !== "finished" || focusInfo?.id === game.id) return;
  try {
    focusInfo = { id: game.id, ...(await api(`/api/games/${game.id}/focus`)) };
    renderBanner();
  } catch { /* focus summary is optional */ }
}

async function refreshSparring() {
  sparring = await api("/api/sparring");
  els.drills.hidden = !sparring.available || !sparring.drills?.length;
  if (!sparring.available) return;
  els.drillList.innerHTML = "";
  for (const d of sparring.drills) {
    const li = document.createElement("li");
    const stats = d.played
      ? `${d.played} played · scored ${d.score} (Stockfish odds ${d.expected.toFixed(1)})`
      : `${d.positions} positions`;
    li.innerHTML = `<div><div class="drill-name"></div><div class="hint"></div></div><button>Practice</button>`;
    li.querySelector(".drill-name").textContent = nameOf(d.tag);
    li.querySelector(".hint").textContent = stats;
    const btn = li.querySelector("button");
    btn.disabled = !d.positions;
    btn.addEventListener("click", () => {
      game = null;
      run(api("/api/drills", { method: "POST", body: JSON.stringify({ tag: d.tag }) }));
    });
    els.drillList.appendChild(li);
  }
  onOpponentChange(false);
}

function resultText(g) {
  if (g.result === "1/2-1/2") return "Draw";
  const whiteWon = g.result === "1-0";
  const iWon = (g.player_color === "white") === whiteWon;
  return iWon ? "You won" : "You lost";
}

async function refreshGames() {
  const list = await api("/api/games?limit=30");
  els.games.innerHTML = "";
  for (const g of list) {
    const li = document.createElement("li");
    const outcome = g.status === "ongoing" ? "ongoing" : resultText(g);
    const cls = outcome === "You won" ? "win" : outcome === "You lost" ? "loss" : "";
    li.innerHTML = `<span class="${cls}"></span><span class="meta"></span>`;
    li.children[0].textContent = outcome;
    li.children[1].textContent =
      `#${g.id} · ${g.player_color} vs ${oppLabel(g.opponent)}${g.opponent_elo ? " " + g.opponent_elo : ""} · ${Math.ceil(g.num_moves / 2)} moves`;
    li.addEventListener("click", () => loadGame(g.id));
    els.games.appendChild(li);
  }
}

function oppLabel(name) {
  return opponents[name]?.label ?? name;
}

async function loadOpponents() {
  const list = await api("/api/opponents");
  opponents = Object.fromEntries(list.map((o) => [o.name, o]));
  const select = els.form.opponent;
  select.innerHTML = "";
  for (const o of list) select.add(new Option(o.label, o.name));
  onOpponentChange();
}

function onOpponentChange(resetSlider = true) {
  const o = opponents[els.form.opponent.value];
  if (!o) return;
  const slider = els.form.elo;
  slider.min = o.min_elo;
  slider.max = o.max_elo;
  if (resetSlider) slider.value = DEFAULT_ELO[o.name] ?? Math.round((o.min_elo + o.max_elo) / 2);
  els.eloOut.textContent = slider.value;
  if (o.name === "sparring") {
    // The sparring partner sets its own level and picks a weak spot to aim at.
    els.eloLabel.hidden = true;
    els.eloFixed.hidden = false;
    els.focusLabel.hidden = false;
    if (sparring?.available) {
      els.eloFixed.textContent = `Adapts to you: plays at ${sparring.rating} now ` +
        `(${sparring.games} sparring game${sparring.games === 1 ? "" : "s"}). Aims at your weak spots.`;
      const sel = els.form.focus;
      const keep = sel.value;
      sel.innerHTML = "";
      sel.add(new Option(sparring.next_focus ? `Auto (next: ${nameOf(sparring.next_focus)})` : "Auto", "auto"));
      for (const t of sparring.targets) sel.add(new Option(nameOf(t.tag), t.tag));
      if ([...sel.options].some((op) => op.value === keep)) sel.value = keep;
      if (!sparring.targets.length) els.eloFixed.textContent += " (Open the weakness map once to find them.)";
    }
    return;
  }
  els.focusLabel.hidden = true;
  // Personal models play at the rating I had when they were trained: no slider.
  const fixed = o.min_elo === o.max_elo;
  els.eloLabel.hidden = fixed;
  els.eloFixed.hidden = !fixed;
  els.eloFixed.textContent = fixed ? `Plays like you did at ${o.min_elo}.` : "";
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

async function refreshPersonal() {
  const p = await api("/api/personal");
  els.personal.hidden = !p.available;
  if (!p.available) return;
  const left = Math.max(0, p.every - p.games_since_last);
  let summary;
  if (p.training) {
    summary = "Training a new version on all your games (a few minutes)";
  } else if (!p.versions.length) {
    summary = "No personal model yet. Train one from your imported games.";
  } else {
    const v = p.versions[0];
    summary = `<strong>v${v.version}</strong> predicts <strong>${pct(v.val_acc)}</strong> of your moves ` +
      `(base model ${pct(v.base_val_acc)}). Next update after ${left} more game${left === 1 ? "" : "s"}.`;
  }
  els.personalSummary.innerHTML = summary;
  els.personalSummary.classList.toggle("thinking", p.training);
  if (p.error) els.personalSummary.textContent = `Last training failed: ${p.error}`;
  els.personalVersions.innerHTML = "";
  for (const v of p.versions) {
    const li = document.createElement("li");
    li.innerHTML = "<span></span><span class=gain></span>";
    li.children[0].textContent = v.label;
    li.children[1].textContent = `${pct(v.val_acc)} (${v.val_acc >= v.base_val_acc ? "+" : ""}${((v.val_acc - v.base_val_acc) * 100).toFixed(1)})`;
    els.personalVersions.appendChild(li);
  }
  els.personalTrain.disabled = p.training;
  els.personalTrain.textContent = p.versions.length ? "Retrain now" : "Train now";
  const wasTraining = personalTraining;
  personalTraining = p.training;
  if (p.training) setTimeout(refreshPersonal, 5000);
  else if (wasTraining) loadOpponents();  // a new version is ready to play
}
let personalTraining = false;

// ---------- Actions ----------

async function run(promise) {
  busy = true;
  render();
  try {
    game = await promise;
  } catch (err) {
    els.status.textContent = `Error: ${err.message}`;
  } finally {
    busy = false;
    render();
    refreshGames();
    if (game?.status === "finished") {
      refreshPersonal();
      refreshSparring();
      loadFocus();
    }
  }
}

async function onUserMove(orig, dest) {
  let uci = orig + dest;
  if (isPromotion(dest)) {
    const piece = await pickPromotion();
    uci += piece;
    const role = { q: "queen", r: "rook", b: "bishop", n: "knight" }[piece];
    cg.setPieces(new Map([[dest, { role, color: game.player_color, promoted: true }]]));
  }
  await run(api(`/api/games/${game.id}/move`, { method: "POST", body: JSON.stringify({ uci }) }));
}

function isPromotion(dest) {
  const piece = cg.state.pieces.get(dest);
  return piece && piece.role === "pawn" && (dest[1] === "8" || dest[1] === "1");
}

function pickPromotion() {
  const white = game.player_color === "white";
  const choices = [
    ["q", white ? "♕" : "♛"],
    ["r", white ? "♖" : "♜"],
    ["b", white ? "♗" : "♝"],
    ["n", white ? "♘" : "♞"],
  ];
  return new Promise((resolve) => {
    els.promotion.innerHTML = "";
    for (const [code, glyph] of choices) {
      const b = document.createElement("button");
      b.textContent = glyph;
      b.style.fontSize = "44px";
      b.addEventListener("click", () => {
        els.promotion.hidden = true;
        resolve(code);
      });
      els.promotion.appendChild(b);
    }
    els.promotion.hidden = false;
  });
}

async function loadGame(id) {
  game = await api(`/api/games/${id}`);
  render();
  loadFocus();
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const data = new FormData(els.form);
  const body = {
    color: data.get("color"),
    opponent: data.get("opponent"),
    elo: Number(data.get("elo")),
    focus: data.get("focus") || null,
  };
  game = null;
  run(api("/api/games", { method: "POST", body: JSON.stringify(body) }));
});

els.form.opponent.addEventListener("change", () => onOpponentChange());

els.personalTrain.addEventListener("click", async () => {
  await api("/api/personal/train", { method: "POST" });
  refreshPersonal();
});

els.form.elo.addEventListener("input", () => {
  els.eloOut.textContent = els.form.elo.value;
});

els.resign.addEventListener("click", () => {
  if (!game || !confirm("Resign this game?")) return;
  run(api(`/api/games/${game.id}/resign`, { method: "POST" }));
});

// Resume the most recent unfinished game, if any.
(async () => {
  await loadOpponents();
  refreshPersonal();
  refreshSparring();
  await refreshGames();
  const list = await api("/api/games?limit=1");
  if (list.length && list[0].status === "ongoing") loadGame(list[0].id);
})();
