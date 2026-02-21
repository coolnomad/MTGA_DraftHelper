const state = {
  sessionId: null,
  apiBase: "",
  lockedCsvText: "",
  wobbleCsvText: "",
  ignoredCsvText: "",
  data: null,
  suggestions: null,
  beamResult: null,
  parseWarnings: null,
  selectedTopDeck: 0,
};

const byId = (id) => document.getElementById(id);

function normalizeApiBase(v) {
  return String(v || "").trim().replace(/\/+$/, "");
}

function loadApiBase() {
  const q = new URLSearchParams(window.location.search).get("api");
  if (q && q.trim()) {
    const fromQuery = normalizeApiBase(q);
    state.apiBase = fromQuery;
    localStorage.setItem("deckbuildApiBase", fromQuery);
    return;
  }
  state.apiBase = normalizeApiBase(localStorage.getItem("deckbuildApiBase") || "");
}

function saveApiBase() {
  const val = normalizeApiBase(byId("apiBase").value);
  state.apiBase = val;
  localStorage.setItem("deckbuildApiBase", val);
}

function apiUrl(path) {
  if (!path) return path;
  if (/^https?:\/\//i.test(path)) return path;
  if (!state.apiBase) return path;
  if (path.startsWith("/")) return `${state.apiBase}${path}`;
  return `${state.apiBase}/${path}`;
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(apiUrl(path), opts);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status}: ${txt}`);
  }
  return res.json();
}

function imgFor(card) {
  return card.image_url || card.card_image_url || "";
}

function cardRow(card, actions) {
  const row = document.createElement("div");
  row.className = "card";

  const img = document.createElement("img");
  img.src = imgFor(card);
  img.alt = card.name;
  row.appendChild(img);

  const body = document.createElement("div");
  body.className = "card-body";
  body.innerHTML = `<div class="name">${card.name}</div><div class="count">x${card.count}</div>`;

  const btns = document.createElement("div");
  btns.className = "actions";
  for (const a of actions) {
    const b = document.createElement("button");
    b.textContent = a.label;
    b.onclick = a.onClick;
    btns.appendChild(b);
  }
  body.appendChild(btns);
  row.appendChild(body);
  return row;
}

function renderSuggestions() {
  const summary = byId("optSummary");
  const tbody = byId("suggestTable").querySelector("tbody");
  summary.textContent = "";
  tbody.innerHTML = "";
  if (!state.suggestions) return;
  const cur = state.suggestions.current || {};
  const sug = state.suggestions.suggestions || [];
  const warns = (cur.warnings || []).join(" | ");
  summary.textContent = `Current rank_score: ${(cur.rank_score ?? 0).toFixed(5)} | Suggestions: ${sug.length}${warns ? " | " + warns : ""}`;
  for (const s of sug) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${s.remove}</td><td>${s.add}</td><td>${s.delta.toFixed(5)}</td><td>${s.new_rank_score.toFixed(5)}</td>`;
    tbody.appendChild(tr);
  }
}

function selectedOptimizedDeck() {
  if (!state.beamResult || !Array.isArray(state.beamResult.top_decks)) return null;
  const idx = Math.max(0, Math.min(state.selectedTopDeck, state.beamResult.top_decks.length - 1));
  return state.beamResult.top_decks[idx] || null;
}

function renderBeam() {
  const summary = byId("beamSummary");
  const pathRoot = byId("beamPath");
  const select = byId("optimizedDeckSelect");
  const score = byId("optimizedScore");
  const preview = byId("optimizedPreview");
  summary.textContent = "";
  pathRoot.innerHTML = "";
  select.innerHTML = "";
  score.textContent = "";
  preview.textContent = "";
  if (!state.beamResult) return;

  const start = state.beamResult.start || {};
  const topDecks = state.beamResult.top_decks || [];
  const best = topDecks[0] || state.beamResult.best || {};
  const path = best.path || state.beamResult.path || [];
  summary.textContent = `Beam start: ${(start.rank_score ?? 0).toFixed(5)} | best: ${(best.rank_score ?? 0).toFixed(5)} | swaps: ${path.length}`;

  for (const step of path) {
    const li = document.createElement("li");
    li.textContent = `- ${step.remove} + ${step.add} (delta ${Number(step.delta || 0).toFixed(5)})`;
    pathRoot.appendChild(li);
  }

  for (let i = 0; i < topDecks.length; i++) {
    const deck = topDecks[i];
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Deck #${deck.rank} (${Number(deck.rank_score || 0).toFixed(5)})`;
    select.appendChild(opt);
  }

  if (topDecks.length > 0) {
    state.selectedTopDeck = Math.max(0, Math.min(state.selectedTopDeck, topDecks.length - 1));
    select.value = String(state.selectedTopDeck);
  }

  const chosen = selectedOptimizedDeck();
  if (chosen) {
    score.textContent = `Selected rank: #${chosen.rank} | rank_score: ${Number(chosen.rank_score || 0).toFixed(5)} | swaps: ${(chosen.path || []).length}`;
    const lines = Object.entries(chosen.deck_counts || {})
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      .map(([name, count]) => `${count} ${name}`);
    preview.textContent = lines.join("\n");
  }
}

async function setMinLock(card, minCount) {
  state.data = await api("/api/set_min_lock", "POST", {
    session_id: state.sessionId,
    card,
    min_count: Math.max(0, parseInt(minCount || "0", 10) || 0),
  });
}

async function toggleLockAll(card, lock) {
  state.data = await api("/api/toggle_lock_all", "POST", {
    session_id: state.sessionId,
    card,
    lock: !!lock,
  });
}

async function setLandSwapOnly(card, flag) {
  state.data = await api("/api/set_land_swap_only", "POST", {
    session_id: state.sessionId,
    card,
    flag: !!flag,
  });
}

async function move(card, from_zone, to_zone) {
  try {
    state.data = await api("/api/move", "POST", {
      session_id: state.sessionId,
      card_id: card,
      from_zone,
      to_zone,
    });
    render();
  } catch (err) {
    alert(err.message);
  }
}

function render() {
  if (!state.data) return;
  byId("sessionId").textContent = state.sessionId;
  byId("deckCount").textContent = state.data.deck_count;

  const parseWarnings = byId("parseWarnings");
  const warns = state.parseWarnings || state.data.parse_warnings || { unparsed_lines: [], unknown_cards: [] };
  const parts = [];
  if ((warns.unparsed_lines || []).length > 0) {
    parts.push(`Unparsed: ${(warns.unparsed_lines || []).join(" | ")}`);
  }
  if ((warns.unknown_cards || []).length > 0) {
    parts.push(`Unknown cards: ${(warns.unknown_cards || []).join(", ")}`);
  }
  parseWarnings.textContent = parts.join(" | ");

  const lockedRoot = byId("lockedCards");
  const wobbleRoot = byId("wobbleCards");
  const ignoredRoot = byId("ignoredCards");
  const basicsRoot = byId("basicsBar");
  lockedRoot.innerHTML = "";
  wobbleRoot.innerHTML = "";
  ignoredRoot.innerHTML = "";
  basicsRoot.innerHTML = "";

  for (const card of state.data.locked || []) {
    const row = cardRow(card, [
      { label: "+", onClick: () => move(card.name, "wobble", "locked") },
      { label: "-", onClick: () => move(card.name, "locked", "wobble") },
      { label: "To Ignorable", onClick: () => move(card.name, "locked", "ignored") },
    ]);

    const body = row.querySelector(".card-body");
    const minLocked = (state.data.min_locked_counts && state.data.min_locked_counts[card.name]) || 0;
    const lockWrap = document.createElement("div");
    lockWrap.className = "lock-row";

    const lockChk = document.createElement("input");
    lockChk.type = "checkbox";
    lockChk.checked = minLocked >= card.count && card.count > 0;
    lockChk.title = "Lock all current copies";
    lockChk.onchange = async () => {
      try {
        await toggleLockAll(card.name, lockChk.checked);
        render();
      } catch (err) {
        alert(err.message);
      }
    };

    const minInput = document.createElement("input");
    minInput.type = "number";
    minInput.min = "0";
    minInput.max = String(card.count);
    minInput.value = String(Math.min(minLocked, card.count));
    minInput.title = "Minimum copies to keep";
    minInput.onchange = async () => {
      let v = parseInt(minInput.value || "0", 10) || 0;
      if (v < 0) v = 0;
      if (v > card.count) v = card.count;
      minInput.value = String(v);
      try {
        await setMinLock(card.name, v);
        render();
      } catch (err) {
        alert(err.message);
      }
    };

    const minLbl = document.createElement("span");
    minLbl.textContent = minLocked > 0 ? `min: ${minLocked}` : "min: 0";

    const lso = document.createElement("input");
    lso.type = "checkbox";
    lso.checked = !!(state.data.land_swap_only && state.data.land_swap_only[card.name]);
    lso.title = "Land-only swap";
    lso.onchange = async () => {
      try {
        await setLandSwapOnly(card.name, lso.checked);
        render();
      } catch (err) {
        alert(err.message);
      }
    };

    lockWrap.appendChild(document.createTextNode("Lock all "));
    lockWrap.appendChild(lockChk);
    lockWrap.appendChild(document.createTextNode("  min "));
    lockWrap.appendChild(minInput);
    lockWrap.appendChild(minLbl);
    lockWrap.appendChild(document.createTextNode("  land-only "));
    lockWrap.appendChild(lso);
    body.appendChild(lockWrap);
    lockedRoot.appendChild(row);
  }

  for (const card of state.data.wobble || []) {
    const row = cardRow(card, [
      { label: "To Main", onClick: () => move(card.name, "wobble", "locked") },
      { label: "To Ignorable", onClick: () => move(card.name, "wobble", "ignored") },
    ]);

    const body = row.querySelector(".card-body");
    const lso = document.createElement("div");
    lso.className = "lock-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!(state.data.land_swap_only && state.data.land_swap_only[card.name]);
    cb.onchange = async () => {
      try {
        await setLandSwapOnly(card.name, cb.checked);
        render();
      } catch (err) {
        alert(err.message);
      }
    };
    lso.appendChild(document.createTextNode("land-only "));
    lso.appendChild(cb);
    body.appendChild(lso);
    wobbleRoot.appendChild(row);
  }

  for (const card of state.data.ignored || []) {
    ignoredRoot.appendChild(cardRow(card, [
      { label: "To Swappable", onClick: () => move(card.name, "ignored", "wobble") },
    ]));
  }

  const basics = ["Island", "Swamp", "Forest", "Mountain", "Plains"];
  for (const name of basics) {
    const b = document.createElement("button");
    b.textContent = `${name} +`;
    b.onclick = () => move(name, "pool", "locked");
    basicsRoot.appendChild(b);
  }

  renderSuggestions();
  renderBeam();
}

async function newSession() {
  const baseP = parseFloat(byId("baseP").value || "0.55");
  const res = await api("/api/session", "POST", { base_p_user: baseP });
  state.sessionId = res.session_id;
  state.data = res.state;
  state.suggestions = null;
  state.beamResult = null;
  state.parseWarnings = null;
  state.selectedTopDeck = 0;
  render();
}

async function loadPool() {
  if (!state.sessionId) await newSession();
  const lockedListText = byId("lockedList").value;
  const wobbleListText = byId("wobbleList").value;
  const ignoredListText = byId("ignoredList").value;

  const out = await api("/api/load_pool", "POST", {
    session_id: state.sessionId,
    locked_list_text: lockedListText,
    locked_csv_text: state.lockedCsvText,
    wobble_list_text: wobbleListText,
    wobble_csv_text: state.wobbleCsvText,
    ignored_list_text: ignoredListText,
    ignored_csv_text: state.ignoredCsvText,
  });
  state.data = out.state || out;
  state.parseWarnings = out.parse_warnings || null;

  const baseP = parseFloat(byId("baseP").value || "0.55");
  state.data = await api("/api/set_base_p", "POST", {
    session_id: state.sessionId,
    base_p_user: baseP,
  });
  state.suggestions = null;
  state.beamResult = null;
  state.selectedTopDeck = 0;
  render();
}

async function evaluateDeck() {
  if (!state.sessionId) return;
  const basePUser = parseFloat(byId("baseP").value || "0.55");
  const out = await api("/api/evaluate", "POST", {
    session_id: state.sessionId,
    base_p_values: [0.4, 0.5, 0.6, basePUser],
  });
  const tbody = byId("evalTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const p of out.predictions || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.base_p.toFixed(2)}</td><td>${p.deck_bump.toFixed(5)}</td>`;
    tbody.appendChild(tr);
  }
}

async function suggestImprovements() {
  if (!state.sessionId) return;
  const basePUser = parseFloat(byId("baseP").value || "0.55");
  const rankMode = byId("rankMode").value || "user";
  const topK = parseInt(byId("topK").value || "15", 10);
  state.suggestions = await api("/api/suggest_swaps", "POST", {
    session_id: state.sessionId,
    top_k: topK,
    rank_mode: rankMode,
    base_p_values: [0.4, 0.5, 0.6, basePUser],
  });
  renderSuggestions();
}

async function autoIterate() {
  if (!state.sessionId) return;
  const basePUser = parseFloat(byId("baseP").value || "0.55");
  const rankMode = byId("rankMode").value || "user";
  const out = await api("/api/auto_iterate", "POST", {
    session_id: state.sessionId,
    max_steps: 10,
    rank_mode: rankMode,
    base_p_values: [0.4, 0.5, 0.6, basePUser],
  });
  state.data = out.state;
  state.suggestions = null;
  state.beamResult = null;
  render();
}

async function optimizeBeam() {
  if (!state.sessionId) return;
  const basePUser = parseFloat(byId("baseP").value || "0.55");
  const rankMode = byId("rankMode").value || "user";
  const beamTopK = parseInt(byId("beamTopK").value || "5", 10);
  state.beamResult = await api("/api/optimize_beam", "POST", {
    session_id: state.sessionId,
    steps: 8,
    beam_width: 10,
    top_children_per_parent: 60,
    R: 12,
    rank_mode: rankMode,
    mode_removable: "auto",
    include_basic_adds: false,
    include_basic_tweaks: false,
    base_p_values: [0.4, 0.5, 0.6, basePUser],
    return_top_k: Math.max(1, beamTopK || 5),
  });
  state.selectedTopDeck = 0;
  renderBeam();
}

async function applyBeamPath() {
  if (!state.sessionId) return;
  const selected = selectedOptimizedDeck();
  if (!selected) return;
  state.data = await api("/api/apply_optimized_deck", "POST", {
    session_id: state.sessionId,
    deck_counts: selected.deck_counts || {},
  });
  state.suggestions = null;
  render();
}

async function applyOptimizedDeck() {
  if (!state.sessionId) return;
  const selected = selectedOptimizedDeck();
  if (!selected) return;
  state.data = await api("/api/apply_optimized_deck", "POST", {
    session_id: state.sessionId,
    deck_counts: selected.deck_counts || {},
  });
  state.suggestions = null;
  render();
}

byId("lockedCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  state.lockedCsvText = f ? await f.text() : "";
});

byId("wobbleCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  state.wobbleCsvText = f ? await f.text() : "";
});

byId("ignoredCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  state.ignoredCsvText = f ? await f.text() : "";
});

byId("saveApiBaseBtn").onclick = () => {
  saveApiBase();
  newSession().catch((e) => alert(e.message));
};

byId("newSessionBtn").onclick = () => newSession().catch((e) => alert(e.message));
byId("loadPoolBtn").onclick = () => loadPool().catch((e) => alert(e.message));
byId("evalBtn").onclick = () => evaluateDeck().catch((e) => alert(e.message));
byId("suggestBtn").onclick = () => suggestImprovements().catch((e) => alert(e.message));
byId("iterateBtn").onclick = () => autoIterate().catch((e) => alert(e.message));
byId("beamBtn").onclick = () => optimizeBeam().catch((e) => alert(e.message));
byId("applyPathBtn").onclick = () => applyBeamPath().catch((e) => alert(e.message));
byId("applyDeckBtn").onclick = () => applyOptimizedDeck().catch((e) => alert(e.message));
byId("optimizedDeckSelect").onchange = () => {
  state.selectedTopDeck = parseInt(byId("optimizedDeckSelect").value || "0", 10) || 0;
  renderBeam();
};

loadApiBase();
byId("apiBase").value = state.apiBase;
newSession().catch((e) => console.error(e));
