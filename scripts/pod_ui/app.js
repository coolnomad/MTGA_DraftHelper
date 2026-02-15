let sessionId = null;
let useCardImage = false;
let cardSize = 160;

const statusEl = document.getElementById("status");
const draftView = document.getElementById("draftView");
const deckView = document.getElementById("deckView");
const packGrid = document.getElementById("packGrid");
const poolList = document.getElementById("poolList");
const sideList = document.getElementById("sideList");
const packTitle = document.getElementById("packTitle");
const cardSizeInput = document.getElementById("cardSize");
const cardImageToggle = document.getElementById("showCardImage");
const deckEffectEl = document.getElementById("deckEffect");
const deckBumpEl = document.getElementById("deckBump");
const deckSideList = document.getElementById("deckSideList");
const deckMainList = document.getElementById("deckMainList");
const deckEffectBuildEl = document.getElementById("deckEffectBuilder");
const deckBumpBuildEl = document.getElementById("deckBumpBuilder");
const deckCountBuildEl = document.getElementById("deckCountBuilder");
const basicPalette = document.getElementById("basicPalette");

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function createSession() {
  const format = document.getElementById("format").value.trim();
  const seed = Number(document.getElementById("seed").value) || 1337;
  const humanSeat = Number(document.getElementById("humanSeat").value) || 0;
  const bots = document
    .getElementById("bots")
    .value.trim()
    .split(",")
    .map((b) => b.trim())
    .filter(Boolean);
  setStatus("Starting session.");
  const res = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, seed, human_seat: humanSeat, bot_policies: bots }),
  });
  if (!res.ok) {
    setStatus("Failed to start session");
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  setStatus("Session ready");
  renderState(data.state);
}

async function fetchState() {
  if (!sessionId) return;
  const res = await fetch(`/api/state?session_id=${sessionId}`);
  if (!res.ok) {
    setStatus("Session missing");
    return;
  }
  const state = await res.json();
  renderState(state);
}

async function pickCard(cardId) {
  if (!sessionId) return;
  setStatus("Submitting pick.");
  const res = await fetch("/api/pick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, card_id: cardId }),
  });
  if (!res.ok) {
    setStatus("Pick failed");
    return;
  }
  const state = await res.json();
  renderState(state);
}

function renderState(state) {
  window.latestDeckCount = state.deck_count || 0;
  if (state.status === "finished") {
    draftView.classList.add("hidden");
    deckView.classList.remove("hidden");
    packTitle.textContent = "Draft finished - build your deck";
    packGrid.innerHTML = "<div class='finished'>Draft complete. Build your deck below.</div>";
    renderPack([]);
    renderDeckBuilder(state.pool_sideboard || [], state.pool_main || [], state.basic_lands || []);
    renderDeckStats(state.deck_effect, state.deck_bump);
    setStatus("Deck build mode");
    return;
  }
  draftView.classList.remove("hidden");
  deckView.classList.add("hidden");
  packTitle.textContent = `Pack ${state.pack_number} | Pick ${state.pick_number}`;
  renderPack(state.pack || []);
  renderPool(state.pool_main || [], "main");
  renderSideboard(state.pool_sideboard || []);
  renderDeckStats(state.deck_effect, state.deck_bump);
  setStatus("Awaiting pick");
}

function renderPack(cards) {
  packGrid.innerHTML = "";
  cards.forEach((card) => {
    const cardEl = document.createElement("div");
    cardEl.className = "card";
    cardEl.onclick = () => pickCard(card.id);
    const img = document.createElement("img");
    const primary = useCardImage ? card.card_image_url || card.image_url : card.image_url || card.card_image_url;
    img.src = primary || card.art_uri || "/static/placeholder.svg";
    img.alt = card.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = card.name;
    const metrics = document.createElement("div");
    metrics.className = "metrics";
    if (card.projected_deck_effect !== null && card.projected_deck_effect !== undefined) {
      const b1 = document.createElement("div");
      b1.className = "badge";
      b1.textContent = `Effect ${Number(card.projected_deck_effect).toFixed(3)}`;
      metrics.appendChild(b1);
    }
    if (card.projected_deck_bump !== null && card.projected_deck_bump !== undefined) {
      const b2 = document.createElement("div");
      b2.className = "badge";
      b2.textContent = `Bump ${Number(card.projected_deck_bump).toFixed(3)}`;
      metrics.appendChild(b2);
    }
    if (card.projected_deck_effect_delta !== null && card.projected_deck_effect_delta !== undefined) {
      const b3 = document.createElement("div");
      b3.className = "badge";
      b3.textContent = `?Eff ${Number(card.projected_deck_effect_delta).toFixed(3)}`;
      metrics.appendChild(b3);
    }
    if (card.projected_deck_bump_delta !== null && card.projected_deck_bump_delta !== undefined) {
      const b4 = document.createElement("div");
      b4.className = "badge";
      b4.textContent = `?Bump ${Number(card.projected_deck_bump_delta).toFixed(3)}`;
      metrics.appendChild(b4);
    }
    meta.appendChild(name);
    meta.appendChild(metrics);
    cardEl.appendChild(img);
    cardEl.appendChild(meta);
    packGrid.appendChild(cardEl);
  });
}

function buildPoolRow(item, moveTo) {
  const row = document.createElement("div");
  row.className = "pool-item";
  const img = document.createElement("img");
  const primary = useCardImage ? item.card_image_url || item.image_url : item.image_url || item.card_image_url;
  img.src = primary || item.art_uri || "/static/placeholder.svg";
  img.alt = item.name;
  const details = document.createElement("div");
  details.className = "details";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = item.name;
  const count = document.createElement("div");
  count.className = "count";
  count.textContent = `x${item.count}`;
  details.appendChild(name);
  details.appendChild(count);
  row.appendChild(img);
  row.appendChild(details);
  row.onclick = () => moveCard(item.id, moveTo);
  return row;
}

function renderPool(pool, area) {
  poolList.innerHTML = "";
  pool.forEach((item) => {
    const row = buildPoolRow(item, area === "main" ? "sideboard" : "main");
    poolList.appendChild(row);
  });
}

function renderSideboard(pool) {
  sideList.innerHTML = "";
  pool.forEach((item) => {
    const row = buildPoolRow(item, "main");
    sideList.appendChild(row);
  });
}

function renderDeckBuilder(sideboardPool, mainDeck, basicLands) {
  deckSideList.innerHTML = "";
  sideboardPool.forEach((item) => {
    const row = buildPoolRow(item, "main");
    deckSideList.appendChild(row);
  });
  deckMainList.innerHTML = "";
  let mainCount = 0;
  mainDeck.forEach((item) => {
    mainCount += Number(item.count || 0);
    const row = buildPoolRow(item, "sideboard");
    deckMainList.appendChild(row);
  });
  window.latestDeckCount = mainCount;
  if (deckCountBuildEl) {
    deckCountBuildEl.textContent = `Cards: ${mainCount}`;
  }
  if (basicPalette) {
    basicPalette.innerHTML = "";
    basicLands.forEach((card) => {
      const cardEl = document.createElement("div");
      cardEl.className = "basic-card";
      const img = document.createElement("img");
      const primary = useCardImage ? card.card_image_url || card.image_url : card.image_url || card.card_image_url;
      img.src = primary || card.art_uri || "/static/placeholder.svg";
      img.alt = card.name;
      const meta = document.createElement("div");
      meta.className = "meta";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = card.name;
      const count = document.createElement("div");
      count.className = "count";
      count.textContent = `Main: ${card.count || 0}`;
      const actions = document.createElement("div");
      actions.className = "basic-actions";
      const addBtn = document.createElement("button");
      addBtn.textContent = "+";
      addBtn.onclick = () => moveCard(card.id, "main");
      const removeBtn = document.createElement("button");
      removeBtn.textContent = "-";
      removeBtn.onclick = () => moveCard(card.id, "sideboard");
      actions.appendChild(addBtn);
      actions.appendChild(removeBtn);
      meta.appendChild(name);
      meta.appendChild(count);
      meta.appendChild(actions);
      cardEl.appendChild(img);
      cardEl.appendChild(meta);
      basicPalette.appendChild(cardEl);
    });
  }
}

function renderDeckStats(effect, bump) {
  const effectTxt = effect !== undefined && effect !== null ? Number(effect).toFixed(3) : "--";
  const bumpTxt = bump !== undefined && bump !== null ? Number(bump).toFixed(3) : "--";
  deckEffectEl.textContent = `Effect: ${effectTxt}`;
  deckBumpEl.textContent = `Bump: ${bumpTxt}`;
  if (deckEffectBuildEl) deckEffectBuildEl.textContent = `Effect: ${effectTxt}`;
  if (deckBumpBuildEl) deckBumpBuildEl.textContent = `Bump: ${bumpTxt}`;
  if (deckCountBuildEl && window.latestDeckCount !== undefined) {
    deckCountBuildEl.textContent = `Cards: ${window.latestDeckCount}`;
  }
}

async function moveCard(cardId, to) {
  if (!sessionId) return;
  const res = await fetch(`/api/move?session_id=${sessionId}&card_id=${encodeURIComponent(cardId)}&to=${to}`, { method: "POST" });
  if (!res.ok) {
    setStatus("Move failed");
    return;
  }
  const state = await res.json();
  renderState(state);
}

document.getElementById("startBtn").addEventListener("click", () => {
  createSession();
});

cardImageToggle.addEventListener("change", (e) => {
  useCardImage = e.target.checked;
  fetchState();
});

cardSizeInput.addEventListener("input", (e) => {
  cardSize = Number(e.target.value) || 160;
  document.documentElement.style.setProperty("--card-width", `${cardSize}px`);
});

// preload placeholder
const img = new Image();
img.src = "/static/placeholder.svg";
