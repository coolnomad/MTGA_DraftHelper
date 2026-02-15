let sessionId = null;
let useCardImage = false;

const statusEl = document.getElementById("status");
const mainListEl = document.getElementById("mainList");
const sideListEl = document.getElementById("sideList");
const basicPaletteEl = document.getElementById("basicPalette");
const deckCountEl = document.getElementById("deckCount");
const poolMetaEl = document.getElementById("poolMeta");
const scoreBoxEl = document.getElementById("scoreBox");

function setStatus(message) {
  statusEl.textContent = message;
}

async function createSession() {
  const setCode = document.getElementById("setCode").value.trim() || "FIN";
  const seed = Number(document.getElementById("seed").value) || 1337;
  setStatus("Loading random pool.");
  const res = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ set_code: setCode, seed }),
  });
  if (!res.ok) {
    setStatus("Failed to load pool");
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  renderState(data.state);
  setStatus("Pool loaded");
}

async function rerollPool() {
  if (!sessionId) {
    setStatus("Load a pool first");
    return;
  }
  const seed = Number(document.getElementById("seed").value) || 1337;
  setStatus("Loading another random pool.");
  const res = await fetch("/api/random_pool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, seed }),
  });
  if (!res.ok) {
    setStatus("Failed to reroll pool");
    return;
  }
  const state = await res.json();
  renderState(state);
  setStatus("New pool loaded");
}

async function submitScore() {
  if (!sessionId) {
    setStatus("Load a pool first");
    return;
  }
  setStatus("Scoring deck.");
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    setStatus("Score request failed");
    return;
  }
  const state = await res.json();
  renderState(state);
  setStatus("Score updated");
}

async function moveCard(cardId, to) {
  if (!sessionId) return;
  const res = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, card_id: cardId, to }),
  });
  if (!res.ok) {
    setStatus("Move failed");
    return;
  }
  const state = await res.json();
  renderState(state);
}

function renderCardRow(item, moveTo) {
  const row = document.createElement("button");
  row.className = "pool-item";
  row.onclick = () => moveCard(item.id, moveTo);
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
  return row;
}

function renderBasics(basicLands) {
  basicPaletteEl.innerHTML = "";
  basicLands.forEach((card) => {
    const item = document.createElement("div");
    item.className = "basic-card";
    const img = document.createElement("img");
    const primary = useCardImage ? card.card_image_url || card.image_url : card.image_url || card.card_image_url;
    img.src = primary || card.art_uri || "/static/placeholder.svg";
    img.alt = card.name;
    const info = document.createElement("div");
    info.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = card.name;
    const count = document.createElement("div");
    count.className = "count";
    count.textContent = `Main: ${card.count || 0}`;
    const actions = document.createElement("div");
    actions.className = "actions";
    const addBtn = document.createElement("button");
    addBtn.textContent = "+";
    addBtn.onclick = () => moveCard(card.id, "main");
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "-";
    removeBtn.onclick = () => moveCard(card.id, "sideboard");
    actions.appendChild(addBtn);
    actions.appendChild(removeBtn);
    info.appendChild(name);
    info.appendChild(count);
    info.appendChild(actions);
    item.appendChild(img);
    item.appendChild(info);
    basicPaletteEl.appendChild(item);
  });
}

function renderScore(score) {
  if (!score) {
    scoreBoxEl.innerHTML = "<div>`Submit for Scoring` to evaluate current main deck.</div>";
    return;
  }
  const examples = score.ignored_examples && score.ignored_examples.length
    ? score.ignored_examples.join(", ")
    : "(none)";
  scoreBoxEl.innerHTML = `
    <div><strong>p_hat:</strong> ${Number(score.p_hat).toFixed(4)}</div>
    <div><strong>bump:</strong> ${Number(score.bump).toFixed(4)}</div>
    <div><strong>deck_count:</strong> ${score.deck_count}</div>
    <div><strong>known_cards_count:</strong> ${score.known_cards_count}</div>
    <div><strong>ignored_cards_count:</strong> ${score.ignored_cards_count}</div>
    <div><strong>ignored_examples:</strong> ${examples}</div>
  `;
}

function renderState(state) {
  mainListEl.innerHTML = "";
  sideListEl.innerHTML = "";

  (state.pool_main || []).forEach((item) => {
    mainListEl.appendChild(renderCardRow(item, "sideboard"));
  });
  (state.pool_sideboard || []).forEach((item) => {
    sideListEl.appendChild(renderCardRow(item, "main"));
  });
  renderBasics(state.basic_lands || []);
  renderScore(state.score);

  const deckCount = Number(state.deck_count || 0);
  const deckFlag = deckCount === 40 ? " ok" : "";
  deckCountEl.className = `chip${deckFlag}`;
  deckCountEl.textContent = `Cards: ${deckCount} / 40`;
  poolMetaEl.textContent = `Pool row ${state.pool_index} | Pool cards ${state.pool_size}`;
}

document.getElementById("startBtn").addEventListener("click", () => createSession());
document.getElementById("rerollBtn").addEventListener("click", () => rerollPool());
document.getElementById("scoreBtn").addEventListener("click", () => submitScore());
document.getElementById("showCardImage").addEventListener("change", (event) => {
  useCardImage = event.target.checked;
  if (sessionId) {
    fetch(`/api/state?session_id=${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((state) => {
        if (state) renderState(state);
      });
  }
});
