const state = {
  sessionId: null,
  lockedCsvText: "",
  wobbleCsvText: "",
  poolCsvText: "",
  data: null,
};

const byId = (id) => document.getElementById(id);

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
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

function render() {
  if (!state.data) return;
  byId("sessionId").textContent = state.sessionId;
  byId("deckCount").textContent = state.data.deck_count;

  const poolRoot = byId("poolCards");
  const lockedRoot = byId("lockedCards");
  const wobbleRoot = byId("wobbleCards");
  const basicsRoot = byId("basicsBar");
  poolRoot.innerHTML = "";
  lockedRoot.innerHTML = "";
  wobbleRoot.innerHTML = "";
  basicsRoot.innerHTML = "";

  for (const card of state.data.pool) {
    poolRoot.appendChild(cardRow(card, [
      { label: "Add to Locked", onClick: () => move(card.name, "pool", "locked") },
      { label: "Add to Wobble", onClick: () => move(card.name, "pool", "wobble") },
    ]));
  }

  for (const card of state.data.locked) {
    lockedRoot.appendChild(cardRow(card, [
      { label: "+", onClick: () => move(card.name, "pool", "locked") },
      { label: "-", onClick: () => move(card.name, "locked", "pool") },
      { label: "Move to Wobble", onClick: () => move(card.name, "locked", "wobble") },
      { label: "Remove", onClick: () => move(card.name, "locked", "pool") },
    ]));
  }

  for (const card of state.data.wobble) {
    wobbleRoot.appendChild(cardRow(card, [
      { label: "Move to Locked", onClick: () => move(card.name, "wobble", "locked") },
      { label: "Remove", onClick: () => move(card.name, "wobble", "pool") },
    ]));
  }

  const basics = ["Island", "Swamp", "Forest", "Mountain", "Plains"];
  for (const name of basics) {
    const b = document.createElement("button");
    b.textContent = `${name} +`;
    b.onclick = () => move(name, "pool", "locked");
    basicsRoot.appendChild(b);
  }
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

async function newSession() {
  const baseP = parseFloat(byId("baseP").value || "0.55");
  const res = await api("/api/session", "POST", { base_p_user: baseP });
  state.sessionId = res.session_id;
  state.data = res.state;
  render();
}

async function loadPool() {
  if (!state.sessionId) await newSession();
  const lockedListText = byId("lockedList").value;
  const wobbleListText = byId("wobbleList").value;
  const poolListText = byId("poolList").value;
  state.data = await api("/api/load_pool", "POST", {
    session_id: state.sessionId,
    list_text: poolListText,
    csv_text: state.poolCsvText,
    locked_list_text: lockedListText,
    locked_csv_text: state.lockedCsvText,
    wobble_list_text: wobbleListText,
    wobble_csv_text: state.wobbleCsvText,
  });
  const baseP = parseFloat(byId("baseP").value || "0.55");
  state.data = await api("/api/set_base_p", "POST", {
    session_id: state.sessionId,
    base_p_user: baseP,
  });
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
  for (const p of out.predictions) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.base_p.toFixed(2)}</td><td>${p.deck_bump.toFixed(5)}</td>`;
    tbody.appendChild(tr);
  }
}

byId("lockedCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    state.lockedCsvText = "";
    return;
  }
  state.lockedCsvText = await f.text();
});

byId("wobbleCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    state.wobbleCsvText = "";
    return;
  }
  state.wobbleCsvText = await f.text();
});

byId("poolCsvFile").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    state.poolCsvText = "";
    return;
  }
  state.poolCsvText = await f.text();
});

byId("newSessionBtn").onclick = () => newSession().catch((e) => alert(e.message));
byId("loadPoolBtn").onclick = () => loadPool().catch((e) => alert(e.message));
byId("evalBtn").onclick = () => evaluateDeck().catch((e) => alert(e.message));

newSession().catch((e) => console.error(e));
