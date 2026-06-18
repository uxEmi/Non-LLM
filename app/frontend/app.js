const API_URL  = "http://localhost:8000/predict";
const CONFIDENCE_THRESHOLD = 0.50;

const TEAMS = {
  "Loans":                       "Credite",
  "Credit Reporting":            "Raportare credit",
  "Bank Accounts and Services":  "Conturi și servicii bancare",
  "Debt Collection":             "Colectare debite",
  "Credit Card Services":        "Servicii carduri de credit",
};
const REVIEW_KEY = "__review__";
const REVIEW_LABEL_RO = "Verificare umană";

const EXAMPLES = [
  "Am fost taxat de două ori pentru taxa anuală a cardului și vreau banii înapoi.",
  "Primesc apeluri zilnice de la o firmă de recuperări pentru o datorie pe care nu o recunosc.",
  "Raportul meu de credit arată un cont pe care nu l-am deschis niciodată.",
  "Aplicația pentru creditul ipotecar mi-a fost respinsă fără nicio explicație.",
  "Mi s-au perceput comisioane de administrare neautorizate la contul curent.",
];

let currentModel = "svm";

const $ = (id) => document.getElementById(id);
const queuesEl = $("queues");
const routeBtn = $("route-btn");
const ticketEl = $("ticket");
const badgeEl  = $("board-tag");
const resultEl = $("readout-grid");
const emptyEl  = $("readout-empty");
const ringEl   = $("r-ring");

const RING_CIRC = 2 * Math.PI * 52;
ringEl.style.strokeDasharray = RING_CIRC;
ringEl.style.strokeDashoffset = RING_CIRC;

const queueEls = {};
function buildQueues() {
  Object.entries(TEAMS).forEach(([enKey, ro]) => {
    const el = document.createElement("div");
    el.className = "queue";
    el.setAttribute("role", "listitem");
    el.dataset.key = enKey;
    el.innerHTML = `<span class="queue-node"></span><span class="queue-name">${ro}</span>`;
    queuesEl.appendChild(el);
    queueEls[enKey] = el;
  });

  const div = document.createElement("div");
  div.className = "queues-divider";
  div.textContent = "prag";
  queuesEl.appendChild(div);

  const rev = document.createElement("div");
  rev.className = "queue is-review";
  rev.setAttribute("role", "listitem");
  rev.dataset.key = REVIEW_KEY;
  rev.innerHTML = `<span class="queue-node"></span><span class="queue-name">${REVIEW_LABEL_RO}</span>`;
  queuesEl.appendChild(rev);
  queueEls[REVIEW_KEY] = rev;
}

function buildChips() {
  EXAMPLES.forEach((txt) => {
    const c = document.createElement("button");
    c.className = "chip";
    c.type = "button";
    c.textContent = txt.length > 46 ? txt.slice(0, 44) + "…" : txt;
    c.title = txt;
    c.addEventListener("click", () => { ticketEl.value = txt; ticketEl.focus(); });
    $("chips").appendChild(c);
  });
}

function wireModelGroup() {
  document.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((b) => b.setAttribute("aria-checked", "false"));
      btn.setAttribute("aria-checked", "true");
      currentModel = btn.dataset.model;
    });
  });
}

async function predict(text, model) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  if (!res.ok) throw new Error("Backend " + res.status);
  return res.json();
}

function countUp(el, to, dur = 700) {
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(to * eased);
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function setBadge(text, kind) {
  badgeEl.textContent = text;
  badgeEl.classList.remove("is-routed", "is-review");
  if (kind) badgeEl.classList.add(kind);
}

function render(result) {
  const review = result.confidence < CONFIDENCE_THRESHOLD;
  const key = review ? REVIEW_KEY : result.predicted_team;
  const targetEl = queueEls[key] || queueEls[REVIEW_KEY];

  emptyEl.hidden = true;
  resultEl.hidden = false;
  resultEl.querySelectorAll(".tile").forEach((t) => {
    t.style.animation = "none"; void t.offsetWidth; t.style.animation = "";
  });

  Object.values(queueEls).forEach((el) => el.classList.remove("active", "dim"));
  Object.values(queueEls).forEach((el) => { if (el !== targetEl) el.classList.add("dim"); });
  targetEl.classList.add("active");

  const teamRo = review ? REVIEW_LABEL_RO : TEAMS[result.predicted_team];
  $("r-team").textContent = teamRo;
  $("r-team").style.color = review ? "var(--alert)" : "var(--text)";
  $("r-sub").textContent = review ? "sub pragul de încredere" : `rutat de ${result.model.toUpperCase()}`;
  const dot = $("r-dot");
  dot.style.background = review ? "var(--alert)" : "var(--accent)";
  dot.style.boxShadow = review ? "0 0 0 4px var(--alert-soft)" : "0 0 0 4px var(--accent-soft)";
  setBadge(review ? "verificare" : "rutat", review ? "is-review" : "is-routed");

  const confPct = Math.round(result.confidence * 100);
  countUp($("r-conf"), confPct);
  ringEl.style.stroke = review ? "var(--alert)" : "url(#ringGrad)";
  requestAnimationFrame(() => {
    ringEl.style.strokeDashoffset = RING_CIRC * (1 - result.confidence);
  });

  countUp($("r-lat"), result.latency_ms);
  $("r-model").textContent = result.model.toUpperCase();

  const wait = result.estimated_wait_min ?? 0;
  countUp($("r-wait"), wait);
  $("r-wait").style.color = wait <= 5 ? "#1FA971" : wait <= 15 ? "#E8890B" : "var(--alert)";

  renderChips(review ? [] : (result.top_words || []));
  renderHeatmap(result.text_en, result.word_weights, TEAMS[result.predicted_team] || result.predicted_team);
  updateArena(result);
}

function renderChips(words) {
  const why = $("r-why"), wrap = $("r-words");
  wrap.innerHTML = "";
  if (words && words.length) {
    words.forEach((w) => {
      const s = document.createElement("span");
      s.className = "why-word";
      s.textContent = w;
      wrap.appendChild(s);
    });
    why.hidden = false;
  } else {
    why.hidden = true;
  }
}

function normWord(tok) {
  return tok.toLowerCase()
    .replace(/[ăâ]/g, "a").replace(/î/g, "i").replace(/[șş]/g, "s").replace(/[țţ]/g, "t")
    .replace(/[^a-z0-9]+/g, "");
}

function renderHeatmap(text, weights, teamRo) {
  const tile = $("r-heat-tile"), box = $("r-heat");
  if (!weights || !Object.keys(weights).length || !text) { tile.hidden = true; return; }
  box.innerHTML = "";
  text.split(/(\s+)/).forEach((tok) => {
    if (/^\s+$/.test(tok) || tok === "") { box.appendChild(document.createTextNode(tok)); return; }
    const key = normWord(tok);
    const w = weights[key];
    const span = document.createElement("span");
    span.className = "heat-word";
    span.textContent = tok;
    if (w !== undefined && Math.abs(w) > 0.06) {
      const a = Math.min(Math.abs(w), 1) * 0.6;
      span.style.background = w > 0 ? `rgba(31,169,113,${a})` : `rgba(229,72,77,${a})`;
      span.title = `${w > 0 ? "susține" : "împotriva"} ${teamRo} (${w.toFixed(2)})`;
    }
    box.appendChild(span);
  });
  tile.hidden = false;
}

async function explainRo(text, model, review) {
  try {
    const res = await fetch(API_EXPLAIN, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderChips(review ? [] : (data.top_words || []));
    renderHeatmap(data.heat_text, data.word_weights, TEAMS[data.predicted_team] || data.predicted_team);
  } catch (err) { /* keep English fallback */ }
}

const CLASS_LIST = Object.keys(TEAMS);
const ARENA_SHORT = {
  "Loans": "Credite",
  "Credit Reporting": "Raportare",
  "Bank Accounts and Services": "Conturi",
  "Debt Collection": "Colectare",
  "Credit Card Services": "Carduri",
};
const MODEL_COLORS = { svm: "#F59E0B", xgboost: "#FB7185", bilstm: "#2BB3C0" };

const API_TRACE = API_URL.replace(/\/predict$/, "/trace");
const API_EXPLAIN = API_URL.replace(/\/predict$/, "/explain");
const arenaCanvas = $("arena-canvas");
const actx = arenaCanvas.getContext("2d");
let arenaAnchors = [];
let arenaOrbs = [];
let arenaRAF = null;
let trailPoints = [];
let traceTimer = null;
let lastTrace = null;
let lastTraceModel = "svm";

function hexA(hex, a) {
  const v = Math.max(0, Math.min(255, Math.round(a * 255))).toString(16).padStart(2, "0");
  return hex + v;
}

function arenaLayout() {
  const w = arenaCanvas.clientWidth || 600;
  const h = Math.max(320, Math.min(420, w * 0.62));
  const dpr = window.devicePixelRatio || 1;
  arenaCanvas.width = w * dpr;
  arenaCanvas.height = h * dpr;
  arenaCanvas.style.height = h + "px";
  actx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cx = w / 2, cy = h / 2 + 6, R = Math.min(w, h) * 0.33;
  arenaAnchors = CLASS_LIST.map((c, i) => {
    const a = -Math.PI / 2 + i * 2 * Math.PI / 5;
    return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), label: ARENA_SHORT[c], cx, cy };
  });
}

function orbPos(probs) {
  let x = 0, y = 0;
  CLASS_LIST.forEach((c, i) => { const p = probs[c] || 0; x += p * arenaAnchors[i].x; y += p * arenaAnchors[i].y; });
  return { x, y };
}

function drawArena() {
  const w = arenaCanvas.clientWidth, h = parseFloat(arenaCanvas.style.height);
  actx.clearRect(0, 0, w, h);

  actx.strokeStyle = "rgba(140,110,70,.16)";
  actx.lineWidth = 1;
  actx.beginPath();
  arenaAnchors.forEach((a, i) => { i ? actx.lineTo(a.x, a.y) : actx.moveTo(a.x, a.y); });
  actx.closePath();
  actx.stroke();

  if (trailPoints.length > 1) {
    const color = arenaOrbs.length ? arenaOrbs[0].color : "#F59E0B";
    for (let k = 1; k < trailPoints.length; k++) {
      actx.strokeStyle = hexA(color, (k / trailPoints.length) * 0.55);
      actx.lineWidth = 2.5;
      actx.lineCap = "round";
      actx.beginPath();
      actx.moveTo(trailPoints[k - 1].x, trailPoints[k - 1].y);
      actx.lineTo(trailPoints[k].x, trailPoints[k].y);
      actx.stroke();
    }
  } else if (arenaOrbs.length === 1 && arenaOrbs[0].src) {
    const o = arenaOrbs[0];
    CLASS_LIST.forEach((c, i) => {
      const p = o.src[c] || 0;
      actx.strokeStyle = `rgba(245,158,11,${Math.max(0.05, p * 0.7)})`;
      actx.lineWidth = 1 + p * 6;
      actx.beginPath();
      actx.moveTo(arenaAnchors[i].x, arenaAnchors[i].y);
      actx.lineTo(o.x, o.y);
      actx.stroke();
    });
  }

  arenaAnchors.forEach((a) => {
    actx.beginPath();
    actx.arc(a.x, a.y, 6, 0, Math.PI * 2);
    actx.fillStyle = "#CBB082";
    actx.fill();
    actx.font = "600 12.5px Inter, sans-serif";
    actx.fillStyle = "#7C6F60";
    actx.textAlign = a.x < a.cx - 6 ? "right" : a.x > a.cx + 6 ? "left" : "center";
    actx.textBaseline = a.y < a.cy ? "bottom" : "top";
    actx.fillText(a.label, a.x + (a.x - a.cx) * 0.14, a.y + (a.y - a.cy) * 0.16);
  });

  arenaOrbs.forEach((o) => {
    const g = actx.createRadialGradient(o.x, o.y, 1, o.x, o.y, o.r * 2.6);
    g.addColorStop(0, o.color);
    g.addColorStop(0.45, o.color + "77");
    g.addColorStop(1, o.color + "00");
    actx.fillStyle = g;
    actx.beginPath();
    actx.arc(o.x, o.y, o.r * 2.6, 0, Math.PI * 2);
    actx.fill();
    actx.fillStyle = o.color;
    actx.beginPath();
    actx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
    actx.fill();
    actx.strokeStyle = "#fff";
    actx.lineWidth = 2;
    actx.stroke();
  });
}

function startArena() {
  if (arenaRAF) return;
  const step = () => {
    let moving = false;
    arenaOrbs.forEach((o) => {
      o.x += (o.tx - o.x) * 0.12;
      o.y += (o.ty - o.y) * 0.12;
      if (Math.abs(o.tx - o.x) > 0.4 || Math.abs(o.ty - o.y) > 0.4) moving = true;
    });
    drawArena();
    arenaRAF = moving ? requestAnimationFrame(step) : null;
  };
  arenaRAF = requestAnimationFrame(step);
}

function setArenaSingle(probs, model) {
  arenaLayout();
  trailPoints = [];
  const t = orbPos(probs);
  const color = MODEL_COLORS[model] || "#F59E0B";
  const c = arenaAnchors[0];
  arenaOrbs = [{ x: c.cx, y: c.cy, tx: t.x, ty: t.y, color, r: 13, src: probs }];
  startArena();
}

function cancelTrajectory() {
  if (traceTimer) { clearTimeout(traceTimer); traceTimer = null; }
  if (arenaRAF) { cancelAnimationFrame(arenaRAF); arenaRAF = null; }
  trailPoints = [];
}

function playTrajectory(steps, model) {
  arenaLayout();
  cancelTrajectory();
  const positions = steps.map((s) => orbPos(s.probabilities));
  if (!positions.length) return;
  const color = MODEL_COLORS[model] || "#F59E0B";
  arenaOrbs = [{ x: positions[0].x, y: positions[0].y, tx: positions[0].x, ty: positions[0].y, color, r: 13, src: steps[steps.length - 1].probabilities }];
  let i = 0;
  let done = false;
  const stepMs = Math.max(55, Math.min(150, 3000 / positions.length));
  const advance = () => {
    arenaOrbs[0].tx = positions[i].x;
    arenaOrbs[0].ty = positions[i].y;
    i += 1;
    traceTimer = setTimeout(i < positions.length ? advance : () => { done = true; }, stepMs);
  };
  advance();
  const loop = () => {
    const o = arenaOrbs[0];
    o.x += (o.tx - o.x) * 0.2;
    o.y += (o.ty - o.y) * 0.2;
    trailPoints.push({ x: o.x, y: o.y });
    if (trailPoints.length > 280) trailPoints.shift();
    drawArena();
    if (done && Math.abs(o.tx - o.x) < 0.5 && Math.abs(o.ty - o.y) < 0.5) { arenaRAF = null; drawArena(); return; }
    arenaRAF = requestAnimationFrame(loop);
  };
  arenaRAF = requestAnimationFrame(loop);
}

async function updateArena(result) {
  if (!result.probabilities) return;
  $("arena-modal").hidden = false;
  arenaLayout();
  const text = ticketEl.value.trim();
  try {
    const res = await fetch(API_TRACE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: result.model }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.steps && data.steps.length) {
        lastTrace = data.steps;
        lastTraceModel = result.model;
        playTrajectory(data.steps, result.model);
        return;
      }
    }
  } catch (err) { /* fall back to static orb */ }
  setArenaSingle(result.probabilities, result.model);
}

function closeArena() {
  $("arena-modal").hidden = true;
  cancelTrajectory();
}

window.addEventListener("resize", () => {
  if ($("arena-modal").hidden) return;
  arenaLayout();
  arenaOrbs.forEach((o) => { if (o.src) { const t = orbPos(o.src); o.tx = o.x = t.x; o.ty = o.y = t.y; } });
  drawArena();
});

async function route() {
  const rawText = ticketEl.value.trim();
  if (!rawText) { ticketEl.focus(); return; }

  const piiInfo = checkPII(rawText);
  let textToRoute = rawText;
  if (piiInfo.hasPii) {
    ticketEl.value = piiInfo.redactedText;
    textToRoute = piiInfo.redactedText;
    updatePiiWarning();
  }

  routeBtn.disabled = true;
  routeBtn.querySelector(".route-btn-label").textContent = "Se rutează…";

  try {
    const result = await predict(textToRoute, currentModel);
    render(result);
    explainRo(textToRoute, currentModel, result.confidence < CONFIDENCE_THRESHOLD);
    showFaqSuggestions(result.faq_suggestions);
  } catch (err) {
    emptyEl.hidden = false;
    resultEl.hidden = true;
    emptyEl.innerHTML = "Nu am putut contacta backend-ul. Pornește serverul: <code>uvicorn app.backend:app --port 8000</code>.";
  } finally {
    routeBtn.disabled = false;
    routeBtn.querySelector(".route-btn-label").textContent = "Rutează";
  }
}

buildQueues();
buildChips();
wireModelGroup();
routeBtn.addEventListener("click", route);
$("arena-close").addEventListener("click", closeArena);
$("arena-replay").addEventListener("click", () => { if (lastTrace) playTrajectory(lastTrace, lastTraceModel); });
$("arena-backdrop").addEventListener("click", closeArena);
ticketEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") route();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("arena-modal").hidden) closeArena();
});

const piiWarningEl = $("pii-warning");
const piiMessageEl = $("pii-message");
const faqSuggestionsEl = $("faq-suggestions");
const faqListEl        = $("faq-list");
const faqModalEl       = $("faq-modal");
const modalTitleEl     = $("modal-title");
const modalBodyEl      = $("modal-body");
const modalCloseBtn    = $("modal-close-btn");

function checkPII(text) {
  const detectedTypes = [];
  let redacted = text;

  let hasCNP = false;
  redacted = redacted.replace(/\b[1-8]\d{12}\b/g, () => {
    hasCNP = true;
    return '[CNP]';
  });
  if (hasCNP) detectedTypes.push("CNP");

  let hasEmail = false;
  redacted = redacted.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, () => {
    hasEmail = true;
    return '[EMAIL]';
  });
  if (hasEmail) detectedTypes.push("adresă email");

  let hasPhone = false;
  redacted = redacted.replace(/(^|[\s,;.:])((?:\+?40|0)[- ]?[237]\d{2}[- ]?\d{3}[- ]?\d{3})\b/g, (match, prefix, phone) => {
    hasPhone = true;
    const clean = phone.replace(/[\s-]/g, '');
    const last3 = clean.slice(-3);
    return `${prefix}[TELEFON: *******${last3}]`;
  });
  if (hasPhone) detectedTypes.push("număr telefon");

  let hasCard = false;
  redacted = redacted.replace(/\b((?:\d[ -]*?){13,16})\b/g, (match, card) => {
    const clean = card.replace(/[\s-]/g, '');
    if (clean.length >= 13 && clean.length <= 16) {
      hasCard = true;
      const last4 = clean.slice(-4);
      return `[CARD: ****${last4}]`;
    }
    return match;
  });
  if (hasCard) detectedTypes.push("număr card");

  return {
    hasPii: detectedTypes.length > 0,
    redactedText: redacted,
    detectedTypes: detectedTypes
  };
}

function updatePiiWarning() {
  const text = ticketEl.value;
  const piiInfo = checkPII(text);
  if (piiInfo.hasPii) {
    piiMessageEl.textContent = `Am detectat date sensibile (${piiInfo.detectedTypes.join(', ')}). Le vom securiza la trimitere.`;
    piiWarningEl.hidden = false;
  } else {
    piiWarningEl.hidden = true;
  }
}

function showFaqSuggestions(suggestions) {
  faqListEl.innerHTML = "";
  
  if (!suggestions || suggestions.length === 0) {
    faqSuggestionsEl.hidden = true;
    return;
  }
  
  suggestions.forEach((faq) => {
    const card = document.createElement("button");
    card.className = "faq-card";
    card.type = "button";
    card.textContent = faq.q;
    card.addEventListener("click", () => {
      openFaqModal(faq.q, faq.a);
    });
    faqListEl.appendChild(card);
  });
  
  faqSuggestionsEl.hidden = false;
}

function openFaqModal(title, body) {
  modalTitleEl.textContent = title;
  modalBodyEl.innerHTML = body;
  faqModalEl.hidden = false;
}

function closeFaqModal() {
  faqModalEl.hidden = true;
}

let debounceTimer;
function debouncePredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    const text = ticketEl.value.trim();
    if (text.length < 15) {
      faqSuggestionsEl.hidden = true;
      return;
    }
    
    try {
      const result = await predict(text, currentModel);
      showFaqSuggestions(result.faq_suggestions);
    } catch (e) {
    }
  }, 500);
}
ticketEl.addEventListener("input", () => {
  updatePiiWarning();
  debouncePredict();
});
modalCloseBtn.addEventListener("click", closeFaqModal);
faqModalEl.addEventListener("click", (e) => {
  if (e.target === faqModalEl) closeFaqModal();
});

const API_CONF = API_URL.replace(/\/predict$/, "/confusion/");
let confModel = "svm";
let confData = null;

function openConfusion() {
  $("conf-modal").hidden = false;
  loadConfusion(confModel);
}
function closeConfusion() { $("conf-modal").hidden = true; }

async function loadConfusion(model) {
  confModel = model;
  document.querySelectorAll("#conf-models .seg").forEach((b) => b.setAttribute("aria-checked", b.dataset.cm === model ? "true" : "false"));
  const grid = $("conf-grid"), sub = $("conf-sub"), ex = $("conf-examples");
  ex.hidden = true;
  grid.innerHTML = "";
  sub.textContent = "Se calculează…";
  try {
    const res = await fetch(API_CONF + model);
    const data = await res.json();
    if (!data.available) { sub.textContent = "Indisponibil — necesită data/test.csv local."; return; }
    confData = data;
    sub.textContent = `Acuratețe ${(data.accuracy * 100).toFixed(1)}% pe ${data.total} tichete de test. Apasă o celulă roșie (din afara diagonalei).`;
    renderConfGrid(data);
  } catch (err) {
    sub.textContent = "Nu am putut contacta backend-ul.";
  }
}

function renderConfGrid(data) {
  const grid = $("conf-grid");
  const C = data.classes, M = data.matrix;
  let maxOff = 1, maxDiag = 1;
  C.forEach((_, i) => C.forEach((_, j) => {
    if (i === j) maxDiag = Math.max(maxDiag, M[i][j]);
    else maxOff = Math.max(maxOff, M[i][j]);
  }));
  grid.style.gridTemplateColumns = `minmax(78px,auto) repeat(${C.length}, 1fr)`;
  grid.innerHTML = "";
  const corner = document.createElement("div");
  corner.className = "conf-corner";
  corner.textContent = "real ↓ / prezis →";
  grid.appendChild(corner);
  C.forEach((c) => {
    const h = document.createElement("div");
    h.className = "conf-colh";
    h.textContent = ARENA_SHORT[c] || c;
    grid.appendChild(h);
  });
  C.forEach((rc, i) => {
    const rh = document.createElement("div");
    rh.className = "conf-rowh";
    rh.textContent = ARENA_SHORT[rc] || rc;
    grid.appendChild(rh);
    C.forEach((cc, j) => {
      const cell = document.createElement("div");
      cell.className = "conf-cell" + (i === j ? " diag" : "");
      cell.textContent = M[i][j];
      if (i === j) {
        const a = Math.min(0.8, (M[i][j] / maxDiag) * 0.7 + 0.12);
        cell.style.background = `rgba(31,169,113,${a})`;
      } else if (M[i][j] > 0) {
        const a = (M[i][j] / maxOff) * 0.72 + 0.06;
        cell.style.background = `rgba(229,72,77,${a})`;
        cell.classList.add("clickable");
        cell.addEventListener("click", () => showConfExamples(i, j));
      }
      grid.appendChild(cell);
    });
  });
}

function showConfExamples(i, j) {
  if (!confData) return;
  const ex = $("conf-examples");
  const list = confData.examples[`${i},${j}`] || [];
  const trueRo = TEAMS[confData.classes[i]] || confData.classes[i];
  const predRo = TEAMS[confData.classes[j]] || confData.classes[j];
  ex.innerHTML = "";
  const title = document.createElement("div");
  title.className = "conf-ex-title";
  title.innerHTML = `Tichete reale din <b>${trueRo}</b> rutate greșit la <b>${predRo}</b>:`;
  ex.appendChild(title);
  if (!list.length) {
    const d = document.createElement("div"); d.className = "conf-ex"; d.textContent = "(fără exemple)"; ex.appendChild(d);
  } else {
    list.forEach((t) => { const d = document.createElement("div"); d.className = "conf-ex"; d.textContent = t; ex.appendChild(d); });
  }
  const note = document.createElement("div");
  note.className = "conf-note";
  note.textContent = "Date de test (text original, în engleză).";
  ex.appendChild(note);
  ex.hidden = false;
}

$("confusion-btn").addEventListener("click", openConfusion);
$("conf-close").addEventListener("click", closeConfusion);
$("conf-backdrop").addEventListener("click", closeConfusion);
document.querySelectorAll("#conf-models .seg").forEach((b) => b.addEventListener("click", () => loadConfusion(b.dataset.cm)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("conf-modal").hidden) closeConfusion(); });
