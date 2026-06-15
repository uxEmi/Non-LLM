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

const RING_CIRC = 2 * Math.PI * 52;   // r=52 in the SVG
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

  // Reveal + replay the bento entrance animation
  emptyEl.hidden = true;
  resultEl.hidden = false;
  resultEl.querySelectorAll(".tile").forEach((t) => {
    t.style.animation = "none"; void t.offsetWidth; t.style.animation = "";
  });

  // Destination highlight
  Object.values(queueEls).forEach((el) => el.classList.remove("active", "dim"));
  Object.values(queueEls).forEach((el) => { if (el !== targetEl) el.classList.add("dim"); });
  targetEl.classList.add("active");

  // Team hero
  const teamRo = review ? REVIEW_LABEL_RO : TEAMS[result.predicted_team];
  $("r-team").textContent = teamRo;
  $("r-team").style.color = review ? "var(--alert)" : "var(--text)";
  $("r-sub").textContent = review ? "sub pragul de încredere" : `rutat de ${result.model.toUpperCase()}`;
  const dot = $("r-dot");
  dot.style.background = review ? "var(--alert)" : "var(--accent)";
  dot.style.boxShadow = review ? "0 0 0 4px var(--alert-soft)" : "0 0 0 4px var(--accent-soft)";
  setBadge(review ? "verificare" : "rutat", review ? "is-review" : "is-routed");

  // Confidence gauge
  const confPct = Math.round(result.confidence * 100);
  countUp($("r-conf"), confPct);
  ringEl.style.stroke = review ? "var(--alert)" : "url(#ringGrad)";
  requestAnimationFrame(() => {
    ringEl.style.strokeDashoffset = RING_CIRC * (1 - result.confidence);
  });

  // Stats
  countUp($("r-lat"), result.latency_ms);
  $("r-model").textContent = result.model.toUpperCase();
}

async function route() {
  const text = ticketEl.value.trim();
  if (!text) { ticketEl.focus(); return; }

  routeBtn.disabled = true;
  routeBtn.querySelector(".route-btn-label").textContent = "Se rutează…";

  try {
    const result = await predict(text, currentModel);
    render(result);
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
ticketEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") route();
});
