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

  // 1. CNP (13 digits starting with 1-8)
  let hasCNP = false;
  redacted = redacted.replace(/\b[1-8]\d{12}\b/g, () => {
    hasCNP = true;
    return '[CNP]';
  });
  if (hasCNP) detectedTypes.push("CNP");

  // 2. Email
  let hasEmail = false;
  redacted = redacted.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, () => {
    hasEmail = true;
    return '[EMAIL]';
  });
  if (hasEmail) detectedTypes.push("adresă email");

  // 3. Phone (Romanian landline/mobile formats, keeping 3 digits)
  let hasPhone = false;
  redacted = redacted.replace(/(^|[\s,;.:])((?:\+?40|0)[- ]?[237]\d{2}[- ]?\d{3}[- ]?\d{3})\b/g, (match, prefix, phone) => {
    hasPhone = true;
    const clean = phone.replace(/[\s-]/g, '');
    const last3 = clean.slice(-3);
    return `${prefix}[TELEFON: *******${last3}]`;
  });
  if (hasPhone) detectedTypes.push("număr telefon");

  // 4. Credit Card (13-16 digits, with optional spaces/dashes)
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

const FAQ_DATABASE = {
  "Loans": [
    {
      q: "Cum funcționează perioada de grație pentru refinanțare?",
      a: "<p>Perioada de grație îți permite să amâni plata principalului sau a întregii rate pentru o perioadă determinată de timp (de obicei între 1 și 6 luni).</p><p>În această perioadă, dobânda se poate acumula în continuare în funcție de contract. Vă recomandăm să contactați departamentul de credite pentru o simulare detaliată a refinanțării.</p>"
    },
    {
      q: "Ce documente sunt necesare pentru creditul de nevoi personale?",
      a: "<p>Pentru a aplica pentru un credit de nevoi personale, aveți nevoie de:</p><ul><li>Actul de identitate (buletinul)</li><li>Acordul de interogare ANAF (pentru verificarea automată a veniturilor)</li><li>În unele cazuri, adeverință de salariu sau extras de cont pe ultimele 3 luni (dacă nu aveți venituri raportate la ANAF)</li></ul>"
    },
    {
      q: "Cum pot solicita o amânare a ratelor (moratoriu)?",
      a: "<p>Dacă întâmpinați dificultăți financiare temporare, puteți depune o cerere de restructurare sau amânare a ratelor.</p><p>Cererea trebuie să includă documente justificative (cum ar fi decizia de șomaj, concediu medical prelungit sau alte dovezi de scădere a veniturilor). Cererea se depune online în secțiunea de Suport sau în orice sucursală.</p>"
    }
  ],
  "Credit Reporting": [
    {
      q: "Cum pot contesta o înregistrare negativă la Biroul de Credit?",
      a: "<p>Dacă considerați că datele raportate la Biroul de Credit sunt eronate (de exemplu, o restanță care a fost plătită la timp), aveți dreptul să contestați înregistrarea.</p><p>Contestația se poate depune direct către banca noastră (prin intermediul acestui portal de suport) sau direct către Biroul de Credit. Banca va analiza istoricul plăților și va efectua rectificarea în maxim 30 de zile dacă eroarea se confirmă.</p>"
    },
    {
      q: "Cât timp rămân datele mele înscrise în Biroul de Credit?",
      a: "<p>Conform reglementărilor în vigoare, datele negative (informațiile despre restanțe) sunt păstrate în baza de date a Biroului de Credit timp de <b>4 ani</b> de la data plății ultimei restanțe sau de la data ultimei actualizări transmise.</p>"
    },
    {
      q: "Cum pot obține raportul meu de credit gratuit?",
      a: "<p>Fiecare cetățean are dreptul la o interogare gratuită a propriei situații din Biroul de Credit o dată pe lună.</p><p>Puteți descărca raportul instant creându-vă un cont direct pe site-ul oficial al Biroului de Credit (www.biroul-de-credit.ro), secțiunea Persoane Fizice.</p>"
    }
  ],
  "Bank Accounts and Services": [
    {
      q: "Cum pot închide un cont bancar deschis online?",
      a: "<p>Închiderea unui cont curent se poate solicita direct din aplicația de Mobile/Internet Banking, accesând setările contului și selectând 'Închidere cont'.</p><p><b>Notă:</b> Pentru a putea închide contul, soldul acestuia trebuie să fie exact 0 (să nu fie pe minus și să nu aibă fonduri rămase) și să nu aibă atașate depozite active sau servicii conexe nereziliate.</p>"
    },
    {
      q: "Care sunt comisioanele aplicabile pentru retrageri externe?",
      a: "<p>Retragerile de numerar de la ATM-urile din Uniunea Europeană în euro sunt comisionate la fel ca retragerile naționale (comision 0% pentru majoritatea pachetelor de cont premium).</p><p>Pentru retragerile în afara UE sau în alte valute, se aplică un comision de 1.5% din suma retrasă + o taxă fixă de 10 RON per tranzacție, plus eventuale comisioane de conversie valutară.</p>"
    },
    {
      q: "Cum activez plățile contactless pe noul meu card?",
      a: "<p>Pentru a activa funcția contactless pe un card nou emis, prima tranzacție trebuie realizată fizic la un terminal POS sau la un ATM introducând cardul în fantă și introducând codul PIN.</p><p>După această primă tranzacție de confirmare a codului PIN, plățile contactless (inclusiv Apple Pay sau Google Pay) vor fi complet funcționale.</p>"
    }
  ],
  "Debt Collection": [
    {
      q: "Ce drepturi am când sunt contactat de o agenție de recuperare?",
      a: "<p>Agențiile de recuperare a debitelor trebuie să respecte legislația privind protecția consumatorilor. Acestea nu au voie:</p><ul><li>Să vă contacteze în afara orelor 08:00 - 20:00</li><li>Să vă hărțuiască sau să folosească un limbaj amenințător</li><li>Să contacteze vecinii, rudele sau angajatorul dumneavoastră pentru a divulga detalii despre datorie</li></ul><p>Dacă sesizați abuzuri, puteți înregistra o reclamație oficială.</p>"
    },
    {
      q: "Cum pot stabili un plan de eșalonare a datoriei?",
      a: "<p>Dacă nu puteți achita datoria integral, echipa noastră de colectare vă poate ajuta să stabiliți un angajament de plată eșalonat (rate lunare accesibile).</p><p>Trimiteți-ne o solicitare prin acest tichet menționând suma lunară pe care o puteți plăti realist, iar un consultant vă va contacta pentru a semna acordul de eșalonare.</p>"
    },
    {
      q: "Ce fac dacă primesc o notificare de executare silită greșită?",
      a: "<p>Dacă ați primit o notificare pentru o datorie pe care ați achitat-o deja sau care aparține altei persoane, vă rugăm să atașați o copie a notificării și dovada plății în acest tichet.</p><p>Echipa noastră va sista imediat procedurile de colectare și va clarifica situația cu executorul judecătoresc în cel mai scurt timp.</p>"
    }
  ],
  "Credit Card Services": [
    {
      q: "Cum disput o tranzacție neautorizată (chargeback)?",
      a: "<p>Dacă observați pe extrasul de cont tranzacții pe care nu le recunoașteți sau bunuri/servicii plătite care nu v-au fost livrate, aveți dreptul de a iniția o dispută comercială (procedura de chargeback).</p><p>Pentru a începe, specificați data tranzacției, suma și numele comerciantului în reclamația dumneavoastră. Vă recomandăm să blocați cardul imediat din aplicația mobilă pentru a preveni alte tranzacții.</p>"
    },
    {
      q: "Cum pot modifica limita de credit pentru cardul meu?",
      a: "<p>Modificarea (majorarea sau micșorarea) limitei de credit atașate cardului se poate solicita direct din aplicația de Internet Banking sau prin depunerea unei solicitări scrise.</p><p>Pentru majorarea limitei, banca va efectua o nouă analiză a veniturilor dumneavoastră raportate la ANAF.</p>"
    },
    {
      q: "Ce trebuie să fac dacă mi-a fost blocat sau furat cardul?",
      a: "<p>Dacă ați pierdut cardul sau bănuiți că a fost furat, trebuie să îl blocați imediat. O puteți face instant din Mobile/Internet Banking la secțiunea setări card.</p><p>Dacă nu aceți acces la aplicație, sunați de urgență la numărul nostru de asistență carduri: <b>+40 21 304 8100</b> (disponibil non-stop) pentru blocarea cardului și reemiterea unuia nou.</p>"
    }
  ]
};

function showFaqSuggestions(teamKey) {
  const faqs = FAQ_DATABASE[teamKey] || [];
  faqListEl.innerHTML = "";
  
  if (faqs.length === 0) {
    faqSuggestionsEl.hidden = true;
    return;
  }
  
  faqs.forEach((faq) => {
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
      showFaqSuggestions(result.predicted_team);
    } catch (e) {
      // Quietly ignore background prediction errors (e.g. offline server)
    }
  }, 500);
}

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
    c.addEventListener("click", () => {
      ticketEl.value = txt;
      ticketEl.focus();
      updatePiiWarning();
      debouncePredict();
    });
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

  // Keywords
  const kwContainer = $("r-keywords");
  kwContainer.innerHTML = "";
  if (result.keywords && result.keywords.length > 0) {
    result.keywords.forEach((kw) => {
      const chip = document.createElement("span");
      chip.className = "keyword-chip";
      const scoreStr = kw.score >= 0.001 ? kw.score.toFixed(3) : kw.score.toFixed(5);
      chip.innerHTML = `${kw.word} <span class="keyword-score">${scoreStr}</span>`;
      kwContainer.appendChild(chip);
    });
  } else {
    const emptySpan = document.createElement("span");
    emptySpan.style.color = "var(--faint)";
    emptySpan.style.fontSize = "13px";
    emptySpan.textContent = "Indisponibil pentru acest model";
    kwContainer.appendChild(emptySpan);
  }
}

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
ticketEl.addEventListener("input", () => {
  updatePiiWarning();
  debouncePredict();
});
ticketEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") route();
});

modalCloseBtn.addEventListener("click", closeFaqModal);
faqModalEl.addEventListener("click", (e) => {
  if (e.target === faqModalEl) {
    closeFaqModal();
  }
});
