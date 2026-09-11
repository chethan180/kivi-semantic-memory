/* ==========================================================================
   Kivi — interface

   No framework, no build step, no CDN. Small render functions over fetch calls.
   The reviewer runs one command and the product works, offline included.
   ========================================================================== */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const api = async (path, options) => {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
};

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const when = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(+d)) return iso;
  return d.toLocaleString(undefined, {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
  });
};

const day = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(+d) ? String(iso).slice(0, 10)
    : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
};

/* ------------------------------------------------------------ navigation -- */

let currentView = 'ask';

function show(view) {
  currentView = view;
  $$('.view').forEach((el) => { el.hidden = el.id !== `view-${view}`; });
  $$('.nav').forEach((b) => b.setAttribute('aria-current', String(b.dataset.view === view)));
  if (view === 'dictation') loadRecords(true);
  if (view === 'knows') loadKnows();
  if (view === 'inspect') loadInspect();
}

$$('.nav').forEach((b) => b.addEventListener('click', () => show(b.dataset.view)));

/* ---------------------------------------------------------------- drawer -- */

const drawer = $('#drawer');

function openDrawer(title, html) {
  $('#drawer-title').textContent = title;
  $('#drawer-body').innerHTML = html;
  drawer.hidden = false;
}
function closeDrawer() { drawer.hidden = true; }

drawer.addEventListener('click', (e) => { if (e.target.dataset.close !== undefined) closeDrawer(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

/* Highlight the exact words a memory was drawn from. Provenance you can see
   beats provenance you are promised. */
function highlight(text, span) {
  if (!span) return esc(text);
  const norm = (s) => s.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
  const target = norm(span);
  const words = text.split(/(\s+)/);
  for (let start = 0; start < words.length; start++) {
    let acc = '';
    for (let end = start; end < words.length; end++) {
      acc += words[end];
      if (norm(acc) === target) {
        return esc(words.slice(0, start).join('')) +
               `<mark class="evidence">${esc(acc)}</mark>` +
               esc(words.slice(end + 1).join(''));
      }
      if (norm(acc).length > target.length + 12) break;
    }
  }
  return esc(text);
}

async function showRecord(recordId, evidenceSpan) {
  openDrawer(recordId, '<p class="empty">Loading…</p>');
  try {
    const { record, learned, ignored } = await api(`/records/${recordId}`);
    const said = record.app === 'hey_kivi';
    openDrawer(said ? 'You told Kivi' : recordId, `
      <div class="meta-line">
        ${esc(when(record.ts))}
        ${said
          ? '&nbsp;·&nbsp; <span class="said">you said this to Kivi directly</span>'
          : `&nbsp;·&nbsp; ${esc(record.app || '')}${
              record.destination ? `&nbsp;→&nbsp; ${esc(record.destination)}` : ''}${
              record.speech_act ? `&nbsp;·&nbsp; ${esc(record.speech_act.replace(/_/g, ' '))}` : ''}`}
      </div>
      <blockquote class="quote${said ? ' said-quote' : ''}">${highlight(record.formatted_text, evidenceSpan)}</blockquote>

      ${said ? '' : `<h4 class="why-h">As spoken</h4>
      <p class="record-raw">${esc(record.raw_asr)}</p>`}

      ${learned.length ? `
        <h4 class="why-h" style="margin-top:20px">What Kivi took from this</h4>
        ${learned.map((f) => `
          <div class="card" style="margin-bottom:6px">
            <p class="card-title">${esc(f.statement)}</p>
            <p class="card-sub">${esc(f.fact_id)} · ${esc(f.status)}</p>
          </div>`).join('')}` : ''}

      ${ignored.length ? `
        <h4 class="why-h" style="margin-top:20px">What Kivi deliberately ignored</h4>
        <table><tbody>${ignored.map((d) => `
          <tr>
            <td><span class="reason">${esc(d.reason_code)}</span></td>
            <td>${esc(d.reason_detail || '')}</td>
          </tr>`).join('')}</tbody></table>` : ''}
    `);
  } catch (err) {
    openDrawer(recordId, `<p class="empty">Couldn't load that record: ${esc(err.message)}</p>`);
  }
}

async function showFact(factId) {
  openDrawer(factId, '<p class="empty">Loading…</p>');
  try {
    const { fact, revisions, sources } = await api(`/memory/facts/${factId}`);
    openDrawer(fact.statement, `
      <div class="meta-line">${esc(fact.claim_key)} · ${esc(fact.status)}</div>

      <h4 class="why-h">How this changed</h4>
      <ul class="timeline">
        ${revisions.map((r) => `
          <li class="${r.status === 'active' ? 'current' : ''}">
            <strong>${esc(r.value_display)}</strong>
            ${r.status === 'active' ? ' — current' : r.valid_to ? ` — until ${esc(day(r.valid_to))}` : ''}
            <br>from ${esc(day(r.valid_from))}, said in
            <button class="cite" data-record="${esc(r.source_record_id)}"
                    data-span="${esc(r.evidence_span)}">${esc(r.source_record_id)}</button>
          </li>`).join('')}
      </ul>

      <h4 class="why-h" style="margin-top:20px">Everything behind this</h4>
      ${sources.map((s) => `
        <div class="card" style="margin-bottom:6px">
          <div class="record-head">
            <span>${esc(when(s.ts))}</span>
            <span class="record-app">${esc(s.app || '')}</span>
            <span>${esc(s.contribution)}</span>
          </div>
          <p style="margin:0">${highlight(s.formatted_text, s.evidence_span)}</p>
        </div>`).join('')}
    `);
  } catch (err) {
    openDrawer(factId, `<p class="empty">Couldn't load that: ${esc(err.message)}</p>`);
  }
}

document.addEventListener('click', (e) => {
  const cite = e.target.closest('[data-record]');
  if (cite) { showRecord(cite.dataset.record, cite.dataset.span); return; }
  const factBtn = e.target.closest('[data-fact]');
  if (factBtn) showFact(factBtn.dataset.fact);
});

/* ================================================================ ask ==== */

const SUGGESTIONS = [
  "Who's running the pricing redesign now?",
  'When does mobile onboarding ship?',
  'Why did the pricing redesign slip?',
  'Who is my manager?',
  'When is the Acme renewal?',
  'What did Acme say about the security review?',
];

function renderSuggestions() {
  $('#suggestions').innerHTML = SUGGESTIONS
    .map((q) => `<button class="suggestion">${esc(q)}</button>`).join('');
  $$('.suggestion').forEach((b) => b.addEventListener('click', () => {
    $('#ask-input').value = b.textContent;
    $('#ask-form').requestSubmit();
  }));
}

/* Turn [rec_0123] into something you can open. A citation you cannot follow is
   decoration; one you can follow is evidence. */
function linkCitations(text, sources) {
  const spans = Object.fromEntries(
    (sources || []).map((s) => [s.ref, s.kind === 'record' ? '' : ''])
  );
  // say_* ids are things the person told Kivi in conversation. They are cited
  // exactly like dictations — the point of storing a correction as a record is
  // that "because you told me" is checkable — but labelled so the person can
  // tell at a glance which kind of evidence they are looking at.
  return esc(text).replace(/\[([a-zA-Z]{1,4}_?[a-z0-9]+)\]/g, (_, ref) => {
    const source = (sources || []).find((s) => s.ref === ref);
    const isSaid = ref.startsWith('say_');
    const attr = source && source.kind === 'fact'
      ? `data-fact="${esc(ref)}"`
      : `data-record="${esc(ref)}"`;
    const label = isSaid ? 'you told Kivi' : ref;
    return `<button class="cite${isSaid ? ' cite-said' : ''}" ${attr}>${esc(label)}</button>`;
  });
}

function renderAnswer(turn, data) {
  const kindClass = data.abstained ? 'abstained'
    : (data.used_facts || []).some((f) => f.status === 'disputed') ? 'disputed' : '';

  const paragraphs = data.answer.split(/\n{2,}/)
    .map((p) => `<p>${linkCitations(p, data.sources)}</p>`).join('');

  const pill = data.abstained
    ? '<span class="pill quiet">Not in your history</span>'
    : `<span class="pill grounded">${data.citations.length} source${data.citations.length === 1 ? '' : 's'}</span>`;

  turn.querySelector('.answer').outerHTML = `
    <div class="answer ${kindClass}">
      <div class="answer-text">${paragraphs}</div>
      <div class="answer-meta">
        ${pill}
        ${data.dropped_citations && data.dropped_citations.length
          ? `<span class="pill warn">${data.dropped_citations.length} unverifiable reference${data.dropped_citations.length === 1 ? '' : 's'} removed</span>`
          : ''}
        <span class="spacer"></span>
        <span>${data.latency_ms} ms</span>
        <button class="linky" data-why>Why this answer</button>
      </div>
      <div class="why" hidden></div>
    </div>`;

  const answerEl = turn.querySelector('.answer');
  answerEl.querySelector('[data-why]').addEventListener('click', async (e) => {
    const why = answerEl.querySelector('.why');
    why.hidden = !why.hidden;
    e.target.textContent = why.hidden ? 'Why this answer' : 'Hide';
    if (!why.dataset.loaded) { why.dataset.loaded = '1'; why.innerHTML = await explain(data); }
  });
}

async function explain(data) {
  let trace = null;
  try { trace = await api(`/inspect/traces/${data.trace_id}`); } catch {}

  const facts = (data.used_facts || []).map((f) =>
    `<li>${esc(f.statement)}${f.detail ? ` <span style="color:var(--ink-faint)">${esc(f.detail)}</span>` : ''}</li>`).join('');

  const constraints = trace?.constraints?.explanations || [];
  const timings = trace?.latency || {};

  return `
    ${data.abstained ? `
      <h4>Why Kivi didn't answer</h4>
      <p style="margin:0 0 11px">${esc(data.abstain_reason)}</p>` : ''}

    ${facts ? `<h4>What it knew</h4><ul>${facts}</ul>` : ''}

    ${constraints.length ? `
      <h4>How it read your question</h4>
      <ul>${constraints.map((c) => `<li>${esc(c)}</li>`).join('')}</ul>` : ''}

    ${data.sources?.length ? `
      <h4>What it looked at</h4>
      <ul>${data.sources.map((s) => `
        <li><code>${esc(s.ref)}</code> — ${esc((s.text || '').slice(0, 90))}…</li>`).join('')}</ul>` : ''}

    ${data.dropped_citations?.length ? `
      <h4>Removed before you saw it</h4>
      <ul>${data.dropped_citations.map((c) =>
        `<li><code>${esc(c)}</code> — cited but not in what was retrieved, so it was stripped</li>`).join('')}</ul>` : ''}

    <h4>Cost of this answer</h4>
    <ul>
      <li>retrieval ${Object.entries(timings).filter(([k]) => k !== 'total')
        .map(([k, v]) => `${esc(k)} ${v}ms`).join(', ') || '—'}</li>
      <li>${data.cost_usd ? `$${data.cost_usd.toFixed(6)}` : 'no model call'} · trace <code>${esc(data.trace_id)}</code></li>
    </ul>`;
}

/* One conversation at a time. The id is held in memory only, so a reload starts
   fresh — a conversation is a working context, not a record you accumulate.
   Anything worth keeping across conversations has become a fact by then, and
   facts are read from memory by every session. */
let sessionId = null;

function resetConversation() {
  sessionId = null;
  $('#conversation').innerHTML = '';
  $('#ask-input').focus();
}

$('#new-chat')?.addEventListener('click', resetConversation);

$('#ask-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('#ask-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  $('#ask-button').disabled = true;

  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `
    <div class="question">${esc(question)}</div>
    <div class="answer"><span class="thinking">Looking through your dictations</span></div>`;
  $('#conversation').prepend(turn);

  try {
    const data = await api('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    // The server settles the id on the first turn; every later question in this
    // conversation carries it, and no other conversation ever sees these turns.
    if (data.session_id) sessionId = data.session_id;
    renderAnswer(turn, data);
    $('#new-chat').hidden = false;
  } catch (err) {
    turn.querySelector('.answer').outerHTML =
      `<div class="answer abstained"><div class="answer-text"><p>Something went wrong: ${esc(err.message)}</p></div></div>`;
  } finally {
    $('#ask-button').disabled = false;
    input.focus();
  }
});

/* ========================================================== dictation ==== */

let recOffset = 0;
let recApp = null;

async function loadRecords(reset) {
  if (reset) { recOffset = 0; $('#record-list').innerHTML = ''; }
  const q = $('#rec-search').value.trim();
  const params = new URLSearchParams({ limit: '40', offset: String(recOffset) });
  if (recApp) params.set('app_name', recApp);
  if (q) params.set('q', q);

  const data = await api(`/records?${params}`);

  if (reset && !$('#app-filters').dataset.built) {
    $('#app-filters').dataset.built = '1';
    $('#app-filters').innerHTML = data.apps
      .map((a) => `<button class="chip" data-app="${esc(a)}" aria-pressed="false">${esc(a)}</button>`).join('');
    $$('#app-filters .chip').forEach((c) => c.addEventListener('click', () => {
      recApp = recApp === c.dataset.app ? null : c.dataset.app;
      $$('#app-filters .chip').forEach((x) =>
        x.setAttribute('aria-pressed', String(x.dataset.app === recApp)));
      loadRecords(true);
    }));
  }

  const showRaw = $('#show-raw').checked;
  const html = data.records.map((r) => `
    <article class="record" data-record="${esc(r.record_id)}">
      <div class="record-head">
        <span>${esc(when(r.ts))}</span>
        <span class="record-app">${esc(r.app || '')}</span>
        <span>${esc(r.destination || '')}</span>
      </div>
      <p class="record-text">${esc(r.formatted_text)}</p>
      ${showRaw ? `<p class="record-raw">${esc(r.raw_asr)}</p>` : ''}
    </article>`).join('');

  $('#record-list').insertAdjacentHTML('beforeend', html);
  recOffset += data.records.length;
  $('#load-more').hidden = data.records.length < 40;

  if (!$('#record-list').children.length) {
    $('#record-list').innerHTML = '<p class="empty">No dictations match that.</p>';
  }
}

$('#rec-search').addEventListener('input', debounce(() => loadRecords(true), 250));
$('#show-raw').addEventListener('change', () => loadRecords(true));
$('#load-more').addEventListener('click', () => loadRecords(false));

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/* ============================================================== knows ==== */

async function loadKnows() {
  const body = $('#knows-body');
  body.innerHTML = '<p class="empty">Loading…</p>';
  const data = await api('/memory');

  const decisions = data.needs_decision.map((f) => `
    <div class="card decision" data-fact-card="${esc(f.fact_id)}">
      <p class="card-title">You've said two different things</p>
      <p class="card-sub">${esc(f.statement)}</p>
      <div class="actions" data-resolve="${esc(f.fact_id)}">
        <button class="action" data-fact="${esc(f.fact_id)}">See both</button>
      </div>
    </div>`).join('');

  const factCard = (f) => `
    <div class="card">
      <p class="card-title">${esc(f.statement)}</p>
      <p class="card-sub">
        Learned from ${f.source_count} dictation${f.source_count === 1 ? '' : 's'}${
          f.revisions > 1 ? ` · changed ${f.revisions - 1} time${f.revisions === 2 ? '' : 's'}` : ''
        }${f.last_changed ? ` · last ${esc(day(f.last_changed))}` : ''}
      </p>
      <div class="actions">
        <button class="action" data-fact="${esc(f.fact_id)}">Where from</button>
        <button class="action danger" data-forget="${esc(f.fact_id)}">Forget this</button>
      </div>
    </div>`;

  const byAnchor = {};
  data.facts.filter((f) => f.status === 'active').forEach((f) => {
    const key = f.anchor_name || 'Other';
    (byAnchor[key] ||= []).push(f);
  });

  const prefCard = (p) => `
    <div class="card">
      <p class="card-title">${esc(p.statement)}</p>
      <p class="card-sub">
        <span class="pref-state" style="color:${p.state === 'active' ? 'var(--accent)' : 'var(--ink-faint)'}">
          ${esc(p.state)}</span>
        · ${p.origin === 'explicit' ? 'you told Kivi this' : `noticed ${p.evidence_count} time${p.evidence_count === 1 ? '' : 's'}`}
        ${p.state === 'candidate' ? ' · not being applied yet' : ''}
      </p>
      <div class="actions">
        <button class="action" data-toggle-pref="${esc(p.pref_id)}">
          ${p.state === 'active' ? 'Stop doing this' : 'Start doing this'}
        </button>
      </div>
    </div>`;

  body.innerHTML = `
    ${decisions ? `<section class="section"><h2>Needs your decision</h2>${decisions}</section>` : ''}

    <section class="section">
      <h2>How you like things done</h2>
      ${data.preferences.length ? data.preferences.map(prefCard).join('')
        : '<p class="empty">Nothing learned yet.</p>'}
    </section>

    ${Object.entries(byAnchor).map(([name, facts]) => `
      <section class="section">
        <h2>${esc(name)}</h2>
        ${facts.map(factCard).join('')}
      </section>`).join('') || '<p class="empty">Kivi hasn\'t learned anything yet.</p>'}
  `;

  $$('[data-forget]').forEach((b) => b.addEventListener('click', async () => {
    if (!confirm('Kivi will stop believing this. Your dictation is untouched.')) return;
    await api(`/memory/facts/${b.dataset.forget}`, { method: 'DELETE' });
    loadKnows();
  }));

  $$('[data-toggle-pref]').forEach((b) => b.addEventListener('click', async () => {
    await api(`/memory/preferences/${b.dataset.togglePref}/toggle`, { method: 'POST' });
    loadKnows();
  }));

  $('#decision-badge').hidden = !data.needs_decision.length;
  $('#decision-badge').textContent = data.needs_decision.length;
}

/* ============================================================ inspect ==== */

async function loadInspect() {
  const body = $('#inspect-body');
  body.innerHTML = '<p class="empty">Loading…</p>';
  const [summary, decisions, traces] = await Promise.all([
    api('/inspect/summary'),
    api('/inspect/decisions?outcome=rejected&limit=60'),
    api('/inspect/traces?limit=15'),
  ]);

  const c = summary.counts;
  const stat = (k, n) => `<div class="stat"><div class="n">${n ?? 0}</div><div class="k">${k}</div></div>`;

  body.innerHTML = `
    <section class="section">
      <h2>Memory</h2>
      <div class="grid">
        ${stat('dictations', c.records)}
        ${stat('facts', c.facts)}
        ${stat('revisions', c.fact_revisions)}
        ${stat('entities', c.entities)}
        ${stat('preferences', c.preferences)}
        ${stat('edges', c.edges)}
        ${stat('database KiB', summary.db_kib)}
      </div>
    </section>

    <section class="section">
      <h2>What it deliberately ignored</h2>
      <table>
        <thead><tr><th>Reason</th><th class="num">Count</th></tr></thead>
        <tbody>${summary.reasons.map((r) => `
          <tr><td><span class="reason">${esc(r.reason_code)}</span></td>
              <td class="num">${r.n}</td></tr>`).join('')}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Reconciliation</h2>
      <table>
        <thead><tr><th>Operation</th><th class="num">Count</th></tr></thead>
        <tbody>${summary.operations.map((r) => `
          <tr><td><span class="reason kept">${esc(r.op)}</span></td>
              <td class="num">${r.n}</td></tr>`).join('')}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Model usage</h2>
      <div class="scroll-x"><table>
        <thead><tr><th>Purpose</th><th class="num">Calls</th><th class="num">Tokens in</th>
          <th class="num">Tokens out</th><th class="num">Avg ms</th><th class="num">Cost</th></tr></thead>
        <tbody>${summary.usage_by_purpose.map((u) => `
          <tr><td>${esc(u.purpose)}</td><td class="num">${u.calls}</td>
              <td class="num">${u.tokens_in.toLocaleString()}</td>
              <td class="num">${u.tokens_out.toLocaleString()}</td>
              <td class="num">${u.avg_latency_ms}</td>
              <td class="num">$${Number(u.cost_usd).toFixed(4)}</td></tr>`).join('')}</tbody>
      </table></div>
    </section>

    <section class="section">
      <h2>Recent answers</h2>
      <div class="scroll-x"><table>
        <thead><tr><th>Question</th><th>Outcome</th><th class="num">ms</th><th class="num">Cost</th></tr></thead>
        <tbody>${traces.traces.map((t) => {
          const lat = (() => { try { return JSON.parse(t.latency_json).total; } catch { return ''; } })();
          return `<tr>
            <td>${esc(t.query)}</td>
            <td>${t.abstained
              ? `<span class="reason">abstained</span> <span style="color:var(--ink-faint)">${esc(t.abstain_reason || '')}</span>`
              : '<span class="reason kept">answered</span>'}</td>
            <td class="num">${lat}</td>
            <td class="num">$${Number(t.cost_usd || 0).toFixed(5)}</td>
          </tr>`;
        }).join('') || '<tr><td colspan="4" class="empty">No questions asked yet.</td></tr>'}</tbody>
      </table></div>
    </section>

    <section class="section">
      <h2>Refused candidates</h2>
      <div class="scroll-x"><table>
        <thead><tr><th>Record</th><th>Reason</th><th>Why</th></tr></thead>
        <tbody>${decisions.decisions.map((d) => `
          <tr>
            <td><button class="cite" data-record="${esc(d.record_id)}">${esc(d.record_id)}</button></td>
            <td><span class="reason">${esc(d.reason_code)}</span></td>
            <td>${esc(d.reason_detail || '')}</td>
          </tr>`).join('')}</tbody>
      </table></div>
    </section>`;
}

/* =============================================================== boot ==== */

async function boot() {
  renderSuggestions();
  show('ask');
  $('#ask-input').focus();

  try {
    const h = await api('/health');
    const rail = $('#rail-status');
    rail.classList.add(h.unprocessed > 0 ? 'busy' : 'ok');
    $('#status-text').textContent =
      `${h.records.toLocaleString()} dictations · ${h.facts} facts` +
      (h.unprocessed ? ` · ${h.unprocessed} still processing` : '');

    const mem = await api('/memory');
    if (mem.needs_decision.length) {
      $('#decision-badge').hidden = false;
      $('#decision-badge').textContent = mem.needs_decision.length;
    }
  } catch (err) {
    $('#status-text').textContent = 'backend unreachable';
  }
}

boot();
