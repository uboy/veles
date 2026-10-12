/* Админ-панель veles (/admin): обзор, документы (+удаление, v2: чекбоксы,
 * массовая смена группы и массовое удаление), задачи (+v2: Отменить/
 * Перезапустить/Удалить, бейджи «уже в базе» и «дубль задачи»), группы,
 * тестовый вопрос, загрузка. Токен админа (VELES_API_TOKEN) вводится на
 * странице и хранится в localStorage; каждый API-запрос несёт
 * Authorization: Bearer <токен>, в URL токен не используется. Вкладки -
 * показ/скрытие секций без перезагрузки; данные из ответов экранируются esc().
 * Открытые эндпоинты (healthz/readyz/openapi.json) читаются без токена. */
"use strict";

const TOK_KEY = "veles_admin_token";
const $ = (id) => document.getElementById(id);

function token() { return localStorage.getItem(TOK_KEY) || ""; }

function authHeaders() {
  const h = {};
  const t = token();
  if (t) h["Authorization"] = "Bearer " + t;
  return h;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function dash(v) { return (v === undefined || v === null || v === "") ? "-" : v; }

class ApiError extends Error {
  constructor(status, detail) {
    super(detail || ("отказ " + status));
    this.status = status;
    this.detail = detail || "";
  }
}

function errText(e) {
  if (e instanceof ApiError) {
    if (e.status === 401) return "введите токен админа";
    if (e.status === 429) return "слишком много запросов - повторите позже";
    if (e.status === 0) return "API недоступен";
    return e.detail || ("отказ " + e.status);
  }
  return String(e);
}

function banner(on) { $("apibanner").hidden = !on; }

function statusLine(text) { $("apistatus").textContent = text; }

function authFail() {
  // сохранённое значение не стираем - после ввода верного токена не набирать заново
  const msg = $("authmsg");
  msg.textContent = "Токен недействителен или истёк - введите заново.";
  msg.hidden = false;
  $("token").focus();
}

function authOk() {
  $("authmsg").hidden = true;
  banner(false);
}

/* JSON-запрос к API с bearer-токеном; ошибка - ApiError с detail бэкенда
 * (тексты бэкенда русские - показываются как есть). */
async function api(path, opts) {
  const o = Object.assign({}, opts || {});
  o.headers = Object.assign(authHeaders(), o.headers || {});
  let r;
  try {
    r = await fetch(path, o);
  } catch (e) {
    banner(true);
    throw new ApiError(0, "сеть недоступна");
  }
  if (r.status === 401) { authFail(); throw new ApiError(401, "unauthorized"); }
  if (r.status === 429) throw new ApiError(429, "Слишком много запросов - повторите позже");
  if (!r.ok) {
    let detail = "";
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (e) { /* тело не JSON */ }
    throw new ApiError(r.status, detail);
  }
  banner(false);
  return r.json();
}

/* Открытые эндпоинты - без токена и без authFail (401/5xx там не бывает). */
async function publicJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new ApiError(r.status, "");
  return r.json();
}

// ---- общие данные: stats кормит «Обзор», «Документы» и «Группы» -----------

let STATS = null;
let STATS_AT = "";
let GROUPS = [];

async function loadStats() {
  STATS = await api("/v1/knowledge/stats");
  STATS_AT = new Date().toLocaleTimeString();
  return STATS;
}

function errCell(e) {
  const t = esc(errText(e));
  return '<span class="' + (e instanceof ApiError && e.status === 401 ? "warn" : "bad") + '">' +
    t + "</span>";
}

// ---- Обзор ----------------------------------------------------------------

async function loadOverview() {
  try {
    const h = await publicJson("/healthz");
    $("ov-health").innerHTML = '<span class="ok">жив</span>' +
      (h.index_loaded ? "" : ' <span class="warn">индекс не загружен</span>');
  } catch (e) {
    $("ov-health").innerHTML = '<span class="bad">нет ответа</span>';
  }
  try {
    const r = await fetch("/readyz");
    $("ov-ready").innerHTML = r.ok
      ? '<span class="ok">готов</span>'
      : '<span class="warn">индекс загружается</span>';
  } catch (e) {
    $("ov-ready").innerHTML = '<span class="bad">нет ответа</span>';
  }
  try {
    const spec = await publicJson("/openapi.json");
    $("ov-version").textContent = (spec.info && spec.info.version) || "-";
  } catch (e) {
    $("ov-version").textContent = "-";
  }
  try {
    const m = await api("/v1/models");
    $("ov-model").textContent = (m.data || []).map((x) => x.id).join(", ") || "-";
  } catch (e) {
    $("ov-model").innerHTML = errCell(e);
  }
  try {
    const s = await loadStats();
    $("ov-docs").textContent = s.totals.documents;
    $("ov-nodes").textContent = s.totals.nodes;
    $("ov-superseded").textContent = s.totals.superseded_nodes;
    $("ov-embed").textContent = dash(s.embed_model);
  } catch (e) {
    ["ov-docs", "ov-nodes", "ov-superseded", "ov-embed"].forEach(
      (id) => { $(id).innerHTML = errCell(e); });
  }
  statusLine("API: последнее обращение " + new Date().toLocaleTimeString());
}

// ---- Документы ------------------------------------------------------------

function renderDocs() {
  const box = $("docsbox");
  if (!STATS) { box.innerHTML = "<p><i>нет данных</i></p>"; return; }
  const q = $("docfilter").value.trim().toLowerCase();
  const docs = (STATS.documents || []).filter(
    (d) => !q || String(d.title || "").toLowerCase().includes(q));
  const rows = docs.map((d) =>
    "<tr><td><input type=\"checkbox\" class=\"selrow\" data-doc=\"" + esc(d.document_id) +
    "\" data-title=\"" + esc(dash(d.title)) + '"></td><td>' + esc(dash(d.title)) +
    "</td><td>" + esc((d.groups || []).join(", ")) +
    "</td><td>" + d.nodes + "</td><td><code>" + esc(d.document_id) + "</code></td>" +
    '<td><button type="button" class="del" data-doc="' + esc(d.document_id) +
    '" data-title="' + esc(dash(d.title)) + '">Удалить</button></td></tr>').join("");
  box.innerHTML = rows
    ? "<table><thead><tr><th><input type=\"checkbox\" id=\"selall\" title=\"выбрать все\"></th>" +
      "<th>Название</th><th>Группы</th><th>Узлов</th><th>id</th><th></th></tr>" +
      "</thead><tbody>" + rows + "</tbody></table>"
    : "<p><i>документы не найдены</i></p>";
  $("docmsg").textContent = "Всего: " + STATS.totals.documents + " документов / " +
    STATS.totals.nodes + " узлов · загружено " + STATS_AT;
  updateBulkPanel();
}

async function loadDocs() {
  try {
    await loadStats();
    authOk();
  } catch (e) {
    $("docmsg").textContent = errText(e);
  }
  renderDocs();
  renderGroups();
}

async function deleteDoc(docId, title) {
  if (!confirm("Удалить «" + title + "»? Узлы выйдут из поиска сразу, канон переедет в trash.")) {
    return;
  }
  const msg = $("docmsg");
  msg.textContent = "удаляю...";
  try {
    const res = await api("/v1/documents/" + encodeURIComponent(docId), { method: "DELETE" });
    msg.textContent = "Удалено: узлов выведено - " + res.nodes +
      ", ревизий в trash - " + res.revisions + ", каталог - " + res.trash;
    await loadDocs();
  } catch (e) {
    msg.textContent = "Не удалось: " + errText(e);
  }
}

// Массовые операции (v2): чекбоксы строк + панель «сменить группу / удалить».
// Смена группы одиночного документа - той же панелью при одном выбранном.
function selectedDocs() {
  return Array.prototype.map.call(
    document.querySelectorAll("#docsbox input.selrow:checked"),
    (el) => ({ id: el.dataset.doc, title: el.dataset.title }));
}

function updateBulkPanel() {
  const panel = $("bulkpanel");
  if (!panel) return;
  const anyRows = document.querySelector("#docsbox input.selrow") !== null;
  const n = selectedDocs().length;
  panel.hidden = !anyRows;
  $("bulkprogress").textContent = "";
  $("bulkgroup").disabled = !n;
  $("bulkregroup").disabled = !n || !$("bulkgroup").value;
  $("bulkdelete").disabled = !n;
}

async function runBulk(kind) {
  const docs = selectedDocs();
  if (!docs.length) return;
  const msg = $("docmsg");
  if (kind === "regroup") {
    const opt = $("bulkgroup").selectedOptions[0];
    if (!opt || !opt.value) { msg.textContent = "выберите группу"; return; }
    if (!confirm("Сменить группу у " + docs.length + " документ(ов) на «" +
        opt.textContent + "»? Права применятся сразу.")) return;
  } else if (!confirm("Удалить " + docs.length + " документ(ов)? Узлы выйдут " +
      "из поиска сразу, канон переедет в trash.")) {
    return;
  }
  let ok = 0;
  const failed = [];
  for (let i = 0; i < docs.length; i++) {
    $("bulkprogress").textContent = "обрабатываю " + (i + 1) + " из " + docs.length + "...";
    try {
      if (kind === "regroup") {
        await api("/v1/documents/" + encodeURIComponent(docs[i].id) + "/groups", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ groups: [$("bulkgroup").value] }),
        });
      } else {
        await api("/v1/documents/" + encodeURIComponent(docs[i].id), { method: "DELETE" });
      }
      ok += 1;
    } catch (e) {
      failed.push(docs[i].title + " - " + errText(e));
      if (e instanceof ApiError && e.status === 401) {
        $("bulkprogress").textContent = "";
        msg.textContent = "Остановлено: " + errText(e);
        return;
      }
    }
  }
  $("bulkprogress").textContent = "";
  msg.textContent = kind === "regroup"
    ? "Группы сменены: " + ok + " из " + docs.length +
      (failed.length ? "; с ошибками: " + failed.join("; ") : "")
    : "Удалено: " + ok + " из " + docs.length +
      (failed.length ? "; с ошибками: " + failed.join("; ") : "");
  await loadDocs();
}

// ---- Задачи ---------------------------------------------------------------

function statusBadge(s) {
  const cls = { done: "ok", failed: "bad", running: "warn", queued: "dim" }[s] || "dim";
  return '<span class="' + cls + '">' + esc(dash(s)) + "</span>";
}

/* Бейджи v2: «уже в базе» (exists_in_base) и «дубль задачи №N» (duplicate_of). */
function jobBadges(j) {
  let out = "";
  if (j.exists_in_base) {
    out += ' <span class="ok" title="документ уже лежит в каноне">уже в базе</span>';
  }
  if (j.duplicate_of) {
    out += ' <span class="warn" title="та же задача: ' + esc(j.duplicate_of) +
      '">дубль задачи №' + esc(String(j.duplicate_of).slice(0, 8)) + "</span>";
  }
  return out;
}

/* Кнопки по статусу (v2): Отменить - queued/running, Перезапустить - failed,
 * Удалить - завершённые (done/failed). */
function jobButtons(j) {
  const btn = (act, label, cls) =>
    '<button type="button" class="jobact' + (cls ? " " + cls : "") +
    '" data-act="' + act + '" data-job="' + esc(j.job_id) + '">' + label + "</button>";
  if (j.status === "queued" || j.status === "running") return btn("cancel", "Отменить");
  if (j.status === "failed") return btn("retry", "Перезапустить") + " " + btn("delete", "Удалить", "del");
  if (j.status === "done") return btn("delete", "Удалить", "del");
  return "";
}

function renderJobs(jobs) {
  const box = $("jobsbox");
  if (!jobs.length) { box.innerHTML = "<p><i>задач нет</i></p>"; return; }
  const rows = jobs.map((j) =>
    "<tr data-job=\"" + esc(j.job_id) + "\"><td>" +
    esc(String(dash(j.created_at)).replace("T", " ").slice(0, 19)) + "</td><td>" +
    esc(j.title || j.filename || "-") + jobBadges(j) + "</td><td>" + esc(j.user_id || "-") +
    "</td><td>" + statusBadge(j.status) + "</td><td>" + esc(dash(j.action)) +
    "</td><td>" + esc(dash(j.revision_id)) + "</td><td>" + (j.nodes || 0) +
    "</td><td>" + esc(String(j.error || "").slice(0, 120)) + "</td><td>" +
    jobButtons(j) + "</td></tr>").join("");
  box.innerHTML = "<table><thead><tr><th>Создана</th><th>Файл</th><th>Пользователь</th>" +
    "<th>Статус</th><th>Действие</th><th>Ревизия</th><th>Узлов</th><th>Ошибка</th>" +
    "<th>Операции</th></tr>" +
    "</thead><tbody>" + rows + "</tbody></table>" +
    "<p><small>клик по строке - детали задачи; бейдж «уже в базе» - документ уже в каноне, " +
    "«дубль» - другая активная задача с тем же файлом</small></p>";
}

/* Действия над задачей (v2): cancel/retry - POST, delete - DELETE;
 * автосписок обновляется после действия, не теряя состояние страницы. */
async function jobAction(jobId, act) {
  const note = $("jobsnote");
  note.textContent = act === "cancel" ? "отменяю..." :
    act === "retry" ? "перезапускаю..." : "удаляю...";
  try {
    if (act === "cancel") {
      await api("/v1/documents/jobs/" + encodeURIComponent(jobId) + "/cancel", { method: "POST" });
      note.textContent = "задача отменена";
    } else if (act === "retry") {
      const r = await api("/v1/documents/jobs/" + encodeURIComponent(jobId) + "/retry", { method: "POST" });
      note.textContent = "задача перезапущена: " + r.job_id;
    } else {
      await api("/v1/documents/jobs/" + encodeURIComponent(jobId), { method: "DELETE" });
      note.textContent = "задача удалена из журнала";
    }
    loadJobs();
  } catch (e) {
    note.textContent = "не удалось: " + errText(e);
  }
}

let jobsTimer = null;

function scheduleJobsPoll(need) {
  // автопов раз в 5 с, пока есть queued/running и задан токен; сбой - стоп
  if (jobsTimer) { clearTimeout(jobsTimer); jobsTimer = null; }
  if (need && token()) {
    jobsTimer = setTimeout(() => { jobsTimer = null; loadJobs(); }, 5000);
  }
}

async function loadJobs() {
  const note = $("jobsnote");
  try {
    const d = await api("/v1/documents/jobs");
    authOk();
    renderJobs(d.jobs || []);
    const active = (d.jobs || []).filter(
      (j) => j.status === "queued" || j.status === "running").length;
    note.textContent = active ? "активных: " + active + " - обновление раз в 5 с" : "";
    scheduleJobsPoll(active > 0);
  } catch (e) {
    note.textContent = errText(e);
    scheduleJobsPoll(false);
  }
}

async function showJob(jobId) {
  const box = $("jobdetail");
  box.textContent = "загружаю задачу...";
  try {
    const j = await api("/v1/documents/jobs/" + encodeURIComponent(jobId));
    // В значениях ячеек - ТОЛЬКО esc(), без исключений: title приходит из
    // формы загрузки без санитизации (documents.py) и может начинаться с
    // "<span" - сырая вставка «красивого бейджа» была бы HTML-инъекцией.
    // Бейдж статуса - отдельная строка таблицы из esc-безопасных кусков.
    const rows = [
      ["файл", j.filename], ["название", j.title], ["документ", j.document_id],
      ["пользователь", j.user_id], ["размер, байт", j.size_bytes],
      ["группы", (j.groups || []).join(", ")], ["действие", j.action],
      ["ревизия", j.revision_id], ["узлов", j.nodes], ["sha256", j.sha256],
      ["создана", j.created_at], ["старт", j.started_at],
      ["завершена", j.finished_at], ["ошибка", j.error],
    ];
    box.innerHTML = "<h3>Задача <code>" + esc(j.job_id) + "</code></h3><table><tbody>" +
      "<tr><th>статус</th><td>" + statusBadge(j.status) + jobBadges(j) + "</td></tr>" +
      rows.map((kv) => "<tr><th>" + esc(kv[0]) + "</th><td>" + esc(dash(kv[1])) +
        "</td></tr>").join("") +
      "</tbody></table>";
  } catch (e) {
    box.textContent = "не удалось получить задачу: " + errText(e);
  }
}

// ---- Группы ---------------------------------------------------------------

function renderGroups() {
  const box = $("groupsbox");
  if (!GROUPS.length) {
    if (token()) box.innerHTML = "<p><i>группы не получены - нужен токен</i></p>";
    return;
  }
  const docs = (STATS && STATS.documents) || [];
  const rows = GROUPS.map((g) => {
    const n = docs.filter((d) => (d.groups || []).indexOf(g.id) >= 0).length;
    return "<tr><td><code>" + esc(g.id) + "</code></td><td>" + esc(g.title) +
      "</td><td>" + n + "</td></tr>";
  }).join("");
  box.innerHTML = "<table><thead><tr><th>id</th><th>Название</th><th>Документов</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table>" +
    "<p><small>Документов в группе - по текущим узлам, видимым владельцу токена.</small></p>";
}

async function loadGroups() {
  try {
    const d = await api("/v1/groups");
    authOk();
    GROUPS = d.groups || [];
  } catch (e) {
    $("groupsbox").innerHTML = "<p><i>" + esc(errText(e)) + "</i></p>";
    return;
  }
  renderGroups();
  const sel = $("upgroup");
  sel.innerHTML = '<option value="">личный документ</option>' + GROUPS.map(
    (g) => '<option value="' + esc(g.id) + '">' + esc(g.title + " (" + g.id + ")") +
      "</option>").join("");
  // массовая панель «Документов» - тот же список групп, что у загрузки
  const bulk = $("bulkgroup");
  bulk.innerHTML = '<option value="">- выберите группу -</option>' + GROUPS.map(
    (g) => '<option value="' + esc(g.id) + '">' + esc(g.title + " (" + g.id + ")") +
      "</option>").join("");
  updateBulkPanel();
}

// ---- Вопрос ---------------------------------------------------------------

// Ссылка источника безопасна для панели только подписанная (exp+sig):
// static-режим без VELES_LINK_SECRET строит source_url с token=<bearer> в
// query (attach_citation_links, mint_token_links=static) - клик кладёт токен
// в историю браузера и логи /v1/source, вопреки обещанию панели «токен в URL
// не используется». Ссылки с token= панель не рендерит - источник текстом.
function safeSourceUrl(u) {
  if (!u) return "";
  try {
    return new URL(u, location.origin).searchParams.has("token") ? "" : u;
  } catch (e) {
    return "";
  }
}

function renderAsk(res) {
  const out = $("askout");
  const badge = res.decision === "answer"
    ? '<span class="ok">ответ найден</span>'
    : res.decision === "warn_conflict"
      ? '<span class="warn">возможен конфликт редакций</span>'
      : '<span class="bad">ответа нет</span>';
  let html = "<p>" + badge + "</p>";
  if (res.answer) {
    html += '<div class="answer">' + esc(res.answer).replace(/\n/g, "<br>") + "</div>";
  }
  const cits = res.citations || [];
  if (cits.length) {
    let suppressed = 0;
    html += "<h3>Цитаты</h3><ol>" + cits.map((c) => {
      const where = c.lines && c.lines.length >= 2
        ? ", строки " + c.lines[0] + "-" + c.lines[1] : "";
      const label = "[" + esc(c.n) + "] «" + esc(dash(c.title)) + "» " +
        esc(dash(c.source_id)) + esc(where);
      const url = safeSourceUrl(c.source_url);
      if (c.source_url && !url) suppressed += 1;
      return "<li>" + (url
        ? '<a href="' + esc(url) + '" target="_blank" rel="noopener">' + label + "</a>"
        : label) + "</li>";
    }).join("") + "</ol>";
    if (suppressed) {
      html += "<p><small>Ссылки на источники скрыты: фасад строит их с токеном " +
        "в URL (static-режим без VELES_LINK_SECRET). Задайте VELES_LINK_SECRET - " +
        "панель будет открывать источники по подписанным ссылкам.</small></p>";
    }
  }
  const diag = res.diagnostics || {};
  const bits = ["уверенность: " + dash(diag.confidence),
    "синтез LLM: " + (diag.llm_used ? "да" : "нет")];
  if (diag.llm_error) bits.push("ошибка синтеза: " + diag.llm_error);
  if (diag.reason) bits.push("причина: " + diag.reason);
  if (diag.conflicts && diag.conflicts.length) {
    bits.push("конфликты: " + diag.conflicts.map((c) =>
      "«" + (c.title || c.document_id) + "» (" + c.revision_id + ", " + c.signal + ")"
    ).join("; "));
  }
  html += "<p><small>" + esc(bits.join(" · ")) + "</small></p>";
  out.innerHTML = html;
}

async function ask(ev) {
  ev.preventDefault();
  const q = $("askq").value.trim();
  if (!q) { $("askout").innerHTML = "<p><i>введите вопрос</i></p>"; return; }
  const headers = { "Content-Type": "application/json" };
  // act-as принимается только с владельческим bearer; без него молча игнорируется
  const as = $("askas").value.trim();
  if (as) headers["X-Veles-On-Behalf-Of"] = as;
  $("askout").textContent = "спрашиваю...";
  try {
    const res = await api("/v1/knowledge/query", {
      method: "POST",
      headers,
      body: JSON.stringify({
        query: q,
        top_k: parseInt($("asktopk").value, 10) || 5,
        use_llm: true,
      }),
    });
    renderAsk(res);
  } catch (e) {
    $("askout").innerHTML = "<p>" + esc(errText(e)) + "</p>";
  }
}

// ---- Загрузка -------------------------------------------------------------

async function upload(ev) {
  ev.preventDefault();
  const inp = $("upfile");
  const msg = $("upmsg");
  if (!inp.files.length) { msg.textContent = "выберите файл"; return; }
  const fname = inp.files[0].name;
  const fd = new FormData();
  fd.append("file", inp.files[0]);
  const title = $("uptitle").value.trim();
  if (title) fd.append("title", title);
  const group = $("upgroup").value;
  if (group) fd.append("group", group);
  msg.textContent = "отправляю " + fname + "...";
  try {
    const r = await fetch("/v1/documents", { method: "POST", headers: authHeaders(), body: fd });
    if (r.status === 401) { authFail(); msg.textContent = "введите токен админа"; return; }
    if (!r.ok) {
      let detail = "";
      try {
        const j = await r.json();
        detail = typeof j.detail === "string" ? j.detail : "";
      } catch (e) { /* тело не JSON */ }
      msg.textContent = "отказ " + r.status + (detail ? ": " + detail : "");
      return;
    }
    const job = await r.json();
    authOk();
    msg.textContent = "принято, задача " + job.job_id + " - смотрите вкладку «Задачи»";
    showTab("jobs");
    loadJobs();
  } catch (e) {
    banner(true);
    msg.textContent = "сервис загрузки недоступен: " + e;
  }
}

// ---- вкладки и старт ------------------------------------------------------

function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll("main .tab").forEach((s) => {
    s.hidden = s.id !== "tab-" + name;
  });
  // данные вкладки обновляются при открытии; вопрос/загрузка - формы, не грузим
  if (name === "overview") loadOverview();
  if (name === "docs") loadDocs();
  if (name === "jobs") loadJobs();
  if (name === "groups") loadGroups();
}

function saveToken() {
  localStorage.setItem(TOK_KEY, $("token").value.trim());
  authOk();
  statusLine("токен сохранён, обновляю...");
  loadOverview();
  loadDocs();
  loadJobs();
  loadGroups();
}

window.addEventListener("DOMContentLoaded", () => {
  $("token").value = token();
  $("tokensave").addEventListener("click", saveToken);
  $("token").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); saveToken(); }
  });
  $("tabs").addEventListener("click", (ev) => {
    const b = ev.target.closest("button[data-tab]");
    if (b) showTab(b.dataset.tab);
  });
  $("docfilter").addEventListener("input", renderDocs);
  $("docsrefresh").addEventListener("click", loadDocs);
  $("jobsrefresh").addEventListener("click", loadJobs);
  // v2: выбор строк («выбрать все» - через шапку таблицы) и массовые операции
  $("docsbox").addEventListener("change", (ev) => {
    if (ev.target.id === "selall") {
      document.querySelectorAll("#docsbox input.selrow").forEach(
        (el) => { el.checked = ev.target.checked; });
    }
    updateBulkPanel();
  });
  $("bulkgroup").addEventListener("change", updateBulkPanel);
  $("bulkregroup").addEventListener("click", () => runBulk("regroup"));
  $("bulkdelete").addEventListener("click", () => runBulk("delete"));
  $("docsbox").addEventListener("click", (ev) => {
    const b = ev.target.closest("button.del");
    if (b) deleteDoc(b.dataset.doc, b.dataset.title);
  });
  // v2: кнопки задач приоритетнее клика по строке (детали)
  $("jobsbox").addEventListener("click", (ev) => {
    const b = ev.target.closest("button.jobact");
    if (b) { jobAction(b.dataset.job, b.dataset.act); return; }
    const tr = ev.target.closest("tr[data-job]");
    if (tr) showJob(tr.dataset.job);
  });
  $("askform").addEventListener("submit", ask);
  $("upform").addEventListener("submit", upload);
  loadOverview();
  loadDocs();
  loadJobs();
  loadGroups();
});
