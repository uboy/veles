/* Панель загрузки документов (/upload). Токен берётся из ?token= ссылки
 * (владелец); в identity-режимах владельцу достаточно открыть страницу с
 * ?token=<VELES_API_TOKEN>, обычным пользователям - через Telegram/API. */
"use strict";

const qs = new URLSearchParams(location.search);
const TOK = qs.get("token") || "";
const H = TOK ? { "Authorization": "Bearer " + TOK } : {};

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function loadDocs() {
  try {
    const r = await fetch("/v1/knowledge/stats", { headers: H });
    if (r.status === 401) { document.getElementById("msg").textContent = "нет доступа: откройте страницу с ?token=<VELES_API_TOKEN>"; return; }
    const d = await r.json();
    const rows = (d.documents || []).map(doc =>
      "<tr><td>" + esc(doc.title) + "</td><td>" + doc.nodes + "</td><td>" +
      esc(doc.document_id) + "</td></tr>").join("");
    document.getElementById("docs").innerHTML =
      "<table><tr><th>Документ</th><th>Узлов</th><th>id</th></tr>" + rows + "</table>" +
      "<p>Всего: " + d.totals.documents + " документов / " + d.totals.nodes + " узлов</p>";
  } catch (e) {
    document.getElementById("msg").textContent = "список недоступен: " + e;
  }
}

async function pollJob(jobId, fname) {
  const box = document.getElementById("msg");
  for (let i = 0; i < 300; i++) {
    let rec = null;
    try {
      const r = await fetch("/v1/documents/jobs/" + jobId, { headers: H });
      if (r.status === 404) { box.textContent = "❌ " + fname + ": джоб потерян (сервис перезапускался)"; return; }
      rec = await r.json();
      if (rec.status === "done" || rec.status === "failed") break;
    } catch (e) { /* фасад мог перезапускаться - ждём дальше */ }
    box.textContent = "⏳ " + fname + ": " + (rec ? rec.status : "в очереди") + "...";
    await new Promise(res => setTimeout(res, 2000));
  }
  if (rec && rec.status === "done") {
    box.textContent = "✅ " + fname + ": загружено (" + (rec.revision_id || "") + "), документ в поиске";
    loadDocs();
  } else if (rec && rec.status === "failed") {
    box.textContent = "❌ " + fname + ": " + (rec.error || "не удалось обработать");
  } else {
    box.textContent = "⏳ " + fname + " всё ещё обрабатывается - проверьте позже";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("upform").addEventListener("submit", async ev => {    ev.preventDefault();
    const inp = document.getElementById("file");
    const box = document.getElementById("msg");
    if (!inp.files.length) { box.textContent = "выберите файл"; return; }
    const fname = inp.files[0].name;
    const fd = new FormData();
    fd.append("file", inp.files[0]);
    box.textContent = "отправляю " + fname + "...";
    let r;
    try {
      r = await fetch("/v1/documents", { method: "POST", headers: H, body: fd });
    } catch (e) {
      box.textContent = "❌ сервис загрузки недоступен: " + e;
      return;
    }
    if (r.status === 401) { box.textContent = "нет доступа: откройте страницу с ?token=<VELES_API_TOKEN>"; return; }
    if (r.status === 403 || r.status === 413) { box.textContent = "❌ " + (await r.json()).detail; return; }
    if (!r.ok) { box.textContent = "❌ отказ " + r.status + ": " + (await r.json()).detail; return; }
    const job = await r.json();
    box.textContent = "принято, обработка...";
    pollJob(job.job_id, fname);
  });
  loadDocs();
});
  document.getElementById("refresh").addEventListener("click", loadDocs);
