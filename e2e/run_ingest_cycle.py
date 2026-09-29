#!/usr/bin/env python3
"""E2E-цикл сохранения (K13): «сохранялись данные в базу правильно» — no-loss.

Запуск НА стенд (не pytest), tmp VELES_DATA, порт 18460, настоящий эмбеддер:
  CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 python3 e2e/run_ingest_cycle.py \
      [--repo ~/proj/veles-bench/veles] [--port 18460] [--keep]

Шаги (чек-лист PASS/FAIL, вердикт и exit-code в конце):
  1. tmp-окружение: канон с seed-документом, полный индекс настоящим
     SentenceTransformers-эмбеддером (как пересборка прода);
  2. старт фасада на 127.0.0.1:<port>;
  3. загрузка txt (multipart), md (JSON) и маленького pdf (multipart) через
     POST /v1/documents -> джобы done;
  4. NO-LOSS: узлы документа в ИНДЕКСЕ (GET /v1/knowledge/stats) == числу узлов
     в каноне минус root - для каждого формата;
  5. поиск находит каждый документ по уникальному маркеру;
  6. повторная загрузка txt (изменённый) -> ревизия r2: action=updated,
     r1 в каноне superseded, в индексе superseded-узлы == узлам r1,
     новый маркер ищется, старый - НЕ виден;
  7. рестарт фасада -> документ на месте (K3/K4: load_index + overlay),
     поиск работает;
  8. DELETE -> 0 узлов: документа нет в stats, поиск пуст, канон в trash
     (все ревизии), L0 удалён, события в ingest-audit.jsonl.

Stdlib-only (urllib/subprocess): работает системным python3. Секретов не
требует (loopback static-owner без bearer). Прод не трогает: всё в tmp-каталоге,
удаляется в конце (или --keep).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

results: list[dict] = []


def record(item: str, name: str, expected: str, actual: str, ok: bool, note: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    results.append({"id": item, "name": name, "expected": expected,
                    "actual": actual, "status": status, "note": note})
    line = f"[{status}] {item} {name}: {actual}"
    if note:
        line += f" | {note}"
    print(line, flush=True)


# ------------------------------------------------------------------- HTTP

def http(method: str, url: str, *, headers: dict | None = None, body: bytes | None = None,
         timeout: float = 60) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # статус >= 400 - тоже валидный ответ
        return e.code, e.read().decode("utf-8", "replace")


def http_json(method: str, url: str, payload: dict | None = None,
              timeout: float = 120) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    code, text = http(method, url, headers=headers, body=data, timeout=timeout)
    try:
        return code, json.loads(text)
    except ValueError:
        return code, {"_raw": text[:300]}


def multipart(fields: dict[str, str], file_field: str, filename: str,
              content: bytes, boundary: str) -> bytes:
    """multipart/form-data вручную (stdlib): поля + один файл."""
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n").encode("utf-8")
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
    out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(out)


def doc_dir_name(document_id: str) -> str:
    """Та же формула имени каталога, что canonical.store._doc_dir (дублируется,
    чтобы скрипт был stdlib-only)."""
    import hashlib

    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in document_id)
    if len(safe) <= 120:
        return safe
    suffix = hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:105]}-{suffix}"


# --------------------------------------------------------------- PDF (малыш)

def minimal_pdf(lines: list[str]) -> bytes:
    """Минимальный валидный одностраничный PDF (Helvetica, WinAnsi/ASCII):
    заголовок + абзацы >= 25 символов (docling отбрасывает короче)."""
    content_parts = ["BT /F1 16 Tf 72 760 Td (K13 Ingest Integrity Check) Tj ET"]
    y = 720
    for line in lines:
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_parts.append(f"BT /F1 11 Tf 72 {y} Td ({safe}) Tj ET")
        y -= 20
    stream = "\n".join(content_parts).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    return bytes(out)


# ------------------------------------------------------------- subprocesses

def veles_env(data_dir: Path) -> dict:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VELES_"):
            del env[key]
    env.update({
        "VELES_DATA": str(data_dir),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


class Facade:
    """Фасад veles как дочерний процесс (tmp VELES_DATA, loopback без токена)."""

    def __init__(self, py: str, repo: Path, port: int, data_dir: Path):
        self.proc: subprocess.Popen | None = None
        self.cmd = [py, "-m", "veles", "serve", "--host", "127.0.0.1", "--port", str(port)]
        self.base = f"http://127.0.0.1:{port}"
        self.cwd, self.env = repo, veles_env(data_dir)

    def start(self, timeout_s: float = 420) -> None:
        self.proc = subprocess.Popen(
            self.cmd, cwd=str(self.cwd), env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"фасад умер при старте, код {self.proc.returncode}")
            try:
                code, _ = http("GET", f"{self.base}/healthz", timeout=5)
                if code == 200:
                    return
            except OSError:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"фасад не поднялся за {timeout_s}s")

    def wait_ready(self, timeout_s: float = 420) -> bool:
        """Пайплайн собран (эмбеддер загружен): stats отвечает 200."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code, _ = http_json("GET", f"{self.base}/v1/knowledge/stats", timeout=30)
            if code == 200:
                return True
            time.sleep(2.0)
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None


def wait_job(base: str, job_id: str, timeout_s: float = 600) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code, job = http_json("GET", f"{base}/v1/documents/jobs/{job_id}", timeout=30)
        if code == 200 and job.get("status") in ("done", "failed"):
            return job
        time.sleep(1.0)
    return {"status": "timeout", "error": f"джоб не завершился за {timeout_s}s"}


def stats_by_doc(base: str) -> tuple[dict[str, int], dict]:
    code, stats = http_json("GET", f"{base}/v1/knowledge/stats", timeout=120)
    if code != 200:
        return {}, stats
    return {d["document_id"]: d["nodes"] for d in stats.get("documents", [])}, stats


def canon_node_count(data_dir: Path, document_id: str) -> int | None:
    path = data_dir / "canonical" / doc_dir_name(document_id) / "nodes.jsonl"
    if not path.exists():
        return None
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def search(base: str, query: str) -> list[dict]:
    code, resp = http_json("POST", f"{base}/v1/knowledge/query",
                           {"query": query, "top_k": 5, "use_llm": False}, timeout=180)
    if code != 200:
        return [{"_http_error": code, "_body": str(resp)[:200]}]
    return resp.get("citations", [])


# --------------------------------------------------------------- сценарий

def bootstrap_index(py: str, repo: Path, data_dir: Path) -> tuple[bool, str]:
    """Seed-документ в канон + полный индекс настоящим эмбеддером (как пересборка)."""
    script = f"""
import sys
sys.path.insert(0, {str(repo)!r})
from pathlib import Path
from veles.config import EMBED_MODEL, EMBED_DIM
from veles.canonical.store import save_document
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.indexing.embedder import SentenceTransformersProvider

corpus = Path({str(data_dir)!r}) / "bootstrap-corpus"
seed = corpus / "seed" / "k13-seed-doc.md"
seed.parent.mkdir(parents=True, exist_ok=True)
seed.write_text(
    "# K13 seed\\n\\n"
    "Seed-документ для стартового индекса цикла сохранения K13.\\n"
    "Сид-маркер поиска: базелин-фиксатор контура контроля.\\n",
    encoding="utf-8", newline="\\n")
doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
save_document(doc, by="k13-e2e")
embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
bundle = build_index(embedder, NumpyFlatStore(dim=EMBED_DIM))
bundle.save(Path({str(data_dir)!r}) / "index",
            embed_meta={{"model": EMBED_MODEL, "backend": "numpy-flat"}})
print("bootstrap ok:", len(bundle.node_meta), "узлов")
"""
    r = subprocess.run([py, "-c", script], cwd=str(repo), env=veles_env(data_dir),
                       capture_output=True, text=True, timeout=900)
    ok = r.returncode == 0 and "bootstrap ok" in r.stdout
    return ok, (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E-цикл сохранения veles (K13)")
    ap.add_argument("--repo", default=os.getcwd(), help="чекаут veles (python -m veles)")
    ap.add_argument("--python", default=sys.executable, help="интерпретатор с зависимостями")
    ap.add_argument("--port", type=int, default=18460)
    ap.add_argument("--keep", action="store_true", help="не удалять tmp-окружение в конце")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    tmp_parent = Path(tempfile.gettempdir())
    data_dir = tmp_parent / f"k13-ingest-e2e-{uuid.uuid4().hex[:8]}"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    (data_dir / "canonical").mkdir(parents=True)
    facade = Facade(args.python, repo, args.port, data_dir)
    print(f"tmp VELES_DATA: {data_dir}; репо: {repo}; порт: {args.port}", flush=True)

    docs = {
        "txt": {
            "filename": "k13-proverka-celikosti.txt",
            "v1": ("K13 проверка целостности сохранения.\n\n"
                   "Квантомассовый кворум-форж стабилизатора настроен на 42 контура "
                   "и не теряет ни одного узла при загрузке.\n"),
            "v2": ("K13 проверка целостности сохранения, вторая редакция.\n\n"
                   "Нумератор-гамма заменяет кворум-форж и подтверждает полную "
                   "ревизию документа без потерь узлов.\n"),
            "marker_v1": "кворум-форж",
            "marker_v2": "нумератор-гамма",
            "mode": "multipart-txt",
        },
        "md": {
            "filename": None,  # JSON-загрузка: имя из slug заголовка
            "title": "K13 markdown документ",
            "v1": ("# K13 markdown\n\nОктопус-транспайлер ревизионного контура "
                   "сохраняет каждый абзац в канон без потерь и дублей.\n"),
            "marker_v1": "октопус-транспайлер",
            "mode": "json-md",
        },
        "pdf": {
            "filename": "k13-integrity-quantumforge.pdf",
            "v1": None,  # байты минимального PDF генерируются ниже
            "marker_v1": "quantumforge",
            "mode": "multipart-pdf",
        },
    }
    docs["pdf"]["content"] = minimal_pdf([
        "Quantumforge ledger persistence paragraph one describing stored nodes.",
        "Quantumforge ledger persistence paragraph two describing index counts.",
        "Quantumforge ledger persistence paragraph three describing restarts.",
    ])

    doc_ids: dict[str, str] = {}
    doc_id_txt = ""
    try:
        # 1. tmp-окружение и старт фасада
        ok, note = bootstrap_index(args.python, repo, data_dir)
        record("B1", "seed + полный индекс настоящим эмбеддером",
               "bootstrap ok", note or "нет вывода", ok)
        if not ok:
            print("ВЕРДИКТ: КРАСНЫЙ (bootstrap не прошёл)", flush=True)
            return 1
        facade.start()
        record("B2", "старт фасада", "/healthz 200",
               f"pid {facade.proc.pid}, порт {args.port}", True)
        ready = facade.wait_ready()
        record("B3", "пайплайн готов (эмбеддер загружен)", "stats 200",
               "готов" if ready else "таймаут", ready)
        if not ready:
            print("ВЕРДИКТ: КРАСНЫЙ (пайплайн не собрался)", flush=True)
            return 1

        # 2. загрузка txt/md/pdf и NO-LOSS по каждому
        upload_plan = [
            ("C1", "txt", docs["txt"]),
            ("C2", "md", docs["md"]),
            ("C3", "pdf", docs["pdf"]),
        ]
        for item, kind, spec in upload_plan:
            if spec["mode"] == "json-md":
                code, resp = http_json(
                    "POST", f"{facade.base}/v1/documents",
                    {"text": spec["v1"], "title": spec["title"]}, timeout=120)
                job_id, document_id = resp.get("job_id"), resp.get("document_id")
            else:
                boundary = uuid.uuid4().hex
                content = (spec["v1"].encode("utf-8") if kind == "txt" else spec["content"])
                body = multipart({}, "file", spec["filename"], content, boundary)
                code, text = http(
                    "POST", f"{facade.base}/v1/documents",
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    body=body, timeout=120)
                resp = json.loads(text) if text.strip().startswith("{") else {}
                job_id, document_id = resp.get("job_id"), resp.get("document_id")
            if code != 202 or not job_id:
                record(item, f"загрузка {kind}", "202 с job_id",
                       f"HTTP {code}: {str(resp)[:150]}", False)
                continue
            job = wait_job(facade.base, job_id)
            ok = job.get("status") == "done" and bool(document_id)
            record(item, f"джоб {kind} done", "done",
                   f"{job.get('status')} action={job.get('action')} "
                   f"nodes={job.get('nodes')}", ok, job.get("error", "")[:120])
            if not ok:
                continue
            doc_ids[kind] = document_id
            idx, _ = stats_by_doc(facade.base)
            in_index = idx.get(document_id)
            in_canon = canon_node_count(data_dir, document_id)
            # no-loss: в индексе не-root узлы, в каноне на 1 root больше
            ok = in_canon is not None and in_index == in_canon - 1
            record(f"N{item[1:]}", f"NO-LOSS {kind}: индекс == канон-root",
                   f"{in_canon - 1} == {in_canon - 1}",
                   f"индекс {in_index}, канон {in_canon}, джоб {job.get('nodes')}", ok)
            cites = search(facade.base, spec["marker_v1"])
            ok = any(c.get("document_id") == document_id for c in cites
                     if isinstance(c, dict))
            record(f"S{item[1:]}", f"поиск находит {kind}", f"цитата {document_id}",
                   f"цитат {len([c for c in cites if isinstance(c, dict)])}, "
                   f"наш документ: {any(c.get('document_id') == document_id for c in cites if isinstance(c, dict))}",
                   ok)
        doc_id_txt = doc_ids.get("txt", "")

        # 3. ревизия r2 txt: updated, r1 superseded, старый маркер не виден
        if doc_id_txt:
            boundary = uuid.uuid4().hex
            body = multipart({}, "file", docs["txt"]["filename"],
                             docs["txt"]["v2"].encode("utf-8"), boundary)
            code, text = http("POST", f"{facade.base}/v1/documents",
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                              body=body, timeout=120)
            resp = json.loads(text) if text.strip().startswith("{") else {}
            job = wait_job(facade.base, resp.get("job_id", "")) if resp.get("job_id") else {}
            ok = job.get("status") == "done" and job.get("action") == "updated" \
                and job.get("revision_id") == "r2"
            record("R1", "повторная загрузка -> ревизия r2",
                   "done/updated/r2",
                   f"{job.get('status')}/{job.get('action')}/{job.get('revision_id')}", ok,
                   job.get("error", "")[:120])
            # канон: r1 заархивирован superseded, текущая r2
            ddir = data_dir / "canonical" / doc_dir_name(doc_id_txt)
            r1_manifest = ddir / "revision-r1" / "manifest.json"
            cur_manifest = ddir / "manifest.json"
            m1 = json.loads(r1_manifest.read_text(encoding="utf-8")) if r1_manifest.exists() else {}
            mc = json.loads(cur_manifest.read_text(encoding="utf-8")) if cur_manifest.exists() else {}
            r1_nodes = sum(1 for ln in (ddir / "revision-r1" / "nodes.jsonl")
                           .read_text(encoding="utf-8").splitlines() if ln.strip()) \
                if (ddir / "revision-r1" / "nodes.jsonl").exists() else -1
            ok = m1.get("status") == "superseded" and mc.get("revision_id") == "r2" \
                and mc.get("previous_revision") == "r1"
            record("R2", "канон: r1 superseded, текущая r2 с previous_revision=r1",
                   "superseded/r2/r1",
                   f"{m1.get('status')}/{mc.get('revision_id')}/"
                   f"{mc.get('previous_revision')}, узлов r1: {r1_nodes}", ok)
            # индекс: r2 полный, r1 - superseded-узлы, старого в поиске нет
            idx, stats = stats_by_doc(facade.base)
            cur_nodes = canon_node_count(data_dir, doc_id_txt) or 0
            superseded = stats.get("totals", {}).get("superseded_nodes", -1)
            ok = idx.get(doc_id_txt) == cur_nodes - 1 and superseded == r1_nodes - 1
            record("R3", "индекс: r2 полный, r1 не виден (superseded)",
                   f"текущих {cur_nodes - 1}, superseded {r1_nodes - 1}",
                   f"текущих {idx.get(doc_id_txt)}, superseded {superseded}", ok)
            cites_new = search(facade.base, docs["txt"]["marker_v2"])
            ok = any(c.get("document_id") == doc_id_txt for c in cites_new
                     if isinstance(c, dict))
            record("R4", "поиск: новый маркер r2 находится", "цитата есть",
                   f"цитат {len(cites_new)}, наш документ: "
                   f"{any(c.get('document_id') == doc_id_txt for c in cites_new if isinstance(c, dict))}",
                   ok)
            cites_old = search(facade.base, docs["txt"]["marker_v1"])
            ok = not any(c.get("document_id") == doc_id_txt for c in cites_old
                         if isinstance(c, dict))
            record("R5", "поиск: старый маркер r1 больше не виден", "0 цитат нашего документа",
                   f"цитат {len(cites_old)}, наш документ: "
                   f"{any(c.get('document_id') == doc_id_txt for c in cites_old if isinstance(c, dict))}",
                   ok)

            # 4. рестарт фасада: документ на месте (K3/K4 механика)
            facade.stop()
            facade.start()
            ok = facade.wait_ready()
            record("D1", "рестарт фасада", "healthz + stats 200",
                   "готов" if ok else "таймаут", ok)
            idx, _ = stats_by_doc(facade.base)
            ok = idx.get(doc_id_txt) == cur_nodes - 1
            record("D2", "после рестарта: r2 в индексе, узлы те же",
                   f"{cur_nodes - 1}", f"{idx.get(doc_id_txt)}", ok)
            cites = search(facade.base, docs["txt"]["marker_v2"])
            ok = any(c.get("document_id") == doc_id_txt for c in cites
                     if isinstance(c, dict))
            record("D3", "после рестарта: поиск находит", "цитата есть",
                   f"цитат {len(cites)}, наш документ: "
                   f"{any(c.get('document_id') == doc_id_txt for c in cites if isinstance(c, dict))}",
                   ok)

            # 5. DELETE: 0 узлов, trash, L0 удалён, аудит-события
            code, resp = http_json("DELETE", f"{facade.base}/v1/documents/{doc_id_txt}",
                                   timeout=60)
            ok = code == 200 and resp.get("status") == "deleted" \
                and resp.get("nodes", 0) >= cur_nodes - 1
            record("E1", "DELETE документа", "200 status=deleted, nodes>0",
                   f"HTTP {code}: {str(resp)[:150]}", ok)
            idx, _ = stats_by_doc(facade.base)
            ok = doc_id_txt not in idx
            record("E2", "после DELETE: документа нет в индексе", "0 узлов",
                   f"узлов {idx.get(doc_id_txt, 0)}", ok)
            cites = search(facade.base, docs["txt"]["marker_v2"])
            ok = not any(c.get("document_id") == doc_id_txt for c in cites
                         if isinstance(c, dict))
            record("E3", "после DELETE: поиск пуст", "0 цитат",
                   f"цитат {len(cites)}, наш документ: "
                   f"{any(c.get('document_id') == doc_id_txt for c in cites if isinstance(c, dict))}",
                   ok)
            trash_exists = any((data_dir / "trash").glob(f"*-{doc_dir_name(doc_id_txt)}")) \
                if (data_dir / "trash").exists() else False
            canon_gone = not (data_dir / "canonical" / doc_dir_name(doc_id_txt)).exists()
            # L0 живёт по document_id (uploads/<owner>/<файл>), как discard_l0_file
            l0_gone = not (data_dir / "uploads" / doc_id_txt).exists()
            record("E4", "канон в trash, каталога и L0 нет", "trash+нет каталога+нет L0",
                   f"trash: {trash_exists}, каталог: {not canon_gone}, L0: {not l0_gone}",
                   trash_exists and canon_gone and l0_gone)
            audit_log = data_dir / "canonical" / "ingest-audit.jsonl"
            events = audit_log.read_text(encoding="utf-8") if audit_log.exists() else ""
            has_ingest = '"event": "ingest"' in events
            has_delete = '"event": "delete"' in events
            record("E5", "события ingest/delete в ingest-audit.jsonl", "оба есть",
                   f"ingest: {has_ingest}, delete: {has_delete}",
                   has_ingest and has_delete)
    finally:
        facade.stop()
        if args.keep:
            print(f"tmp-окружение оставлено: {data_dir}", flush=True)
        else:
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"tmp-окружение удалено: {data_dir}", flush=True)

    fails = [r for r in results if r["status"] == "FAIL"]
    print("\n=== ИТОГ e2e-цикл сохранения (K13) ===", flush=True)
    for r in results:
        print(f"[{r['status']}] {r['id']} {r['name']}: {r['actual'][:110]}", flush=True)
    print(f"счёт: PASS={len(results) - len(fails)}, FAIL={len(fails)}", flush=True)
    print("ВЕРДИКТ: ЗЕЛЁНЫЙ" if not fails else "ВЕРДИКТ: КРАСНЫЙ", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
