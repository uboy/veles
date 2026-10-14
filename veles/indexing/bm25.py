"""BM25 (Okapi, k1/b по канону) поверх простой русской токенизации.

Хранение - CSR на чистом numpy (scipy в зависимостях нет): строки - документы,
колонки - термины; data float64 (tf), indices int32 (колонка термина), indptr
int64. Словари tfs v1 (list[dict[str, int]], ~1.2-1.6 ГБ RAM на прода-корпусе
233 775 узлов) заменены плоскими массивами (~0.3 ГБ на том же корпусе).

Диск - bm25.npz (np.savez_compressed) + СИНХРОННЫЙ bm25.json в payload v1:
каталог после любой записи обязан читаться старым кодом без пересборки
(откат на v1 не заперт; дельты загрузок после сохранения живут в overlay.npz,
формат оверлея не менялся). При загрузке bm25.json читается, если npz нет,
и конвертируется в CSR в памяти (полная обратная совместимость: стенды bm1/bm2
поднимаются без пересборки).

Инвариант K4: в add() ids дописывается ПОСЛЕДНИМ; search в другом потоке берёт
n_docs = len(ids) и читает только уже дописанные строки CSR - индекс безопасно
читать, пока воркер очереди загрузок его дополняет. Рост массивов - замена
ссылки на больший массив с копированием префикса: любая версия массива -
валидный префикс данных, поэтому смесь версий у читателя консистентна.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Строчные токены; для среза без стемминга (калибруется в Ф2)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# стартовые ёмкости массивов CSR; растут удвоением - амортизированно O(1) на документ
_INIT_DOC_CAP = 256
_INIT_NNZ_CAP = 8192
_INIT_TERM_CAP = 1024

# обязательные поля bm25.npz
_NPZ_KEYS = ("k1", "b", "avgdl", "doc_lens", "df", "indptr", "indices", "data", "terms", "ids")


class _NpzFormatError(ValueError):
    """Наша ошибка формата bm25.npz (отличаем от чужих ValueError из numpy,
    которые оборачиваем в сообщение с путём файла)."""


def _json_buffer(value) -> np.ndarray:
    """JSON-строка как uint8-массив: unicode-dtype (<U*n) раздувает файл в разы
    на фиксированной ширине, а uint8-буфер читается без allow_pickle."""
    return np.frombuffer(
        json.dumps(value, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
    )


def _json_from_buffer(arr: np.ndarray, name: str, path: Path):
    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
        raise _NpzFormatError(
            f"bm25.npz {path}: поле '{name}' ожидается uint8-буфером JSON, "
            f"получен dtype={getattr(arr, 'dtype', type(arr))}"
        )
    try:
        return json.loads(arr.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise _NpzFormatError(
            f"bm25.npz {path}: поле '{name}' не читается как JSON: {e}"
        ) from e


def _check_field(npz, name: str, dtype, path: Path) -> np.ndarray:
    arr = npz[name]
    if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype):
        raise _NpzFormatError(
            f"bm25.npz {path}: поле '{name}' ожидается dtype={np.dtype(dtype)}, "
            f"получено {getattr(arr, 'dtype', type(arr))}"
        )
    return arr


class Bm25Index:
    """CSR (строки - документы, колонки - термины) + df по колонкам.

    Формула Okapi BM25 и арифметика float64 - те же, что в v1; вклады каждого
    дока суммируются в порядке терминов запроса (повторы запроса учитываются,
    без set). idf считается math.log, как в v1: np.log может отличаться на ULP,
    а результаты search() обязаны совпадать со старой реализацией бит-в-бит.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = float(k1), float(b)
        # K4: ids - маркер видимости документа, дописывается последним;
        # читатели берут n_docs = len(ids)
        self.ids: list[str] = []
        # CSR с запасом ёмкости; заполнено: строки [0, len(ids)),
        # nnz = indptr[len(ids)], термины [0, _n_terms)
        self.doc_lens = np.zeros(_INIT_DOC_CAP, dtype=np.int64)
        self.indptr = np.zeros(_INIT_DOC_CAP + 1, dtype=np.int64)
        self.indices = np.zeros(_INIT_NNZ_CAP, dtype=np.int32)
        self.data = np.zeros(_INIT_NNZ_CAP, dtype=np.float64)
        self.df = np.zeros(_INIT_TERM_CAP, dtype=np.int64)
        # термин -> колонка CSR (= индекс в df); колонки только дописываются
        self.term_to_col: dict[str, int] = {}
        self._n_terms = 0
        self.avgdl = 0.0

    # --- ёмкости: замена ссылки на больший массив, префикс копируется ---
    # (читатель со старой версией массива видит валидный префикс - K4)

    def _ensure_doc_cap(self, needed: int) -> None:
        cap = len(self.indptr) - 1
        if needed <= cap:
            return
        new_cap = max(cap * 2, needed, _INIT_DOC_CAP)
        indptr = np.zeros(new_cap + 1, dtype=np.int64)
        indptr[: cap + 1] = self.indptr[: cap + 1]
        self.indptr = indptr
        doc_lens = np.zeros(new_cap, dtype=np.int64)
        doc_lens[:cap] = self.doc_lens[:cap]
        self.doc_lens = doc_lens

    def _ensure_nnz_cap(self, needed: int) -> None:
        cap = len(self.indices)
        if needed <= cap:
            return
        new_cap = max(cap * 2, needed, _INIT_NNZ_CAP)
        nnz = int(self.indptr[len(self.ids)])
        indices = np.zeros(new_cap, dtype=np.int32)
        indices[:nnz] = self.indices[:nnz]
        self.indices = indices
        data = np.zeros(new_cap, dtype=np.float64)
        data[:nnz] = self.data[:nnz]
        self.data = data

    def _ensure_term_cap(self, needed: int) -> None:
        if needed <= len(self.df):
            return
        new_cap = max(len(self.df) * 2, needed, _INIT_TERM_CAP)
        df = np.zeros(new_cap, dtype=np.int64)
        df[: self._n_terms] = self.df[: self._n_terms]
        self.df = df

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        for nid, text in zip(ids, texts):
            tokens = tokenize(text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # df пересчитывается как в v1: до того, как документ станет видим
            for term in tf:
                col = self.term_to_col.get(term)
                if col is None:
                    col = self._n_terms
                    self._ensure_term_cap(col + 1)
                    self._n_terms = col + 1
                    self.term_to_col[term] = col
                self.df[col] += 1
            self._ensure_doc_cap(len(self.ids) + 1)
            i = len(self.ids)
            start = int(self.indptr[i])
            self._ensure_nnz_cap(start + len(tf))
            self.indices[start : start + len(tf)] = [self.term_to_col[t] for t in tf]
            self.data[start : start + len(tf)] = list(tf.values())
            self.indptr[i + 1] = start + len(tf)
            self.doc_lens[i] = len(tokens)
            # ids дописывается ПОСЛЕДНИМ в итерации (K4): search в другом потоке
            # берёт n_docs = len(ids) и индексирует CSR/doc_lens по i < n_docs -
            # безопасно читать индекс, пока воркер очереди загрузок его дополняет
            self.ids.append(nid)
        n = len(self.ids)
        self.avgdl = float(self.doc_lens[:n].sum() / n) if n else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        # Снимок состояния (K4): n_docs - по len(ids) в момент старта, но не
        # больше строк в снятом indptr; все версии массивов префиксны, поэтому
        # indptr одной версии + data/indices/doc_lens любой (новее) - консистентны
        n_ids = len(self.ids)
        if n_ids == 0:
            return []
        indptr = self.indptr
        n_docs = min(n_ids, len(indptr) - 1)
        indices = self.indices
        data = self.data
        doc_lens = self.doc_lens
        nnz = int(indptr[n_docs])
        k1, b = self.k1, self.b
        avg = self.avgdl or 1
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in tokenize(query):
            col = self.term_to_col.get(term)
            if col is None:
                continue
            df_t = int(self.df[col])
            # idf - math.log, как в v1 (см. докстринг класса про бит-в-бит)
            idf = math.log(1 + (n_docs - df_t + 0.5) / (df_t + 0.5))
            # позиции вхождений термина по всем документам (колонка == col)
            pos = np.nonzero(indices[:nnz] == col)[0]
            if pos.size == 0:
                continue
            docs = np.searchsorted(indptr[: n_docs + 1], pos, side="right") - 1
            tf = data[pos]
            denom = tf + k1 * (1 - b + b * doc_lens[docs] / avg)
            # вклад и порядок суммирования - как в v1: вклады дока складываются
            # в порядке терминов запроса (в строке термин встречается один раз,
            # поэтому += по docs без потерь)
            scores[docs] += idf * tf * (k1 + 1) / denom
        good = np.nonzero(scores > 0)[0]
        if good.size == 0:
            return []
        vals = scores[good]
        # стабильная сортировка по убыванию скорa: при равных скорах - по
        # индексу дока (то же, что list.sort по -score в v1)
        order = np.argsort(-vals, kind="stable")[:top_k]
        return [(self.ids[int(good[j])], float(vals[j])) for j in order]

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        n = len(self.ids)
        nnz = int(self.indptr[n])
        terms: list[str | None] = [None] * self._n_terms
        for term, col in self.term_to_col.items():
            terms[col] = term
        np.savez_compressed(
            dir_path / "bm25.npz",
            k1=np.float64(self.k1),
            b=np.float64(self.b),
            avgdl=np.float64(self.avgdl),
            doc_lens=self.doc_lens[:n],
            df=self.df[: self._n_terms],
            indptr=self.indptr[: n + 1],
            indices=self.indices[:nnz],
            data=self.data[:nnz],
            terms=_json_buffer(terms),
            ids=_json_buffer(self.ids),
        )
        # bm25.json пишется СИНХРОННО с npz (payload ровно формата v1): удалять
        # его нельзя - после записи каталог стал бы односторонним, откат кода
        # на v1 (load_index читает json.loads bm25.json) падал бы FileNotFoundError
        # и закрывался только пересборкой старым кодом (ревью раунд 1). tfs
        # собирается из CSR только на время сохранения - в памяти живого индекса
        # словарей нет; дельты последующих загрузок v1 догонит через overlay.npz.
        indptr_l = self.indptr[: n + 1].tolist()
        cols_l = self.indices[:nnz].tolist()
        vals_l = self.data[:nnz].tolist()
        payload = {
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "doc_lens": self.doc_lens[:n].tolist(),
            "tfs": [
                {terms[c]: int(v) for c, v in zip(cols_l[indptr_l[i] : indptr_l[i + 1]], vals_l[indptr_l[i] : indptr_l[i + 1]])}
                for i in range(n)
            ],
            "df": {terms[c]: int(self.df[c]) for c in range(self._n_terms)},
            "avgdl": self.avgdl,
        }
        (dir_path / "bm25.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: Path) -> "Bm25Index":
        """bm25.npz, если есть; иначе старый bm25.json (конвертация в CSR в памяти)."""
        dir_path = Path(dir_path)
        npz_path = dir_path / "bm25.npz"
        if npz_path.exists():
            return cls._load_npz(npz_path)
        legacy = dir_path / "bm25.json"
        if legacy.exists():
            return cls._load_json(legacy)
        raise FileNotFoundError(
            f"в {dir_path} нет ни bm25.npz, ни bm25.json - индекс не собран? (python -m veles index)"
        )

    @classmethod
    def _load_npz(cls, path: Path) -> "Bm25Index":
        try:
            with np.load(path, allow_pickle=False) as npz:
                missing = [k for k in _NPZ_KEYS if k not in npz]
                if missing:
                    raise _NpzFormatError(
                        f"bm25.npz {path}: нет полей {', '.join(missing)} - файл битый или чужой"
                    )
                k1 = float(npz["k1"])
                b = float(npz["b"])
                avgdl = float(npz["avgdl"])
                ids = _json_from_buffer(npz["ids"], "ids", path)
                terms = _json_from_buffer(npz["terms"], "terms", path)
                if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                    raise _NpzFormatError(f"bm25.npz {path}: ids - не список строк")
                if not isinstance(terms, list) or not all(isinstance(x, str) for x in terms):
                    raise _NpzFormatError(f"bm25.npz {path}: terms - не список строк")
                doc_lens = _check_field(npz, "doc_lens", np.int64, path)
                df = _check_field(npz, "df", np.int64, path)
                indptr = _check_field(npz, "indptr", np.int64, path)
                indices = _check_field(npz, "indices", np.int32, path)
                data = _check_field(npz, "data", np.float64, path)
        except _NpzFormatError:
            raise
        except Exception as e:  # noqa: BLE001 - битый файл -> внятная ошибка, не мусор
            raise ValueError(f"bm25.npz {path} не читается (битый файл?): {e}") from e
        n = len(ids)
        n_terms = len(terms)
        if indptr.ndim != 1 or len(indptr) != n + 1:
            raise ValueError(f"bm25.npz {path}: len(indptr)={len(indptr)} != len(ids)+1={n + 1}")
        if int(indptr[0]) != 0 or bool(np.any(np.diff(indptr) < 0)):
            raise ValueError(f"bm25.npz {path}: indptr не монотонен или не начинается с 0")
        nnz = int(indptr[-1])
        if len(indices) != nnz or len(data) != nnz:
            raise ValueError(
                f"bm25.npz {path}: len(indices)={len(indices)}, len(data)={len(data)} "
                f"!= indptr[-1]={nnz}"
            )
        if len(doc_lens) != n:
            raise ValueError(f"bm25.npz {path}: len(doc_lens)={len(doc_lens)} != len(ids)={n}")
        if len(df) != n_terms:
            raise ValueError(f"bm25.npz {path}: len(df)={len(df)} != len(terms)={n_terms}")
        if nnz and (int(indices.min()) < 0 or int(indices.max()) >= n_terms):
            raise ValueError(f"bm25.npz {path}: indices выходят за диапазон [0, {n_terms})")
        idx = cls(k1=k1, b=b)
        idx.ids = ids
        idx._n_terms = n_terms
        idx.term_to_col = {t: c for c, t in enumerate(terms)}
        idx.df = df
        idx.doc_lens = doc_lens
        idx.indptr = indptr
        idx.indices = indices
        idx.data = data
        idx.avgdl = avgdl
        return idx

    @classmethod
    def _load_json(cls, path: Path) -> "Bm25Index":
        """Старый формат v1 (tfs-список словарей) -> CSR в памяти. Порядок
        терминов в строке - порядок ключей dict (как лежали в json); формула
        поиска от порядка терминов в строке не зависит."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = payload["ids"]
        tfs = payload["tfs"]
        df_dict = payload["df"]
        doc_lens = payload["doc_lens"]
        n = len(ids)
        if not (n == len(tfs) == len(doc_lens)):
            raise ValueError(
                f"bm25.json {path}: длины ids/tfs/doc_lens расходятся "
                f"({n}/{len(tfs)}/{len(doc_lens)})"
            )
        idx = cls(k1=float(payload["k1"]), b=float(payload["b"]))
        idx.avgdl = float(payload["avgdl"])
        idx.ids = list(ids)
        idx._n_terms = len(df_dict)
        idx.term_to_col = {t: c for c, t in enumerate(df_dict)}
        idx.df = np.array(list(df_dict.values()), dtype=np.int64)
        idx.doc_lens = np.asarray(doc_lens, dtype=np.int64)
        try:
            idx.indices = np.array(
                [idx.term_to_col[t] for row in tfs for t in row], dtype=np.int32
            )
            idx.data = np.array([c for row in tfs for c in row.values()], dtype=np.float64)
        except KeyError as e:
            raise ValueError(
                f"bm25.json {path}: термин {e} из tfs отсутствует в df - индекс битый"
            ) from e
        idx.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum([len(row) for row in tfs], dtype=np.int64, out=idx.indptr[1:])
        if len(idx.indices) != int(idx.indptr[-1]):
            raise ValueError(f"bm25.json {path}: nnz не сходится с indptr - индекс битый")
        return idx
