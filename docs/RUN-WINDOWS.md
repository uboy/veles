# Запуск на Windows из исходников (проверенный путь)

Дата: 2026-10-08. Весь путь ниже исполнен на Windows 11, Python 3.11,
без Docker: от пустой папки до ответа по HTTP.

Нужно: Python 3.11 или новее (python.org, при установке отметить
«Add python.exe to PATH») и Git.

## 1. Получить код и установить

```powershell
git clone https://github.com/uboy/veles.git
cd veles
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[st,faiss]"
```

`st` - поиск по смыслу (при первом запуске скачает модель ~450 МБ
в C:\Users\<вы>\.cache\huggingface), `faiss` - быстрый поиск по векторам.

## 2. Загрузить первые документы

Документы - файлы .txt и .md. Положите их в папку корпуса в подкаталог
`05_vyshka_academic\lectures` (сейчас загрузчик смотрит только туда;
путь задаётся переменной VELES_CORPUS_ROOT):

```powershell
mkdir C:\veles-data\corpus\05_vyshka_academic\lectures
copy "C:\мои-документы\*.md" C:\veles-data\corpus\05_vyshka_academic\lectures\

$env:VELES_DATA = "C:\veles-data"
$env:VELES_CORPUS_ROOT = "C:\veles-data\corpus"
.venv\Scripts\veles ingest --no-manifest-check   # файлы -> внутренний формат
.venv\Scripts\veles index                        # сборка индекса поиска
```

После этого добавлять новые файлы можно по одному, без пересборки:

```powershell
.venv\Scripts\veles add "C:\мои-документы\заметка.md" --title "Заметка"
```

Проверка: `.venv\Scripts\veles ask "ваш вопрос"` - покажет найденное место
и цитату. Состояние базы: `veles status`.

## 3. Запустить сервер

```powershell
.venv\Scripts\veles serve --port 8400
```

Сервер отвечает на http://127.0.0.1:8400 (проверка: /healthz). Вопрос по
HTTP (файлом, чтобы не бороться с кавычками PowerShell):

```powershell
'{"query": "ваш вопрос"}' | Set-Content q.json
curl.exe -s -X POST http://127.0.0.1:8400/v1/knowledge/query `
  -H "Content-Type: application/json" --data-binary "@q.json"
```

В ответе: решение (answer - есть ответ), текст ответа и цитаты с указанием
документа и строк. Документ можно открыть целиком:
`/v1/source/<документ>`.

## 4. Подключить языковую модель для составления ответов

Без этого шага база отвечает цитатами из документов - этого достаточно
для поиска. Чтобы сервер составлял связный ответ, укажите модель по
OpenAI-совместимому протоколу (переменные окружения перед запуском):

```powershell
$env:VELES_LLM_BASE_URL = "http://<адрес-модели>/v1"   # например llama-server
$env:VELES_LLM_MODEL    = "gemma4-31b"                 # имя модели на том сервере
$env:VELES_LLM_API_KEY  = "ключ, если требуется"
.venv\Scripts\veles serve --port 8400
```

Подойдёт и облачный API. Помните: вопросы и найденные цитаты уходят на
тот адрес, где стоит модель.

## 5. Веб-чат (по желанию)

Веб-интерфейс - отдельная программа Open WebUI. Проще всего через
Docker Desktop: контейнер open-webui на порту 8300, в его настройках
добавляется OpenAI-совместимое подключение к http://127.0.0.1:8400/v1.
Готовая интеграция (функция veles_knowledge) лежит в
integrations/openwebui/.

## Примечания

- Сервер по умолчанию слушает только 127.0.0.1 (доступ с этого же
  компьютера). Чтобы открыть базу другим устройствам в сети, нужен
  VELES_API_TOKEN: без него сервер с внешним адресом не стартует
  (защита от «открыл базу всей сети, забыв про пароль»).
- Сколько это занимает: индекс на 127 тыс. фрагментов - 3.9 ГБ памяти,
  поиск 193 мс (замеры в docs/REQUIREMENTS.md).
- Есть и установщик deploy/kit/windows/install.ps1 (профили lite и
  personal) - но он не создаёт начальный индекс и ссылается на
  отсутствующий run-facade.ps1; до починки рабочим является путь выше.
