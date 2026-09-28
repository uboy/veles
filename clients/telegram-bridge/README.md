# Telegram-мост dm_chat_ai_helper

Персональный Telegram-бот поверх Open WebUI (фаза Ф4, карточка corp-ai-kb).
Тонкий long-polling клиент: весь интеллект (модели, маршрутизация) - в Open WebUI,
бот только держит диалог и ходит в его API.

- Base-модель - GLM через z.ai coding plan (соединение в Open WebUI).
- `/kb <вопрос>` - вопрос в базу знаний (модель `veles-knowledge` фасада Veles
  через то же соединение Open WebUI).
- `/new` - сброс истории чата; история держится в памяти процесса (последние
  `BRIDGE_HISTORY` сообщений, дефолт 16).
- Персональный режим: при `PAIR_MODE=1` первый `/start` спаривает бота с чатом
  (state file), сообщения остальных чатов игнорируются и логируются.
- **Форматирование** (перенос пайплайна донора `telegram_bot_ai`): ответ модели
  конвертируется markdown → Telegram-HTML (`tg_format.py`: заголовки/**bold**/
  *italic*/списки → •/code-блоки с экранированием) и отправляется с
  `parse_mode="HTML"`; длинные тексты режутся по абзацам (лимит 3400 plain);
  отказ парсинга Telegram → тот же кусок отправляется plain-fallback'ом.
- **Provenance**: ответ базовой модели подписывается источником — «ответ общей
  модели {model}, без обращения к базе знаний» (`BRIDGE_SOURCE_FOOTER`, пустая
  строка выключает). Ответы `/kb` содержат блок «Источники:» от фасада Veles.

## Переменные окружения (только EnvironmentFile, не в репозитории)

| Переменная | Обязательна | Смысл |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | да | токен бота от @BotFather |
| `OPENWEBUI_BASE_URL` | нет | URL Open WebUI (дефолт `http://127.0.0.1:8300`) |
| `OPENWEBUI_API_KEY` | да | API-ключ пользователя Open WebUI (Settings → Account) |
| `OPENWEBUI_MODEL` | нет | id модели по умолчанию (пусто - дефолт WebUI) |
| `OPENWEBUI_KB_MODEL` | нет | id модели базы знаний для `/kb` |
| `VELES_BASE_URL` | нет | фасад Veles для /stats /sources /toc (дефолт `http://127.0.0.1:8400`) |
| `VELES_API_TOKEN` | нет | bearer фасада (тот же `VELES_API_TOKEN` из veles-facade.env) |
| `BRIDGE_SYSTEM_PROMPT` | нет | системный промпт общего чата |
| `TG_PROXY` | нет | HTTPS/SOCKS прокси для Telegram API (дефолт - напрямую) |
| `BRIDGE_STATE_FILE` | нет | файл pairing-state (дефолт `bridge-state.json` рядом) |
| `BRIDGE_HISTORY` | нет | глубина истории общего чата |
| `BRIDGE_SOURCE_FOOTER` | нет | HTML-подпись provenance на ответах базовой модели; пустая = выключить |
| `PAIR_MODE` | нет | `1` (дефолт) - спарить первого `/start` |

Зависимости: `python >= 3.11`, `httpx` (прокси вида `socks5h://` требует дополнительно `httpx[socks]`).

Боевой запуск на стенд и все файлы юнитов - `docs/DEPLOY-стенд.md` в корне репозитория.

## Трекер (задачи и заметки)

Полное описание - архитектура, структура хранения, формат карточек, напоминалки, интеграции: [TRACKER.md](TRACKER.md).
