# Phubot

A local AI agent in Go. Talks to LM Studio, browses the web, remembers conversations.

## What it does

- **ReAct loop** — the LLM reasons, calls tools, and loops until it has an answer (max 5 rounds)
- **Browser automation** — controls Chrome via CDP to scrape JS-rendered pages (flight prices, etc.)
- **Persistent memory** — conversation history stored in a WAL, with automatic context compaction
- **Telegram bot** — full gateway with vision support, user auth, and message splitting
- **Task scheduler** — "check prices every 2 hours" actually works

## Setup

You need Go 1.26+, LM Studio with a tool-calling model loaded on port 1234, and Chrome/Chromium for browser tools.

```bash
git clone https://github.com/tillknuesting/phubot.git
cd phubot
make install
make build
```

## Run

```bash
# CLI
./phubot

# Telegram
./phubot -telegram $TELEGRAM_TOKEN -allowed 123456789
```

### Env vars

```bash
LM_STUDIO_URL=http://localhost:1234/v1   # default
TELEGRAM_TOKEN=your-bot-token
ALLOWED_USERS=123456789,987654321        # comma-separated
```

### Model

Default is `qwen3.5-9b-mlx`. Change in `main.go`:
```go
const DefaultModel = "your-model-name"
```

## Tools

**`browser_search_flights`** — Opens Chrome, navigates to a flight search site, extracts prices/times/airlines via `innerText`. Params: `origin`, `destination`, `date`, `adults`, `url`, `wait_seconds`.

**`schedule_task`** — Run a prompt on a recurring interval. Actions: `schedule`, `cancel`, `list`. Params: `action`, `task_id`, `prompt`, `interval`.

## Architecture

```
User → Gateway (CLI/Telegram) → ReAct Engine → LLM (LM Studio)
                                      ↓
                                 Tool Registry
                                ┌─────────────┐
                                │ browser      │
                                │ scheduler    │
                                └─────────────┘
                                      ↓
                              State/Memory (WAL)
```

5 layers: Gateway → ReAct Engine → Tool Registry → Browser Engine → Memory.

## Dev

```bash
make check    # fmt + vet + test + race
make test     # tests only
make race     # tests with race detector
make tidy     # fix + fmt + check
```

## Project structure

```
main.go              Agent, tools, ReAct loop
telegram.go          Telegram gateway
main_test.go         Unit tests
integration_test.go  Integration tests
fuzz_test.go         Fuzz tests
spec.md              Full system spec
Makefile             Build & quality targets
docs/
  ARCHITECTURE.md    Design decisions
  DEVLOG.md          Change log
```

## License

MIT
