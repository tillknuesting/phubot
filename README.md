# Phubot - Personal AI Assistant

A persistent, autonomous personal AI agent in Go inspired by OpenClaw and Nous Research's Hermes Agent.

## Overview

Phubot operates a **ReAct (Reason + Act) loop**, maintains conversational memory, and actuates in the real world using a dynamic Tool Registry. The primary capability is real web browsing via Chrome DevTools Protocol (CDP) to navigate dynamic websites and extract live information.

## Features

### Core Capabilities
- **ReAct Loop**: Reason + Act execution cycle with circuit breaker (max 5 iterations)
- **Tool Registry**: Dynamic plugin system for extending capabilities
- **Browser Automation**: Real Chrome/Chromium control via CDP (chromedp)
- **Persistent Memory**: Write-Ahead Log (WAL) for conversation history
- **Context Compaction**: Automatic token management for local LLMs
- **Loop Detection**: Prevents infinite tool call cycles

### Communication Gateways
- **CLI**: Interactive command-line interface
- **Telegram**: Full bot support with vision capabilities

### Advanced Features
- **Scheduler**: Periodic task execution (e.g., "check prices every 2 hours")
- **Rate Limiting**: API call throttling
- **Vision Support**: Image analysis with vision-capable models
- **Memory Rotation**: Automatic archival of long-term memory

## Technology Stack

- **Language**: Go 1.26
- **LLM API**: LM Studio Local Server (`http://localhost:1234/v1`)
- **LLM SDK**: `github.com/sashabaranov/go-openai`
- **Browser Automation**: `github.com/chromedp/chromedp`
- **Telegram**: `github.com/go-telegram-bot-api/telegram-bot-api/v5`
- **Tokenizer**: `github.com/pkoukk/tiktoken-go`

**Recommended Models**: Tool-calling tuned local models (e.g., Hermes-2-Pro, Qwen-2.5-Coder, Qwen3.5-9B-MLX)

## Quick Start

### Prerequisites

1. **Go 1.26+**
   ```bash
   go version
   ```

2. **LM Studio** with a tool-calling model
   - Download from: https://lmstudio.ai/
   - Load a model like `Qwen2.5-Coder-7B` or `Hermes-2-Pro-Llama-3`
   - Start local server on port 1234

3. **Chrome/Chromium** (for browser automation)
   ```bash
   # macOS
   brew install chromium
   
   # Linux
   sudo apt install chromium-browser
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/tillknuesting/phubot.git
cd phubot

# Install dependencies
make install

# Build binary
make build
```

### Usage

#### CLI Mode

```bash
# Run interactive CLI
./phubot

# Or use make
make run
```

#### Telegram Mode

```bash
# Set your Telegram bot token
export TELEGRAM_TOKEN="your-bot-token-here"

# Run with Telegram
./phubot -telegram $TELEGRAM_TOKEN

# Or use make
make run-telegram
```

#### Example Interactions

```
You: Check flight prices from HKT to FRA on 2026-04-15
Agent: [Uses browser_search_flights tool]
       I found several flights starting at €542...

You: Check this every 2 hours
Agent: [Uses schedule_task tool]
       I've scheduled this task to run every 2 hours.

You: What scheduled tasks do you have?
Agent: You have 1 active task:
       - "flight-check" running every 2h (last run: 10min ago)
```

## Architecture

Phubot follows a strict 5-layer architecture:

### 1. Tool Registry (Skills)
Tools are isolated capabilities implementing the `Tool` interface:
```go
type Tool interface {
    Name() string
    Definition() openai.Tool
    Execute(args string) (string, error)
}
```

### 2. Gateway (I/O Boundary)
Multiple communication channels (CLI, Telegram) with unified agent interface.

### 3. State & Memory Management
- Conversation history: `[]openai.ChatCompletionMessage`
- Context compaction when tokens exceed threshold
- Persistent WAL storage in `.phubot/history.wal`

### 4. Real-World Actuation (Browser Engine)
- Chrome DevTools Protocol via chromedp
- 15-second timeout on all browser operations
- JavaScript extraction of `innerText` (not raw HTML)
- Text truncation to 4000-6000 characters

### 5. ReAct Engine (The Brain)
Main execution loop:
1. Append user message to history
2. Request LLM generation with tools
3. If text response → return to user
4. If tool calls → execute tools, append results, loop to step 2
5. Circuit breaker: abort after 5 iterations

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.

## Development

### Code Quality

Built-in Go tooling with automated checks:

```bash
# Quick checks (fast)
make quick

# Full quality suite
make check

# Modernize and clean
make tidy

# View all options
make help
```

### Pre-Commit Hook

Automatically runs on every commit:
- Code formatting (gofmt)
- Modernization (go fix)
- Static analysis (go vet)
- Quick tests

### Testing

```bash
# Run all tests
make test

# Run with race detector
make race

# Run specific test
go test -run TestChat_ToolCall -v
```

### Project Structure

```
phubot/
├── main.go              # Core agent implementation
├── telegram.go          # Telegram gateway
├── main_test.go         # Unit tests
├── integration_test.go  # Integration tests
├── fuzz_test.go         # Fuzzing tests
├── spec.md              # System specification
├── Makefile             # Build automation
├── .gitignore           # Git ignore rules
└── docs/
    ├── ARCHITECTURE.md  # Architecture decisions
    └── DEVLOG.md        # Development blog/log
```

## Configuration

### Environment Variables

```bash
# LM Studio server (default: http://127.0.0.1:1234/v1)
LM_STUDIO_URL=http://localhost:1234/v1

# Telegram bot token
TELEGRAM_TOKEN=your-bot-token

# Allowed Telegram user IDs (comma-separated)
ALLOWED_USERS=123456789,987654321
```

### Model Configuration

The default model is `qwen3.5-9b-mlx`. Change in `main.go`:
```go
const DefaultModel = "your-model-name"
```

## Available Tools

### `browser_search_flights`
Search for live flight prices using Chrome automation.
- **Parameters**: `origin`, `destination`, `date`, `adults`, `url`, `wait_seconds`
- **Returns**: Extracted flight information (prices, times, airlines)

### `schedule_task`
Schedule periodic task execution.
- **Parameters**: `action` (schedule/cancel/list), `task_id`, `prompt`, `interval`
- **Returns**: Task status and confirmation

## Roadmap

- [ ] Discord gateway
- [ ] Voice interface (Whisper + TTS)
- [ ] File operations tool
- [ ] Email integration
- [ ] Calendar integration
- [ ] Multi-agent collaboration
- [ ] Plugin system for custom tools

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run quality checks (`make check`)
5. Commit (pre-commit hook runs automatically)
6. Push to the branch
7. Open a Pull Request

## Documentation

- [Architecture Decisions](docs/ARCHITECTURE.md) - Why we built it this way
- [Development Log](docs/DEVLOG.md) - Change history and rationale
- [System Specification](spec.md) - Original requirements

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenClaw architecture inspiration
- Nous Research's Hermes Agent
- Go team for excellent tooling
- LM Studio for local LLM serving

## Support

- **Issues**: https://github.com/tillknuesting/phubot/issues
- **Discussions**: https://github.com/tillknuesting/phubot/discussions

---

**Built with ❤️ using Go 1.26**
