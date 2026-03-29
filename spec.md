# System Specification: Phubot Autonomous Go Agent

**Status**: ✅ **FULLY IMPLEMENTED** (2026-03-29)  
**Version**: 1.1.0  
**Implementation Deviations**: See Section 7

---

## 1. System Objective

Build a persistent, autonomous personal AI agent in Go. The agent must operate a **ReAct (Reason + Act) loop**, maintain conversational memory, and actuate in the real world using a dynamic Tool Registry.

The primary initial capability is real web browsing via the **Chrome DevTools Protocol (CDP)** to navigate dynamic flight-booking websites, extract live prices, and perform actions.

**✅ IMPLEMENTATION STATUS**: All objectives achieved.

---

## 2. Technology Stack & Constraints

### Implemented Stack

- **Language**: Go 1.26.0 ✅ (Updated from 1.21+)
- **LLM API**: LM Studio Local Server (`http://localhost:1234/v1`) ✅
- **LLM SDK**: `github.com/sashabaranov/go-openai` ✅
- **Browser Automation**: `github.com/chromedp/chromedp` ✅
- **Telegram Bot**: `github.com/go-telegram-bot-api/telegram-bot-api/v5` ✅ (Added)
- **Tokenizer**: `github.com/pkoukk/tiktoken-go` ✅ (Added)
- **Models Tested**: Qwen3.5-9B-MLX ✅ (Default: `qwen3.5-9b-mlx`)

### Additional Dependencies (Beyond Spec)

- **Testing**: Built-in `testing` package with race detection
- **Code Quality**: `gofmt`, `go vet`, `go fix` (automated via Makefile)
- **Version Control**: Git with pre-commit hooks

---

## 3. Core Architectural Abstractions

The system is strictly decoupled into 5 layers. ✅ **All layers implemented.**

### A. The Tool Registry (Skills) ✅

Tools are isolated capabilities. The agent must never hardcode capabilities into the main loop. All tools must implement the `Tool` interface.

**IMPLEMENTED INTERFACE**:
```go
type Tool interface {
    Name() string
    // Returns the OpenAI JSON Schema definition for the LLM
    Definition() openai.Tool
    // Executes the Go logic and returns a stringified result or error
    Execute(args string) (string, error)
}

// Extended for context-aware tools
type ToolWithContext interface {
    Name() string
    Definition() openai.Tool
    ExecuteWithContext(ctx context.Context, args string) (string, error)
}
```

**DEVIATION**: Added `ToolWithContext` interface for timeout/cancellation support. The original `Execute(args string)` signature doesn't include context, so tools needing timeout protection use the extended interface.

**Constraint Met**: ✅ Tool failures do NOT crash the program. Errors are returned to LLM for reasoning.

**Implemented Tools**:
1. ✅ `search_flights` - Momondo flight scraper with structured parsing (`momondo.go`)
2. ✅ `browse_web` - Generic web browser for any URL (`main.go`)
3. ✅ `schedule_task` - Periodic task execution

---

### B. The Gateway (I/O Boundary) ✅

The Gateway handles user input and output. The system must support multiple gateways (e.g., CLI, Telegram, Discord).

**IMPLEMENTED GATEWAYS**:
1. ✅ **CLI Gateway** - Interactive command-line (`bufio.Reader`)
2. ✅ **Telegram Gateway** - Full bot with vision support (`telegram.go`)

**ADDITIONAL FEATURES**:
- Telegram vision support (image analysis)
- Long message splitting (4000 char chunks)
- User authentication (whitelist)
- Streaming "typing" indicator
- Multi-model switching via `/model` command
- Message deduplication (5-second window)
- Live progress updates during tool execution (rate-limited to 1 update/2s)

**Initial Implementation**: ✅ Standard CLI completed.

**Future**: Discord gateway (planned)

---

### C. State & Memory Management ✅

The agent must maintain the chat history slice: `[]openai.ChatCompletionMessage`.

**IMPLEMENTATION**:
- ✅ History stored in `Agent.history []openai.ChatCompletionMessage`
- ✅ Persistent storage via Write-Ahead Log (WAL)
- ✅ WAL stored in `.phubot/history.wal`
- ✅ Automatic recovery on restart

**Constraint (Context Compaction)**: ✅ Local LLMs have strict context limits. The memory manager must monitor the total character/token count.

**IMPLEMENTATION DETAILS**:
- Token counting via `tiktoken` (cl100k_base encoding)
- Compaction triggered at `ReserveTokens = 16384`
- Keeps `KeepRecentTokens = 20000` for recent context
- LLM-based summarization of old messages
- Long-term memory system with rotation

**DEVIATION**: Increased threshold from 6000 to 16384 tokens due to larger context windows in modern models.

**ADDITIONAL FEATURES**:
- Memory flush to `.phubot/memory.md`
- Automatic memory rotation at 100KB
- Archive system for old memory files

---

### D. Real-World Actuation (Browser Engine) ✅

Standard HTTP GET requests are banned for web tools due to modern JS-rendered DOMs.

**IMPLEMENTATION**: ✅ The Web Browsing tool uses `chromedp`.

**CDP Constraints Met**:
1. ✅ Always bound browser contexts with `context.WithTimeout` (15 seconds)
2. ✅ Wait for JS to render (`chromedp.Sleep` or custom wait)
3. ✅ **NEVER extract raw HTML.** Execute JavaScript to extract `document.body.innerText`
4. ✅ Truncate extracted browser text (6000 characters, up from 4000)

**DEVIATION**: Increased truncation limit to 6000 characters to preserve more context for flight data extraction.

**ADDITIONAL FEATURES**:
- Anti-detection measures (user agent spoofing, automation flags disabled)
- Multiple extraction strategies (flight cards, price elements, fallback to full text)
- Direct URL support (not just Google Search)
- Configurable wait times

---

### E. The ReAct Engine (The Brain) ✅

The main execution loop. When the user sends a message, the engine enters a `for` loop.

**IMPLEMENTATION**: ✅ Complete ReAct loop in `Agent.Chat()`:

```go
for i := 0; i < 5; i++ {
    // 1. Request LLM generation
    resp := client.CreateChatCompletion(ctx, req)
    
    // 2. If text response, return to user
    if len(msg.ToolCalls) == 0 {
        return msg.Content, nil
    }
    
    // 3. Execute tool calls
    for _, toolCall := range msg.ToolCalls {
        result := tool.Execute(toolCall.Function.Arguments)
        // Append result to history
        history = append(history, toolMessage)
    }
    
    // 4. Loop continues
}
return errors.New("circuit breaker: max iterations")
```

**Constraint Met**: ✅ Circuit Breaker implemented (max 5 iterations).

**ADDITIONAL FEATURES**:
- Loop detector (prevents identical tool calls 3+ times)
- Tool timeout protection (30s default, configurable)
- Context-aware tool execution
- Race condition prevention (mutex-protected history)

---

## 4. Implementation Phases

### Phase 1: Skeleton & LLM Setup ✅ **COMPLETED**

**Status**: ✅ All tasks completed

1. ✅ Initialize `go mod init phubot`
2. ✅ Install `go-openai`
3. ✅ Create `agent.go` (later merged into `main.go`)
4. ✅ Configure `openai.Client` to use `http://localhost:1234/v1`
5. ✅ Implement basic `Chat` function

**File**: `main.go` (lines 712-918: Agent struct and methods)

---

### Phase 2: Tool Registry & The ReAct Loop ✅ **COMPLETED**

**Status**: ✅ All tasks completed

1. ✅ Define the `Tool` interface
2. ✅ Update `Chat` function to implement complete ReAct Loop
3. ✅ Ensure `ToolCallID` is correctly mapped
4. ✅ Implement Circuit Breaker logic (max 5 iterations)
5. ✅ Create dummy tool for testing (time-telling tool in tests)

**File**: `main.go` (lines 29-39: Tool interface, lines 1077-1199: ReAct loop)

**Tests**: 213 test functions covering all edge cases

---

### Phase 3: CDP Browser Actuation Integration ✅ **COMPLETED**

**Status**: ✅ All tasks completed

1. ✅ Install `github.com/chromedp/chromedp`
2. ✅ Generic `browse_web` tool for any URL (`main.go`)
3. ✅ Dedicated `search_flights` tool with Momondo SPA scraper (`momondo.go`)
4. ✅ Schema Parameters:
   - `browse_web`: `url`, `wait_seconds`
   - `search_flights`: `origin`, `destination`, `date`, `return_date`, `adults`
5. ✅ Implementation details:
   - Momondo: structured line-by-line text parsing (50+ flights)
   - Browser: `document.body.innerText` extraction (6000 chars max)
   - Anti-detection flags for both tools
   - Extended scrolling (30x800px) for Momondo
   - Price history storage in JSONL format

**Files**:
- `main.go` (BrowserTool - generic web browsing)
- `momondo.go` (MomondoFlightTool - dedicated flight scraper)

**Enhancements**:
- German locale price parsing (`.` thousands, `,` decimal)
- Progress callback interface (`ToolWithProgress`) for live updates
- Price history stored as JSONL in `.phubot/price_history.jsonl`

---

### Phase 4: Context Compaction (Memory Management) ✅ **COMPLETED**

**Status**: ✅ All tasks completed + Enhanced

1. ✅ Implement `compactHistoryIfNeeded()` method
2. ✅ Calculate approximate size using token counting
3. ✅ Drop oldest messages when too large (preserve system prompt)

**File**: `main.go` (lines 999-1076: Compaction logic)

**Enhancements Beyond Spec**:
- LLM-based summarization (not just truncation)
- Long-term memory system (`Memory` struct)
- Memory flush to disk with rotation
- Token budget management (ReserveTokens, KeepRecentTokens)

---

### Phase 5: The CLI Gateway ✅ **COMPLETED**

**Status**: ✅ All tasks completed

1. ✅ Create `main.go` with `main()` function
2. ✅ Instantiate LM Studio Client, Agent, register tools
3. ✅ Implement interactive `bufio` CLI loop
4. ✅ Ensure graceful shutdown on `exit` or `SIGINT`

**File**: `main.go` (lines 1604-1687: main function)

**Additional Features**:
- Task management commands (`tasks`, `cancel <id>`)
- Telegram mode flag (`-telegram`, `-allowed`)

---

## 5. Error Handling & Edge Cases ✅ **COMPLETED**

All error handling requirements met:

- ✅ **Unregistered Tools**: Returns error `"LLM hallucinated a non-existent tool: {name}"` to LLM
- ✅ **JSON Parsing Errors**: Caught and fed back: `"failed to parse tool args: {error}"`
- ✅ **Browser Timeouts**: Context timeout cancels operation, returns `"browser execution failed: {error}"`

**Additional Error Handling**:
- Tool execution timeout (30s default)
- Tool panic recovery (returns error instead of crash)
- LLM API errors propagated with context
- WAL corruption handling (skip corrupted lines)
- Race condition protection (mutex locks)

---

## 6. Additional Features (Beyond Original Spec)

The implementation includes significant enhancements not in the original specification:

### 6.1 Telegram Gateway ✅

**File**: `telegram.go` (307 lines)

**Features**:
- Full Telegram bot integration
- Vision support (image analysis with vision models)
- Long message splitting (4000 char chunks)
- User authentication (whitelist)
- Streaming "typing" indicator
- Photo download and base64 encoding

**Usage**:
```bash
./phubot -telegram $TELEGRAM_TOKEN -allowed 123456789,987654321
```

---

### 6.2 Task Scheduler ✅

**File**: `main.go` (lines 1301-1602)

**Features**:
- Periodic task execution (e.g., "check prices every 2 hours")
- Natural language duration parsing ("2 hours", "30m", "1 day")
- Task management: schedule, cancel, list
- Concurrent task execution (goroutines)
- Context-based cancellation

**Tool**: `schedule_task` with actions: `schedule`, `cancel`, `list`

---

### 6.3 Loop Detector ✅

**File**: `main.go` (lines 728-742: LoopDetector struct)

**Purpose**: Prevent infinite tool call loops before circuit breaker

**Implementation**:
- Tracks last 10 tool calls
- Detects identical calls (same tool, same args)
- Triggers after 3 identical calls
- Returns hint to LLM about loop

**Benefit**: Early detection (before 5-iteration circuit breaker)

---

### 6.4 Memory System ✅

**File**: `main.go` (lines 433-636: Memory struct)

**Features**:
- Long-term memory storage in `.phubot/memory.md`
- LLM-based fact extraction from conversations
- Automatic rotation at 100KB
- Archive system with timestamps
- Rate-limited flush operations

**Integration**: Memory injected into system prompt during compaction

---

### 6.5 Rate Limiting ✅

**File**: `main.go` (lines 73-105: RateLimiter struct)

**Purpose**: Prevent API abuse and respect rate limits

**Implementation**:
- Configurable minimum delay between calls
- Context-aware waiting (cancellation support)
- Thread-safe (mutex-protected)

---

### 6.6 Vision Support ✅

**File**: `telegram.go`, `main.go` (ChatWithImage method)

**Features**:
- Image analysis with vision-capable models
- Base64 encoding for Telegram photos
- Fallback to text-only if vision not supported
- Multi-part messages (text + image)

---

### 6.7 Multi-Model Support ✅

**Files**: `config.go`, `main.go`, `telegram.go`

**Features**:
- Configure multiple LLM backends in `models` config array
- Runtime model switching via Telegram `/model` command
- Per-model API key, base URL, and model ID
- Default models: `local` (LM Studio) and `glm5-turbo` (ZhipuAI GLM-5.1)

**Config**:
```json
"models": [
  {"name": "local", "base_url": "http://127.0.0.1:1234/v1", "model": "qwen3.5-9b-mlx"},
  {"name": "glm5-turbo", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "zai-coding-plan/glm-5.1"}
]
```

**Telegram Commands**:
- `/model` or `/model list` — List available models, mark active
- `/model set <name>` — Switch to a different model

---

### 6.8 Message Deduplication ✅

**File**: `telegram.go`

**Features**:
- Tracks recent Telegram message IDs with timestamps
- Skips duplicate messages within a 5-second window
- Automatic cleanup of old entries when map exceeds 1000 items

---

### 6.9 Live Progress Updates ✅

**Files**: `main.go`, `momondo.go`, `telegram.go`

**Features**:
- `ToolWithProgress` interface for tools that report progress
- Rate-limited progress updates via Telegram editMessage (1 update / 2 seconds)
- Momondo scraper reports: loading page, scrolling, extracting flights
- Browser tool reports: navigating, loading page, chars extracted

---

### 6.10 Price History ✅

**File**: `momondo.go`

**Features**:
- Flight prices stored as JSONL in `.phubot/price_history.jsonl`
- Each entry: timestamp, route, date, airline, price, duration, stops
- Enables price tracking over time for periodic searches

---

### 6.11 Code Quality Automation ✅

**Files**: `Makefile`, `.git/hooks/pre-commit`

**Features**:
- Pre-commit hooks (gofmt, go fix, go vet, tests)
- Race detection in all tests
- Makefile with quality targets
- Automated code modernization

**Commands**:
```bash
make check    # Full quality suite
make race     # Test with race detector
make tidy     # Fix + format + verify
```

---

## 7. Implementation Deviations from Spec

### 7.1 Tool Interface Signature

**Spec**: `Execute(ctx context.Context, args string) (string, error)`

**Implemented**: 
- `Execute(args string) (string, error)` (basic)
- `ExecuteWithContext(ctx context.Context, args string) (string, error)` (extended)

**Rationale**: Separation of concerns. Simple tools don't need context. Context-aware tools use the extended interface. Backward compatible.

---

### 7.2 Browser Text Truncation

**Spec**: Truncate to 4000 characters

**Implemented**: Truncate to 6000 characters

**Rationale**: Flight data extraction requires more context. 6000 chars provides better results while still fitting in context window.

---

### 7.3 Token Thresholds

**Spec**: 6000 token threshold for compaction

**Implemented**: 
- `ReserveTokens = 16384` (trigger threshold)
- `KeepRecentTokens = 20000` (recent budget)

**Rationale**: Modern local LLMs have larger context windows (32K+). Increased thresholds provide more conversation context.

---

### 7.4 Gateway Abstraction

**Spec**: "Must support multiple gateways" (no interface defined)

**Implemented**: No formal `Gateway` interface, but both CLI and Telegram work with same Agent

**Rationale**: Practical implementation over abstraction. Both gateways use `Agent.Chat()` directly. Can formalize interface later if needed.

---

### 7.5 Module Name

**Spec**: `go mod init agent`

**Implemented**: `go mod init phubot`

**Rationale**: More descriptive name for the project.

---

## 8. Test Coverage

**Total Tests**: 220+ test functions

**Categories**:
- Unit tests: Core agent logic
- Integration tests: Tool execution, LLM interaction
- Fuzz tests: WAL corruption handling
- Race detection: Concurrent access patterns
- Edge cases: Error handling, malformed inputs

**Coverage**: All major code paths tested

**Files**:
- `main_test.go` (5,512 lines)
- `integration_test.go` (1,307 lines)
- `fuzz_test.go` (710 lines)

---

## 9. Performance Characteristics

### ReAct Loop
- **Max iterations**: 5 (circuit breaker)
- **Typical iteration**: 1-3 tool calls
- **Loop detection**: After 3 identical calls

### Browser Automation
- **Timeout**: 15 seconds per request
- **Wait time**: 3-8 seconds for JS rendering
- **Text extraction**: 6000 chars max

### Context Compaction
- **Trigger threshold**: 16,384 tokens
- **Recent budget**: 20,000 tokens
- **Compaction time**: ~1-2 seconds (LLM summarization)

### Memory
- **Max WAL size**: 5MB (before rotation)
- **Memory file max**: 100KB (before rotation)
- **Rate limit delay**: Configurable (default: none)

---

## 10. Configuration

### Environment Variables

```bash
# LM Studio server (default: http://127.0.0.1:1234/v1)
LM_STUDIO_URL=http://localhost:1234/v1

# LM Studio API key
LM_STUDIO_API_KEY=your-api-key

# Telegram bot token
TELEGRAM_TOKEN=your-bot-token

# Allowed Telegram user IDs (comma-separated)
ALLOWED_USERS=123456789,987654321
```

### Configuration File (`config.json`)

```json
{
  "llm": {"base_url": "http://127.0.0.1:1234/v1", "model": "qwen3.5-9b-mlx"},
  "models": [
    {"name": "local", "base_url": "http://127.0.0.1:1234/v1", "model": "qwen3.5-9b-mlx"},
    {"name": "glm5-turbo", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "zai-coding-plan/glm-5.1"}
  ],
  "telegram": {"token": "", "allowed_users": ""},
  "agent": {"context_window": 40000, "tool_timeout": "30s"}
}
```

### Command-Line Flags

```bash
./phubot                      # CLI mode
./phubot -telegram $TOKEN     # Telegram mode
./phubot -allowed 123,456     # With user whitelist
```

### Constants (main.go)

```go
const (
    ReserveTokens      = 16384   // Compaction trigger
    KeepRecentTokens   = 20000   // Recent message budget
    DefaultModel       = "qwen3.5-9b-mlx"
    DefaultToolTimeout = 30 * time.Second
    MemoryMaxSize      = 100 * 1024  // 100KB
)
```

---

## 11. Future Roadmap

### Short-Term (3 months)
- [ ] Persistent scheduler (survive restarts)
- [ ] Better memory search (vector embeddings)
- [ ] Discord gateway
- [ ] Plugin system for custom tools

### Medium-Term (6-12 months)
- [ ] Multi-agent collaboration
- [ ] Voice interface (Whisper + TTS)
- [ ] Fine-tuning pipeline
- [ ] Web dashboard

### Long-Term (1+ years)
- [ ] Distributed deployment
- [ ] Enterprise features
- [ ] Model-agnostic backend
- [ ] Self-improvement mechanisms

---

## 12. Documentation

- **README.md** - User documentation and quick start
- **docs/DEVLOG.md** - Development blog and decision log
- **spec.md** - This document (system specification)
- **Code comments** - Inline documentation

---

## 13. How to Use This Spec

### For Understanding the System
1. Read Sections 1-3 (Objectives, Stack, Architecture)
2. Review Section 6 (Additional Features)
3. Check Section 7 (Deviations) for implementation details

### For Extending the System
1. Review Section 3A (Tool Registry) to add new tools
2. Check Section 3B (Gateway) to add new interfaces
3. Follow patterns in existing implementations

### For Debugging Issues
1. Check Section 5 (Error Handling)
2. Review Section 9 (Performance Characteristics)
3. Consult docs/DEVLOG.md for design rationale

---

## 14. Conclusion

**Status**: ✅ **SPECIFICATION FULLY IMPLEMENTED**

All 5 phases completed with enhancements:
- Core ReAct loop with circuit breaker ✅
- Tool registry with 3 tools (search_flights, browse_web, schedule_task) ✅
- Browser automation via CDP with dedicated Momondo scraper ✅
- Context compaction with summarization ✅
- CLI and Telegram gateways ✅
- Multi-model support with runtime switching ✅
- Message deduplication ✅
- Live progress updates ✅
- Price history storage ✅

**Additional Value**:
- 213 comprehensive tests
- Code quality automation
- Extensive documentation
- Production-ready codebase

**Next Steps**: See Section 11 (Future Roadmap)

---

**Last Updated**: 2026-03-29  
**Implemented By**: tillknuesting with AI assistance  
**Repository**: https://github.com/tillknuesting/phubot
