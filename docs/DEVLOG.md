# Development Log (DEVLOG)

> **Purpose**: This document serves as a development blog and decision log, recording all architectural choices, implementation rationale, and change history. It is designed to be consumed by both humans and LLMs for understanding the system's evolution.

**Last Updated**: 2026-03-28  
**Version**: 1.0.0  
**Maintainer**: tillknuesting

---

## Table of Contents

- [Project Genesis](#project-genesis)
- [Architecture Decisions](#architecture-decisions)
- [Implementation Log](#implementation-log)
- [Design Patterns](#design-patterns)
- [Lessons Learned](#lessons-learned)
- [Future Considerations](#future-considerations)

---

## Project Genesis

### Initial Vision (2026-03-01)

**Goal**: Build a persistent, autonomous personal AI agent in Go that can browse the web, maintain conversation history, and execute tasks autonomously.

**Inspiration Sources**:
- Modular tool system
- ReAct loop pattern
- Local LLM capabilities (LM Studio)

**Why Go?**
- Fast compilation and execution
- Excellent concurrency primitives (goroutines, channels)
- Strong standard library
- Built-in testing and tooling
- Single binary deployment
- No runtime dependencies

**Why Local LLMs?**
- Privacy (no data leaves the machine)
- Cost (no API fees)
- Control (model selection, fine-tuning)
- Offline capability
- Speed (no network latency)

---

## Architecture Decisions

### ADR-001: ReAct Loop Implementation

**Date**: 2026-03-01  
**Status**: Accepted  
**Decision**: Implement ReAct (Reason + Act) loop with circuit breaker

**Rationale**:
- Separates reasoning from action execution
- Allows multi-step tool orchestration
- Circuit breaker prevents infinite loops (max 5 iterations)
- Mirrors human problem-solving: think → act → observe → repeat

**Implementation**:
```go
for i := 0; i < 5; i++ {
    // 1. Get LLM response
    resp := llm.Chat(history, tools)
    
    // 2. If text response, return to user
    if len(resp.ToolCalls) == 0 {
        return resp.Content
    }
    
    // 3. Execute tool calls
    for _, call := range resp.ToolCalls {
        result := tool.Execute(call.Args)
        history.Append(result)
    }
    
    // 4. Loop back for next iteration
}
```

**Consequences**:
- ✅ Clear execution flow
- ✅ Prevents runaway tool chains
- ⚠️ Complex tasks may need >5 steps (user must break down)
- ⚠️ Requires state management across iterations

---

### ADR-002: Tool Registry Pattern

**Date**: 2026-03-01  
**Status**: Accepted  
**Decision**: Use interface-based tool registry with dynamic registration

**Rationale**:
- Extensibility: Add new tools without modifying core agent
- Isolation: Tool failures don't crash the agent
- Testability: Mock tools for unit testing
- LLM compatibility: OpenAI tool schema integration

**Interface Design**:
```go
type Tool interface {
    Name() string                          // Unique identifier
    Definition() openai.Tool               // JSON Schema for LLM
    Execute(args string) (string, error)   // Execution logic
}

type ToolWithContext interface {
    ExecuteWithContext(ctx context.Context, args string) (string, error)
}
```

**Why Two Interfaces?**
- `Tool`: Simple tools without context needs
- `ToolWithContext`: Tools requiring timeout/cancellation
- Backward compatible: All tools can implement `Tool`
- Context-aware tools get timeout protection

**Tool Registration**:
```go
agent := NewAgent(client, wal)
agent.RegisterTool(&CDPBrowserFlightTool{})
agent.RegisterTool(&SchedulerTool{})
```

**Consequences**:
- ✅ Plugin architecture
- ✅ Easy to add tools
- ✅ Tool isolation
- ⚠️ Requires JSON argument parsing
- ⚠️ Tool naming conflicts must be managed

---

### ADR-003: Write-Ahead Log (WAL) for Persistence

**Date**: 2026-03-02  
**Status**: Accepted  
**Decision**: Use append-only WAL for conversation history

**Rationale**:
- Durability: Survives crashes and restarts
- Simplicity: Append-only is fast and reliable
- Recovery: Can replay history from disk
- Audit: Full conversation history preserved

**Storage Format**:
```
{"role":"system","content":"You are..."}
{"role":"user","content":"Hello"}
{"role":"assistant","content":"Hi! How can I help?"}
{"role":"tool","name":"browser","content":"<results>","tool_call_id":"c1"}
```

**Rotation Strategy**:
- Max size: 5MB (`WALMaxSize`)
- Trim when exceeded
- Keep newest messages
- Preserve system prompt

**Why Not Database?**
- SQLite: Overkill for simple append/read
- Bolt/Badger: Additional dependency
- JSON lines: Human-readable, easy to debug

**Consequences**:
- ✅ Crash-safe
- ✅ Simple implementation
- ✅ Human-readable
- ⚠️ File size grows without rotation
- ⚠️ Concurrent writes need mutex

---

### ADR-004: Context Compaction Strategy

**Date**: 2026-03-02  
**Status**: Accepted  
**Decision**: Automatic context compaction when token count exceeds threshold

**Rationale**:
- Local LLMs have strict context limits (4K-32K tokens)
- Long conversations exhaust context window
- Need to preserve important information while staying within limits

**Compaction Algorithm**:
1. **Token Counting**: Use tiktoken (cl100k_base encoding)
2. **Threshold Check**: If `tokens > reserveTokens`, trigger compaction
3. **Memory Flush**: Extract facts to long-term memory
4. **Summarization**: LLM summarizes old messages
5. **Truncation**: Keep recent messages + summary

**Token Budget**:
```go
ReserveTokens    = 16384  // Trigger compaction at this limit
KeepRecentTokens = 20000  // Budget for recent messages
```

**Why Not Simple Truncation?**
- Loses important context
- Breaks conversation continuity
- No learning from past interactions

**Why LLM Summarization?**
- Preserves key information
- Context compression (1000s → 100s of tokens)
- Extracts facts for long-term memory

**Consequences**:
- ✅ Automatic memory management
- ✅ Preserves important context
- ⚠️ Requires LLM call for summarization
- ⚠️ Summary quality depends on model
- ⚠️ Compaction latency (~1-2s)

---

### ADR-005: Chrome DevTools Protocol (CDP) for Browser Automation

**Date**: 2026-03-02  
**Status**: Accepted  
**Decision**: Use chromedp for browser automation instead of HTTP requests

**Rationale**:
- Modern websites use JavaScript rendering
- HTTP requests miss dynamic content
- CDP controls real Chrome/Chromium browser
- Can handle SPA (Single Page Applications)

**Implementation Details**:
```go
// Anti-detection measures
opts := []chromedp.ExecAllocatorOption{
    chromedp.Flag("headless", false),  // Visible for debugging
    chromedp.Flag("disable-blink-features", "AutomationControlled"),
    chromedp.UserAgent("Mozilla/5.0..."),
}

// Extract text, not HTML
chromedp.Evaluate(`
    document.body.innerText  // NOT innerHTML
`, &result)
```

**Why Not Selenium/Puppeteer?**
- Selenium: Requires external server, heavy
- Puppeteer: Node.js-only
- chromedp: Pure Go, lightweight, fast

**Text Extraction Strategy**:
- Use `innerText` (not `innerHTML`)
- Truncate to 4000-6000 characters
- Save context window space
- Avoid HTML parsing overhead

**Timeout Strategy**:
- 15-second context timeout
- Wait 3-8 seconds for JS rendering
- Cancel on timeout, return error to LLM

**Consequences**:
- ✅ Handles dynamic websites
- ✅ Real browser behavior
- ⚠️ Requires Chrome/Chromium installed
- ⚠️ Slower than HTTP (3-8s per request)
- ⚠️ Memory overhead (browser process)

---

### ADR-006: Rate Limiting and Circuit Breaking

**Date**: 2026-03-02  
**Status**: Accepted  
**Decision**: Implement rate limiter and circuit breaker for API calls

**Rationale**:
- Prevent API abuse (local or external)
- Avoid LLM provider rate limits
- Protect against runaway tool chains
- Graceful degradation under load

**Rate Limiter Implementation**:
```go
type RateLimiter struct {
    mu       sync.Mutex
    minDelay time.Duration
    lastCall time.Time
}

func (r *RateLimiter) Wait(ctx context.Context) error {
    wait := time.Until(r.lastCall.Add(r.minDelay))
    if wait > 0 {
        select {
        case <-time.After(wait):
        case <-ctx.Done():
            return ctx.Err()
        }
    }
    r.lastCall = time.Now()
    return nil
}
```

**Circuit Breaker**:
- Max 5 iterations in ReAct loop
- Prevents infinite tool call chains
- Forces LLM to conclude or error

**Consequences**:
- ✅ API abuse prevention
- ✅ Graceful degradation
- ✅ Predictable behavior
- ⚠️ May delay legitimate requests
- ⚠️ Adds complexity to call paths

---

### ADR-007: Telegram Gateway Implementation

**Date**: 2026-03-03  
**Status**: Accepted  
**Decision**: Implement Telegram bot as alternative to CLI

**Rationale**:
- Mobile access to AI assistant
- Always-on availability
- Rich media support (images, files)
- Familiar interface for users

**Features**:
- Text chat with streaming "typing" indicator
- Image analysis (vision models)
- Long message splitting (4000 char chunks)
- User authentication (whitelist)

**Vision Support**:
```go
func (t *TelegramBot) handleMessage(msg *tgbotapi.Message) {
    if msg.Photo != nil {
        // Download image
        imageBase64 := t.downloadPhotoAsBase64(msg.Photo)
        
        // Use vision model
        reply := t.agent.ChatWithImage(ctx, caption, imageBase64)
        
        // Send response
        t.editMessage(chatID, statusMsgID, reply)
    }
}
```

**Authentication**:
- Optional user ID whitelist
- Unauthorized users rejected
- Per-user conversation isolation (future)

**Consequences**:
- ✅ Mobile access
- ✅ Vision capabilities
- ⚠️ Requires bot token
- ⚠️ Rate limits (Telegram API)
- ⚠️ Always-on process (daemon)

---

### ADR-008: Scheduler for Periodic Tasks

**Date**: 2026-03-03  
**Status**: Accepted  
**Decision**: Implement task scheduler for periodic execution

**Rationale**:
- Users want recurring tasks ("check prices every 2 hours")
- Manual scheduling is tedious
- Native Go implementation (no cron dependency)

**Implementation**:
```go
type Scheduler struct {
    agent   *Agent
    tasks   map[string]*ScheduledTask
    ctx     context.Context
    mu      sync.RWMutex
}

func (s *Scheduler) Schedule(id, prompt string, interval time.Duration) error {
    task := &ScheduledTask{
        ID:       id,
        Prompt:   prompt,
        Interval: interval,
        Active:   true,
    }
    
    go s.runTask(task)  // Spawn goroutine
    s.tasks[id] = task
    return nil
}
```

**Duration Parsing**:
- Natural language: "2 hours", "30 minutes", "1 day"
- Abbreviations: "2h", "30m", "1d"
- Validation: No zero or negative intervals

**Task Management**:
- `schedule`: Create new periodic task
- `cancel`: Stop and remove task
- `list`: Show all tasks with status

**Consequences**:
- ✅ Autonomous task execution
- ✅ Natural language scheduling
- ⚠️ Tasks lost on restart (not persistent)
- ⚠️ No distributed scheduling
- ⚠️ Memory leak if tasks not cancelled

---

### ADR-009: Loop Detection for Tool Calls

**Date**: 2026-03-03  
**Status**: Accepted  
**Decision**: Detect and prevent infinite tool call loops

**Rationale**:
- LLM may repeat same tool call with same args
- Infinite loops waste API calls and time
- Circuit breaker alone insufficient (allows 5 loops)

**Detection Algorithm**:
```go
type LoopDetector struct {
    recentToolCalls []toolCallRecord
    maxHistory      int
    loopThreshold   int
}

func (ld *LoopDetector) detectLoop(toolName, args string) (bool, string) {
    // Count recent occurrences of this exact call
    count := 0
    for _, record := range ld.recentToolCalls {
        if record.name == toolName && record.args == args {
            count++
        }
    }
    
    if count >= ld.loopThreshold {
        return true, "Detected loop: same tool called 3+ times"
    }
    return false, ""
}
```

**Configuration**:
- `maxHistory`: 10 recent tool calls
- `loopThreshold`: 3 identical calls = loop

**Consequences**:
- ✅ Prevents infinite loops
- ✅ Early detection (before circuit breaker)
- ⚠️ May block legitimate retries
- ⚠️ Simple heuristic (not semantic analysis)

---

### ADR-010: Memory System for Long-Term Context

**Date**: 2026-03-03  
**Status**: Accepted  
**Decision**: Implement persistent memory system with rotation

**Rationale**:
- Conversation history is ephemeral
- Important facts should persist across sessions
- Context compaction loses information without storage

**Memory Structure**:
```
.phubot/
├── memory.md           # Current memory (facts, state)
├── memory.md.summary   # Rotated summary
└── archive/
    ├── memory_20260301.md
    └── memory_20260315.md
```

**Flush Strategy**:
1. LLM extracts facts from conversation
2. Append to `memory.md`
3. Rotate when size > 100KB
4. Archive old memory with timestamp
5. Create new summary file

**Integration**:
```go
// During compaction
if a.memory != nil {
    a.memory.Flush(ctx, client, model, history)
    // Memory now contains key facts
    // Inject into system prompt
    newPrompt := a.buildSystemPrompt()
}
```

**Consequences**:
- ✅ Long-term context preservation
- ✅ Cross-session learning
- ⚠️ Requires LLM call for extraction
- ⚠️ Memory file grows without rotation
- ⚠️ No semantic search (plain text)

---

## Implementation Log

### 2026-03-01: Initial Skeleton

**Tasks**:
- [x] Initialize Go module (`go mod init phubot`)
- [x] Install go-openai SDK
- [x] Create agent.go with Agent struct
- [x] Configure LM Studio client (localhost:1234)
- [x] Basic Chat function (no ReAct loop yet)

**Code Written**: ~200 lines  
**Tests**: 5 basic tests  
**Time**: 2 hours

**Challenges**:
- OpenAI SDK expects remote API, needed to override BaseURL
- Figuring out correct message format for tools

**Key Learning**: Test with mock LLM server first, then integrate real LM Studio

---

### 2026-03-02: Tool Registry & ReAct Loop

**Tasks**:
- [x] Define Tool interface
- [x] Implement ReAct loop in Chat()
- [x] Map ToolCallID correctly
- [x] Implement circuit breaker (5 iterations)
- [x] Create dummy_time_tool for testing

**Code Written**: ~400 lines  
**Tests**: 25 tests  
**Time**: 4 hours

**Challenges**:
- Tool results must have matching ToolCallID
- Tool calls can be multiple in one response
- Error handling: tool errors should feed back to LLM, not crash

**Key Learning**: Always append tool results to history before next LLM call

---

### 2026-03-02: CDP Browser Integration

**Tasks**:
- [x] Install chromedp
- [x] Create CDPBrowserFlightTool
- [x] Implement navigation and text extraction
- [x] Add timeout (15s) and truncation (4000 chars)
- [x] Test with Google Flights, Momondo

**Code Written**: ~180 lines  
**Tests**: 8 tests  
**Time**: 3 hours

**Challenges**:
- Anti-bot detection on flight sites
- JS rendering timing (how long to wait?)
- Extracting useful data from noisy page text

**Key Learning**: Use `chromedp.Evaluate` with custom JS to extract structured data, not just `innerText`

---

### 2026-03-02: Context Compaction

**Tasks**:
- [x] Implement token counting with tiktoken
- [x] Create compactHistoryIfNeeded()
- [x] Implement summarization with LLM
- [x] Test with long conversations

**Code Written**: ~150 lines  
**Tests**: 12 tests  
**Time**: 2 hours

**Challenges**:
- Token counting is approximate (different tokenizers)
- Summarization quality varies by model
- When to trigger compaction (threshold tuning)

**Key Learning**: Reserve more tokens than you think (16K buffer for 32K context)

---

### 2026-03-03: CLI Gateway

**Tasks**:
- [x] Create main() with interactive CLI
- [x] Implement bufio reader loop
- [x] Add graceful shutdown on "exit"
- [x] Register tools in main()

**Code Written**: ~100 lines  
**Tests**: 3 tests  
**Time**: 1 hour

**Challenges**:
- None (straightforward)

**Key Learning**: Keep CLI minimal, all logic in Agent

---

### 2026-03-03: Telegram Gateway

**Tasks**:
- [x] Create TelegramBot struct
- [x] Implement message handling
- [x] Add vision support (image download + base64)
- [x] User authentication (whitelist)
- [x] Long message splitting

**Code Written**: ~300 lines  
**Tests**: 10 tests  
**Time**: 3 hours

**Challenges**:
- Telegram API quirks (update polling)
- Image download and encoding
- Long messages need splitting (4000 char limit)

**Key Learning**: Use "typing" indicator for better UX during long LLM calls

---

### 2026-03-03: Scheduler

**Tasks**:
- [x] Create Scheduler struct
- [x] Implement Schedule/Cancel/List
- [x] Parse natural language durations
- [x] Spawn goroutines for periodic execution
- [x] Create SchedulerTool for LLM

**Code Written**: ~200 lines  
**Tests**: 15 tests  
**Time**: 2 hours

**Challenges**:
- Goroutine lifecycle management
- Context cancellation for stopping tasks
- Duration parsing edge cases

**Key Learning**: Always use context for goroutine cancellation

---

### 2026-03-03: Loop Detector

**Tasks**:
- [x] Create LoopDetector struct
- [x] Implement detectLoop()
- [x] Record tool call history
- [x] Integrate into ReAct loop

**Code Written**: ~80 lines  
**Tests**: 8 tests  
**Time**: 1 hour

**Challenges**:
- None (straightforward)

**Key Learning**: Simple heuristics work well for loop detection

---

### 2026-03-03: Memory System

**Tasks**:
- [x] Create Memory struct
- [x] Implement Flush() with LLM extraction
- [x] Add file rotation logic
- [x] Integrate into compaction

**Code Written**: ~150 lines  
**Tests**: 10 tests  
**Time**: 2 hours

**Challenges**:
- File rotation edge cases
- Rate limiting memory flushes
- Extracting useful facts vs noise

**Key Learning**: Use structured prompts for fact extraction

---

### 2026-03-27: Test Suite Expansion

**Tasks**:
- [x] Write comprehensive unit tests
- [x] Add integration tests
- [x] Create fuzz tests for WAL
- [x] Expand to 213 test functions

**Code Written**: ~4500 lines (tests)  
**Tests**: 213 tests  
**Time**: 6 hours

**Challenges**:
- Mocking LLM responses
- Testing concurrent code (race detector)
- Fuzz test design for WAL

**Key Learning**: Table-driven tests scale better than individual test functions

---

### 2026-03-27: Code Quality Automation

**Tasks**:
- [x] Create Makefile with quality targets
- [x] Implement pre-commit hooks
- [x] Add go fix integration
- [x] Configure race detector in tests

**Code Written**: ~200 lines (Makefile, hooks)  
**Tests**: 0 (tooling)  
**Time**: 2 hours

**Challenges**:
- Hook script edge cases
- Go fix modernization conflicts
- Test caching issues

**Key Learning**: Automate everything that can be automated

---

### 2026-03-28: Documentation & Open Source

**Tasks**:
- [x] Write comprehensive README
- [x] Create DEVLOG (this document)
- [x] Document architecture decisions
- [x] Push to GitHub

**Code Written**: ~1000 lines (docs)  
**Tests**: 0 (docs)  
**Time**: 3 hours

**Challenges**:
- Explaining complex architecture simply
- Balancing detail vs readability
- Making docs LLM-friendly

**Key Learning**: Write docs for your future self (and LLM assistants)

---

## Design Patterns

### Pattern 1: Repository Pattern (Tool Registry)

**Purpose**: Decouple tool implementation from agent logic

**Implementation**:
```go
type Agent struct {
    tools map[string]Tool
}

func (a *Agent) RegisterTool(t Tool) {
    a.tools[t.Name()] = t
}
```

**Benefits**:
- Easy to add new tools
- Tools are isolated
- Testable with mocks

---

### Pattern 2: Gateway Pattern (I/O Boundary)

**Purpose**: Separate communication channels from core logic

**Implementation**:
```go
// CLI Gateway
func main() {
    agent := NewAgent(client, wal)
    for {
        text := readUserInput()
        reply := agent.Chat(ctx, text)
        fmt.Println(reply)
    }
}

// Telegram Gateway
func (t *TelegramBot) Run() {
    for update := range t.bot.Updates {
        reply := t.agent.Chat(ctx, update.Message.Text)
        t.sendMessage(reply)
    }
}
```

**Benefits**:
- Same agent, multiple interfaces
- Easy to add new gateways
- Core logic unchanged

---

### Pattern 3: Strategy Pattern (Summarization)

**Purpose**: Allow custom summarization strategies

**Implementation**:
```go
type SummarizeFunc func(ctx context.Context, prompt string, messages []Message) (string, error)

type Agent struct {
    summarizer SummarizeFunc  // Pluggable strategy
}

func (a *Agent) defaultSummarizer(...) { /* LLM summarization */ }
```

**Benefits**:
- Swap summarization method
- Test with mock summarizer
- Future: Different models for summarization

---

### Pattern 4: Circuit Breaker Pattern

**Purpose**: Prevent infinite loops and resource exhaustion

**Implementation**:
```go
for i := 0; i < 5; i++ {
    resp := llm.Chat(history)
    if len(resp.ToolCalls) == 0 {
        return resp.Content
    }
    executeTools(resp.ToolCalls)
}
return errors.New("circuit breaker: max iterations")
```

**Benefits**:
- Predictable resource usage
- Fails fast
- Forces LLM to conclude

---

### Pattern 5: Observer Pattern (Scheduler)

**Purpose**: Notify when tasks complete/fail

**Implementation**:
```go
type ScheduledTask struct {
    ID       string
    Prompt   string
    Interval time.Duration
    RunCount int
    LastRun  time.Time
    Active   bool
}

func (s *Scheduler) runTask(task *ScheduledTask) {
    ticker := time.NewTicker(task.Interval)
    for {
        select {
        case <-ticker.C:
            s.agent.Chat(ctx, task.Prompt)
            task.RunCount++
        case <-s.ctx.Done():
            return
        }
    }
}
```

**Benefits**:
- Decoupled task execution
- Easy monitoring
- Clean shutdown

---

## Lessons Learned

### Technical Lessons

1. **Token Counting is Approximate**
   - Different tokenizers give different counts
   - Always reserve buffer (10-20%)
   - Test with real model, not just estimates

2. **Tool Errors Must Feed Back to LLM**
   - Don't crash on tool failure
   - Return error as tool result
   - Let LLM reason about failure and retry

3. **Context Cancellation is Critical**
   - All goroutines must respect context
   - Prevents goroutine leaks
   - Enables clean shutdown

4. **Browser Automation is Fragile**
   - Websites change frequently
   - Anti-bot measures evolve
   - Need fallback strategies

5. **Local LLMs Have Limits**
   - Tool calling quality varies
   - Context windows are small
   - Summarization can lose detail

---

### Process Lessons

1. **Test Early, Test Often**
   - Write tests alongside code
   - Mock external dependencies
   - Run race detector frequently

2. **Automate Everything**
   - Pre-commit hooks save time
   - CI/CD catches issues early
   - Code quality automation pays off

3. **Document Decisions**
   - Future you will thank present you
   - LLMs need context too
   - ADRs (Architecture Decision Records) are gold

4. **Iterate Incrementally**
   - Start simple, add complexity
   - Refactor when patterns emerge
   - Don't over-engineer early

5. **Real-World Testing Matters**
   - Unit tests don't catch everything
   - Integration tests reveal edge cases
   - Fuzz testing finds crashes

---

## Future Considerations

### Short-Term (Next 3 Months)

1. **Persistent Scheduler**
   - Save tasks to disk
   - Survive restarts
   - SQLite backend?

2. **Better Memory Search**
   - Vector embeddings for facts
   - Semantic search
   - Relevance ranking

3. **Discord Gateway**
   - Similar to Telegram
   - Voice channel support?
   - Server management tools?

4. **Plugin System**
   - Dynamic tool loading
   - Third-party tools
   - Tool marketplace?

---

### Medium-Term (6-12 Months)

1. **Multi-Agent Collaboration**
   - Specialized agents (browser, email, calendar)
   - Inter-agent communication
   - Task delegation

2. **Voice Interface**
   - Whisper for STT
   - TTS for responses
   - Real-time conversation

3. **Fine-Tuning Pipeline**
   - Collect user corrections
   - Fine-tune model on interactions
   - Personalized agent

4. **Web Dashboard**
   - View conversation history
   - Manage scheduled tasks
   - Configure settings

---

### Long-Term (1+ Years)

1. **Distributed Deployment**
   - Run on cloud
   - Multiple users
   - Scalable architecture

2. **Enterprise Features**
   - Role-based access control
   - Audit logging
   - Compliance tools

3. **Model Agnostic**
   - Support multiple LLM backends
   - Switch models per task
   - Ensemble approaches

4. **Self-Improvement**
   - Learn from user feedback
   - Optimize prompts automatically
   - Continuous improvement

---

## Consumption Guide for LLMs

### How to Read This Document

**For Understanding the System**:
1. Read [Project Genesis](#project-genesis) for motivation
2. Review [Architecture Decisions](#architecture-decisions) for design rationale
3. Check [Implementation Log](#implementation-log) for evolution
4. Examine [Lessons Learned](#lessons-learned) for pitfalls

**For Making Changes**:
1. Check relevant ADR (Architecture Decision Record)
2. Review implementation log for context
3. Understand design patterns used
4. Consider future roadmap alignment

**For Debugging Issues**:
1. Check [Lessons Learned](#lessons-learned) for known pitfalls
2. Review ADR for design constraints
3. Check implementation log for recent changes
4. Consider design pattern implications

### Key Concepts to Understand

1. **ReAct Loop**: Reason → Act → Observe → Repeat (max 5 times)
2. **Tool Registry**: Dynamic plugin system for capabilities
3. **WAL**: Append-only conversation history storage
4. **Context Compaction**: Automatic token management
5. **Circuit Breaker**: Prevents infinite loops
6. **Gateway Pattern**: Multiple interfaces (CLI, Telegram) to same agent

### Code Navigation Tips

- **Core Logic**: `main.go` - Agent struct, ReAct loop
- **Tools**: `main.go` - Tool implementations at bottom
- **Persistence**: `main.go` - WAL implementation
- **Telegram**: `telegram.go` - Bot gateway
- **Tests**: `*_test.go` - Comprehensive test coverage

---

## Changelog

### v1.0.0 (2026-03-28)

**Initial Release**

**Features**:
- ReAct loop with circuit breaker
- Tool registry with 2 tools (browser, scheduler)
- WAL persistence for conversation history
- Context compaction with summarization
- Telegram gateway with vision support
- Scheduler for periodic tasks
- Loop detector for tool calls
- Memory system with rotation
- Comprehensive test suite (213 tests)

**Code Quality**:
- Pre-commit hooks
- Automated formatting (gofmt, go fix)
- Race detection in tests
- Makefile for automation

**Documentation**:
- README.md
- DEVLOG.md (this document)
- Architecture decisions
- Code comments

---

## Entry: Momondo Scraping POC Results (2026-03-28)

### ADR: Momondo Scraping Strategy

**Context**: Need to reliably scrape flight data from momondo.de for the `search_flights` tool.

**POC Results** (`cmd/momondo-poc/main.go`):
- **CSS selectors `div.c5NSH-card` all return 0 after React hydration** — SSR class names are replaced
- **`div[class*="card--hidden"]` also returns 0** — hydration removes these
- **XHR interception via prototype patching failed** — installed after page load, wiped on reload
- **BUT: `document.body.innerText` contains ALL flight data in structured text**:
  - Flight times: `14:35 – 20:55`, `19:10 – 21:40`
  - Airlines: `Batik Air`, `AirAsia`, `Malaysia Airlines`, `Firefly`, `Thai Lion Air`
  - Prices: `59 €`, `73 €`, `98 €`, `104 €`, `117 €`
  - Duration: `1:30 Std.`, `5:20 Std.`
  - Stops: `Nonstop`, `1 Stopp` (German UI)
  - Routes: `Phuket International - Kuala Lumpur International`

**Decision**: Parse `innerText` with regex instead of DOM selectors. This is:
1. **Robust** — immune to CSS class name changes
2. **Simple** — no selector maintenance
3. **Complete** — all data present in text

**Body text structure** (each flight is a block):
```
19:10 – 21:40                    # times
Phuket International            # origin
-                               # separator
Kuala Lumpur International      # destination
Nonstop                         # stops
1:30 Std.                       # duration
Batik Air                       # airline
59 €                            # price
/Person                         # per person
Gesamt: 236 €                   # total
```

**Wait strategy**: The search takes ~10-15 seconds to fully load. The page shows a "Flugprognose" bar chart and price summary BEFORE individual results load. We need to wait until body text contains flight result patterns (price + time + airline).

**Headless mode**: Works fine. No captcha observed.

---

**End of Development Log**

*This document is a living record. Update it with every significant change.*
