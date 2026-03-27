


Here is an extensive `spec.md` designed specifically for an AI coding agent (like Cursor, Claude Engineer, or Aider). 

You can save this as `spec.md` in your project root and simply prompt your coding agent: *"Read `spec.md` and implement Phase 1 and Phase 2."*

***

```markdown
# System Specification: OpenClaw-Style Autonomous Go Agent

## 1. System Objective
Build a persistent, autonomous personal AI agent in Go. The architecture is inspired by **OpenClaw** and **Nous Research's Hermes Agent**. The agent must operate a **ReAct (Reason + Act) loop**, maintain conversational memory, and actuate in the real world using a dynamic Tool Registry. 

The primary initial capability is real web browsing via the **Chrome DevTools Protocol (CDP)** to navigate dynamic flight-booking websites, extract live prices, and perform actions.

## 2. Technology Stack & Constraints
- **Language**: Go (1.21+)
- **LLM API**: LM Studio Local Server (`http://localhost:1234/v1`).
- **LLM SDK**: `github.com/sashabaranov/go-openai` (Overridden to point to LM Studio).
- **Browser Automation**: `github.com/chromedp/chromedp` (For CDP control).
- **Models Expected**: Tool-calling tuned local models (e.g., Hermes-2-Pro, Qwen-2.5-Coder).

---

## 3. Core Architectural Abstractions

The system is strictly decoupled into 5 layers. The coding agent MUST adhere to these boundaries.

### A. The Tool Registry (Skills)
Tools are isolated capabilities. The agent must never hardcode capabilities into the main loop. All tools must implement the `Tool` interface.
```go
type Tool interface {
	Name() string
	// Returns the OpenAI JSON Schema definition for the LLM
	Definition() openai.Tool
	// Executes the Go logic and returns a stringified result or error
	Execute(ctx context.Context, args string) (string, error)
}
```
**Constraint**: Tool failures must NOT crash the program. The error string must be returned to the LLM so it can reason about the failure and try again.

### B. The Gateway (I/O Boundary)
The Gateway handles user input and output. The system must support multiple gateways (e.g., CLI, Telegram, Discord). 
**Initial Implementation**: Standard CLI (`os.Stdin` / `os.Stdout`).

### C. State & Memory Management
The agent must maintain the chat history slice: `[]openai.ChatCompletionMessage`.
**Constraint (Context Compaction)**: Local LLMs have strict context limits. The memory manager must monitor the total character/token count. If history exceeds a threshold (e.g., 6000 tokens), older messages must be summarized or truncated, preserving the System Prompt.

### D. Real-World Actuation (Browser Engine)
Standard HTTP GET requests are banned for web tools due to modern JS-rendered DOMs. 
The Web Browsing tool MUST use `chromedp`.
**CDP Constraints**:
1. Always bound browser contexts with `context.WithTimeout` (e.g., 15 seconds).
2. Wait for JS to render (`chromedp.Sleep` or `chromedp.WaitVisible`).
3. **NEVER extract raw HTML.** Execute JavaScript to extract `document.body.innerText` to save context window space.
4. Truncate extracted browser text to a maximum of 4000 characters before returning it to the LLM.

### E. The ReAct Engine (The Brain)
The main execution loop. When the user sends a message, the engine enters a `for` loop:
1. Append User Message to history.
2. Request LLM generation (passing History + Tool Definitions).
3. If LLM returns normal text -> Append to history, return to User, break loop.
4. If LLM returns `ToolCalls` -> 
    - Append LLM's ToolCall message to history.
    - Execute the requested Go `Tool.Execute()`.
    - Append the result as `Role: "tool"` with matching `ToolCallID`.
    - Loop continues to Step 2.
**Constraint**: Implement a Circuit Breaker. The loop must abort after `MaxIterations` (e.g., 5) to prevent infinite loops.

---

## 4. Implementation Phases

Coding Agent, please execute the following phases sequentially:

### Phase 1: Skeleton & LLM Setup
1. Initialize `go mod init agent`.
2. Install `go-openai`.
3. Create `agent.go`. Implement the `Agent` struct holding `history` and `tools map[string]Tool`.
4. Configure the `openai.Client` to use `http://localhost:1234/v1` (LM Studio).
5. Implement the basic `Chat` function (without the ReAct loop yet) to ensure standard chat works.

### Phase 2: Tool Registry & The ReAct Loop
1. Define the `Tool` interface.
2. Update the `Chat` function in `agent.go` to implement the complete ReAct Loop (Concept E).
3. Ensure the `ToolCallID` is correctly mapped when appending tool results to `history`.
4. Implement the Circuit Breaker logic (max 5 iterations).
5. Create a `dummy_tool.go` (e.g., a simple time-telling tool) to verify the LLM can trigger and read tool calls.

### Phase 3: CDP Browser Actuation Integration
1. Install `github.com/chromedp/chromedp`.
2. Create `browser_tool.go` implementing `Tool`.
3. Name: `browser_search_flights`.
4. Schema Parameters: `origin` (string), `destination` (string), `date` (string).
5. Implementation details for `Execute()`:
   - Parse JSON args.
   - Construct URL: `https://www.google.com/search?q=flights+from+{origin}+to+{destination}+on+{date}`
   - Launch `chromedp` context with a 15-second timeout.
   - Navigate, wait 3 seconds, extract `innerText`.
   - Truncate result to 4000 characters and return.

### Phase 4: Context Compaction (Memory Management)
1. Implement a method in the `Agent` struct: `func (a *Agent) CompactHistoryIfNeeded()`.
2. Calculate the approximate size of `a.history` before sending requests to the LLM.
3. If the history is too large, drop the oldest user/assistant message pairs (excluding the system prompt and recent tool calls) to keep the context window safe for LM Studio.

### Phase 5: The CLI Gateway
1. Create `main.go`.
2. Instantiate the configured LM Studio Client, the Agent, and register the `browser_search_flights` tool.
3. Implement an interactive `bufio` CLI loop allowing the user to type prompts.
4. Ensure graceful shutdown on `exit` or `SIGINT`.

---

## 5. Error Handling & Edge Cases
- **Unregistered Tools**: If the LLM hallucinates a tool name, the framework must catch this, return a string error `"Tool [Name] not found"`, and feed it back to the LLM as a Tool Role message. Do not panic.
- **JSON Parsing Errors**: Local LLMs occasionally output malformed JSON tool arguments. Catch `json.Unmarshal` errors and feed them back: `"Invalid JSON args provided: {error}"`.
- **Browser Timeouts**: If `chromedp` hangs, the context timeout must cancel the operation. Return `"Browser timed out navigating to URL"` to the LLM.
```

### How to use this:
1. Save the above text into a file named `spec.md`.
2. Open your preferred AI coding environment (like Cursor, or attach it to Claude).
3. Say: *"Read `spec.md`. Please complete Phase 1 and Phase 2."*
4. Once it finishes, say: *"Great, now implement Phase 3 and 4."* 

By giving the agent these explicit constraints (especially the LM Studio overrides and the CDP `innerText` truncation rule), you prevent the AI from making common beginner mistakes with local agent frameworks!
