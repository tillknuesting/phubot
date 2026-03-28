/*
Package main implements Phubot, a persistent autonomous AI assistant.

Phubot is a personal AI agent that operates a ReAct (Reason + Act) loop,
maintains conversational memory, and actuates in the real world using a
dynamic Tool Registry.

Architecture Overview:

The system is organized into 5 strict layers:
 1. Tool Registry - Isolated capabilities (browser, scheduler, etc.)
 2. Gateway - I/O boundary (CLI, Telegram)
 3. State & Memory - Conversation history and long-term memory
 4. Browser Engine - Real-world actuation via Chrome DevTools Protocol
 5. ReAct Engine - The main reasoning and execution loop

Key Features:
  - ReAct loop with circuit breaker (max 5 iterations)
  - Persistent conversation history via Write-Ahead Log (WAL)
  - Automatic context compaction for local LLMs
  - Tool registry with dynamic registration
  - Browser automation with anti-detection
  - Task scheduler for periodic execution
  - Long-term memory with rotation
  - Telegram gateway with vision support

Usage:

		# CLI mode
		./phubot

		# Telegram mode
		./phubot -telegram $TOKEN -allowed 123456789

	 Configuration:

		config.json         - Main config file (run 'phubot -init' to create)
		LM_STUDIO_API_KEY   - LLM API key (overrides config file)
		LM_STUDIO_URL       - LLM server URL (overrides config file)
		TELEGRAM_TOKEN      - Telegram bot token (overrides config file)
		ALLOWED_USERS       - Comma-separated Telegram user IDs (overrides config file)

		# CLI flags (override everything)
		./phubot -config path/to/config.json
		./phubot -telegram $TOKEN -allowed 123456789

See README.md for complete documentation.
*/
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/chromedp/chromedp"
	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"
)

// ==========================================
// OPENCLAW CONCEPT 1: THE SKILL / TOOL REGISTRY
// ==========================================

// Tool defines the interface for pluggable capabilities.
// All tools must implement this interface to be registered with the Agent.
// Tools are isolated capabilities that can be dynamically added without
// modifying the core agent logic.
type Tool interface {
	// Name returns the unique identifier for this tool.
	// This is used by the LLM to reference the tool in tool calls.
	Name() string

	// Definition returns the OpenAI tool definition with JSON Schema.
	// This describes the tool's parameters to the LLM, enabling it to
	// generate properly structured tool calls.
	Definition() openai.Tool

	// Execute runs the tool's logic with the provided JSON arguments.
	// The args string contains JSON-encoded parameters matching the schema
	// from Definition(). Returns a string result or an error.
	//
	// IMPORTANT: Tool errors should NOT crash the program. Instead, return
	// the error as a string so the LLM can reason about the failure and retry.
	Execute(args string) (string, error)
}

// ToolWithContext is an extended Tool interface that supports context
// for timeout and cancellation. Tools that perform long-running operations
// (like browser automation) should implement this interface instead of Tool.
//
// The Agent will automatically detect and use ExecuteWithContext when available,
// providing timeout protection via the Agent's toolTimeout configuration.
type ToolWithContext interface {
	Name() string
	Definition() openai.Tool

	// ExecuteWithContext runs the tool with context support for cancellation.
	// The context should be respected for all long-running operations.
	// Implementations should select on ctx.Done() to enable graceful cancellation.
	ExecuteWithContext(ctx context.Context, args string) (string, error)
}

// ==========================================
// CONFIGURATION CONSTANTS
// ==========================================

// WAL configuration
var (
	// WALMaxSize is the maximum size of the WAL file before rotation (5MB).
	// When exceeded, older messages are trimmed to keep the file size manageable.
	WALMaxSize = 5 * 1024 * 1024

	// WALDir is the directory where persistent data is stored.
	// Default: .phubot in the current working directory.
	WALDir = ".phubot"

	// WALFile is the filename for the Write-Ahead Log.
	// Contains the full conversation history in JSON Lines format.
	WALFile = "history.wal"
)

// Agent configuration constants
const (
	// DefaultContextWindow is the default context window size in tokens.
	// This should match or be less than your model's actual context window.
	// Default: 40000 tokens (suitable for most modern models, lower than OpenClaw's 200K)
	DefaultContextWindow = 40000

	// ReserveTokensRatio is the ratio of context window that triggers compaction.
	// When tokens exceed (ContextWindow * ReserveTokensRatio), compaction starts.
	// Default: 0.70 (70% of context window, more aggressive than OpenClaw's 0.75)
	ReserveTokensRatio = 0.70

	// KeepRecentTokensRatio is the ratio of context window reserved for recent messages.
	// During compaction, this portion of recent context is always preserved.
	// Default: 0.85 (85% of context window, more aggressive pruning of old messages)
	KeepRecentTokensRatio = 0.85

	// MemoryFlushThreshold is the token count remaining when memory flush triggers.
	// When (CurrentTokens > Reserve - MemoryFlushThreshold), flush memory before compaction.
	// Default: 4000 tokens (give LLM time to write durable notes before context is lost)
	MemoryFlushThreshold = 4000

	// DefaultModel is the LLM model identifier used for chat completions.
	// This should match the model loaded in LM Studio.
	// Default: "qwen3.5-9b-mlx" (Qwen3.5 9B MLX-optimized variant)
	DefaultModel = "qwen3.5-9b-mlx"

	// DefaultToolTimeout is the maximum duration for tool execution.
	// Tools that exceed this timeout return an error to the LLM.
	// Default: 30 seconds (sufficient for most operations including browser)
	DefaultToolTimeout = 30 * time.Second

	// MemoryMaxSize is the maximum size of the memory file before rotation.
	// When exceeded, the memory file is archived and a new one is created.
	// Default: 100KB
	MemoryMaxSize = 100 * 1024

	// CompactionMinDelay is the minimum time between memory flush operations.
	// This prevents excessive LLM calls for fact extraction during rapid conversation.
	// Default: 5 seconds
	CompactionMinDelay = 5 * time.Second

	// SummaryPrompt is the instruction given to the LLM for summarizing old messages.
	// It emphasizes preserving key facts while condensing the conversation.
	SummaryPrompt = "Summarize the key facts, decisions, and context from this conversation. Be concise but preserve important details like names, dates, numbers, and decisions made."

	// MemoryFlushPrompt is the instruction for extracting facts to long-term memory.
	// Facts extracted here persist across conversation sessions in the memory file.
	MemoryFlushPrompt = "Extract all important facts, state, and context from this conversation that should be remembered long-term. Format as bullet points. Include names, numbers, decisions, and any information that would be important to know if the conversation were summarized."
)

// Tool result pruning configuration (Tier 1 - OpenClaw style)
type PruningConfig struct {
	Mode                 string  // "off", "conservative", "aggressive"
	SoftTrimRatio        float64 // Trim when this ratio of context is tool results (0.25 = 25%)
	HardClearRatio       float64 // Clear when this ratio of context is tool results (0.40 = 40%)
	SoftTrimMaxChars     int     // Maximum chars to keep in soft-trimmed results
	SoftTrimHeadChars    int     // Characters to keep from start of result
	SoftTrimTailChars    int     // Characters to keep from end of result
	HardClearPlaceholder string  // Placeholder text when hard-clearing results
}

// Default pruning configuration
var DefaultPruningConfig = PruningConfig{
	Mode:                 "aggressive", // More aggressive than OpenClaw
	SoftTrimRatio:        0.20,         // Trim when 20% of context is tool results
	HardClearRatio:       0.35,         // Clear when 35% is tool results
	SoftTrimMaxChars:     3000,         // Keep max 3000 chars in trimmed results
	SoftTrimHeadChars:    1000,         // Keep first 1000 chars
	SoftTrimTailChars:    1000,         // Keep last 1000 chars
	HardClearPlaceholder: "[Previous tool result cleared to save context]",
}

// ==========================================
// TOKEN COUNTING
// ==========================================

// globalTiktokenEncoding is the shared tokenizer instance for token counting.
// Uses cl100k_base encoding (same as GPT-3.5/4) for approximate token counts.
// Initialized lazily via sync.Once for thread safety.
var globalTiktokenEncoding *tiktoken.Tiktoken
var globalTiktokenOnce sync.Once

// getTiktokenEncoding returns the global tiktoken encoding instance.
// Initializes on first call using cl100k_base encoding (compatible with most modern LLMs).
// Returns nil if initialization fails (fallback to character-based estimation).
func getTiktokenEncoding() *tiktoken.Tiktoken {
	globalTiktokenOnce.Do(func() {
		tke, err := tiktoken.GetEncoding("cl100k_base")
		if err != nil {
			log.Printf("[Tiktoken] Failed to get encoding, using fallback: %v", err)
			return
		}
		globalTiktokenEncoding = tke
	})
	return globalTiktokenEncoding
}

// ==========================================
// RATE LIMITING
// ==========================================

// RateLimiter provides simple rate limiting for API calls.
// Ensures minimum delay between operations to prevent API abuse.
// Thread-safe via mutex protection.
type RateLimiter struct {
	mu       sync.Mutex    // Protects concurrent access
	minDelay time.Duration // Minimum time between calls
	lastCall time.Time     // Timestamp of last successful call
}

// NewRateLimiter creates a new rate limiter with the specified minimum delay.
// Negative delays are treated as zero (no rate limiting).
func NewRateLimiter(minDelay time.Duration) *RateLimiter {
	if minDelay < 0 {
		minDelay = 0
	}
	return &RateLimiter{minDelay: minDelay}
}

// Wait blocks until the minimum delay has elapsed since the last call.
// Returns nil on success, or ctx.Err() if the context is cancelled.
// Thread-safe: can be called from multiple goroutines.
func (r *RateLimiter) Wait(ctx context.Context) error {
	if r.minDelay <= 0 {
		return nil
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	wait := time.Until(r.lastCall.Add(r.minDelay))
	if wait <= 0 {
		r.lastCall = time.Now()
		return nil
	}

	select {
	case <-time.After(wait):
		r.lastCall = time.Now()
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ==========================================
// WAL (Write-Ahead Log) - Persistent Conversation History
// ==========================================

// WAL implements a Write-Ahead Log for durable conversation history storage.
// It provides crash-safe persistence of all chat messages in an append-only format.
//
// Storage Format:
//   - JSON Lines format (one JSON object per line)
//   - Each line is a complete openai.ChatCompletionMessage
//   - File location: .phubot/history.wal
//   - Automatic rotation when size exceeds WALMaxSize (5MB)
//
// Thread Safety:
//   - All operations are protected by mutex
//   - Safe for concurrent use from multiple goroutines
//
// Recovery:
//   - On startup, LoadAll() replays the entire conversation history
//   - Corrupted lines are skipped with a warning log
//   - If file doesn't exist, returns empty slice (new session)
type WAL struct {
	mu   sync.Mutex // Protects concurrent file access
	path string     // Full path to WAL file
}

// OpenWAL creates or opens the Write-Ahead Log in the default location.
// Creates the .phubot directory if it doesn't exist.
// Returns an error if the directory cannot be created.
func OpenWAL() (*WAL, error) {
	dir := WALDir
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}
	path := filepath.Join(dir, WALFile)
	return &WAL{path: path}, nil
}

// Append adds a new message to the Write-Ahead Log.
// The message is serialized to JSON and appended as a new line.
// Automatically trims old messages if file size exceeds WALMaxSize.
//
// Thread-safe: can be called from multiple goroutines.
// Returns an error if file operations fail, but never panics.
func (w *WAL) Append(msg openai.ChatCompletionMessage) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal WAL entry: %w", err)
	}

	f, err := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open WAL for append: %w", err)
	}
	defer f.Close()

	data = append(data, '\n')
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("failed to write WAL entry: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("failed to sync WAL: %w", err)
	}

	w.trimIfTooLarge()
	return nil
}

// LoadAll reads all messages from the Write-Ahead Log.
// Returns the complete conversation history in chronological order.
//
// Behavior:
//   - Returns empty slice if file doesn't exist (new session)
//   - Skips corrupted lines with a warning log (resilient to partial writes)
//   - Uses buffered scanning for large files (up to 1MB per line)
//
// Thread-safe: can be called from multiple goroutines.
func (w *WAL) LoadAll() ([]openai.ChatCompletionMessage, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := os.ReadFile(w.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read WAL: %w", err)
	}

	var messages []openai.ChatCompletionMessage
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // Support large lines
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue // Skip blank lines
		}
		var msg openai.ChatCompletionMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			log.Printf("[WAL] skipping corrupted line: %v", err)
			continue // Skip corrupted entries, don't fail
		}
		messages = append(messages, msg)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan WAL: %w", err)
	}
	return messages, nil
}

func (w *WAL) trimIfTooLarge() {
	info, err := os.Stat(w.path)
	if err != nil || info.Size() <= int64(WALMaxSize) {
		return
	}

	data, err := os.ReadFile(w.path)
	if err != nil {
		return
	}

	lines := bytes.Count(data, []byte{'\n'})
	if lines <= 1 {
		return
	}

	keepFromLine := lines / 2
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	lineIdx := 0
	var kept []byte
	for scanner.Scan() {
		if lineIdx >= keepFromLine {
			kept = append(kept, scanner.Bytes()...)
			kept = append(kept, '\n')
		}
		lineIdx++
	}

	tmpPath := w.path + ".trim"
	if err := os.WriteFile(tmpPath, kept, 0644); err != nil {
		log.Printf("[WAL] failed to write trimmed WAL: %v", err)
		return
	}
	if err := os.Rename(tmpPath, w.path); err != nil {
		os.Remove(tmpPath)
		log.Printf("[WAL] failed to rename trimmed WAL: %v", err)
		return
	}
	log.Printf("[WAL] WARNING: trimmed WAL from %d to %d lines (in-memory history may diverge until next compaction)", lines, lineIdx-keepFromLine)
}

func (w *WAL) Rewrite(messages []openai.ChatCompletionMessage) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	var buf bytes.Buffer
	for _, msg := range messages {
		data, err := json.Marshal(msg)
		if err != nil {
			return fmt.Errorf("failed to marshal WAL entry for rewrite: %w", err)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}

	tmpPath := w.path + ".tmp"
	tmpFile, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("failed to create temp WAL: %w", err)
	}
	if _, err := tmpFile.Write(buf.Bytes()); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to write temp WAL: %w", err)
	}
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to sync temp WAL: %w", err)
	}
	tmpFile.Close()

	if err := os.Rename(tmpPath, w.path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename WAL: %w", err)
	}

	return nil
}

// ==========================================
// OPENCLAW CONCEPT 2: REAL-WORLD ACTUATION (CDP BROWSER)
// ==========================================

type CDPBrowserFlightTool struct{}

func (t *CDPBrowserFlightTool) Name() string { return "browser_search_flights" }

func (t *CDPBrowserFlightTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: "Opens a Chrome browser via CDP to search for live flight prices. Can use direct URL or search via Bing.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url":          map[string]any{"type": "string", "description": "Direct URL to navigate to (optional, takes precedence over search)"},
					"origin":       map[string]any{"type": "string", "description": "Origin city/airport code"},
					"destination":  map[string]any{"type": "string", "description": "Destination city/airport code"},
					"date":         map[string]any{"type": "string", "description": "Date in YYYY-MM-DD format"},
					"adults":       map[string]any{"type": "string", "description": "Number of adults (default 2)"},
					"wait_seconds": map[string]any{"type": "string", "description": "Seconds to wait for page load (default 8)"},
				},
				"required": []string{},
			},
		},
	}
}

func (t *CDPBrowserFlightTool) Execute(args string) (string, error) {
	var parsedArgs map[string]string
	if err := json.Unmarshal([]byte(args), &parsedArgs); err != nil {
		return "", fmt.Errorf("failed to parse tool args: %v", err)
	}

	waitSecs := 8
	if ws, ok := parsedArgs["wait_seconds"]; ok {
		if v, err := strconv.Atoi(ws); err == nil && v > 0 {
			waitSecs = v
		}
	}

	var targetURL string
	if directURL, ok := parsedArgs["url"]; ok && directURL != "" {
		targetURL = directURL
		log.Printf("[Browser] Using direct URL: %s", targetURL)
	} else {
		origin := parsedArgs["origin"]
		dest := parsedArgs["destination"]
		date := parsedArgs["date"]
		adults := parsedArgs["adults"]
		if adults == "" {
			adults = "2"
		}

		if origin != "" && dest != "" && date != "" {
			targetURL = fmt.Sprintf(
				"https://www.momondo.de/flight-search/%s-%s/%s/%sadults?sort=price_a",
				origin, dest, date, adults,
			)
			log.Printf("[Browser] Constructed Momondo URL: %s", targetURL)
		} else {
			query := "flight prices"
			if origin != "" {
				query = fmt.Sprintf("flights from %s", origin)
			}
			if dest != "" {
				query = fmt.Sprintf("%s to %s", query, dest)
			}
			if date != "" {
				query = fmt.Sprintf("%s on %s", query, date)
			}
			targetURL = "https://www.bing.com/search?q=" + url.QueryEscape(query)
			log.Printf("[Browser] Using Bing search: %s", targetURL)
		}
	}

	log.Printf("[Browser] Launching Chrome with anti-detection...")

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", false),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.Flag("disable-extensions", false),
		chromedp.Flag("disable-plugins", false),
		chromedp.Flag("disable-images", false),
		chromedp.Flag("disable-web-security", false),
		chromedp.Flag("disable-infobars", true),
		chromedp.Flag("disable-breakpoints", true),
		chromedp.Flag("enable-automation", false),
		chromedp.Flag("no-first-run", true),
		chromedp.Flag("no-default-browser-check", true),
		chromedp.Flag("disable-popup-blocking", false),
		chromedp.Flag("disable-notifications", false),
		chromedp.Flag("disable-hang-monitor", true),
		chromedp.Flag("disable-prompt-on-repost", true),
		chromedp.Flag("disable-background-timer-throttling", true),
		chromedp.Flag("disable-renderer-backgrounding", true),
		chromedp.Flag("disable-backgrounding-occluded-windows", true),
		chromedp.Flag("disable-ipc-flooding-protection", true),
		chromedp.Flag("disable-component-update", true),
		chromedp.Flag("disable-domain-reliability", true),
		chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
		chromedp.Flag("window-size", "1920,1080"),
		chromedp.Flag("start-maximized", true),
		chromedp.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
		chromedp.WindowSize(1920, 1080),
	)
	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	ctx, cancel := chromedp.NewContext(allocCtx, chromedp.WithLogf(log.Printf))
	defer cancel()

	timeout := time.Duration(waitSecs+15) * time.Second
	ctx, cancelTimeout := context.WithTimeout(ctx, timeout)
	defer cancelTimeout()

	var flightData string

	err := chromedp.Run(ctx,
		chromedp.Navigate(targetURL),
		chromedp.Sleep(2*time.Second),
		chromedp.Evaluate(`
			Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
			Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
			Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'de']});
			window.chrome = {runtime: {}};
		`, nil),
		chromedp.Sleep(1*time.Second),
		chromedp.Sleep(time.Duration(max(waitSecs-3, 2))*time.Second),
		chromedp.Evaluate(`
			(() => {
				const results = [];
				
				const flightCards = document.querySelectorAll('[class*="resultInner"], [class*="flight-card"], [class*="FlightResult"], [class*="ResultEntry"], [data-result-id]');
				
				if (flightCards.length > 0) {
					flightCards.forEach((card, i) => {
						const text = card.innerText || card.textContent;
						if (text && text.trim()) {
							results.push('Flight ' + (i+1) + ':\n' + text.trim().substring(0, 500));
						}
					});
					if (results.length > 0) return 'FLIGHTS FOUND:\n\n' + results.slice(0, 10).join('\n\n');
				}
				
				const priceElements = document.querySelectorAll('[class*="price"], [class*="Price"]');
				if (priceElements.length > 0) {
					priceElements.forEach((el, i) => {
						const text = el.innerText || el.textContent;
						if (text && (text.includes('€') || text.includes('$') || text.includes('£'))) {
							results.push(text.trim().substring(0, 200));
						}
					});
					if (results.length > 0) return 'PRICES FOUND:\n' + results.slice(0, 15).join('\n');
				}
				
				return document.body.innerText.substring(0, 6000);
			})()
		`, &flightData),
	)

	if err != nil {
		return "", fmt.Errorf("browser execution failed: %v", err)
	}

	if flightData == "" {
		chromedp.Run(ctx, chromedp.Evaluate(`document.body.innerText.substring(0, 4000)`, &flightData))
	}

	if len(flightData) > 6000 {
		flightData = flightData[:6000]
	}

	log.Printf("[Browser] Page scraped successfully (%d chars)", len(flightData))

	return "BROWSER EXTRACTED:\n" + flightData, nil
}

// ==========================================
// OPENCLAW CONCEPT 6: MEMORY (Durable State)
// ==========================================
// OPENCLAW CONCEPT 6: MEMORY (Durable State)
// ==========================================

type Memory struct {
	mu        sync.Mutex
	baseDir   string
	maxSize   int
	lastFlush time.Time
	minDelay  time.Duration
}

func NewMemory(baseDir string) *Memory {
	return &Memory{
		baseDir:  baseDir,
		maxSize:  MemoryMaxSize,
		minDelay: CompactionMinDelay,
	}
}

func NewMemoryWithoutRateLimit(baseDir string) *Memory {
	return &Memory{
		baseDir:  baseDir,
		maxSize:  MemoryMaxSize,
		minDelay: 0,
	}
}

func (m *Memory) SetMinDelay(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.minDelay = d
}

func (m *Memory) Flush(ctx context.Context, client *openai.Client, model string, history []openai.ChatCompletionMessage) error {
	if len(history) < 3 {
		return nil
	}

	m.mu.Lock()
	if time.Since(m.lastFlush) < m.minDelay {
		m.mu.Unlock()
		log.Printf("[Memory] Skipping flush - minimum delay not elapsed")
		return nil
	}
	m.lastFlush = time.Now()
	m.mu.Unlock()

	if err := os.MkdirAll(m.baseDir, 0755); err != nil {
		return fmt.Errorf("failed to create memory dir: %w", err)
	}

	serialized := serializeMessages(history)

	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: MemoryFlushPrompt},
			{Role: openai.ChatMessageRoleUser, Content: serialized},
		},
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("memory flush LLM call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("memory flush: LLM returned no choices")
	}

	content := resp.Choices[0].Message.Content
	if content == "" {
		return nil
	}

	entry := fmt.Sprintf("\n## Memory Flush (%s)\n%s\n", time.Now().Format(time.RFC3339), content)

	mainFile := filepath.Join(m.baseDir, "MEMORY.md")
	if err := m.appendWithRotation(mainFile, entry); err != nil {
		return err
	}

	datedFile := filepath.Join(m.baseDir, time.Now().Format("2006-01-02")+".md")
	f, err := os.OpenFile(datedFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open dated memory file: %w", err)
	}
	if _, err := f.WriteString(entry); err != nil {
		f.Close()
		return fmt.Errorf("failed to write dated memory file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("failed to close dated memory file: %w", err)
	}

	return nil
}

func (m *Memory) appendWithRotation(filePath, content string) error {
	info, err := os.Stat(filePath)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to stat memory file: %w", err)
	}

	if err == nil && info.Size() > int64(m.maxSize) {
		if err := m.rotateMemory(filePath); err != nil {
			log.Printf("[Memory] Rotation failed, continuing with append: %v", err)
		}
	}

	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open MEMORY.md: %w", err)
	}
	if _, err := f.WriteString(content); err != nil {
		f.Close()
		return fmt.Errorf("failed to write MEMORY.md: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("failed to close MEMORY.md: %w", err)
	}
	return nil
}

func (m *Memory) rotateMemory(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}

	archiveName := fmt.Sprintf("MEMORY-%s.md", time.Now().Format("2006-01-02-150405"))
	archivePath := filepath.Join(m.baseDir, "archive", archiveName)

	if err := os.MkdirAll(filepath.Dir(archivePath), 0755); err != nil {
		return err
	}

	if err := os.WriteFile(archivePath, data, 0644); err != nil {
		return err
	}

	summaryFile := filePath + ".summary"
	if err := os.WriteFile(summaryFile, fmt.Appendf(nil, "# Memory Summary (rotated %s)\n\nSee archive/%s for full history.\n\n", time.Now().Format(time.RFC3339), archiveName), 0644); err != nil {
		log.Printf("[Memory] Failed to create summary file: %v", err)
	}

	log.Printf("[Memory] Rotated %s to archive/%s (size: %d bytes)", filePath, archiveName, len(data))

	header := fmt.Sprintf("# Memory (rotated %s)\nSee archive/%s for full history.\n", time.Now().Format(time.RFC3339), archiveName)
	if err := os.WriteFile(filePath, []byte(header), 0644); err != nil {
		log.Printf("[Memory] Failed to truncate after rotation: %v", err)
	}

	return nil
}

func (m *Memory) ReadMemory() (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	mainFile := filepath.Join(m.baseDir, "MEMORY.md")
	data, err := os.ReadFile(mainFile)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(data), nil
}

func (m *Memory) CleanupOldArchives(maxAge time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	archiveDir := filepath.Join(m.baseDir, "archive")
	entries, err := os.ReadDir(archiveDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	cutoff := time.Now().Add(-maxAge)
	deleted := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			path := filepath.Join(archiveDir, entry.Name())
			if err := os.Remove(path); err != nil {
				log.Printf("[Memory] Failed to delete old archive %s: %v", path, err)
			} else {
				deleted++
			}
		}
	}

	if deleted > 0 {
		log.Printf("[Memory] Cleaned up %d old archive(s)", deleted)
	}
	return nil
}

// ==========================================
// TOKEN COUNTING & COMPACTION HELPERS
// ==========================================

func serializeMessages(messages []openai.ChatCompletionMessage) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("[%s]: %s\n", m.Role, m.Content))
		if len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				sb.WriteString(fmt.Sprintf("  [tool_call:%s] %s\n", tc.Function.Name, tc.Function.Arguments))
			}
		}
	}
	return sb.String()
}

func countTokens(messages []openai.ChatCompletionMessage) int {
	tke := getTiktokenEncoding()
	if tke == nil {
		total := 0
		for _, m := range messages {
			total += len(m.Content)/4 + 4
			for _, tc := range m.ToolCalls {
				total += len(tc.Function.Name)/4 + len(tc.Function.Arguments)/4 + 4
			}
		}
		return total
	}

	total := 0
	for _, m := range messages {
		tokens := tke.Encode(m.Content, nil, nil)
		total += len(tokens) + 4
		for _, tc := range m.ToolCalls {
			nameTokens := tke.Encode(tc.Function.Name, nil, nil)
			argTokens := tke.Encode(tc.Function.Arguments, nil, nil)
			total += len(nameTokens) + len(argTokens) + 4
		}
	}
	return total
}

func findRecentStart(messages []openai.ChatCompletionMessage, budget int) int {
	if len(messages) == 0 {
		return 0
	}

	tke := getTiktokenEncoding()
	if tke == nil {
		total := 0
		for i := len(messages) - 1; i >= 0; i-- {
			total += len(messages[i].Content)/4 + 4
			if total > budget {
				start := i + 1
				if start >= len(messages) {
					start = len(messages) - 1
				}
				return start
			}
		}
		return 0
	}

	total := 0
	for i := len(messages) - 1; i >= 0; i-- {
		tokens := tke.Encode(messages[i].Content, nil, nil)
		total += len(tokens) + 4
		if total > budget {
			start := i + 1
			if start >= len(messages) {
				start = len(messages) - 1
			}
			return start
		}
	}
	return 0
}

// ==========================================
// OPENCLAW CONCEPT 3: PERSISTENT STATE & MEMORY
// ==========================================

// SummarizeFunc is a pluggable function type for custom summarization strategies.
// This allows different summarization implementations (LLM-based, heuristic, etc.)
// The default implementation uses the LLM to generate summaries.
type SummarizeFunc func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error)

// Agent is the core AI assistant implementing the ReAct (Reason + Act) loop.
// It maintains conversation history, executes tools, and manages memory.
//
// Architecture:
//   - ReAct Loop: Reason → Act (Tool) → Observe → Repeat (max 5 iterations)
//   - Tool Registry: Dynamic capability registration
//   - WAL Persistence: Crash-safe conversation history
//   - Memory System: Long-term fact storage
//   - Context Compaction: Automatic token management
//
// Thread Safety:
//   - All public methods are thread-safe
//   - Protected by sync.RWMutex for concurrent access
//   - Safe to call from multiple goroutines
//
// Usage:
//
//	client := openai.NewClientWithConfig(config)
//	wal, _ := OpenWAL()
//	agent := NewAgent(client, wal)
//	agent.RegisterTool(&MyTool{})
//	reply, _ := agent.Chat(ctx, "Hello")
type Agent struct {
	mu               sync.RWMutex                   // Protects concurrent access to agent state
	client           *openai.Client                 // OpenAI-compatible client (LM Studio)
	tools            map[string]Tool                // Registry of available tools
	history          []openai.ChatCompletionMessage // Conversation history
	wal              *WAL                           // Write-Ahead Log for persistence
	memory           *Memory                        // Long-term memory system
	model            string                         // LLM model identifier
	contextWindow    int                            // Total context window in tokens
	reserveTokens    int                            // Token threshold for compaction trigger (calculated)
	keepRecentTokens int                            // Token budget for recent messages (calculated)
	summarizer       SummarizeFunc                  // Custom summarization function (optional)
	baseSystemPrompt string                         // Base system prompt (without memory)
	toolTimeout      time.Duration                  // Maximum tool execution time
	loopDetector     *LoopDetector                  // Detects infinite tool call loops
	pruningConfig    PruningConfig                  // Tool result pruning configuration
}

// LoopDetector identifies when the LLM is stuck in a repetitive tool call pattern.
// This provides early detection before the circuit breaker (5 iterations) is reached.
//
// Detection Strategy:
//   - Exact match: Same tool + same args called 3+ times
//   - Similar match: Same tool + similar args (80%+ word overlap) 3+ times
//
// When a loop is detected, a hint is returned to the LLM suggesting alternative approaches.
type LoopDetector struct {
	mu              sync.Mutex       // Protects concurrent access
	recentToolCalls []toolCallRecord // Circular buffer of recent tool calls
	maxHistory      int              // Maximum number of calls to track (default: 10)
	loopThreshold   int              // Number of repeats before triggering (default: 3)
}

// toolCallRecord stores information about a single tool invocation.
type toolCallRecord struct {
	toolName string // Name of the tool called
	args     string // JSON-encoded arguments
	result   string // Tool result (or error message)
}

// NewLoopDetector creates a new loop detector with default settings.
func NewLoopDetector() *LoopDetector {
	return &LoopDetector{
		maxHistory:    10, // Track last 10 tool calls
		loopThreshold: 3,  // Trigger after 3 identical calls
	}
}

// record adds a tool call to the detector's history.
// Thread-safe: can be called from multiple goroutines.
func (ld *LoopDetector) record(toolName, args, result string) {
	ld.mu.Lock()
	defer ld.mu.Unlock()

	ld.recentToolCalls = append(ld.recentToolCalls, toolCallRecord{
		toolName: toolName,
		args:     args,
		result:   result,
	})

	// Maintain circular buffer size
	if len(ld.recentToolCalls) > ld.maxHistory {
		ld.recentToolCalls = ld.recentToolCalls[1:]
	}
}

// detectLoop checks if the current tool call would create a loop.
// Returns (true, hint) if a loop is detected, (false, "") otherwise.
//
// Detection Logic:
//  1. Count exact matches (same tool + same args)
//  2. Count similar matches (same tool + similar args)
//  3. If count >= loopThreshold, return hint for LLM
//
// The hint provides actionable advice based on whether previous
// calls resulted in errors or successful extractions.
func (ld *LoopDetector) detectLoop(toolName, args string) (bool, string) {
	ld.mu.Lock()
	defer ld.mu.Unlock()

	// Check for exact matches
	sameCount := 0
	var lastResult string

	for _, call := range ld.recentToolCalls {
		if call.toolName == toolName && call.args == args {
			sameCount++
			lastResult = call.result
		}
	}

	if sameCount >= ld.loopThreshold {
		hint := fmt.Sprintf("[LOOP DETECTED] You've called %q with the same arguments %d times. ", toolName, sameCount)
		if strings.Contains(lastResult, "Error") || strings.Contains(lastResult, "failed") {
			hint += "This approach isn't working. Try a different strategy, different arguments, or explain what you're trying to achieve."
		} else {
			hint += "You already have this information. Use it to answer the user's question or try a different approach."
		}
		return true, hint
	}

	// Check for similar matches (same tool, similar args)
	similarCount := 0
	for _, call := range ld.recentToolCalls {
		if call.toolName == toolName && similarity(call.args, args) > 0.8 {
			similarCount++
		}
	}

	if similarCount >= ld.loopThreshold {
		return true, fmt.Sprintf("[LOOP DETECTED] You've called %q with very similar arguments %d times. Try a completely different approach.", toolName, similarCount)
	}

	return false, ""
}

// similarity calculates the Jaccard similarity between two strings based on word overlap.
// Returns a value between 0.0 (no similarity) and 1.0 (identical).
// Used to detect near-duplicate tool calls that aren't exact matches.
func similarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	s1Words := strings.Fields(strings.ToLower(s1))
	s2Words := strings.Fields(strings.ToLower(s2))

	if len(s1Words) == 0 && len(s2Words) == 0 {
		return 1.0
	}
	if len(s1Words) == 0 || len(s2Words) == 0 {
		return 0.0
	}

	// Calculate Jaccard similarity: intersection / union
	s1Set := make(map[string]bool)
	for _, w := range s1Words {
		s1Set[w] = true
	}

	common := 0
	for _, w := range s2Words {
		if s1Set[w] {
			common++
		}
	}

	// Jaccard = intersection / (|A| + |B| - intersection)
	return float64(2*common) / float64(len(s1Words)+len(s2Words))
}

// NewAgent creates a new Agent with the provided OpenAI client and optional WAL.
// If wal is nil, the agent operates without persistence (session-only).
//
// Default Configuration:
//   - Model: DefaultModel ("qwen3.5-9b-mlx")
//   - ContextWindow: 40000 tokens
//   - ReserveTokens: 70% of context window (calculated)
//   - KeepRecentTokens: 85% of context window (calculated)
//   - ToolTimeout: 30 seconds
//   - LoopDetector: enabled with default settings
//   - Pruning: aggressive mode (20% soft trim, 35% hard clear)
//
// The agent automatically:
//   - Loads conversation history from WAL (if provided)
//   - Injects a system prompt with memory (if available)
//   - Initializes the tool registry (empty, use RegisterTool)
func NewAgent(client *openai.Client, wal *WAL) *Agent {
	// Calculate dynamic thresholds based on context window
	contextWindow := DefaultContextWindow
	reserveTokens := int(float64(contextWindow) * ReserveTokensRatio)
	keepRecentTokens := int(float64(contextWindow) * KeepRecentTokensRatio)

	a := &Agent{
		client:           client,
		tools:            make(map[string]Tool),
		wal:              wal,
		model:            DefaultModel,
		contextWindow:    contextWindow,
		reserveTokens:    reserveTokens,
		keepRecentTokens: keepRecentTokens,
		baseSystemPrompt: "You are an autonomous OpenClaw-style personal assistant. You have access to a web browser tool. Always use it if the user asks for real-time data like flight prices. Extract the data cleanly.",
		toolTimeout:      DefaultToolTimeout,
		loopDetector:     NewLoopDetector(),
		pruningConfig:    DefaultPruningConfig,
	}

	// Initialize memory system if WAL is present
	if wal != nil {
		a.memory = NewMemory(filepath.Join(filepath.Dir(wal.path), "memory"))
	}

	systemPrompt := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: a.buildSystemPrompt(),
	}

	if wal != nil {
		loaded, err := wal.LoadAll()
		if err != nil {
			log.Printf("[Agent] failed to load WAL: %v", err)
		} else if len(loaded) > 0 {
			a.history = loaded
			foundSystem := false
			for i, m := range a.history {
				if m.Role == openai.ChatMessageRoleSystem {
					foundSystem = true
					a.history[i].Content = a.buildSystemPrompt()
					break
				}
			}
			if !foundSystem {
				a.history = append([]openai.ChatCompletionMessage{systemPrompt}, a.history...)
				if err := wal.Append(systemPrompt); err != nil {
					log.Printf("[Agent] WAL append error for system prompt: %v", err)
				}
			}
			log.Printf("[Agent] restored %d messages from WAL", len(a.history))
			return a
		}
	}

	a.history = []openai.ChatCompletionMessage{}
	a.appendHistory(systemPrompt)
	return a
}

func (a *Agent) buildSystemPrompt() string {
	base := a.baseSystemPrompt
	if a.memory == nil {
		return base
	}
	memContent, err := a.memory.ReadMemory()
	if err != nil || memContent == "" {
		return base
	}
	return fmt.Sprintf("%s\n\n## Remembered Context:\n%s", base, memContent)
}

func (a *Agent) RefreshSystemPrompt() {
	a.mu.Lock()
	defer a.mu.Unlock()

	newPrompt := a.buildSystemPrompt()
	for i := range a.history {
		if a.history[i].Role == openai.ChatMessageRoleSystem {
			a.history[i].Content = newPrompt
			break
		}
	}
	a.updateSystemPromptInWAL()
}

func (a *Agent) appendHistory(msg openai.ChatCompletionMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.history = append(a.history, msg)
	if a.wal != nil {
		if err := a.wal.Append(msg); err != nil {
			log.Printf("[Agent] WAL append error: %v", err)
		}
	}
}

func (a *Agent) RegisterTool(t Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools[t.Name()] = t
}

func (a *Agent) SetToolTimeout(timeout time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.toolTimeout = timeout
}

func (a *Agent) setModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if model != "" {
		a.model = model
	}
}

func (a *Agent) setContextWindow(window int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if window > 0 {
		a.contextWindow = window
		a.reserveTokens = int(float64(window) * ReserveTokensRatio)
		a.keepRecentTokens = int(float64(window) * KeepRecentTokensRatio)
	}
}

func (a *Agent) setPruningConfig(pc PruningConfig) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pruningConfig = pc
}

func (a *Agent) GetHistory() []openai.ChatCompletionMessage {
	a.mu.RLock()
	defer a.mu.RUnlock()
	result := make([]openai.ChatCompletionMessage, len(a.history))
	copy(result, a.history)
	return result
}

// ClearHistory wipes the conversation history and starts fresh.
// Keeps the system prompt, archives the old WAL, and creates a new empty history.
// Use this to start a new session without losing memory.
func (a *Agent) ClearHistory() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	log.Printf("[Agent] Clearing conversation history...")

	// Archive old WAL if it exists
	if a.wal != nil {
		// Rename old WAL to archive
		archivePath := a.wal.path + ".archived." + time.Now().Format("20060102-150405")
		if err := os.Rename(a.wal.path, archivePath); err != nil && !os.IsNotExist(err) {
			log.Printf("[Agent] Failed to archive old WAL: %v", err)
		} else {
			log.Printf("[Agent] Archived old WAL to %s", archivePath)
		}
	}

	// Keep system prompt
	var systemPrompt openai.ChatCompletionMessage
	for _, m := range a.history {
		if m.Role == openai.ChatMessageRoleSystem {
			systemPrompt = m
			break
		}
	}

	// Reset history with just system prompt
	a.history = []openai.ChatCompletionMessage{systemPrompt}

	// Rewrite WAL with just system prompt
	if a.wal != nil {
		if err := a.wal.Rewrite(a.history); err != nil {
			return fmt.Errorf("failed to rewrite WAL: %w", err)
		}
	}

	log.Printf("[Agent] History cleared, starting fresh session")
	return nil
}

// HistoryStats provides statistics about the conversation history.
type HistoryStats struct {
	MessageCount     int
	TokenCount       int
	ToolResultCount  int
	OldestMessage    time.Time
	ContextWindow    int
	ReserveTokens    int
	KeepRecentTokens int
}

// GetHistoryStats returns statistics about the current conversation history.
func (a *Agent) GetHistoryStats() HistoryStats {
	a.mu.RLock()
	defer a.mu.RUnlock()

	stats := HistoryStats{
		MessageCount:     len(a.history),
		TokenCount:       countTokens(a.history),
		ContextWindow:    a.contextWindow,
		ReserveTokens:    a.reserveTokens,
		KeepRecentTokens: a.keepRecentTokens,
	}

	for _, m := range a.history {
		if m.Role == openai.ChatMessageRoleTool {
			stats.ToolResultCount++
		}
	}

	// Estimate oldest message time (approximate)
	if len(a.history) > 0 {
		stats.OldestMessage = time.Now().Add(-time.Minute * time.Duration(len(a.history)))
	}

	return stats
}

func (a *Agent) updateSystemPromptInWAL() {
	if a.wal == nil {
		return
	}
	for _, m := range a.history {
		if m.Role == openai.ChatMessageRoleSystem {
			if err := a.wal.Rewrite(a.history); err != nil {
				log.Printf("[Agent] failed to persist system prompt update: %v", err)
			}
			break
		}
	}
}

// ==========================================
// OPENCLAW CONCEPT 7: CONTEXT COMPACTION
// ==========================================

func (a *Agent) defaultSummarizer(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
	if a.client == nil {
		return "", fmt.Errorf("no LLM client available for summarization")
	}

	serialized := serializeMessages(messages)
	req := openai.ChatCompletionRequest{
		Model: a.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: prompt},
			{Role: openai.ChatMessageRoleUser, Content: serialized},
		},
	}

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM returned no choices")
	}
	return resp.Choices[0].Message.Content, nil
}

func (a *Agent) generateSummary(ctx context.Context, messages []openai.ChatCompletionMessage) string {
	if len(messages) == 0 {
		return "No previous context."
	}

	fn := a.summarizer
	if fn == nil {
		fn = a.defaultSummarizer
	}

	summary, err := fn(ctx, SummaryPrompt, messages)
	if err != nil {
		log.Printf("[Compaction] Summary generation failed: %v", err)
		return fmt.Sprintf("Summary generation failed. %d messages were compacted.", len(messages))
	}
	return summary
}

func (a *Agent) compactHistoryIfNeeded(ctx context.Context) error {
	a.mu.Lock()

	if a.reserveTokens <= 0 || a.keepRecentTokens <= 0 {
		a.mu.Unlock()
		return nil
	}

	if len(a.history) < 4 {
		a.mu.Unlock()
		return nil
	}

	currentTokens := countTokens(a.history)

	needMemoryFlush := currentTokens > a.reserveTokens-MemoryFlushThreshold && currentTokens < a.reserveTokens && a.memory != nil && a.client != nil
	needCompaction := currentTokens >= a.reserveTokens

	if !needMemoryFlush && !needCompaction {
		a.mu.Unlock()
		return nil
	}

	historySnapshot := make([]openai.ChatCompletionMessage, len(a.history))
	copy(historySnapshot, a.history)
	a.mu.Unlock()

	if needMemoryFlush && !needCompaction {
		log.Printf("[Compaction] Approaching limit (%d/%d tokens), flushing memory before compaction", currentTokens, a.reserveTokens)
		if err := a.memory.Flush(ctx, a.client, a.model, historySnapshot); err != nil {
			log.Printf("[Compaction] Pre-compaction memory flush failed: %v", err)
		} else {
			a.mu.Lock()
			newSysPrompt := a.buildSystemPrompt()
			for i := range a.history {
				if a.history[i].Role == openai.ChatMessageRoleSystem {
					a.history[i].Content = newSysPrompt
					break
				}
			}
			a.mu.Unlock()
		}
		return nil
	}

	log.Printf("[Compaction] Token count %d exceeds reserve %d, starting compaction", currentTokens, a.reserveTokens)

	systemPromptUpdated := false
	if a.memory != nil && a.client != nil {
		if err := a.memory.Flush(ctx, a.client, a.model, historySnapshot); err != nil {
			log.Printf("[Compaction] Memory flush failed (continuing anyway): %v", err)
		} else {
			a.mu.Lock()
			newSysPrompt := a.buildSystemPrompt()
			for i := range a.history {
				if a.history[i].Role == openai.ChatMessageRoleSystem {
					a.history[i].Content = newSysPrompt
					systemPromptUpdated = true
					break
				}
			}
			a.mu.Unlock()
		}
	}

	a.mu.Lock()
	nonSystem := a.history[1:]
	recentStart := findRecentStart(nonSystem, a.keepRecentTokens)

	if recentStart == 0 {
		log.Printf("[Compaction] All messages fit within recent budget, nothing to compact")
		if systemPromptUpdated && a.wal != nil {
			if err := a.wal.Rewrite(a.history); err != nil {
				log.Printf("[Compaction] WAL rewrite for system prompt update failed: %v", err)
			}
		}
		a.mu.Unlock()
		return nil
	}

	oldMessages := make([]openai.ChatCompletionMessage, recentStart)
	copy(oldMessages, nonSystem[:recentStart])
	a.mu.Unlock()

	summary := a.generateSummary(ctx, oldMessages)

	summaryMsg := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: fmt.Sprintf("[CONTEXT SUMMARY]: %s", summary),
	}

	a.mu.Lock()
	var newHistory []openai.ChatCompletionMessage
	newHistory = append(newHistory, a.history[0])
	newHistory = append(newHistory, summaryMsg)
	nonSystem = a.history[1:]
	recentStartNow := findRecentStart(nonSystem, a.keepRecentTokens)
	if recentStartNow > 0 && recentStartNow < len(nonSystem) {
		newHistory = append(newHistory, nonSystem[recentStartNow:]...)
	} else if recentStartNow == 0 {
		newHistory = append(newHistory, nonSystem...)
	}

	beforeCount := len(a.history)
	a.history = newHistory

	if a.wal != nil {
		if err := a.wal.Rewrite(a.history); err != nil {
			log.Printf("[Compaction] WAL rewrite failed: %v", err)
		}
	}

	log.Printf("[Compaction] Reduced from %d to %d messages (%d tokens)", beforeCount, len(a.history), countTokens(a.history))
	a.mu.Unlock()
	return nil
}

// ==========================================
// TOOL RESULT PRUNING (Tier 1 - OpenClaw Style)
// ==========================================

// pruneToolResults trims oversized tool results before sending to LLM.
// This is Tier 1 pruning - happens per-request, in-memory only.
// Does NOT modify on-disk WAL history.
//
// Strategy:
//   - Calculate tool result token ratio
//   - If > SoftTrimRatio: soft trim (keep head + tail)
//   - If > HardClearRatio: hard clear (replace with placeholder)
func (a *Agent) pruneToolResults(history []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	if a.pruningConfig.Mode == "off" {
		return history
	}

	// Calculate total tokens and tool result tokens
	totalTokens := 0
	toolResultTokens := 0
	toolResultIndices := []int{}

	for i, msg := range history {
		msgTokens := countTokens([]openai.ChatCompletionMessage{msg})
		totalTokens += msgTokens

		if msg.Role == openai.ChatMessageRoleTool {
			toolResultTokens += msgTokens
			toolResultIndices = append(toolResultIndices, i)
		}
	}

	if totalTokens == 0 || toolResultTokens == 0 {
		return history
	}

	// Calculate ratio
	toolResultRatio := float64(toolResultTokens) / float64(totalTokens)

	// Check if pruning needed
	if toolResultRatio < a.pruningConfig.SoftTrimRatio {
		return history
	}

	log.Printf("[Pruning] Tool result ratio %.2f%% exceeds threshold %.2f%%, pruning %d tool results",
		toolResultRatio*100, a.pruningConfig.SoftTrimRatio*100, len(toolResultIndices))

	// Create pruned copy
	pruned := make([]openai.ChatCompletionMessage, len(history))
	copy(pruned, history)

	// Determine pruning strategy
	useHardClear := toolResultRatio >= a.pruningConfig.HardClearRatio

	for _, idx := range toolResultIndices {
		if useHardClear {
			// Hard clear: replace entire result with placeholder
			pruned[idx].Content = a.pruningConfig.HardClearPlaceholder
			log.Printf("[Pruning] Hard-cleared tool result at index %d", idx)
		} else {
			content := pruned[idx].Content
			if len(content) > a.pruningConfig.SoftTrimMaxChars {
				headEnd := min(a.pruningConfig.SoftTrimHeadChars, len(content))
				for headEnd > 0 && !utf8.RuneStart(content[headEnd]) {
					headEnd--
				}
				head := content[:headEnd]
				tail := ""
				tailStart := len(content) - a.pruningConfig.SoftTrimTailChars
				for tailStart > headEnd && !utf8.RuneStart(content[tailStart]) {
					tailStart++
				}
				if tailStart > headEnd {
					tail = content[tailStart:]
				}
				pruned[idx].Content = fmt.Sprintf("%s\n\n... [trimmed %d chars] ...\n\n%s",
					head, len(content)-len(head)-len(tail), tail)
				log.Printf("[Pruning] Soft-trimmed tool result at index %d (%d -> %d chars)",
					idx, len(content), len(pruned[idx].Content))
			}
		}
	}

	return pruned
}

// ==========================================
// OPENCLAW CONCEPT 4: THE ReAct (Reason + Act) ENGINE
// ==========================================

func (a *Agent) Chat(ctx context.Context, userInput string) (string, error) {
	a.appendHistory(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userInput,
	})

	if err := a.compactHistoryIfNeeded(ctx); err != nil {
		log.Printf("[Agent] Compaction failed: %v", err)
	}

	var openAITools []openai.Tool
	a.mu.RLock()
	for _, t := range a.tools {
		openAITools = append(openAITools, t.Definition())
	}
	historyCopy := make([]openai.ChatCompletionMessage, len(a.history))
	copy(historyCopy, a.history)
	a.mu.RUnlock()

	// NEW: Prune tool results before sending to LLM (Tier 1)
	historyCopy = a.pruneToolResults(historyCopy)

	for range 5 {
		req := openai.ChatCompletionRequest{
			Model:    a.model,
			Messages: historyCopy,
			Tools:    openAITools,
		}

		resp, err := a.client.CreateChatCompletion(ctx, req)
		if err != nil {
			return "", err
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("LLM returned no choices")
		}

		msg := resp.Choices[0].Message

		if len(msg.ToolCalls) == 0 {
			msg.Content = cleanResponse(msg.Content)
			a.appendHistory(msg)
			return msg.Content, nil
		}

		a.appendHistory(msg)
		historyCopy = append(historyCopy, msg)

		for _, toolCall := range msg.ToolCalls {
			toolName := toolCall.Function.Name
			toolArgs := toolCall.Function.Arguments

			a.mu.RLock()
			tool, exists := a.tools[toolName]
			timeout := a.toolTimeout
			loopDetector := a.loopDetector
			a.mu.RUnlock()

			if !exists {
				toolMsg := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    fmt.Sprintf("Error: tool %q does not exist. Use only the available tools.", toolName),
					ToolCallID: toolCall.ID,
				}
				a.appendHistory(toolMsg)
				historyCopy = append(historyCopy, toolMsg)
				continue
			}

			if isLoop, hint := loopDetector.detectLoop(toolName, toolArgs); isLoop {
				log.Printf("[Agent] Loop detected for tool %q", toolName)
				toolMsg := openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    hint,
					Name:       toolName,
					ToolCallID: toolCall.ID,
				}
				a.appendHistory(toolMsg)
				historyCopy = append(historyCopy, toolMsg)
				continue
			}

			var resultText string
			var err error

			if tc, ok := tool.(ToolWithContext); ok {
				toolCtx, cancel := context.WithTimeout(ctx, timeout)
				resultText, err = tc.ExecuteWithContext(toolCtx, toolArgs)
				cancel()
				if err != nil {
					resultText = fmt.Sprintf("Error executing tool: %v", err)
				}
			} else {
				resultCh := make(chan struct {
					text string
					err  error
				}, 1)

				go func() {
					text, execErr := tool.Execute(toolArgs)
					resultCh <- struct {
						text string
						err  error
					}{text, execErr}
				}()

				select {
				case result := <-resultCh:
					resultText = result.text
					if result.err != nil {
						resultText = fmt.Sprintf("Error executing tool: %v", result.err)
					}
				case <-time.After(timeout):
					resultText = fmt.Sprintf("Error: tool %q timed out after %v", toolName, timeout)
					go func() { <-resultCh }()
				}
			}

			loopDetector.record(toolName, toolArgs, resultText)

			toolMsg := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    resultText,
				Name:       toolName,
				ToolCallID: toolCall.ID,
			}
			a.appendHistory(toolMsg)
			historyCopy = append(historyCopy, toolMsg)
		}

		historyCopy = a.pruneToolResults(historyCopy)
	}

	return "", fmt.Errorf("agent reached maximum thinking steps (circuit breaker triggered)")
}

func (a *Agent) ChatWithImage(ctx context.Context, prompt string, imageBase64 string) (string, error) {
	a.mu.Lock()
	a.history = append(a.history, openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleUser,
		MultiContent: []openai.ChatMessagePart{
			{
				Type: openai.ChatMessagePartTypeText,
				Text: prompt,
			},
			{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    "data:image/jpeg;base64," + imageBase64,
					Detail: openai.ImageURLDetailLow,
				},
			},
		},
	})
	if a.wal != nil {
		if err := a.wal.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "[IMAGE] " + prompt,
		}); err != nil {
			log.Printf("[Agent] WAL append error: %v", err)
		}
	}
	historyCopy := make([]openai.ChatCompletionMessage, len(a.history))
	copy(historyCopy, a.history)
	a.mu.Unlock()

	var openAITools []openai.Tool
	a.mu.RLock()
	for _, t := range a.tools {
		openAITools = append(openAITools, t.Definition())
	}
	a.mu.RUnlock()

	req := openai.ChatCompletionRequest{
		Model:    a.model,
		Messages: historyCopy,
		Tools:    openAITools,
	}

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if strings.Contains(err.Error(), "does not support") || strings.Contains(err.Error(), "not support") {
			log.Printf("[Vision] Model doesn't support images, trying text-only fallback")
			a.mu.Lock()
			a.history = a.history[:len(a.history)-1]
			a.history = append(a.history, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			})
			if a.wal != nil {
				if rwErr := a.wal.Rewrite(a.history); rwErr != nil {
					log.Printf("[Vision] WAL rewrite on fallback failed: %v", rwErr)
				}
			}
			historyCopy = make([]openai.ChatCompletionMessage, len(a.history))
			copy(historyCopy, a.history)
			a.mu.Unlock()

			textReq := openai.ChatCompletionRequest{
				Model:    a.model,
				Messages: historyCopy,
				Tools:    openAITools,
			}
			textResp, textErr := a.client.CreateChatCompletion(ctx, textReq)
			if textErr != nil {
				return "", textErr
			}
			if len(textResp.Choices) == 0 {
				return "", fmt.Errorf("LLM returned no choices")
			}
			textMsg := textResp.Choices[0].Message
			a.appendHistory(openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: textMsg.Content,
			})
			return cleanResponse(textMsg.Content), nil
		}
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM returned no choices")
	}

	msg := resp.Choices[0].Message

	if len(msg.ToolCalls) == 0 {
		a.appendHistory(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: msg.Content,
		})
		return cleanResponse(msg.Content), nil
	}

	a.appendHistory(msg)

	for _, toolCall := range msg.ToolCalls {
		toolName := toolCall.Function.Name
		toolArgs := toolCall.Function.Arguments

		a.mu.RLock()
		tool, exists := a.tools[toolName]
		timeout := a.toolTimeout
		a.mu.RUnlock()

		var resultText string
		if !exists {
			resultText = fmt.Sprintf("Error: tool %q does not exist.", toolName)
		} else {
			if tc, ok := tool.(ToolWithContext); ok {
				toolCtx, cancel := context.WithTimeout(ctx, timeout)
				resultText, _ = tc.ExecuteWithContext(toolCtx, toolArgs)
				cancel()
			} else {
				resultText, _ = tool.Execute(toolArgs)
			}
		}

		toolMsg := openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			Content:    resultText,
			Name:       toolName,
			ToolCallID: toolCall.ID,
		}
		a.appendHistory(toolMsg)
	}

	a.mu.RLock()
	historyCopy = make([]openai.ChatCompletionMessage, len(a.history))
	copy(historyCopy, a.history)
	a.mu.RUnlock()
	historyCopy = a.pruneToolResults(historyCopy)

	followUpReq := openai.ChatCompletionRequest{
		Model:    a.model,
		Messages: historyCopy,
		Tools:    openAITools,
	}
	followUpResp, followUpErr := a.client.CreateChatCompletion(ctx, followUpReq)
	if followUpErr != nil {
		return cleanResponse(msg.Content), nil
	}
	if len(followUpResp.Choices) == 0 {
		return cleanResponse(msg.Content), nil
	}
	followUpMsg := followUpResp.Choices[0].Message
	followUpMsg.Content = cleanResponse(followUpMsg.Content)
	a.appendHistory(followUpMsg)
	return followUpMsg.Content, nil
}

// ==========================================
// OPENCLAW CONCEPT 8: SCHEDULER (Periodic Tasks)
// ==========================================

type ScheduledTask struct {
	ID         string
	Prompt     string
	Interval   time.Duration
	LastRun    time.Time
	NextRun    time.Time
	Active     bool
	CreatedAt  time.Time
	RunCount   int
	cancelFunc context.CancelFunc
}

type Scheduler struct {
	mu    sync.RWMutex
	tasks map[string]*ScheduledTask
	agent *Agent
	ctx   context.Context
}

func NewScheduler(agent *Agent, ctx context.Context) *Scheduler {
	return &Scheduler{
		tasks: make(map[string]*ScheduledTask),
		agent: agent,
		ctx:   ctx,
	}
}

func (s *Scheduler) Schedule(id string, prompt string, interval time.Duration) error {
	if interval <= 0 {
		return fmt.Errorf("interval must be positive")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.tasks[id]; exists {
		return fmt.Errorf("task with id %q already exists", id)
	}

	now := time.Now()
	task := &ScheduledTask{
		ID:        id,
		Prompt:    prompt,
		Interval:  interval,
		NextRun:   now.Add(interval),
		Active:    true,
		CreatedAt: now,
	}

	s.tasks[id] = task

	go s.runTask(task)

	log.Printf("[Scheduler] Scheduled task %q: every %v - %q", id, interval, prompt)
	return nil
}

func (s *Scheduler) runTask(task *ScheduledTask) {
	ticker := time.NewTicker(task.Interval)
	defer ticker.Stop()

	for {
		s.mu.RLock()
		active := task.Active
		s.mu.RUnlock()

		if !active {
			return
		}

		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.executeTask(task)
		}
	}
}

func (s *Scheduler) executeTask(task *ScheduledTask) {
	s.mu.Lock()
	if !task.Active {
		s.mu.Unlock()
		return
	}
	task.LastRun = time.Now()
	task.NextRun = time.Now().Add(task.Interval)
	s.mu.Unlock()

	log.Printf("[Scheduler] Executing task %q: %q", task.ID, task.Prompt)

	reply, err := s.agent.Chat(s.ctx, task.Prompt)
	if err != nil {
		log.Printf("[Scheduler] Task %q failed: %v", task.ID, err)
		return
	}

	s.mu.Lock()
	task.RunCount++
	s.mu.Unlock()

	log.Printf("[Scheduler] Task %q completed (run #%d): %s", task.ID, task.RunCount, truncate(reply, 200))
}

func (s *Scheduler) Cancel(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	task, exists := s.tasks[id]
	if !exists {
		return fmt.Errorf("task %q not found", id)
	}

	task.Active = false
	log.Printf("[Scheduler] Cancelled task %q (ran %d times)", id, task.RunCount)
	return nil
}

func (s *Scheduler) GetTask(id string) (*ScheduledTask, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	task, exists := s.tasks[id]
	if !exists {
		return nil, false
	}
	copy := *task
	return &copy, true
}

func (s *Scheduler) ListTasks() []*ScheduledTask {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*ScheduledTask, 0, len(s.tasks))
	for _, task := range s.tasks {
		copy := *task
		result = append(result, &copy)
	}
	return result
}

func (s *Scheduler) ParseDuration(input string) (time.Duration, error) {
	input = strings.TrimSpace(strings.ToLower(input))

	re := regexp.MustCompile(`^(\d+)\s*(hour|hr|h|minute|min|m|second|sec|s|day|d)s?$`)
	matches := re.FindStringSubmatch(input)
	if matches == nil {
		return 0, fmt.Errorf("invalid duration format: %q (use formats like '2 hours', '30 minutes', '1 day')", input)
	}

	value, _ := strconv.Atoi(matches[1])
	unit := matches[2]

	switch unit {
	case "hour", "hr", "h":
		return time.Duration(value) * time.Hour, nil
	case "minute", "min", "m":
		return time.Duration(value) * time.Minute, nil
	case "second", "sec", "s":
		return time.Duration(value) * time.Second, nil
	case "day", "d":
		return time.Duration(value) * 24 * time.Hour, nil
	default:
		return 0, fmt.Errorf("unknown time unit: %q", unit)
	}
}

func truncate(s string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	if len(s) <= maxLen {
		return s
	}
	trunc := s[:maxLen]
	for len(trunc) > 0 && !utf8.RuneStart(trunc[len(trunc)-1]) {
		trunc = trunc[:len(trunc)-1]
	}
	if len(trunc) == 0 {
		return ""
	}
	return trunc + "..."
}

func cleanResponse(content string) string {
	content = strings.TrimSpace(content)
	for _, suffix := range []string{"<|im_end|>", "</s>", "<|eot_id|>"} {
		content = strings.TrimSuffix(content, suffix)
	}
	content = strings.TrimSpace(content)
	content = stripMarkdown(content)
	return content
}

func stripMarkdown(text string) string {
	re := regexp.MustCompile(`\*\*([^*]+)\*\*`)
	text = re.ReplaceAllString(text, "$1")

	re = regexp.MustCompile(`\*([^*]+)\*`)
	text = re.ReplaceAllString(text, "$1")

	re = regexp.MustCompile(`__([^_]+)__`)
	text = re.ReplaceAllString(text, "$1")

	re = regexp.MustCompile(`_([^_]+)_`)
	text = re.ReplaceAllString(text, "$1")

	re = regexp.MustCompile("`([^`]+)`")
	text = re.ReplaceAllString(text, "$1")

	return text
}

type SchedulerTool struct {
	scheduler *Scheduler
}

func (t *SchedulerTool) Name() string { return "schedule_task" }

func (t *SchedulerTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: "Schedule a task to run periodically. Use this when the user asks to do something repeatedly or at intervals.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"action": map[string]any{
						"type":        "string",
						"enum":        []string{"schedule", "cancel", "list"},
						"description": "Action to perform: schedule, cancel, or list tasks",
					},
					"task_id": map[string]any{
						"type":        "string",
						"description": "Unique identifier for the task (required for schedule and cancel)",
					},
					"prompt": map[string]any{
						"type":        "string",
						"description": "The prompt/instruction to execute periodically (required for schedule)",
					},
					"interval": map[string]any{
						"type":        "string",
						"description": "How often to run (e.g., '2 hours', '30 minutes', '1 day')",
					},
				},
				"required": []string{"action"},
			},
		},
	}
}

func (t *SchedulerTool) Execute(args string) (string, error) {
	var params struct {
		Action   string `json:"action"`
		TaskID   string `json:"task_id"`
		Prompt   string `json:"prompt"`
		Interval string `json:"interval"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("failed to parse args: %v", err)
	}

	switch params.Action {
	case "schedule":
		if params.TaskID == "" || params.Prompt == "" || params.Interval == "" {
			return "", fmt.Errorf("schedule requires task_id, prompt, and interval")
		}
		interval, err := t.scheduler.ParseDuration(params.Interval)
		if err != nil {
			return "", err
		}
		if err := t.scheduler.Schedule(params.TaskID, params.Prompt, interval); err != nil {
			return "", err
		}
		return fmt.Sprintf("Task %q scheduled to run every %v. Use action 'cancel' with task_id to stop it.", params.TaskID, interval), nil

	case "cancel":
		if params.TaskID == "" {
			return "", fmt.Errorf("cancel requires task_id")
		}
		if err := t.scheduler.Cancel(params.TaskID); err != nil {
			return "", err
		}
		return fmt.Sprintf("Task %q cancelled.", params.TaskID), nil

	case "list":
		tasks := t.scheduler.ListTasks()
		if len(tasks) == 0 {
			return "No scheduled tasks.", nil
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Found %d scheduled task(s):\n", len(tasks)))
		for _, task := range tasks {
			status := "active"
			if !task.Active {
				status = "inactive"
			}
			sb.WriteString(fmt.Sprintf("- %s: %q (every %v, %s, runs: %d, next: %s)\n",
				task.ID, task.Prompt, task.Interval, status, task.RunCount, task.NextRun.Format("15:04:05")))
		}
		return sb.String(), nil

	default:
		return "", fmt.Errorf("unknown action: %q", params.Action)
	}
}

// ==========================================
// OPENCLAW CONCEPT 5: THE GATEWAY (I/O BOUNDARY)
// ==========================================

func main() {
	configPath := flag.String("config", "", "Path to config file (JSON)")
	telegramToken := flag.String("telegram", "", "Telegram bot token (overrides config file)")
	allowedUsers := flag.String("allowed", "", "Comma-separated Telegram user IDs (overrides config file)")
	doInit := flag.Bool("init", false, "Create example config.json in current directory")
	flag.Parse()

	if *doInit {
		if err := writeExampleConfig("config.json"); err != nil {
			log.Fatalf("Failed to create config.json: %v", err)
		}
		fmt.Println("Created config.json with example configuration. Edit it with your settings.")
		return
	}

	cfg, loadedFrom, err := findAndLoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	if loadedFrom != "" {
		log.Printf("[Config] Loaded from %s", loadedFrom)
	} else {
		log.Printf("[Config] No config file found, using defaults")
	}

	apiKey := cfg.LLM.APIKey
	if apiKey == "" {
		apiKey = "sk-lm-tF4UHMAz:dhJpiJ5A6MvLrsEMJdU7"
	}
	llmConfig := openai.DefaultConfig(apiKey)
	llmConfig.BaseURL = cfg.LLM.BaseURL

	WALDir = cfg.WAL.Dir
	WALFile = cfg.WAL.File
	WALMaxSize = cfg.WAL.MaxSize

	client := openai.NewClientWithConfig(llmConfig)

	wal, err := OpenWAL()
	if err != nil {
		log.Fatalf("Failed to open WAL: %v", err)
	}

	agent := NewAgent(client, wal)
	agent.setModel(cfg.LLM.Model)
	agent.setContextWindow(cfg.Agent.ContextWindow)
	agent.SetToolTimeout(cfg.Agent.ToolTimeout.ToDuration())
	agent.setPruningConfig(cfg.PruningConfig())
	agent.RegisterTool(&MomondoFlightTool{})
	agent.RegisterTool(&CDPBrowserFlightTool{})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scheduler := NewScheduler(agent, ctx)
	agent.RegisterTool(&SchedulerTool{scheduler: scheduler})

	effectiveToken := *telegramToken
	if effectiveToken == "" {
		effectiveToken = cfg.Telegram.Token
	}

	if effectiveToken != "" {
		var userIDs []int64
		if *allowedUsers != "" {
			userIDs = parseAllowedUsers(*allowedUsers)
		} else {
			userIDs = cfg.AllowedUserIDs()
		}
		bot, err := NewTelegramBot(effectiveToken, agent, scheduler, ctx, userIDs)
		if err != nil {
			log.Fatalf("Failed to create Telegram bot: %v", err)
		}
		log.Println("[Telegram] Bot starting...")
		if err := bot.Run(); err != nil {
			log.Fatalf("Telegram bot error: %v", err)
		}
		return
	}

	fmt.Println("Phubot CLI - Personal AI Assistant")
	fmt.Println("-----------------------------------")
	fmt.Println("Commands: exit, /new, /clear, /stats, tasks, cancel <id>")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("You: ")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)

		if text == "exit" || text == "quit" {
			break
		}
		if text == "" {
			continue
		}

		// Session management commands
		if text == "/new" || text == "/clear" {
			if err := agent.ClearHistory(); err != nil {
				fmt.Printf("Error clearing history: %v\n", err)
			} else {
				fmt.Println("✅ Started fresh session (memory preserved)")
			}
			continue
		}

		if text == "/stats" {
			stats := agent.GetHistoryStats()
			fmt.Printf("📊 Session Statistics:\n")
			fmt.Printf("  Messages: %d\n", stats.MessageCount)
			fmt.Printf("  Tokens: %d / %d (%.1f%%)\n",
				stats.TokenCount, stats.ContextWindow,
				float64(stats.TokenCount)/float64(stats.ContextWindow)*100)
			fmt.Printf("  Tool Results: %d\n", stats.ToolResultCount)
			fmt.Printf("  Reserve Threshold: %d tokens (%.0f%%)\n",
				stats.ReserveTokens, ReserveTokensRatio*100)
			fmt.Printf("  Keep Recent: %d tokens (%.0f%%)\n",
				stats.KeepRecentTokens, KeepRecentTokensRatio*100)
			fmt.Printf("  Compaction: %s\n",
				func() string {
					if stats.TokenCount > stats.ReserveTokens {
						return "⚠️  TRIGGERED (exceeds reserve)"
					}
					approaching := stats.ReserveTokens - MemoryFlushThreshold
					if stats.TokenCount > approaching {
						return fmt.Sprintf("⚠️  APPROACHING (flush at %d tokens)", approaching)
					}
					return "✅ OK"
				}())
			continue
		}

		if text == "tasks" {
			tasks := scheduler.ListTasks()
			if len(tasks) == 0 {
				fmt.Println("No scheduled tasks.")
			} else {
				fmt.Printf("Tasks (%d):\n", len(tasks))
				for _, t := range tasks {
					fmt.Printf("  [%s] %q - every %v (runs: %d)\n", t.ID, t.Prompt, t.Interval, t.RunCount)
				}
			}
			continue
		}

		if after, ok := strings.CutPrefix(text, "cancel "); ok {
			taskID := after
			if err := scheduler.Cancel(taskID); err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Task %q cancelled.\n", taskID)
			}
			continue
		}

		reply, err := agent.Chat(ctx, text)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		fmt.Printf("\nAgent: %s\n\n", reply)
	}
}
