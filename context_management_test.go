package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// ==========================================
// HELPER: create agent with small context for testing
// ==========================================

func newTestAgentWithConfig(client *openai.Client, wal *WAL, contextWindow int, pruningConfig PruningConfig) *Agent {
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
		baseSystemPrompt: "You are a test assistant.",
		toolTimeout:      DefaultToolTimeout,
		loopDetector:     NewLoopDetector(),
		pruningConfig:    pruningConfig,
	}

	if wal != nil {
		a.memory = NewMemoryWithoutRateLimit(filepath.Join(filepath.Dir(wal.path), "memory"))
	}

	systemPrompt := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: a.buildSystemPrompt(),
	}

	if wal != nil {
		loaded, err := wal.LoadAll()
		if err == nil && len(loaded) > 0 {
			a.history = loaded
			return a
		}
	}

	a.history = []openai.ChatCompletionMessage{systemPrompt}
	if wal != nil {
		wal.Append(systemPrompt)
	}
	return a
}

func newDefaultTestAgent(client *openai.Client, wal *WAL) *Agent {
	return newTestAgentWithConfig(client, wal, DefaultContextWindow, DefaultPruningConfig)
}

func makeLongString(length int) string {
	return strings.Repeat("x", length)
}

func makeToolResultMessage(content string) openai.ChatCompletionMessage {
	return openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleTool,
		Content: content,
	}
}

// ==========================================
// TIER 1: PRUNING CONFIG TESTS
// ==========================================

func TestDefaultPruningConfig_Values(t *testing.T) {
	cfg := DefaultPruningConfig

	if cfg.Mode != "aggressive" {
		t.Errorf("expected mode 'aggressive', got %q", cfg.Mode)
	}
	if cfg.SoftTrimRatio != 0.20 {
		t.Errorf("expected SoftTrimRatio 0.20, got %f", cfg.SoftTrimRatio)
	}
	if cfg.HardClearRatio != 0.35 {
		t.Errorf("expected HardClearRatio 0.35, got %f", cfg.HardClearRatio)
	}
	if cfg.SoftTrimMaxChars != 3000 {
		t.Errorf("expected SoftTrimMaxChars 3000, got %d", cfg.SoftTrimMaxChars)
	}
	if cfg.SoftTrimHeadChars != 1000 {
		t.Errorf("expected SoftTrimHeadChars 1000, got %d", cfg.SoftTrimHeadChars)
	}
	if cfg.SoftTrimTailChars != 1000 {
		t.Errorf("expected SoftTrimTailChars 1000, got %d", cfg.SoftTrimTailChars)
	}
	if cfg.HardClearPlaceholder != "[Previous tool result cleared to save context]" {
		t.Errorf("unexpected HardClearPlaceholder: %q", cfg.HardClearPlaceholder)
	}
}

func TestPruningConfig_OffMode_NoPruning(t *testing.T) {
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, PruningConfig{Mode: "off"})

	longResult := makeLongString(50000)
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hi"},
		makeToolResultMessage(longResult),
	}

	result := agent.pruneToolResults(history)
	if result[2].Content != longResult {
		t.Error("off mode should not modify tool results")
	}
}

func TestPruningConfig_OffModePreservesAll(t *testing.T) {
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, PruningConfig{Mode: "off"})

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		makeToolResultMessage("result1"),
		makeToolResultMessage("result2"),
		{Role: openai.ChatMessageRoleUser, Content: "hi"},
	}

	result := agent.pruneToolResults(history)
	for i := range history {
		if result[i].Content != history[i].Content {
			t.Errorf("message %d modified in off mode", i)
		}
	}
}

// ==========================================
// TIER 1: pruneToolResults() TESTS
// ==========================================

func TestPruneToolResults_NoToolResults_NoModification(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
		{Role: openai.ChatMessageRoleUser, Content: "how are you"},
	}

	result := agent.pruneToolResults(history)
	if len(result) != len(history) {
		t.Fatalf("expected %d messages, got %d", len(history), len(result))
	}
	for i := range history {
		if result[i].Content != history[i].Content {
			t.Errorf("message %d content changed unexpectedly", i)
		}
	}
}

func TestPruneToolResults_SmallToolResults_NoModification(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system prompt here"},
		{Role: openai.ChatMessageRoleUser, Content: "check the weather"},
		{Role: openai.ChatMessageRoleAssistant, Content: "let me check", ToolCalls: []openai.ToolCall{{ID: "call_1", Function: openai.FunctionCall{Name: "weather", Arguments: "{}"}}}},
		makeToolResultMessage("Sunny, 72F"),
	}

	result := agent.pruneToolResults(history)
	if result[3].Content != "Sunny, 72F" {
		t.Errorf("small tool result should not be pruned, got: %s", result[3].Content)
	}
}

func TestPruneToolResults_SoftTrim(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.20,
		HardClearRatio:       0.99,
		SoftTrimMaxChars:     200,
		SoftTrimHeadChars:    50,
		SoftTrimTailChars:    50,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	largeNonTool := makeLongString(10000)
	longToolResult := makeLongString(5000)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: largeNonTool},
		{Role: openai.ChatMessageRoleUser, Content: largeNonTool},
		makeToolResultMessage(longToolResult),
	}

	result := agent.pruneToolResults(history)

	if result[2].Content == longToolResult {
		t.Error("expected tool result to be soft-trimmed")
	}
	if !strings.Contains(result[2].Content, "[trimmed") {
		t.Errorf("expected trim marker in output, got: %s", result[2].Content[:min(100, len(result[2].Content))])
	}
	if len(result[2].Content) >= len(longToolResult) {
		t.Errorf("expected trimmed content to be shorter: original=%d, trimmed=%d", len(longToolResult), len(result[2].Content))
	}
}

func TestPruneToolResults_SoftTrim_KeepsHeadAndTail(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.10,
		HardClearRatio:       0.99,
		SoftTrimMaxChars:     300,
		SoftTrimHeadChars:    50,
		SoftTrimTailChars:    50,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	largeNonTool := makeLongString(5000)
	head := "HEAD_" + makeLongString(50)
	tail := "_TAIL" + makeLongString(50)
	middle := "_MIDDLE_" + makeLongString(2000)
	longToolResult := head + middle + tail

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: largeNonTool},
		{Role: openai.ChatMessageRoleUser, Content: largeNonTool},
		makeToolResultMessage(longToolResult),
	}

	result := agent.pruneToolResults(history)

	if !strings.HasPrefix(result[2].Content, head[:min(50, len(head))]) {
		t.Errorf("trimmed result should start with head content, got: %s", result[2].Content[:min(60, len(result[2].Content))])
	}
	if !strings.Contains(result[2].Content, tail[len(tail)-min(50, len(tail)):]) {
		t.Error("trimmed result should contain tail content")
	}
}

func TestPruneToolResults_SoftTrim_ShortResultNotTrimmed(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.10,
		HardClearRatio:       0.90,
		SoftTrimMaxChars:     3000,
		SoftTrimHeadChars:    1000,
		SoftTrimTailChars:    1000,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	shortResult := "short result"
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(shortResult),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != shortResult {
		t.Error("short result below SoftTrimMaxChars should not be trimmed")
	}
}

func TestPruneToolResults_HardClear(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.10,
		SoftTrimMaxChars:     500,
		SoftTrimHeadChars:    200,
		SoftTrimTailChars:    200,
		HardClearPlaceholder: "[CLEARED]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	longToolResult := makeLongString(10000)
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(longToolResult),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != "[CLEARED]" {
		t.Errorf("expected hard clear placeholder, got: %s", result[1].Content)
	}
}

func TestPruneToolResults_HardClear_MultipleToolResults(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.10,
		SoftTrimMaxChars:     500,
		SoftTrimHeadChars:    200,
		SoftTrimTailChars:    200,
		HardClearPlaceholder: "[CLEARED]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	longResult := makeLongString(8000)
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(longResult),
		{Role: openai.ChatMessageRoleAssistant, Content: "ok"},
		makeToolResultMessage(longResult),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != "[CLEARED]" {
		t.Errorf("expected first tool result hard-cleared, got: %s", result[1].Content)
	}
	if result[3].Content == "[CLEARED]" {
		t.Errorf("expected latest tool result to be preserved, but it was hard-cleared")
	}
	if result[3].Content != longResult {
		t.Errorf("expected latest tool result preserved, got: %s", result[3].Content[:min(50, len(result[3].Content))])
	}
}

func TestPruneToolResults_DoesNotModifyOriginal(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	original := "this is the original content that must not change"
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		makeToolResultMessage(original),
	}

	agent.pruneToolResults(history)

	if history[1].Content != original {
		t.Error("pruneToolResults modified the original slice")
	}
}

func TestPruneToolResults_PreservesNonToolMessages(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.10,
		SoftTrimMaxChars:     500,
		SoftTrimHeadChars:    200,
		SoftTrimTailChars:    200,
		HardClearPlaceholder: "[CLEARED]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system prompt"},
		{Role: openai.ChatMessageRoleUser, Content: "user message"},
		{Role: openai.ChatMessageRoleAssistant, Content: "assistant message", ToolCalls: []openai.ToolCall{{ID: "c1", Function: openai.FunctionCall{Name: "test", Arguments: "{}"}}}},
		makeToolResultMessage(makeLongString(8000)),
		{Role: openai.ChatMessageRoleAssistant, Content: "final response"},
	}

	result := agent.pruneToolResults(history)

	if result[0].Content != "system prompt" {
		t.Errorf("system prompt modified: %s", result[0].Content)
	}
	if result[1].Content != "user message" {
		t.Errorf("user message modified: %s", result[1].Content)
	}
	if result[2].Content != "assistant message" {
		t.Errorf("assistant message modified: %s", result[2].Content)
	}
	if result[4].Content != "final response" {
		t.Errorf("final response modified: %s", result[4].Content)
	}
}

func TestPruneToolResults_EmptyHistory(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	result := agent.pruneToolResults([]openai.ChatCompletionMessage{})
	if len(result) != 0 {
		t.Errorf("expected empty result for empty input, got %d messages", len(result))
	}
}

func TestPruneToolResults_BelowThreshold_NoPruning(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.80,
		HardClearRatio:       0.95,
		SoftTrimMaxChars:     500,
		SoftTrimHeadChars:    200,
		SoftTrimTailChars:    200,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	largeNonTool := makeLongString(20000)
	smallTool := "small tool result"

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: largeNonTool},
		makeToolResultMessage(smallTool),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != smallTool {
		t.Error("tool result below threshold should not be pruned")
	}
}

// ==========================================
// TIER 1: PRUNING RATIO CALCULATION TESTS
// ==========================================

func TestPruneToolResults_RatioCalculation(t *testing.T) {
	tests := []struct {
		name            string
		nonToolContent  string
		toolContent     string
		numToolResults  int
		softTrimRatio   float64
		hardClearRatio  float64
		expectTrimmed   bool
		expectHardClear bool
	}{
		{
			name:           "below threshold - no pruning",
			nonToolContent: makeLongString(10000),
			toolContent:    "short",
			numToolResults: 1,
			softTrimRatio:  0.20,
			hardClearRatio: 0.35,
			expectTrimmed:  false,
		},
		{
			name:            "above soft trim - trim",
			nonToolContent:  "short",
			toolContent:     makeLongString(5000),
			numToolResults:  1,
			softTrimRatio:   0.10,
			hardClearRatio:  0.90,
			expectTrimmed:   true,
			expectHardClear: false,
		},
		{
			name:            "above hard clear - hard clear",
			nonToolContent:  "s",
			toolContent:     makeLongString(5000),
			numToolResults:  5,
			softTrimRatio:   0.05,
			hardClearRatio:  0.10,
			expectTrimmed:   true,
			expectHardClear: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := PruningConfig{
				Mode:                 "aggressive",
				SoftTrimRatio:        tt.softTrimRatio,
				HardClearRatio:       tt.hardClearRatio,
				SoftTrimMaxChars:     300,
				SoftTrimHeadChars:    100,
				SoftTrimTailChars:    100,
				HardClearPlaceholder: "[HARD]",
			}
			agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

			history := []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleSystem, Content: tt.nonToolContent},
			}
			for i := 0; i < tt.numToolResults; i++ {
				history = append(history, makeToolResultMessage(tt.toolContent))
			}

			result := agent.pruneToolResults(history)

			for i := 1; i < len(result); i++ {
				if result[i].Role == openai.ChatMessageRoleTool {
					isLatest := i == len(result)-1
					if tt.expectHardClear {
						if isLatest && tt.numToolResults > 1 {
							if result[i].Content == "[HARD]" {
								t.Errorf("expected latest tool result %d to be preserved, but it was hard-cleared", i)
							}
						} else if !isLatest || tt.numToolResults == 1 {
							if result[i].Content != "[HARD]" {
								t.Errorf("expected hard clear for tool result %d", i)
							}
						}
					} else if tt.expectTrimmed {
						if result[i].Content == tt.toolContent && len(tt.toolContent) > 300 {
							t.Errorf("expected tool result %d to be trimmed", i)
						}
					} else {
						if result[i].Content != tt.toolContent {
							t.Errorf("tool result %d should not be modified", i)
						}
					}
				}
			}
		})
	}
}

// ==========================================
// TIER 2: COMPACTION TESTS
// ==========================================

func TestCompactHistoryIfNeeded_ShortHistory_NoCompaction(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hi"},
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_ZeroThreshold_NoCompaction(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	agent.reserveTokens = 0
	agent.keepRecentTokens = 0

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "msg"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "msg"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "msg"},
	)

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 4 {
		t.Fatalf("expected 4 messages with zero thresholds, got %d", len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_BelowThreshold_NoCompaction(t *testing.T) {
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, DefaultPruningConfig)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "short message"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "short reply"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "another message"},
	)

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 4 {
		t.Fatalf("expected 4 messages (below threshold), got %d", len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_TriggeredWithMockLLM(t *testing.T) {
	summaryCalled := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) >= 2 {
			userMsg := req.Messages[len(req.Messages)-1].Content
			if strings.Contains(userMsg, "[user]:") || strings.Contains(userMsg, "[User]:") {
				summaryCalled++
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary of conversation."}},
					},
				}
				json.NewEncoder(w).Encode(resp)
				return
			}
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("This is message number %d with some content to make it longer", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summaryCalled == 0 {
		t.Error("expected summary LLM call during compaction")
	}

	found := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "[CONTEXT SUMMARY]") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected summary message in history after compaction")
	}

	if len(agent.history) >= 21 {
		t.Errorf("expected fewer messages after compaction, got %d", len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_PreservesSystemPrompt(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d with padding to exceed threshold", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("system prompt must be first message after compaction")
	}
}

func TestCompactHistoryIfNeeded_UpdatesWAL(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	for i := range 20 {
		msg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d with padding to exceed threshold", i),
		}
		agent.history = append(agent.history, msg)
		wal.Append(msg)
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	loaded, err := wal.LoadAll()
	if err != nil {
		t.Fatalf("WAL load error: %v", err)
	}

	if len(loaded) != len(agent.history) {
		t.Errorf("WAL (%d messages) out of sync with history (%d messages)", len(loaded), len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_NoWAL(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := newTestAgentWithConfig(client, nil, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d with padding to exceed threshold", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error without WAL: %v", err)
	}
}

// ==========================================
// TIER 2: MEMORY FLUSH BEFORE COMPACTION
// ==========================================

func TestCompactHistoryIfNeeded_MemoryFlushBeforeCompaction(t *testing.T) {
	flushCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		flushCount++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Extracted facts."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	dir := t.TempDir()
	wal := &WAL{path: filepath.Join(dir, "test.wal")}

	agent := newTestAgentWithConfig(client, wal, 500, DefaultPruningConfig)
	agent.memory = NewMemoryWithoutRateLimit(filepath.Join(dir, "memory"))

	for i := range 50 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("Message %d: some content here to build up tokens for the flush test", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if flushCount == 0 {
		t.Error("expected at least one LLM call for memory flush or summary")
	}
}

func TestCompactHistoryIfNeeded_NoMemory_NoFlushCrash(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := newTestAgentWithConfig(client, nil, 200, DefaultPruningConfig)
	agent.memory = nil

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d padding", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("compaction should not crash without memory: %v", err)
	}
}

// ==========================================
// TIER 2: CUSTOM SUMMARIZER
// ==========================================

func TestCompactHistoryIfNeeded_CustomSummarizer(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Default summary"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	customCalled := false
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		customCalled = true
		return "Custom summary from test", nil
	}

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d with padding", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !customCalled {
		t.Error("expected custom summarizer to be called")
	}

	found := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "Custom summary from test") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected custom summary in history")
	}
}

// ==========================================
// TIER 3: ClearHistory() TESTS
// ==========================================

func TestClearHistory_ResetsToSystemPrompt(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "how are you"},
	)

	err := agent.ClearHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(agent.history) != 1 {
		t.Fatalf("expected 1 message (system prompt only), got %d", len(agent.history))
	}
	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("expected system prompt, got role %s", agent.history[0].Role)
	}
}

func TestClearHistory_PreservesSystemPrompt(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	originalPrompt := agent.history[0].Content

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	)

	agent.ClearHistory()

	if agent.history[0].Content != originalPrompt {
		t.Errorf("system prompt changed after clear: expected %q, got %q", originalPrompt, agent.history[0].Content)
	}
}

func TestClearHistory_ArchivesWAL(t *testing.T) {
	wal, cleanup := newTestWAL(t)
	defer cleanup()

	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "test"})
	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "response"})

	agent := newDefaultTestAgent(nil, wal)
	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"},
	)

	err := agent.ClearHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	matches, _ := filepath.Glob(wal.path + ".archived.*")
	if len(matches) == 0 {
		t.Error("expected archived WAL file")
	}
}

func TestClearHistory_RewritesWAL(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "test.wal")
	wal := &WAL{path: walPath}

	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: "system"})
	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old message"})

	agent := newDefaultTestAgent(nil, wal)

	loaded, _ := wal.LoadAll()
	if len(loaded) < 1 {
		t.Fatalf("WAL should have messages before clear, got %d", len(loaded))
	}

	err := agent.ClearHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	loaded, err = wal.LoadAll()
	if err != nil {
		t.Fatalf("WAL load error: %v", err)
	}

	if len(loaded) != 1 {
		t.Fatalf("expected 1 message in new WAL, got %d", len(loaded))
	}
	if loaded[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("expected system prompt in new WAL, got %q", loaded[0].Role)
	}
}

func TestClearHistory_NoWAL_NoCrash(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	err := agent.ClearHistory()
	if err != nil {
		t.Fatalf("ClearHistory without WAL should not crash: %v", err)
	}
}

func TestClearHistory_AllowsNewConversation(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old conversation"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "old response"},
	)

	agent.ClearHistory()

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "new conversation"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "new response"},
	)

	if len(agent.history) != 3 {
		t.Fatalf("expected 3 messages after new conversation, got %d", len(agent.history))
	}
	if agent.history[1].Content != "new conversation" {
		t.Errorf("expected new conversation message, got %q", agent.history[1].Content)
	}
}

// ==========================================
// TIER 3: GetHistoryStats() TESTS
// ==========================================

func TestGetHistoryStats_Basic(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "how are you"},
	)

	stats := agent.GetHistoryStats()

	if stats.MessageCount != 4 {
		t.Errorf("expected 4 messages, got %d", stats.MessageCount)
	}
	if stats.ToolResultCount != 0 {
		t.Errorf("expected 0 tool results, got %d", stats.ToolResultCount)
	}
	if stats.TokenCount <= 0 {
		t.Error("expected positive token count")
	}
}

func TestGetHistoryStats_ToolResultCount(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "check weather"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, Content: "Sunny 72F"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "check again"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, Content: "Rainy 65F"},
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "done"},
	)

	stats := agent.GetHistoryStats()

	if stats.ToolResultCount != 2 {
		t.Errorf("expected 2 tool results, got %d", stats.ToolResultCount)
	}
	if stats.MessageCount != 6 {
		t.Errorf("expected 6 messages, got %d", stats.MessageCount)
	}
}

func TestGetHistoryStats_ThresholdValues(t *testing.T) {
	agent := newTestAgentWithConfig(nil, nil, 10000, DefaultPruningConfig)

	stats := agent.GetHistoryStats()

	if stats.ContextWindow != 10000 {
		t.Errorf("expected ContextWindow 10000, got %d", stats.ContextWindow)
	}
	expectedReserve := int(10000 * ReserveTokensRatio)
	if stats.ReserveTokens != expectedReserve {
		t.Errorf("expected ReserveTokens %d, got %d", expectedReserve, stats.ReserveTokens)
	}
	expectedKeep := int(10000 * KeepRecentTokensRatio)
	if stats.KeepRecentTokens != expectedKeep {
		t.Errorf("expected KeepRecentTokens %d, got %d", expectedKeep, stats.KeepRecentTokens)
	}
}

func TestGetHistoryStats_EmptyHistory(t *testing.T) {
	agent := &Agent{
		history:          []openai.ChatCompletionMessage{},
		contextWindow:    DefaultContextWindow,
		reserveTokens:    int(float64(DefaultContextWindow) * ReserveTokensRatio),
		keepRecentTokens: int(float64(DefaultContextWindow) * KeepRecentTokensRatio),
	}

	stats := agent.GetHistoryStats()

	if stats.MessageCount != 0 {
		t.Errorf("expected 0 messages, got %d", stats.MessageCount)
	}
	if stats.TokenCount != 0 {
		t.Errorf("expected 0 tokens, got %d", stats.TokenCount)
	}
	if stats.ToolResultCount != 0 {
		t.Errorf("expected 0 tool results, got %d", stats.ToolResultCount)
	}
}

func TestGetHistoryStats_TokenCountMatches(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello world"},
	)

	stats := agent.GetHistoryStats()
	expected := countTokens(agent.history)

	if stats.TokenCount != expected {
		t.Errorf("expected token count %d, got %d", expected, stats.TokenCount)
	}
}

// ==========================================
// DYNAMIC THRESHOLD CALCULATION TESTS
// ==========================================

func TestNewAgent_DynamicThresholdCalculation(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)

	expectedReserve := int(float64(DefaultContextWindow) * ReserveTokensRatio)
	expectedKeep := int(float64(DefaultContextWindow) * KeepRecentTokensRatio)

	if agent.contextWindow != DefaultContextWindow {
		t.Errorf("expected contextWindow %d, got %d", DefaultContextWindow, agent.contextWindow)
	}
	if agent.reserveTokens != expectedReserve {
		t.Errorf("expected reserveTokens %d, got %d", expectedReserve, agent.reserveTokens)
	}
	if agent.keepRecentTokens != expectedKeep {
		t.Errorf("expected keepRecentTokens %d, got %d", expectedKeep, agent.keepRecentTokens)
	}
}

func TestNewAgent_DefaultPruningConfig(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)

	if agent.pruningConfig.Mode != DefaultPruningConfig.Mode {
		t.Errorf("expected pruning mode %q, got %q", DefaultPruningConfig.Mode, agent.pruningConfig.Mode)
	}
	if agent.pruningConfig.SoftTrimRatio != DefaultPruningConfig.SoftTrimRatio {
		t.Errorf("expected SoftTrimRatio %f, got %f", DefaultPruningConfig.SoftTrimRatio, agent.pruningConfig.SoftTrimRatio)
	}
	if agent.pruningConfig.HardClearRatio != DefaultPruningConfig.HardClearRatio {
		t.Errorf("expected HardClearRatio %f, got %f", DefaultPruningConfig.HardClearRatio, agent.pruningConfig.HardClearRatio)
	}
}

func TestNewAgent_WithWAL_InitializesMemory(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "test.wal")
	wal := &WAL{path: walPath}

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, wal)

	if agent.memory == nil {
		t.Fatal("expected memory to be initialized when WAL is present")
	}
	if agent.wal == nil {
		t.Fatal("expected WAL to be set")
	}
}

func TestNewAgent_WithoutWAL_NoMemory(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)

	if agent.memory != nil {
		t.Error("expected nil memory without WAL")
	}
}

func TestNewAgent_RestoresFromWAL_ContextMgmt(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "test.wal")
	wal := &WAL{path: walPath}

	msgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}
	for _, m := range msgs {
		wal.Append(m)
	}

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, wal)

	if len(agent.history) != 3 {
		t.Fatalf("expected 3 messages restored from WAL, got %d", len(agent.history))
	}
	if agent.history[1].Content != "hello" {
		t.Errorf("expected restored message 'hello', got %q", agent.history[1].Content)
	}
}

func TestNewAgent_WALWithoutSystem_AddsSystem(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "test.wal")
	wal := &WAL{path: walPath}

	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"})

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, wal)

	if len(agent.history) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(agent.history))
	}
	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("first message should be system prompt")
	}
}

// ==========================================
// CONTEXT WINDOW CONSTANTS TESTS
// ==========================================

func TestContextWindowConstants(t *testing.T) {
	if DefaultContextWindow != 128000 {
		t.Errorf("DefaultContextWindow should be 128000, got %d", DefaultContextWindow)
	}
	if ReserveTokensRatio != 0.50 {
		t.Errorf("ReserveTokensRatio should be 0.50, got %f", ReserveTokensRatio)
	}
	if KeepRecentTokensRatio != 0.60 {
		t.Errorf("KeepRecentTokensRatio should be 0.60, got %f", KeepRecentTokensRatio)
	}
	if MemoryFlushThreshold != 4000 {
		t.Errorf("MemoryFlushThreshold should be 4000, got %d", MemoryFlushThreshold)
	}
}

func TestReserveTokens_CalculatedCorrectly(t *testing.T) {
	tests := []struct {
		contextWindow int
		expected      int
	}{
		{128000, 64000},
		{40000, 20000},
		{10000, 5000},
		{8000, 4000},
		{768, 384},
		{100000, 50000},
	}

	for _, tt := range tests {
		result := int(float64(tt.contextWindow) * ReserveTokensRatio)
		if result != tt.expected {
			t.Errorf("contextWindow=%d: expected reserve %d, got %d", tt.contextWindow, tt.expected, result)
		}
	}
}

func TestKeepRecentTokens_CalculatedCorrectly(t *testing.T) {
	tests := []struct {
		contextWindow int
		expected      int
	}{
		{128000, 76800},
		{40000, 24000},
		{10000, 6000},
		{8000, 4800},
		{768, 460},
	}

	for _, tt := range tests {
		result := int(float64(tt.contextWindow) * KeepRecentTokensRatio)
		if result != tt.expected {
			t.Errorf("contextWindow=%d: expected keepRecent %d, got %d", tt.contextWindow, tt.expected, result)
		}
	}
}

// ==========================================
// INTEGRATION: Chat() with pruning
// ==========================================

func TestChat_AppliesPruningBeforeLLMCall(t *testing.T) {
	receivedHistory := []openai.ChatCompletionMessage{}

	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)
		receivedHistory = req.Messages

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "done"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.10,
		SoftTrimMaxChars:     500,
		SoftTrimHeadChars:    200,
		SoftTrimTailChars:    200,
		HardClearPlaceholder: "[CLEARED]",
	}

	agent := newTestAgentWithConfig(client, nil, DefaultContextWindow, cfg)

	longToolResult := makeLongString(5000)
	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "run tool"},
		makeToolResultMessage(longToolResult),
	)

	_, err := agent.Chat(context.Background(), "now what?")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	for _, m := range receivedHistory {
		if m.Role == openai.ChatMessageRoleTool {
			if m.Content == longToolResult {
				t.Error("tool result should have been pruned before sending to LLM")
			}
			if m.Content == "[CLEARED]" {
				return
			}
		}
	}
	t.Error("no tool result found in LLM request")
}

func TestChat_DoesNotModifyHistoryDuringPruning(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := newDefaultTestAgent(client, nil)

	originalContent := "original tool result content"
	agent.history = append(agent.history,
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "run tool"},
		makeToolResultMessage(originalContent),
	)

	_, err := agent.Chat(context.Background(), "follow up")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	found := false
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleTool && m.Content == originalContent {
			found = true
			break
		}
	}
	if !found {
		t.Error("original history was modified by pruning (pruning should only affect the copy sent to LLM)")
	}
}

func TestChat_CompactionTriggeredDuringConversation(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if strings.Contains(req.Messages[0].Content, "Summarize") ||
			strings.Contains(req.Messages[0].Content, "Extract") ||
			strings.Contains(req.Messages[0].Content, "summarize") ||
			(len(req.Messages) >= 2 && (strings.Contains(req.Messages[len(req.Messages)-1].Content, "[user]:") || strings.Contains(req.Messages[len(req.Messages)-1].Content, "[User]:"))) {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary/facts here."}},
				},
			}
			json.NewEncoder(w).Encode(resp)
			return
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "response"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, PruningConfig{Mode: "off"})

	for i := range 30 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("This is message %d with enough content to use tokens", i),
		})
	}

	_, err := agent.Chat(context.Background(), "trigger compaction")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if callCount < 2 {
		t.Errorf("expected multiple LLM calls (compaction + response), got %d", callCount)
	}
}

// ==========================================
// INTEGRATION: ClearHistory + GetHistoryStats + Chat
// ==========================================

func TestIntegration_ClearAndRestart(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "response"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newDefaultTestAgent(client, wal)

	_, err := agent.Chat(context.Background(), "first message")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	stats1 := agent.GetHistoryStats()
	if stats1.MessageCount != 3 {
		t.Fatalf("expected 3 messages before clear, got %d", stats1.MessageCount)
	}

	agent.ClearHistory()

	stats2 := agent.GetHistoryStats()
	if stats2.MessageCount != 1 {
		t.Fatalf("expected 1 message after clear, got %d", stats2.MessageCount)
	}

	_, err = agent.Chat(context.Background(), "new session message")
	if err != nil {
		t.Fatalf("Chat error after clear: %v", err)
	}

	stats3 := agent.GetHistoryStats()
	if stats3.MessageCount != 3 {
		t.Fatalf("expected 3 messages in new session, got %d", stats3.MessageCount)
	}
}

// ==========================================
// EDGE CASES
// ==========================================

func TestPruneToolResults_AllMessagesAreToolResults(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.10,
		HardClearRatio:       0.50,
		SoftTrimMaxChars:     300,
		SoftTrimHeadChars:    100,
		SoftTrimTailChars:    100,
		HardClearPlaceholder: "[HARD]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	history := []openai.ChatCompletionMessage{
		makeToolResultMessage(makeLongString(5000)),
		makeToolResultMessage(makeLongString(5000)),
		makeToolResultMessage(makeLongString(5000)),
	}

	result := agent.pruneToolResults(history)

	for i, m := range result {
		if m.Role == openai.ChatMessageRoleTool {
			if i == len(result)-1 {
				if m.Content == "[HARD]" {
					t.Errorf("expected latest tool result to be preserved at index %d, but it was hard-cleared", i)
				}
			} else {
				if m.Content != "[HARD]" {
					t.Errorf("expected hard clear for all-tool-result history at index %d, got: %s", i, m.Content[:min(50, len(m.Content))])
				}
			}
		}
	}
}

func TestPruneToolResults_SingleMessage_NoPanic(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	history := []openai.ChatCompletionMessage{
		makeToolResultMessage("single tool result"),
	}

	result := agent.pruneToolResults(history)
	if len(result) != 1 {
		t.Errorf("expected 1 message, got %d", len(result))
	}
}

func TestCompactHistoryIfNeeded_SummaryGenerationFailure(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d with padding", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("compaction should not fail even if LLM fails: %v", err)
	}

	if len(agent.history) == 0 {
		t.Fatal("history should not be empty after failed summary")
	}

	found := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "[CONTEXT SUMMARY]") && strings.Contains(m.Content, "failed") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected fallback summary message when LLM fails")
	}
}

func TestCompactHistoryIfNeeded_CancelledContext(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	wal, cleanup := newTestWAL(t)
	defer cleanup()

	agent := newTestAgentWithConfig(client, wal, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d", i),
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := agent.compactHistoryIfNeeded(ctx)
	if err == nil {
		t.Log("compaction with cancelled context returned nil (acceptable)")
	}
}

func TestCompactHistoryIfNeeded_ConcurrentSafety(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := newTestAgentWithConfig(client, nil, 200, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message %d padding content", i),
		})
	}

	done := make(chan bool, 10)
	for range 10 {
		go func() {
			agent.compactHistoryIfNeeded(context.Background())
			agent.GetHistoryStats()
			done <- true
		}()
	}

	for range 10 {
		<-done
	}
}

// ==========================================
// MEMORY SYSTEM TESTS (for compaction integration)
// ==========================================

func TestMemoryFlush_WritesToDisk(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "- Fact 1\n- Fact 2"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	err := mem.Flush(context.Background(), client, DefaultModel, history)
	if err != nil {
		t.Fatalf("Flush error: %v", err)
	}

	memFile := filepath.Join(dir, "MEMORY.md")
	data, err := os.ReadFile(memFile)
	if err != nil {
		t.Fatalf("failed to read memory file: %v", err)
	}

	if !strings.Contains(string(data), "Fact 1") {
		t.Errorf("expected memory file to contain extracted facts, got: %s", string(data)[:min(200, len(string(data)))])
	}
}

func TestMemoryFlush_TooFewMessages(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	}

	err := mem.Flush(context.Background(), nil, "", history)
	if err != nil {
		t.Fatalf("Flush with few messages should return nil, got: %v", err)
	}

	memFile := filepath.Join(dir, "MEMORY.md")
	if _, err := os.Stat(memFile); !os.IsNotExist(err) {
		t.Error("memory file should not be created with < 3 messages")
	}
}

func TestMemoryFlush_RateLimit(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "fact"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	dir := t.TempDir()
	mem := NewMemory(dir)
	mem.SetMinDelay(1 * time.Hour)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	mem.Flush(context.Background(), client, DefaultModel, history)
	mem.Flush(context.Background(), client, DefaultModel, history)
	mem.Flush(context.Background(), client, DefaultModel, history)

	if callCount > 1 {
		t.Errorf("expected rate limiting to prevent multiple flushes, got %d calls", callCount)
	}
}

func TestMemoryRead_ReturnsContent(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	os.MkdirAll(dir, 0755)
	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("test memory content"), 0644)

	content, err := mem.ReadMemory()
	if err != nil {
		t.Fatalf("ReadMemory error: %v", err)
	}
	if content != "test memory content" {
		t.Errorf("expected 'test memory content', got %q", content)
	}
}

func TestMemoryRead_NoFile(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	content, err := mem.ReadMemory()
	if err != nil {
		t.Fatalf("ReadMemory on non-existent file should not error: %v", err)
	}
	if content != "" {
		t.Errorf("expected empty content, got %q", content)
	}
}

// ==========================================
// countTokens / findRecentStart TESTS
// ==========================================

func TestCountTokens_Empty(t *testing.T) {
	result := countTokens([]openai.ChatCompletionMessage{})
	if result != 0 {
		t.Errorf("expected 0 tokens for empty slice, got %d", result)
	}
}

func TestCountTokens_SingleMessage_CM(t *testing.T) {
	result := countTokens([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	})
	if result <= 0 {
		t.Errorf("expected positive token count, got %d", result)
	}
}

func TestCountTokens_MultipleMessages_CM(t *testing.T) {
	msgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "world"},
	}
	single := countTokens(msgs[:1])
	both := countTokens(msgs)

	if both <= single {
		t.Errorf("expected more tokens for 2 messages (%d) than 1 (%d)", both, single)
	}
}

func TestFindRecentStart_Empty(t *testing.T) {
	result := findRecentStart([]openai.ChatCompletionMessage{}, 1000)
	if result != 0 {
		t.Errorf("expected 0 for empty slice, got %d", result)
	}
}

func TestFindRecentStart_AllFit_CM(t *testing.T) {
	msgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "short"},
		{Role: openai.ChatMessageRoleAssistant, Content: "reply"},
	}

	result := findRecentStart(msgs, 10000)
	if result != 0 {
		t.Errorf("expected 0 when all fit in budget, got %d", result)
	}
}

func TestFindRecentStart_ExceedsBudget(t *testing.T) {
	msgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: makeLongString(5000)},
		{Role: openai.ChatMessageRoleAssistant, Content: makeLongString(5000)},
		{Role: openai.ChatMessageRoleUser, Content: "recent"},
	}

	result := findRecentStart(msgs, 100)
	if result == 0 {
		t.Error("expected non-zero start when messages exceed budget")
	}
	if result >= len(msgs) {
		t.Errorf("start index %d out of range (len=%d)", result, len(msgs))
	}
}

// ==========================================
// BUILD SYSTEM PROMPT TESTS
// ==========================================

func TestBuildSystemPrompt_NoMemory_CM(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	agent.memory = nil

	prompt := agent.buildSystemPrompt()
	if !strings.HasPrefix(prompt, "Current date and time:") {
		t.Errorf("expected date prefix, got %q", prompt)
	}
	if !strings.Contains(prompt, agent.baseSystemPrompt) {
		t.Errorf("without memory, prompt should contain base: got %q", prompt)
	}
}

func TestBuildSystemPrompt_WithMemory_CM(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	os.MkdirAll(dir, 0755)
	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("User likes pizza"), 0644)

	agent := newDefaultTestAgent(nil, nil)
	agent.memory = mem

	prompt := agent.buildSystemPrompt()
	if !strings.Contains(prompt, "User likes pizza") {
		t.Errorf("expected memory content in prompt, got: %s", prompt)
	}
	if !strings.Contains(prompt, "Remembered Context") {
		t.Errorf("expected 'Remembered Context' section, got: %s", prompt)
	}
}

func TestRefreshSystemPrompt_UpdatesHistory_CM(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	os.MkdirAll(dir, 0755)

	agent := newDefaultTestAgent(nil, nil)
	agent.memory = mem

	originalContent := agent.history[0].Content

	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("New fact discovered"), 0644)

	agent.RefreshSystemPrompt()

	if agent.history[0].Content == originalContent {
		t.Error("system prompt in history should be updated after refresh")
	}
	if !strings.Contains(agent.history[0].Content, "New fact discovered") {
		t.Errorf("expected updated prompt to contain new memory: %s", agent.history[0].Content)
	}
}
