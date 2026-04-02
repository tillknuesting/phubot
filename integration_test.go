//go:build integration

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func createIntegrationAgent(t *testing.T, handler http.HandlerFunc) (*Agent, *WAL, *httptest.Server) {
	t.Helper()

	dir := t.TempDir()
	wal := &WAL{path: filepath.Join(dir, "integration.wal")}

	server := httptest.NewServer(handler)

	config := openai.DefaultConfig("test-integration-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, wal)
	return agent, wal, server
}

func assertHistorySequence(t *testing.T, agent *Agent, expectedRoles []string) {
	t.Helper()
	if len(agent.history) != len(expectedRoles) {
		t.Fatalf("history length mismatch: expected %d, got %d", len(expectedRoles), len(agent.history))
	}
	for i, expectedRole := range expectedRoles {
		if agent.history[i].Role != expectedRole {
			t.Fatalf("role mismatch at index %d: expected %s, got %s", i, expectedRole, agent.history[i].Role)
		}
	}
}

func simulateRestart(t *testing.T, wal *WAL, handler http.HandlerFunc) (*Agent, *httptest.Server) {
	t.Helper()

	server := httptest.NewServer(handler)

	config := openai.DefaultConfig("test-integration-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	return NewAgent(client, wal), server
}

// ==========================================
// 1. AGENT LIFECYCLE
// ==========================================

func TestIntegration_AgentLifecycle(t *testing.T) {
	chatCount := 0
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		chatCount++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: fmt.Sprintf("response_%d", chatCount),
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	reply1, err := agent.Chat(context.Background(), "hello")
	if err != nil {
		t.Fatalf("first chat error: %v", err)
	}
	if reply1 != "response_1" {
		t.Fatalf("unexpected first reply: %s", reply1)
	}

	reply2, err := agent.Chat(context.Background(), "how are you")
	if err != nil {
		t.Fatalf("second chat error: %v", err)
	}
	if reply2 != "response_2" {
		t.Fatalf("unexpected second reply: %s", reply2)
	}

	reply3, err := agent.Chat(context.Background(), "goodbye")
	if err != nil {
		t.Fatalf("third chat error: %v", err)
	}
	if reply3 != "response_3" {
		t.Fatalf("unexpected third reply: %s", reply3)
	}

	assertHistorySequence(t, agent, []string{
		openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
	})

	loaded, err := wal.LoadAll()
	if err != nil {
		t.Fatalf("wal load error: %v", err)
	}
	if len(loaded) != 7 {
		t.Fatalf("expected 7 messages in WAL, got %d", len(loaded))
	}
}

// ==========================================
// 2. WAL PERSISTENCE
// ==========================================

func TestIntegration_WALPersistence(t *testing.T) {
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "persisted response",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.Chat(context.Background(), "message 1")
	agent.Chat(context.Background(), "message 2")
	agent.Chat(context.Background(), "message 3")

	originalHistory := make([]openai.ChatCompletionMessage, len(agent.history))
	copy(originalHistory, agent.history)

	restartedAgent, restartServer := simulateRestart(t, wal, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "after restart",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer restartServer.Close()

	if len(restartedAgent.history) != len(originalHistory) {
		t.Fatalf("history length after restart: expected %d, got %d", len(originalHistory), len(restartedAgent.history))
	}

	for i := range originalHistory {
		if restartedAgent.history[i].Role != originalHistory[i].Role {
			t.Fatalf("role mismatch at %d after restart", i)
		}
		if restartedAgent.history[i].Content != originalHistory[i].Content {
			t.Fatalf("content mismatch at %d after restart: expected %q, got %q",
				i, originalHistory[i].Content, restartedAgent.history[i].Content)
		}
	}

	restartedAgent.Chat(context.Background(), "after restart message")

	if len(restartedAgent.history) != len(originalHistory)+2 {
		t.Fatalf("history should have 2 more messages after post-restart chat: got %d vs %d",
			len(restartedAgent.history), len(originalHistory))
	}
}

// ==========================================
// 3. REACT LOOP
// ==========================================

func TestIntegration_ReActLoop(t *testing.T) {
	callCount := 0
	agent, _, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"query": "test"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "react_call_123",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "search_tool",
										Arguments: string(args),
									},
								},
							},
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Based on the search results, here is the answer.",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	var toolExecuted bool
	agent.RegisterTool(&MockTool{
		name: "search_tool",
		executeFn: func(args string) (string, error) {
			toolExecuted = true
			return "search results: found 3 items", nil
		},
	})

	reply, err := agent.Chat(context.Background(), "search for something")
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	if !toolExecuted {
		t.Fatal("tool should have been executed during ReAct loop")
	}
	if callCount != 2 {
		t.Fatalf("expected 2 LLM calls (tool + final), got %d", callCount)
	}
	if reply != "Based on the search results, here is the answer." {
		t.Fatalf("unexpected final reply: %s", reply)
	}

	assertHistorySequence(t, agent, []string{
		openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
		openai.ChatMessageRoleAssistant,
	})

	if agent.history[3].ToolCallID != "react_call_123" {
		t.Fatalf("tool message should have ToolCallID 'react_call_123', got %q", agent.history[3].ToolCallID)
	}
	if agent.history[3].Name != "search_tool" {
		t.Fatalf("tool message should have Name 'search_tool', got %q", agent.history[3].Name)
	}
}

// ==========================================
// 4. MULTIPLE TOOLS
// ==========================================

func TestIntegration_MultipleTools(t *testing.T) {
	callCount := 0
	agent, _, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"x": "y"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_tool_2",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_2",
										Arguments: string(args),
									},
								},
							},
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "tool 2 executed",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	tools := make([]*CountingMockTool, 5)
	for i := 0; i < 5; i++ {
		tools[i] = &CountingMockTool{name: fmt.Sprintf("tool_%d", i)}
		agent.RegisterTool(tools[i])
	}

	_, err := agent.Chat(context.Background(), "use tool 2")
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	for i, tool := range tools {
		count := tool.CallCount()
		if i == 2 {
			if count != 1 {
				t.Fatalf("tool_2 should have been called once, got %d", count)
			}
		} else {
			if count != 0 {
				t.Fatalf("tool_%d should not have been called, got %d", i, count)
			}
		}
	}
}

// ==========================================
// 5. TOOL FAILURE RECOVERY
// ==========================================

func TestIntegration_ToolFailureRecovery(t *testing.T) {
	callCount := 0
	agent, _, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"x": "1"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_failing",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "failing_tool",
										Arguments: string(args),
									},
								},
							},
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else if callCount == 2 {
			args, _ := json.Marshal(map[string]string{"x": "2"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_backup",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "backup_tool",
										Arguments: string(args),
									},
								},
							},
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "recovered using backup tool",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	var failingCalled, backupCalled bool
	agent.RegisterTool(&MockTool{
		name: "failing_tool",
		executeFn: func(args string) (string, error) {
			failingCalled = true
			return "", fmt.Errorf("tool failed intentionally")
		},
	})
	agent.RegisterTool(&MockTool{
		name: "backup_tool",
		executeFn: func(args string) (string, error) {
			backupCalled = true
			return "backup success", nil
		},
	})

	reply, err := agent.Chat(context.Background(), "do something")
	if err != nil {
		t.Fatalf("chat should not error: %v", err)
	}

	if !failingCalled {
		t.Fatal("failing_tool should have been called")
	}
	if !backupCalled {
		t.Fatal("backup_tool should have been called after failure")
	}
	if reply != "recovered using backup tool" {
		t.Fatalf("unexpected reply: %s", reply)
	}

	var foundErrorMsg bool
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleTool && strings.Contains(m.Content, "Error executing tool") {
			foundErrorMsg = true
			break
		}
	}
	if !foundErrorMsg {
		t.Fatal("error message should be in history")
	}

	assertHistorySequence(t, agent, []string{
		openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
		openai.ChatMessageRoleAssistant,
	})
}

// ==========================================
// 6. CIRCUIT BREAKER
// ==========================================

func TestIntegration_CircuitBreaker(t *testing.T) {
	callCount := 0
	agent, _, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		callCount++
		args, _ := json.Marshal(map[string]string{"x": "y"})
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   fmt.Sprintf("call_%d", callCount),
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "loop_tool",
									Arguments: string(args),
								},
							},
						},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.RegisterTool(&MockTool{name: "loop_tool"})
	agent.SetMaxIterations(5)

	_, err := agent.Chat(context.Background(), "trigger loop")
	if err == nil {
		t.Fatal("expected circuit breaker error")
	}
	if !strings.Contains(err.Error(), "circuit breaker") {
		t.Fatalf("expected circuit breaker error, got: %v", err)
	}
	if callCount != 5 {
		t.Fatalf("expected exactly 5 LLM calls, got %d", callCount)
	}

	if len(agent.history) != 12 {
		t.Fatalf("expected 12 messages (sys + u + 5*(a + t) + a_tool_msg), got %d", len(agent.history))
	}
}

// ==========================================
// 7. CONCURRENT SESSIONS
// ==========================================

func TestIntegration_ConcurrentSessions(t *testing.T) {
	var mu sync.Mutex
	callCount := 0

	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		callCount++
		mu.Unlock()
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "concurrent ok",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				_, err := agent.Chat(context.Background(), fmt.Sprintf("goroutine_%d_msg_%d", idx, j))
				if err != nil {
					errors <- fmt.Errorf("goroutine %d msg %d: %w", idx, j, err)
				}
			}
		}(i)
	}
	wg.Wait()
	close(errors)

	errCount := 0
	for err := range errors {
		t.Errorf("concurrent chat error: %v", err)
		errCount++
	}
	if errCount > 0 {
		t.Fatalf("%d concurrent errors occurred", errCount)
	}

	mu.Lock()
	total := callCount
	mu.Unlock()
	if total != 50 {
		t.Fatalf("expected 50 LLM calls (10 goroutines * 5 chats), got %d", total)
	}

	loaded, err := wal.LoadAll()
	if err != nil {
		t.Fatalf("wal load error: %v", err)
	}

	userMsgCount := 0
	for _, m := range loaded {
		if m.Role == openai.ChatMessageRoleUser {
			userMsgCount++
		}
	}
	if userMsgCount != 50 {
		t.Fatalf("expected 50 user messages in WAL, got %d", userMsgCount)
	}

	assistantMsgCount := 0
	for _, m := range loaded {
		if m.Role == openai.ChatMessageRoleAssistant {
			assistantMsgCount++
		}
	}
	if assistantMsgCount != 50 {
		t.Fatalf("expected 50 assistant messages in WAL, got %d", assistantMsgCount)
	}
}

// ==========================================
// 8. HISTORY ACCUMULATION
// ==========================================

func TestIntegration_HistoryAccumulation(t *testing.T) {
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "ok",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	for i := 0; i < 50; i++ {
		_, err := agent.Chat(context.Background(), fmt.Sprintf("message_%d", i))
		if err != nil {
			t.Fatalf("chat %d error: %v", i, err)
		}
	}

	expected := 1 + (2 * 50)
	if len(agent.history) != expected {
		t.Fatalf("expected %d messages in history, got %d", expected, len(agent.history))
	}

	for i := 0; i < 50; i++ {
		userIdx := 1 + (i * 2)
		assistantIdx := 2 + (i * 2)
		if agent.history[userIdx].Content != fmt.Sprintf("message_%d", i) {
			t.Fatalf("user message %d mismatch: expected 'message_%d', got %q", i, i, agent.history[userIdx].Content)
		}
		if agent.history[assistantIdx].Content != "ok" {
			t.Fatalf("assistant message %d mismatch", i)
		}
	}

	loaded, err := wal.LoadAll()
	if err != nil {
		t.Fatalf("wal load error: %v", err)
	}
	if len(loaded) != expected {
		t.Fatalf("expected %d messages in WAL, got %d", expected, len(loaded))
	}
}

// ==========================================
// 9. SYSTEM PROMPT PRESERVATION
// ==========================================

func TestIntegration_SystemPromptPreservation(t *testing.T) {
	dir := t.TempDir()
	wal := &WAL{path: filepath.Join(dir, "no_system.wal")}

	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"})
	wal.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "hi"})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "after injection",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, wal)

	if len(agent.history) != 3 {
		t.Fatalf("expected 3 messages (injected system + user + assistant), got %d", len(agent.history))
	}

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("first message should be system prompt, got %s", agent.history[0].Role)
	}
	if !strings.Contains(agent.history[0].Content, "autonomous") {
		t.Fatalf("system prompt content unexpected: %s", agent.history[0].Content)
	}

	agent.Chat(context.Background(), "new message")

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("system prompt should still be at index 0 after new chat")
	}

	if len(agent.history) != 5 {
		t.Fatalf("expected 5 messages after new chat, got %d", len(agent.history))
	}
}

// ==========================================
// 10. TOOL CALL ID MAPPING
// ==========================================

func TestIntegration_ToolCallIDMapping(t *testing.T) {
	callCount := 0
	agent, _, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args1, _ := json.Marshal(map[string]string{"x": "1"})
			args2, _ := json.Marshal(map[string]string{"x": "2"})
			args3, _ := json.Marshal(map[string]string{"x": "3"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_id_a",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_a",
										Arguments: string(args1),
									},
								},
								{
									ID:   "call_id_b",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_b",
										Arguments: string(args2),
									},
								},
								{
									ID:   "call_id_c",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_c",
										Arguments: string(args3),
									},
								},
							},
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "all three tools executed",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent.RegisterTool(&MockTool{name: "tool_a"})
	agent.RegisterTool(&MockTool{name: "tool_b"})
	agent.RegisterTool(&MockTool{name: "tool_c"})

	reply, err := agent.Chat(context.Background(), "run all tools")
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if reply != "all three tools executed" {
		t.Fatalf("unexpected reply: %s", reply)
	}

	toolMsgs := []openai.ChatCompletionMessage{}
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleTool {
			toolMsgs = append(toolMsgs, m)
		}
	}

	if len(toolMsgs) != 3 {
		t.Fatalf("expected 3 tool messages, got %d", len(toolMsgs))
	}

	toolCallIDs := make(map[string]bool)
	for _, m := range toolMsgs {
		toolCallIDs[m.ToolCallID] = true
	}
	if !toolCallIDs["call_id_a"] || !toolCallIDs["call_id_b"] || !toolCallIDs["call_id_c"] {
		t.Fatalf("missing expected ToolCallIDs, got: %v", toolCallIDs)
	}

	toolNames := make(map[string]bool)
	for _, m := range toolMsgs {
		toolNames[m.Name] = true
	}
	for _, name := range []string{"tool_a", "tool_b", "tool_c"} {
		if !toolNames[name] {
			t.Fatalf("tool name %q not found in tool messages", name)
		}
	}
}

// ==========================================
// 11. BROWSER TOOL
// ==========================================

func TestIntegration_BrowserTool(t *testing.T) {
	testHTML := `<!DOCTYPE html>
<html>
<head><title>Integration Test Page</title></head>
<body>
<h1>Integration Test</h1>
<p>This is a test paragraph for the browser tool.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
<li>Item 3</li>
</ul>
<div id="dynamic">Some content here</div>
</body>
</html>`

	htmlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(testHTML))
	}))
	defer htmlServer.Close()

	t.Logf("HTML test server running at: %s", htmlServer.URL)

	start := time.Now()

	tool := &BrowserTool{}
	args, _ := json.Marshal(map[string]string{
		"url": htmlServer.URL,
	})

	result, err := tool.Execute(string(args))
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("browser tool error: %v", err)
	}
	if elapsed > 20*time.Second {
		t.Fatalf("browser tool took too long: %v", elapsed)
	}

	if !strings.Contains(result, "BROWSER EXTRACTED") {
		t.Fatalf("result should contain 'BROWSER EXTRACTED TEXT', got: %s", result)
	}

	t.Logf("Browser tool completed in %v", elapsed)
}

// ==========================================
// 12. COMPACTION INTEGRATION
// ==========================================

func TestIntegration_CompactionWithLargeHistory(t *testing.T) {
	summaryCallCount := 0
	flushCallCount := 0

	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) >= 2 && req.Messages[0].Role == openai.ChatMessageRoleSystem {
			sysContent := req.Messages[0].Content
			if strings.Contains(sysContent, "long-term") || strings.Contains(sysContent, "Remembered") {
				flushCallCount++
			}
			if strings.Contains(sysContent, "Summarize the key facts") {
				summaryCallCount++
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

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50

	for i := 0; i < 20; i++ {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 50),
		})
	}

	originalLen := len(agent.history)
	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("compaction error: %v", err)
	}

	if len(agent.history) >= originalLen {
		t.Fatalf("history should shrink: before=%d, after=%d", originalLen, len(agent.history))
	}

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("system prompt should remain at index 0")
	}

	foundSummary := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "[CONTEXT SUMMARY]") {
			foundSummary = true
			break
		}
	}
	if !foundSummary {
		t.Fatal("context summary should be in history after compaction")
	}

	loaded, _ := wal.LoadAll()
	if len(loaded) != len(agent.history) {
		t.Fatalf("WAL should match agent history: WAL=%d, history=%d", len(loaded), len(agent.history))
	}

	t.Logf("Compaction: %d -> %d messages, summary calls=%d, flush calls=%d",
		originalLen, len(agent.history), summaryCallCount, flushCallCount)
}

func TestIntegration_CompactionPersistsAcrossRestart(t *testing.T) {
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		return "integration summary", nil
	}

	for i := 0; i < 20; i++ {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 50),
		})
	}

	agent.compactHistoryIfNeeded(context.Background())
	compactedLen := len(agent.history)

	server.Close()

	restartedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "after restart"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer restartedServer.Close()

	config := openai.DefaultConfig("test-integration-key")
	config.BaseURL = restartedServer.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	restartedAgent := NewAgent(client, wal)

	if len(restartedAgent.history) != compactedLen {
		t.Fatalf("restart should restore compacted history: expected %d, got %d", compactedLen, len(restartedAgent.history))
	}

	foundSummary := false
	for _, m := range restartedAgent.history {
		if strings.Contains(m.Content, "integration summary") {
			foundSummary = true
			break
		}
	}
	if !foundSummary {
		t.Fatal("compacted summary should survive restart")
	}
}

func TestIntegration_MultipleCompactionCycles(t *testing.T) {
	summaryNum := 0
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		summaryNum++
		return fmt.Sprintf("summary_cycle_%d", summaryNum), nil
	}

	for cycle := 0; cycle < 3; cycle++ {
		for i := 0; i < 20; i++ {
			agent.history = append(agent.history, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: strings.Repeat("word ", 50),
			})
		}
		beforeLen := len(agent.history)
		agent.compactHistoryIfNeeded(context.Background())
		if len(agent.history) >= beforeLen {
			t.Fatalf("cycle %d: history should shrink", cycle)
		}
	}

	if summaryNum != 3 {
		t.Fatalf("expected 3 summary calls across 3 cycles, got %d", summaryNum)
	}

	foundCycle3 := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "summary_cycle_3") {
			foundCycle3 = true
			break
		}
	}
	if !foundCycle3 {
		t.Fatal("latest summary should be in history after multiple cycles")
	}

	loaded, _ := wal.LoadAll()
	if len(loaded) != len(agent.history) {
		t.Fatalf("WAL should match final history: WAL=%d, history=%d", len(loaded), len(agent.history))
	}
}

func TestIntegration_MemoryFlushCreatesFiles(t *testing.T) {
	dir := t.TempDir()
	_ = &WAL{path: filepath.Join(dir, "memory_flush.wal")}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "- User lives in Berlin\n- Prefers Python over Go"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-integration-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	memDir := filepath.Join(dir, "memory")
	mem := NewMemory(memDir)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system"},
		{Role: openai.ChatMessageRoleUser, Content: "I live in Berlin and prefer Python"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Noted!"},
		{Role: openai.ChatMessageRoleUser, Content: "Also I love coffee"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Got it!"},
	}

	err := mem.Flush(context.Background(), client, DefaultModel, history)
	if err != nil {
		t.Fatalf("flush error: %v", err)
	}

	mainPath := filepath.Join(memDir, "MEMORY.md")
	if _, err := os.Stat(mainPath); os.IsNotExist(err) {
		t.Fatal("MEMORY.md should exist")
	}
	mainContent, _ := os.ReadFile(mainPath)
	if !strings.Contains(string(mainContent), "Berlin") {
		t.Fatal("MEMORY.md should contain Berlin")
	}

	todayFile := filepath.Join(memDir, time.Now().Format("2006-01-02")+".md")
	if _, err := os.Stat(todayFile); os.IsNotExist(err) {
		t.Fatal("dated file should exist")
	}
	datedContent, _ := os.ReadFile(todayFile)
	if !strings.Contains(string(datedContent), "Python") {
		t.Fatal("dated file should contain Python")
	}
}

func TestIntegration_TiktokenAccuracy(t *testing.T) {
	tests := []struct {
		name    string
		content string
		minTok  int
		maxTok  int
	}{
		{"single word", "hello", 1, 5},
		{"short sentence", "The quick brown fox jumps over the lazy dog.", 5, 20},
		{"medium paragraph", strings.Repeat("This is a test sentence with several words in it. ", 10), 50, 300},
		{"code block", "func main() {\n\tfmt.Println(\"hello world\")\n}", 5, 30},
		{"numbers", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", 5, 40},
		{"empty", "", 4, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleUser, Content: tt.content},
			}
			tokens := countTokens(messages)
			if tokens < tt.minTok {
				t.Errorf("token count %d below minimum %d for %q", tokens, tt.minTok, tt.name)
			}
			if tokens > tt.maxTok {
				t.Errorf("token count %d above maximum %d for %q", tokens, tt.maxTok, tt.name)
			}
		})
	}
}

func TestIntegration_CompactionDuringChat(t *testing.T) {
	llmCallCount := 0
	summarizerCalled := false

	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		llmCallCount++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "chat reply"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		summarizerCalled = true
		return "compacted context", nil
	}

	for i := 0; i < 20; i++ {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 50),
		})
	}

	reply, err := agent.Chat(context.Background(), "trigger compaction and chat")
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if reply != "chat reply" {
		t.Fatalf("unexpected reply: %s", reply)
	}
	if !summarizerCalled {
		t.Fatal("summarizer should have been called during chat")
	}

	loaded, _ := wal.LoadAll()
	foundUserMsg := false
	for _, m := range loaded {
		if m.Content == "trigger compaction and chat" {
			foundUserMsg = true
		}
	}
	if !foundUserMsg {
		t.Fatal("user message should be in WAL")
	}

	t.Logf("Total LLM calls: %d (1 summary + 1 chat)", llmCallCount)
}

func TestIntegration_ConcurrentCompaction(t *testing.T) {
	var mu sync.Mutex
	compactCount := 0

	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		mu.Lock()
		compactCount++
		mu.Unlock()
		return "concurrent summary", nil
	}

	for i := 0; i < 20; i++ {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 50),
		})
	}

	var wg sync.WaitGroup
	errors := make(chan error, 10)

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := agent.compactHistoryIfNeeded(context.Background()); err != nil {
				errors <- err
			}
		}()
	}
	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("concurrent compaction error: %v", err)
	}

	if compactCount == 0 {
		t.Fatal("at least one compaction should have occurred")
	}

	loaded, _ := wal.LoadAll()
	if len(loaded) == 0 {
		t.Fatal("WAL should not be empty after concurrent compaction")
	}
	if loaded[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("system prompt should survive concurrent compaction")
	}
}

func TestIntegration_CompactionWithToolHistory(t *testing.T) {
	agent, wal, server := createIntegrationAgent(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		return "tool history summary", nil
	}

	agent.history = append(agent.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: "",
		ToolCalls: []openai.ToolCall{
			{ID: "call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "search", Arguments: `{"q":"test"}`}},
		},
	})
	agent.history = append(agent.history, openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleTool, Content: "search results here", Name: "search", ToolCallID: "call_1",
	})
	for i := 0; i < 15; i++ {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 50),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("compaction error: %v", err)
	}

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatal("system prompt should be preserved")
	}

	loaded, _ := wal.LoadAll()
	if len(loaded) != len(agent.history) {
		t.Fatalf("WAL mismatch: WAL=%d, history=%d", len(loaded), len(agent.history))
	}
}
