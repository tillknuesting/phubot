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

// ==========================================
// MOCK TOOLS
// ==========================================

type MockTool struct {
	name        string
	description string
	params      map[string]any
	required    []string
	executeFn   func(args string) (string, error)
}

func (m *MockTool) Name() string { return m.name }
func (m *MockTool) Execute(args string) (string, error) {
	if m.executeFn != nil {
		return m.executeFn(args)
	}
	return fmt.Sprintf("mock result for %s", m.name), nil
}

func (m *MockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        m.name,
			Description: m.description,
			Parameters: map[string]any{
				"type":       "object",
				"properties": m.params,
				"required":   m.required,
			},
		},
	}
}

// ==========================================
// AGENT STRUCT TESTS
// ==========================================

func TestNewAgent(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)

	if agent == nil {
		t.Fatal("NewAgent returned nil")
	}
	if len(agent.history) != 1 {
		t.Fatalf("expected 1 system message in history, got %d", len(agent.history))
	}
	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("expected first message role to be system, got %s", agent.history[0].Role)
	}
	if agent.tools == nil {
		t.Fatal("tools map should be initialized")
	}
	if len(agent.tools) != 0 {
		t.Fatalf("expected 0 tools, got %d", len(agent.tools))
	}
}

func TestRegisterTool(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	mockTool := &MockTool{name: "test_tool"}
	agent.RegisterTool(mockTool)

	if len(agent.tools) != 1 {
		t.Fatalf("expected 1 registered tool, got %d", len(agent.tools))
	}
	if _, exists := agent.tools["test_tool"]; !exists {
		t.Fatal("tool 'test_tool' not found in registry")
	}
}

func TestRegisterMultipleTools(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "tool_a"})
	agent.RegisterTool(&MockTool{name: "tool_b"})
	agent.RegisterTool(&MockTool{name: "tool_c"})

	if len(agent.tools) != 3 {
		t.Fatalf("expected 3 registered tools, got %d", len(agent.tools))
	}
}

func TestRegisterToolOverwrite(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "tool_a", description: "first"})
	agent.RegisterTool(&MockTool{name: "tool_a", description: "second"})

	if len(agent.tools) != 1 {
		t.Fatalf("expected 1 tool after overwrite, got %d", len(agent.tools))
	}
	if agent.tools["tool_a"].(*MockTool).description != "second" {
		t.Fatal("tool should have been overwritten with second description")
	}
}

// ==========================================
// CHAT METHOD TESTS (with mock LLM server)
// ==========================================

func createMockLLMServer(handler http.HandlerFunc) (*httptest.Server, *openai.Client) {
	server := httptest.NewServer(handler)

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	return server, client
}

func TestChat_SimpleTextResponse(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Hello! How can I help?",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	reply, err := agent.Chat(context.Background(), "hi there")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "Hello! How can I help?" {
		t.Fatalf("unexpected reply: %s", reply)
	}
	if len(agent.history) != 3 {
		t.Fatalf("expected 3 messages (system + user + assistant), got %d", len(agent.history))
	}
}

func TestChat_ToolCallAndResponse(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"query": "test"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_123",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "mock_tool",
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
							Content: "The mock result was successful!",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "mock_tool", executeFn: func(args string) (string, error) {
		return "mock tool executed", nil
	}})

	reply, err := agent.Chat(context.Background(), "do something")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "The mock result was successful!" {
		t.Fatalf("unexpected reply: %s", reply)
	}
	if callCount != 2 {
		t.Fatalf("expected 2 LLM calls, got %d", callCount)
	}
}

func TestChat_CircuitBreaker(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		args, _ := json.Marshal(map[string]string{"query": "loop"})
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

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "loop_tool"})

	_, err := agent.Chat(context.Background(), "trigger loop")
	if err == nil {
		t.Fatal("expected circuit breaker error")
	}
	if !strings.Contains(err.Error(), "circuit breaker") {
		t.Fatalf("expected circuit breaker error, got: %v", err)
	}
	if callCount != 5 {
		t.Fatalf("expected exactly 5 iterations, got %d", callCount)
	}
}

func TestChat_UnregisteredTool(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		args, _ := json.Marshal(map[string]string{"x": "y"})
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_fake",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "nonexistent_tool",
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

	agent := NewAgent(client, nil)

	_, err := agent.Chat(context.Background(), "use a fake tool")
	if err == nil {
		t.Fatal("expected error for unregistered tool")
	}
	if !strings.Contains(err.Error(), "non-existent tool") && !strings.Contains(err.Error(), "hallucinated") {
		t.Fatalf("expected hallucinated tool error, got: %v", err)
	}
}

func TestChat_ToolErrorDoesNotCrash(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_err",
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
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "I see the tool failed. Let me help differently.",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name: "failing_tool",
		executeFn: func(args string) (string, error) {
			return "", fmt.Errorf("something went terribly wrong")
		},
	})

	reply, err := agent.Chat(context.Background(), "use the failing tool")
	if err != nil {
		t.Fatalf("agent should handle tool errors gracefully, got: %v", err)
	}
	if !strings.Contains(reply, "help differently") {
		t.Fatalf("unexpected reply after tool error: %s", reply)
	}
}

func TestChat_HistoryGrows(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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

	agent := NewAgent(client, nil)

	agent.Chat(context.Background(), "first")
	agent.Chat(context.Background(), "second")
	agent.Chat(context.Background(), "third")

	expected := 1 + (2 * 3) // system + (user + assistant) * 3
	if len(agent.history) != expected {
		t.Fatalf("expected %d messages in history, got %d", expected, len(agent.history))
	}
}

func TestChat_ContextCancellation(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "too slow",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := agent.Chat(ctx, "hello")
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

// ==========================================
// CDP BROWSER FLIGHT TOOL TESTS
// ==========================================

func TestCDPBrowserFlightTool_Name(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	if tool.Name() != "browser_search_flights" {
		t.Fatalf("expected name 'browser_search_flights', got '%s'", tool.Name())
	}
}

func TestCDPBrowserFlightTool_Definition(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()

	if def.Type != openai.ToolTypeFunction {
		t.Fatalf("expected tool type 'function', got '%s'", def.Type)
	}
	if def.Function.Name != "browser_search_flights" {
		t.Fatalf("expected function name 'browser_search_flights', got '%s'", def.Function.Name)
	}
	if !strings.Contains(def.Function.Description, "Chrome browser") {
		t.Fatalf("unexpected description: %s", def.Function.Description)
	}

	params, ok := def.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatal("parameters should be a map")
	}
	required, ok := params["required"].([]string)
	if !ok {
		t.Fatal("required should be a string slice")
	}
	if len(required) != 0 {
		t.Fatalf("expected 0 required params (all optional), got %d", len(required))
	}
}

func TestCDPBrowserFlightTool_Execute_MalformedJSON(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	_, err := tool.Execute("not json at all")
	if err == nil {
		t.Fatal("expected error for malformed JSON args")
	}
	if !strings.Contains(err.Error(), "failed to parse tool args") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestCDPBrowserFlightTool_Execute_MissingArgs_NoValidation(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	args, _ := json.Marshal(map[string]string{"origin": "London"})
	result, err := tool.Execute(string(args))
	if err != nil {
		t.Fatalf("tool does not require all args, unexpected error: %v", err)
	}
	if !strings.Contains(result, "BROWSER EXTRACTED") {
		t.Fatalf("expected browser result even with missing args, got: %s", result)
	}
}

// ==========================================
// MOCK TOOL EXECUTE TESTS
// ==========================================

func TestMockTool_Execute(t *testing.T) {
	tool := &MockTool{
		name: "echo",
		executeFn: func(args string) (string, error) {
			var m map[string]string
			json.Unmarshal([]byte(args), &m)
			return m["message"], nil
		},
	}

	args, _ := json.Marshal(map[string]string{"message": "hello world"})
	result, err := tool.Execute(string(args))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello world" {
		t.Fatalf("expected 'hello world', got '%s'", result)
	}
}

func TestMockTool_Execute_Error(t *testing.T) {
	tool := &MockTool{
		name: "error_tool",
		executeFn: func(args string) (string, error) {
			return "", fmt.Errorf("intentional failure")
		},
	}

	_, err := tool.Execute("{}")
	if err == nil {
		t.Fatal("expected error from failing mock tool")
	}
}

// ==========================================
// TOOL DEFINITION SCHEMA TESTS
// ==========================================

func TestToolDefinition_HasRequiredFields(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()

	if def.Function.Name == "" {
		t.Fatal("tool definition must have a name")
	}
	if def.Function.Description == "" {
		t.Fatal("tool definition must have a description")
	}
	if def.Function.Parameters == nil {
		t.Fatal("tool definition must have parameters")
	}
}

func TestToolDefinition_RequiredParamsExist(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()
	params := def.Function.Parameters.(map[string]any)
	required := params["required"].([]string)
	properties := params["properties"].(map[string]any)

	for _, req := range required {
		if _, exists := properties[req]; !exists {
			t.Fatalf("required param '%s' not found in properties", req)
		}
	}
}

// ==========================================
// INTERFACE COMPLIANCE TESTS
// ==========================================

func TestCDPBrowserFlightTool_ImplementsTool(t *testing.T) {
	var _ Tool = &CDPBrowserFlightTool{}
}

func TestMockTool_ImplementsTool(t *testing.T) {
	var _ Tool = &MockTool{}
}

// ==========================================
// WAL TESTS
// ==========================================

func newTestWAL(t *testing.T) (*WAL, func()) {
	t.Helper()
	dir := t.TempDir()
	w := &WAL{path: filepath.Join(dir, "test.wal")}
	return w, func() { os.RemoveAll(dir) }
}

func TestWAL_AppendAndLoad(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	msg := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "hello from WAL",
	}
	if err := w.Append(msg); err != nil {
		t.Fatalf("append error: %v", err)
	}

	loaded, err := w.LoadAll()
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded))
	}
	if loaded[0].Content != "hello from WAL" {
		t.Fatalf("unexpected content: %s", loaded[0].Content)
	}
	if loaded[0].Role != openai.ChatMessageRoleUser {
		t.Fatalf("unexpected role: %s", loaded[0].Role)
	}
}

func TestWAL_MultipleAppends(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system prompt"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi there"},
		{Role: openai.ChatMessageRoleTool, Content: "tool result", ToolCallID: "call_1"},
	}

	for _, m := range messages {
		if err := w.Append(m); err != nil {
			t.Fatalf("append error: %v", err)
		}
	}

	loaded, err := w.LoadAll()
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if len(loaded) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(loaded))
	}
	if loaded[2].Content != "hi there" {
		t.Fatalf("unexpected 3rd message: %s", loaded[2].Content)
	}
}

func TestWAL_EmptyFile(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	loaded, err := w.LoadAll()
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if len(loaded) != 0 {
		t.Fatalf("expected 0 messages from empty WAL, got %d", len(loaded))
	}
}

func TestWAL_PersistsAcrossLoads(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "first"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "response"})

	loaded1, _ := w.LoadAll()
	if len(loaded1) != 2 {
		t.Fatalf("first load: expected 2, got %d", len(loaded1))
	}

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "second"})

	loaded2, _ := w.LoadAll()
	if len(loaded2) != 3 {
		t.Fatalf("second load: expected 3, got %d", len(loaded2))
	}
}

func TestWAL_SkipsCorruptLines(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "valid"})

	f, _ := os.OpenFile(w.path, os.O_APPEND|os.O_WRONLY, 0644)
	f.WriteString("this is not json\n")
	f.WriteString("{\"role\":\"user\",\"content\":\"also bad\"") // no newline, truncated
	f.Close()

	loaded, err := w.LoadAll()
	if err != nil {
		t.Fatalf("load should not fail on corrupt lines: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 valid message, got %d", len(loaded))
	}
}

func TestWAL_TrimOnLargeFile(t *testing.T) {
	origMax := WALMaxSize
	WALMaxSize = 512
	defer func() { WALMaxSize = origMax }()

	w, cleanup := newTestWAL(t)
	defer cleanup()

	for range 20 {
		w.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("x", 100),
		})
	}

	loaded, _ := w.LoadAll()
	if len(loaded) == 0 {
		t.Fatal("WAL should not be empty after trim")
	}
	if len(loaded) >= 20 {
		t.Fatalf("WAL should have been trimmed, still has %d entries", len(loaded))
	}
}

func TestWAL_PreservesToolCallFields(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	msg := openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    "result text",
		Name:       "browser_search_flights",
		ToolCallID: "call_abc123",
	}
	w.Append(msg)

	loaded, _ := w.LoadAll()
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded))
	}
	if loaded[0].ToolCallID != "call_abc123" {
		t.Fatalf("ToolCallID not preserved: %s", loaded[0].ToolCallID)
	}
	if loaded[0].Name != "browser_search_flights" {
		t.Fatalf("Name not preserved: %s", loaded[0].Name)
	}
}

func TestNewAgent_RestoresFromWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: "custom system"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "world"})

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)

	if len(agent.history) != 3 {
		t.Fatalf("expected 3 restored messages, got %d", len(agent.history))
	}
	if agent.history[1].Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", agent.history[1].Content)
	}
}

func TestNewAgent_InjectsSystemPromptIfMissing(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "no system prompt here"})

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)

	if len(agent.history) != 2 {
		t.Fatalf("expected 2 messages (injected system + user), got %d", len(agent.history))
	}
	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("first message should be system prompt, got %s", agent.history[0].Role)
	}
}

func TestAgent_AppendHistory_WritesToWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)
	agent.appendHistory(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "persisted to WAL",
	})

	loaded, _ := w.LoadAll()
	found := false
	for _, m := range loaded {
		if m.Content == "persisted to WAL" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("message not found in WAL after appendHistory")
	}
}

// ==========================================
// WAL EDGE CASE TESTS
// ==========================================

func TestWAL_ConcurrentAppends(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			msg := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("concurrent message %d", idx),
			}
			if err := w.Append(msg); err != nil {
				t.Errorf("concurrent append %d failed: %v", idx, err)
			}
		}(i)
	}
	wg.Wait()

	loaded, err := w.LoadAll()
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if len(loaded) != 50 {
		t.Fatalf("expected 50 messages after concurrent appends, got %d", len(loaded))
	}

	contents := make(map[string]bool)
	for _, m := range loaded {
		contents[m.Content] = true
	}
	for i := range 50 {
		expected := fmt.Sprintf("concurrent message %d", i)
		if !contents[expected] {
			t.Fatalf("missing message: %s", expected)
		}
	}
}

func TestWAL_SpecialCharacters(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	cases := []string{
		"line1\nline2\nline3",
		"tabs\there\tand\tthere",
		"quotes: \"hello\" and 'world'",
		`backslashes: C:\Users\test`,
		"unicode: \u00e9\u00e8\u00ea\u00eb \u4f60\u597d\u4e16\u754c \U0001f600",
		"{json: \"inside content\"}",
		"null \x00 byte",
		"emoji: \U0001f916 \U0001f680 \U0001f30d",
	}

	for _, content := range cases {
		msg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: content,
		}
		if err := w.Append(msg); err != nil {
			t.Fatalf("append failed for content %q: %v", content, err)
		}
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != len(cases) {
		t.Fatalf("expected %d messages, got %d", len(cases), len(loaded))
	}
	for i, content := range cases {
		if loaded[i].Content != content {
			t.Fatalf("roundtrip %d failed:\n  expected: %q\n  got:      %q", i, content, loaded[i].Content)
		}
	}
}

func TestWAL_LargeContent(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	largeContent := strings.Repeat("abcdefghij", 100000) // ~1MB
	msg := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: largeContent,
	}
	if err := w.Append(msg); err != nil {
		t.Fatalf("append large message failed: %v", err)
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded))
	}
	if loaded[0].Content != largeContent {
		t.Fatalf("large content mismatch: expected %d bytes, got %d bytes", len(largeContent), len(loaded[0].Content))
	}
}

func TestWAL_EmptyContent(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	msg := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: "",
	}
	if err := w.Append(msg); err != nil {
		t.Fatalf("append empty content failed: %v", err)
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message, got %d", len(loaded))
	}
	if loaded[0].Content != "" {
		t.Fatalf("expected empty content, got %q", loaded[0].Content)
	}
}

func TestWAL_OnlyBlankLines(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	f, _ := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	f.WriteString("\n\n\n\n")
	f.Close()

	loaded, _ := w.LoadAll()
	if len(loaded) != 0 {
		t.Fatalf("expected 0 messages from blank-only WAL, got %d", len(loaded))
	}
}

func TestWAL_MixedValidAndInvalid(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	f, _ := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	f.WriteString("\n")
	valid1, _ := json.Marshal(openai.ChatCompletionMessage{Role: "user", Content: "valid 1"})
	f.Write(valid1)
	f.WriteString("\n")
	f.WriteString("garbage\n")
	valid2, _ := json.Marshal(openai.ChatCompletionMessage{Role: "assistant", Content: "valid 2"})
	f.Write(valid2)
	f.WriteString("\n")
	f.WriteString("{\"role\":\"user\"}\n") // valid JSON but missing content (still valid struct)
	f.WriteString("more garbage\n")
	valid3, _ := json.Marshal(openai.ChatCompletionMessage{Role: "tool", Content: "valid 3", ToolCallID: "c1"})
	f.Write(valid3)
	f.WriteString("\n")
	f.Close()

	loaded, _ := w.LoadAll()
	if len(loaded) != 4 {
		t.Fatalf("expected 4 valid messages (2 valid + 1 minimal + 1 tool), got %d", len(loaded))
	}
}

// ==========================================
// WAL TRIM TESTS
// ==========================================

func TestWAL_TrimPreservesNewestMessages(t *testing.T) {
	origMax := WALMaxSize
	WALMaxSize = 300
	defer func() { WALMaxSize = origMax }()

	w, cleanup := newTestWAL(t)
	defer cleanup()

	for i := range 10 {
		w.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message_%02d", i),
		})
	}

	loaded, _ := w.LoadAll()
	if len(loaded) == 0 {
		t.Fatal("WAL should not be empty after trim")
	}
	if loaded[len(loaded)-1].Content != "message_09" {
		t.Fatalf("last message should be newest, got %q", loaded[len(loaded)-1].Content)
	}
	if loaded[0].Content == "message_00" {
		t.Fatal("oldest message should have been trimmed away")
	}
}

func TestWAL_NoTrimWhenUnderLimit(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	for i := range 5 {
		w.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("msg_%d", i),
		})
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != 5 {
		t.Fatalf("expected 5 messages (no trim), got %d", len(loaded))
	}
}

// ==========================================
// AGENT + WAL INTEGRATION TESTS
// ==========================================

func TestNewAgent_WithNilWAL(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)

	if agent.wal != nil {
		t.Fatal("wal should be nil when nil is passed")
	}
	if len(agent.history) != 1 {
		t.Fatalf("expected 1 system message, got %d", len(agent.history))
	}
}

func TestAgent_ChatPersistsToWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "persisted reply",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, w)
	agent.Chat(context.Background(), "test persistence")

	loaded, _ := w.LoadAll()
	if len(loaded) < 2 {
		t.Fatalf("expected at least 2 messages in WAL (user + assistant), got %d", len(loaded))
	}

	foundUser := false
	foundAssistant := false
	for _, m := range loaded {
		if m.Role == openai.ChatMessageRoleUser && m.Content == "test persistence" {
			foundUser = true
		}
		if m.Role == openai.ChatMessageRoleAssistant && m.Content == "persisted reply" {
			foundAssistant = true
		}
	}
	if !foundUser {
		t.Fatal("user message not persisted to WAL")
	}
	if !foundAssistant {
		t.Fatal("assistant message not persisted to WAL")
	}
}

func TestAgent_ToolCallPersistsToWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_abc",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "test_tool",
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
							Content: "done",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, w)
	agent.RegisterTool(&MockTool{name: "test_tool", executeFn: func(args string) (string, error) {
		return "tool output", nil
	}})
	agent.Chat(context.Background(), "use tool")

	loaded, _ := w.LoadAll()

	roleSeq := []string{}
	for _, m := range loaded {
		roleSeq = append(roleSeq, m.Role)
	}
	expected := []string{"system", "user", "assistant", "tool", "assistant"}
	if len(roleSeq) != len(expected) {
		t.Fatalf("expected %d roles, got %d: %v", len(expected), len(roleSeq), roleSeq)
	}
	for i, exp := range expected {
		if roleSeq[i] != exp {
			t.Fatalf("role mismatch at %d: expected %s, got %s", i, exp, roleSeq[i])
		}
	}
}

func TestAgent_RestartFromWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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

	agent1 := NewAgent(client, w)
	agent1.Chat(context.Background(), "first message")
	agent1.Chat(context.Background(), "second message")

	agent2 := NewAgent(client, w)
	if len(agent2.history) < 5 {
		t.Fatalf("restored agent should have >= 5 messages (system + 2*(user+assistant)), got %d", len(agent2.history))
	}
	if agent2.history[1].Content != "first message" {
		t.Fatalf("first restored message wrong: %q", agent2.history[1].Content)
	}
	if agent2.history[3].Content != "second message" {
		t.Fatalf("second restored message wrong: %q", agent2.history[3].Content)
	}
}

func TestAgent_NilWALDoesNotCrash(t *testing.T) {
	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	agent.appendHistory(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "this should not crash",
	})
	if len(agent.history) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(agent.history))
	}
}

// ==========================================
// CHAT LOOP ADVANCED TESTS
// ==========================================

func TestChat_MultipleToolCallsInOneResponse(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args1, _ := json.Marshal(map[string]string{"q": "a"})
			args2, _ := json.Marshal(map[string]string{"q": "b"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_a",
										Arguments: string(args1),
									},
								},
								{
									ID:   "call_2",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "tool_b",
										Arguments: string(args2),
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
							Content: "both tools executed",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	var toolACalled, toolBCalled bool
	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "tool_a", executeFn: func(args string) (string, error) {
		toolACalled = true
		return "result_a", nil
	}})
	agent.RegisterTool(&MockTool{name: "tool_b", executeFn: func(args string) (string, error) {
		toolBCalled = true
		return "result_b", nil
	}})

	reply, err := agent.Chat(context.Background(), "run both tools")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !toolACalled || !toolBCalled {
		t.Fatalf("both tools should have been called: a=%v b=%v", toolACalled, toolBCalled)
	}
	if reply != "both tools executed" {
		t.Fatalf("unexpected reply: %s", reply)
	}

	toolMsgCount := 0
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleTool {
			toolMsgCount++
		}
	}
	if toolMsgCount != 2 {
		t.Fatalf("expected 2 tool messages in history, got %d", toolMsgCount)
	}
}

func TestChat_EmptyContentResponse(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	reply, err := agent.Chat(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "" {
		t.Fatalf("expected empty reply, got %q", reply)
	}
}

func TestChat_LLMReturnsHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	_, err := agent.Chat(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error when LLM returns 500")
	}
}

func TestChat_ToolErrorFedBackToLLM(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_err",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "err_tool",
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
			var req openai.ChatCompletionRequest
			json.NewDecoder(r.Body).Decode(&req)

			foundError := false
			for _, m := range req.Messages {
				if m.Role == "tool" && strings.Contains(m.Content, "Error executing tool") {
					foundError = true
					break
				}
			}
			if !foundError {
				t.Errorf("expected tool error to be in history sent to LLM")
			}

			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "saw the error, recovered",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name: "err_tool",
		executeFn: func(args string) (string, error) {
			return "", fmt.Errorf("boom")
		},
	})

	reply, err := agent.Chat(context.Background(), "trigger error")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "saw the error, recovered" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

func TestChat_SendsToolDefinitionsToLLM(t *testing.T) {
	receivedTools := false
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Tools) > 0 {
			receivedTools = true
		}
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

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "test_tool"})
	agent.Chat(context.Background(), "hello")

	if !receivedTools {
		t.Fatal("tool definitions were not sent to the LLM")
	}
}

func TestChat_NoToolsRegistered(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Tools) != 0 {
			t.Fatalf("expected 0 tools in request, got %d", len(req.Tools))
		}
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "no tools needed",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	reply, err := agent.Chat(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "no tools needed" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

// ==========================================
// TOOL DEFINITION ADVANCED TESTS
// ==========================================

func TestToolDefinition_PropertyTypes(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()
	params := def.Function.Parameters.(map[string]any)
	properties := params["properties"].(map[string]any)

	for _, fieldName := range []string{"origin", "destination", "date"} {
		prop, exists := properties[fieldName]
		if !exists {
			t.Fatalf("property %q not found", fieldName)
		}
		propMap, ok := prop.(map[string]any)
		if !ok {
			t.Fatalf("property %q should be a map, got %T", fieldName, prop)
		}
		if propMap["type"] != "string" {
			t.Fatalf("property %q should have type string, got %v", fieldName, propMap["type"])
		}
		if propMap["description"] == nil {
			t.Fatalf("property %q should have a description", fieldName)
		}
	}
}

func TestMockTool_DefaultExecute(t *testing.T) {
	tool := &MockTool{name: "default_tool"}
	result, err := tool.Execute("{}")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "mock result for default_tool" {
		t.Fatalf("unexpected default result: %s", result)
	}
}

func TestMockTool_DefinitionWithNoParams(t *testing.T) {
	tool := &MockTool{name: "no_params", description: "a tool with no params"}
	def := tool.Definition()

	if def.Function.Name != "no_params" {
		t.Fatalf("expected name 'no_params', got %q", def.Function.Name)
	}
	params := def.Function.Parameters.(map[string]any)
	if len(params["required"].([]string)) != 0 {
		t.Fatalf("expected nil required, got %v", params["required"])
	}
}

func TestOpenWAL_CreatesDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	origDir := WALDir
	WALDir = filepath.Join(tmpDir, "nested", "wal_dir")
	defer func() { WALDir = origDir }()

	wal, err := OpenWAL()
	if err != nil {
		t.Fatalf("OpenWAL failed: %v", err)
	}
	wal.Append(openai.ChatCompletionMessage{Role: "user", Content: "test"})
	if _, statErr := os.Stat(filepath.Join(WALDir, WALFile)); statErr != nil {
		t.Fatalf("WAL file should exist after append: %v", statErr)
	}
	wal.LoadAll()
}

func TestNewAgent_EmptyWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)
	if len(agent.history) != 1 {
		t.Fatalf("empty WAL should result in default system prompt only, got %d messages", len(agent.history))
	}
}

func TestNewAgent_DuplicateSystemPrompts(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: "sys1"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: "sys2"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "hello"})

	config := openai.DefaultConfig("test-key")
	config.BaseURL = "http://localhost:9999/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)
	if len(agent.history) != 3 {
		t.Fatalf("should keep all 3 messages including duplicate system, got %d", len(agent.history))
	}
	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("first should still be system, got %s", agent.history[0].Role)
	}
}

func TestChat_StreamingToolCallHistory(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "step 1",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.Chat(context.Background(), "msg1")
	agent.Chat(context.Background(), "msg2")

	if len(agent.history) != 5 {
		t.Fatalf("expected 5 messages (sys + u1 + a1 + u2 + a2), got %d", len(agent.history))
	}
	if agent.history[1].Content != "msg1" {
		t.Fatalf("expected msg1 at index 1, got %q", agent.history[1].Content)
	}
	if agent.history[3].Content != "msg2" {
		t.Fatalf("expected msg2 at index 3, got %q", agent.history[3].Content)
	}
}

// ==========================================
// MOCK TOOL ADVANCED TESTS
// ==========================================

type CountingMockTool struct {
	name      string
	callCount int
	mu        sync.Mutex
	executeFn func(args string) (string, error)
}

func (c *CountingMockTool) Name() string { return c.name }
func (c *CountingMockTool) Execute(args string) (string, error) {
	c.mu.Lock()
	c.callCount++
	c.mu.Unlock()
	if c.executeFn != nil {
		return c.executeFn(args)
	}
	return "ok", nil
}
func (c *CountingMockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        c.name,
			Description: "counting mock tool",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}
func (c *CountingMockTool) CallCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.callCount
}

func TestCountingMockTool_TracksCalls(t *testing.T) {
	tool := &CountingMockTool{name: "counter"}
	tool.Execute(`{}`)
	tool.Execute(`{}`)
	tool.Execute(`{}`)
	if tool.CallCount() != 3 {
		t.Fatalf("expected 3 calls, got %d", tool.CallCount())
	}
}

func TestCountingMockTool_CustomExecute(t *testing.T) {
	tool := &CountingMockTool{
		name: "custom",
		executeFn: func(args string) (string, error) {
			return "custom result", nil
		},
	}
	result, err := tool.Execute(`{}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "custom result" {
		t.Fatalf("expected 'custom result', got %q", result)
	}
	if tool.CallCount() != 1 {
		t.Fatalf("expected 1 call, got %d", tool.CallCount())
	}
}

type SlowMockTool struct {
	name      string
	delay     time.Duration
	executeFn func(args string) (string, error)
}

func (s *SlowMockTool) Name() string { return s.name }
func (s *SlowMockTool) Execute(args string) (string, error) {
	time.Sleep(s.delay)
	if s.executeFn != nil {
		return s.executeFn(args)
	}
	return "slow result", nil
}
func (s *SlowMockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        s.name,
			Description: "slow mock tool",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func TestSlowMockTool_ExecutesWithDelay(t *testing.T) {
	tool := &SlowMockTool{name: "slow", delay: 50 * time.Millisecond}
	start := time.Now()
	tool.Execute(`{}`)
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond {
		t.Fatalf("tool should have taken at least 40ms, took %v", elapsed)
	}
}

type StatefulMockTool struct {
	name   string
	mu     sync.Mutex
	states []string
}

func (s *StatefulMockTool) Name() string { return s.name }
func (s *StatefulMockTool) Execute(args string) (string, error) {
	var m map[string]string
	json.Unmarshal([]byte(args), &m)
	s.mu.Lock()
	s.states = append(s.states, m["value"])
	s.mu.Unlock()
	return fmt.Sprintf("stored: %s", m["value"]), nil
}
func (s *StatefulMockTool) States() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, len(s.states))
	copy(out, s.states)
	return out
}
func (s *StatefulMockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        s.name,
			Description: "stateful mock tool",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"value": map[string]any{"type": "string"},
				},
				"required": []string{"value"},
			},
		},
	}
}

func TestStatefulMockTool_TracksState(t *testing.T) {
	tool := &StatefulMockTool{name: "stateful"}
	tool.Execute(`{"value": "a"}`)
	tool.Execute(`{"value": "b"}`)
	tool.Execute(`{"value": "c"}`)
	states := tool.States()
	if len(states) != 3 {
		t.Fatalf("expected 3 states, got %d", len(states))
	}
	if states[0] != "a" || states[1] != "b" || states[2] != "c" {
		t.Fatalf("unexpected states: %v", states)
	}
}

type ErrorOnceMockTool struct {
	name   string
	called bool
}

func (e *ErrorOnceMockTool) Name() string { return e.name }
func (e *ErrorOnceMockTool) Execute(args string) (string, error) {
	if !e.called {
		e.called = true
		return "", fmt.Errorf("first call always fails")
	}
	return "success on retry", nil
}
func (e *ErrorOnceMockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        e.name,
			Description: "fails once then succeeds",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func TestErrorOnceMockTool_FailsThenSucceeds(t *testing.T) {
	tool := &ErrorOnceMockTool{name: "flaky"}
	_, err1 := tool.Execute(`{}`)
	if err1 == nil {
		t.Fatal("first call should fail")
	}
	result2, err2 := tool.Execute(`{}`)
	if err2 != nil {
		t.Fatalf("second call should succeed, got: %v", err2)
	}
	if result2 != "success on retry" {
		t.Fatalf("unexpected result: %s", result2)
	}
}

type PanickingMockTool struct {
	name string
}

func (p *PanickingMockTool) Name() string { return p.name }
func (p *PanickingMockTool) Execute(args string) (result string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("tool panicked: %v", r)
		}
	}()
	panic("intentional panic")
}
func (p *PanickingMockTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        p.name,
			Description: "panics on execute",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func TestPanickingMockTool_RecoversFromPanic(t *testing.T) {
	tool := &PanickingMockTool{name: "panicker"}
	_, err := tool.Execute(`{}`)
	if err == nil {
		t.Fatal("expected error from panic recovery")
	}
	if !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("expected panic error, got: %v", err)
	}
}

// ==========================================
// CHAT LOOP EDGE CASE TESTS
// ==========================================

func TestChat_CircuitBreakerExactBoundary(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									Name:      "boundary_tool",
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

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "boundary_tool"})

	_, err := agent.Chat(context.Background(), "loop exactly 5 times")
	if err == nil {
		t.Fatal("expected circuit breaker at exactly 5")
	}
	if callCount != 5 {
		t.Fatalf("expected exactly 5 LLM calls at boundary, got %d", callCount)
	}
}

func TestChat_CircuitBreakerThenNormalResponse(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 3 {
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
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "finally resolved",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "loop_tool"})

	reply, err := agent.Chat(context.Background(), "loop 3 times then respond")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "finally resolved" {
		t.Fatalf("unexpected reply: %s", reply)
	}
	if callCount != 4 {
		t.Fatalf("expected 4 LLM calls (3 tool + 1 text), got %d", callCount)
	}
}

func TestChat_LLMReturnsEmptyChoices(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	_, err := agent.Chat(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error when LLM returns empty choices")
	}
}

func TestChat_LLMReturnsMalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("this is not json at all"))
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, nil)
	_, err := agent.Chat(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error when LLM returns malformed JSON")
	}
}

func TestChat_ToolReturnsEmptyResult(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "empty_tool",
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
							Content: "got empty result",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name: "empty_tool",
		executeFn: func(args string) (string, error) {
			return "", nil
		},
	})

	reply, err := agent.Chat(context.Background(), "use empty tool")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "got empty result" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

func TestChat_ToolReturnsVeryLargeResult(t *testing.T) {
	callCount := 0
	largeResult := strings.Repeat("x", 50000)
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "big_tool",
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
							Content: "processed large result",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name:      "big_tool",
		executeFn: func(args string) (string, error) { return largeResult, nil },
	})
	agent.reserveTokens = 100000

	reply, err := agent.Chat(context.Background(), "test")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "processed large result" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

func TestChatWithImage_NoVisionSupport(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error": "Model does not support images"}`))
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "I cannot see images",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.reserveTokens = 100000

	reply, err := agent.ChatWithImage(context.Background(), "What is this?", "base64imagedata")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(reply, "cannot see images") {
		t.Fatalf("expected fallback response, got: %s", reply)
	}
}

func TestChatWithImage_TextFallback(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error": "does not support"}`))
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "I see a cat in the image",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.reserveTokens = 100000

	reply, err := agent.ChatWithImage(context.Background(), "What animal?", "base64cat")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(reply, "cat") {
		t.Fatalf("expected reply about cat, got: %s", reply)
	}
}

func TestChat_SingleToolCallThenText(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"q": "test"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_1",
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
							Content: "here are the results",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "search_tool"})

	reply, err := agent.Chat(context.Background(), "search for something")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "here are the results" {
		t.Fatalf("unexpected reply: %s", reply)
	}
	if callCount != 2 {
		t.Fatalf("expected 2 LLM calls, got %d", callCount)
	}
}

func TestChat_HistoryOrderAfterToolCall(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"q": "test"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_order",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "order_tool",
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
							Content: "done",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "order_tool"})
	agent.Chat(context.Background(), "test order")

	expectedRoles := []string{
		openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
		openai.ChatMessageRoleAssistant,
	}
	if len(agent.history) != len(expectedRoles) {
		t.Fatalf("expected %d messages, got %d", len(expectedRoles), len(agent.history))
	}
	for i, expectedRole := range expectedRoles {
		if agent.history[i].Role != expectedRole {
			t.Fatalf("role mismatch at %d: expected %s, got %s", i, expectedRole, agent.history[i].Role)
		}
	}
	if agent.history[3].ToolCallID != "call_order" {
		t.Fatalf("tool message should have ToolCallID 'call_order', got %q", agent.history[3].ToolCallID)
	}
	if agent.history[3].Name != "order_tool" {
		t.Fatalf("tool message should have Name 'order_tool', got %q", agent.history[3].Name)
	}
}

func TestChat_PassesHistoryToLLM(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if callCount == 2 {
			if len(req.Messages) < 3 {
				t.Fatalf("second call should have history from first, got %d messages", len(req.Messages))
			}
			foundFirstUser := false
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleUser && m.Content == "first message" {
					foundFirstUser = true
				}
			}
			if !foundFirstUser {
				t.Error("first message should be in history on second call")
			}
		}

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

	agent := NewAgent(client, nil)
	agent.Chat(context.Background(), "first message")
	agent.Chat(context.Background(), "second message")
}

func TestChat_SpecialCharactersInUserInput(t *testing.T) {
	specialInput := "hello \"world\" with 'quotes' and \nnewlines\tand\ttabs"
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		lastMsg := req.Messages[len(req.Messages)-1]
		if lastMsg.Content != specialInput {
			t.Errorf("special characters not preserved in user input: got %q", lastMsg.Content)
		}

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

	agent := NewAgent(client, nil)
	reply, err := agent.Chat(context.Background(), specialInput)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "ok" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

func TestChat_VerifiesModelInRequest(t *testing.T) {
	receivedModel := ""
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)
		receivedModel = req.Model

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

	agent := NewAgent(client, nil)
	agent.Chat(context.Background(), "check model")

	if receivedModel != "qwen3.5-9b-mlx" {
		t.Fatalf("expected model 'qwen3.5-9b-mlx' in request, got %q", receivedModel)
	}
}

func TestChat_MultipleChatsAccumulate(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: fmt.Sprintf("reply_%d", len(r.URL.Query())),
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	for i := range 10 {
		agent.Chat(context.Background(), fmt.Sprintf("msg_%d", i))
	}
	expected := 1 + (2 * 10)
	if len(agent.history) != expected {
		t.Fatalf("expected %d messages after 10 chats, got %d", expected, len(agent.history))
	}
}

// ==========================================
// WAL ADDITIONAL EDGE CASES
// ==========================================

func TestWAL_AppendAfterTrim(t *testing.T) {
	origMax := WALMaxSize
	WALMaxSize = 256
	defer func() { WALMaxSize = origMax }()

	w, cleanup := newTestWAL(t)
	defer cleanup()

	for range 20 {
		w.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("x", 50),
		})
	}

	beforeAppend, _ := w.LoadAll()
	beforeCount := len(beforeAppend)

	w.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "after trim message",
	})

	afterLoad, _ := w.LoadAll()
	if len(afterLoad) != beforeCount+1 {
		t.Fatalf("append after trim should add one message: before=%d, after=%d", beforeCount, len(afterLoad))
	}
	if afterLoad[len(afterLoad)-1].Content != "after trim message" {
		t.Fatalf("last message should be 'after trim message', got %q", afterLoad[len(afterLoad)-1].Content)
	}
}

func TestWAL_ConcurrentReadWrite(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	for i := range 10 {
		w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: fmt.Sprintf("init_%d", i)})
	}

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(2)
		go func(idx int) {
			defer wg.Done()
			w.Append(openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("writer_%d", idx),
			})
		}(i)
		go func() {
			defer wg.Done()
			_, _ = w.LoadAll()
		}()
	}
	wg.Wait()

	loaded, _ := w.LoadAll()
	if len(loaded) < 10 {
		t.Fatalf("expected at least 10 messages after concurrent r/w, got %d", len(loaded))
	}
}

func TestWAL_PathWithSpaces(t *testing.T) {
	tmpDir := t.TempDir()
	spaceDir := filepath.Join(tmpDir, "path with spaces", "and more")
	os.MkdirAll(spaceDir, 0755)

	w := &WAL{path: filepath.Join(spaceDir, "test.wal")}
	msg := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "spaces ok"}
	if err := w.Append(msg); err != nil {
		t.Fatalf("append with spaces in path failed: %v", err)
	}
	loaded, _ := w.LoadAll()
	if len(loaded) != 1 || loaded[0].Content != "spaces ok" {
		t.Fatal("roundtrip with spaces in path failed")
	}
}

func TestWAL_VerifyFileOnDisk(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "disk check"})

	raw, err := os.ReadFile(w.path)
	if err != nil {
		t.Fatalf("could not read WAL file: %v", err)
	}
	if !strings.Contains(string(raw), "disk check") {
		t.Fatal("WAL file on disk should contain the message")
	}
	if !strings.HasSuffix(strings.TrimSpace(string(raw)), "}") {
		t.Fatal("WAL file lines should end with }")
	}
}

func TestWAL_MessagesHaveCorrectStructure(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{
		Role:       openai.ChatMessageRoleTool,
		Content:    "structured result",
		Name:       "my_tool",
		ToolCallID: "tc_123",
	})

	loaded, _ := w.LoadAll()
	msg := loaded[0]
	if msg.Role != "tool" {
		t.Fatalf("role mismatch: %s", msg.Role)
	}
	if msg.Name != "my_tool" {
		t.Fatalf("name mismatch: %s", msg.Name)
	}
	if msg.ToolCallID != "tc_123" {
		t.Fatalf("toolcallid mismatch: %s", msg.ToolCallID)
	}
}

// ==========================================
// AGENT + WAL INTEGRATION ADVANCED
// ==========================================

func TestAgent_ChatWithWAL_RestartMidConversation(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	phase := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		phase++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: fmt.Sprintf("phase_%d", phase),
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent1 := NewAgent(client, w)
	reply1, _ := agent1.Chat(context.Background(), "first")
	if reply1 != "phase_1" {
		t.Fatalf("unexpected reply: %s", reply1)
	}

	agent2 := NewAgent(client, w)
	reply2, _ := agent2.Chat(context.Background(), "second after restart")
	if reply2 != "phase_2" {
		t.Fatalf("unexpected reply after restart: %s", reply2)
	}

	if len(agent2.history) < 5 {
		t.Fatalf("restored agent should have accumulated history, got %d", len(agent2.history))
	}
}

func TestAgent_MultipleToolsSameChat(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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

	agent := NewAgent(client, nil)
	for i := range 5 {
		agent.RegisterTool(&MockTool{name: fmt.Sprintf("tool_%d", i)})
	}

	if len(agent.tools) != 5 {
		t.Fatalf("expected 5 tools registered, got %d", len(agent.tools))
	}
}

func TestAgent_OverwrittenToolStillWorks(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
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
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "replace_me",
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
							Content: "done",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	var toolCalled bool
	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{name: "replace_me", executeFn: func(args string) (string, error) {
		toolCalled = true
		return "first", nil
	}})
	agent.RegisterTool(&MockTool{name: "replace_me", executeFn: func(args string) (string, error) {
		toolCalled = true
		return "second", nil
	}})

	agent.Chat(context.Background(), "use tool")
	if !toolCalled {
		t.Fatal("tool should have been called")
	}
}

func TestChat_ToolCallWithComplexArgs(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			complexArgs, _ := json.Marshal(map[string]any{
				"query":   "test search",
				"filters": []string{"a", "b", "c"},
				"nested":  map[string]string{"key": "value"},
				"number":  42.5,
				"boolean": true,
			})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_complex",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "complex_tool",
										Arguments: string(complexArgs),
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
							Content: "complex done",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	var receivedArgs string
	agent := NewAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name: "complex_tool",
		executeFn: func(args string) (string, error) {
			receivedArgs = args
			return "parsed complex", nil
		},
	})

	reply, err := agent.Chat(context.Background(), "complex")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply != "complex done" {
		t.Fatalf("unexpected reply: %s", reply)
	}

	var parsed map[string]any
	json.Unmarshal([]byte(receivedArgs), &parsed)
	if parsed["number"].(float64) != 42.5 {
		t.Fatalf("number arg not preserved: %v", parsed["number"])
	}
	if parsed["boolean"].(bool) != true {
		t.Fatalf("boolean arg not preserved: %v", parsed["boolean"])
	}
	filters := parsed["filters"].([]any)
	if len(filters) != 3 {
		t.Fatalf("filters array not preserved: %v", filters)
	}
}

func TestChat_SystemPromptNotSentAsUserMessage(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		for i, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleSystem && i > 0 {
				t.Errorf("system prompt should only be at index 0, found at %d", i)
			}
		}

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

	agent := NewAgent(client, nil)
	agent.Chat(context.Background(), "check system prompt position")
}

// ==========================================
// BROWSER TOOL DEFINITION TESTS
// ==========================================

func TestBrowserTool_DefinitionPropertiesHaveDescriptions(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()
	params := def.Function.Parameters.(map[string]any)
	properties := params["properties"].(map[string]any)

	expectedProps := map[string]string{
		"url":          "Direct URL to navigate to (optional, takes precedence over search)",
		"origin":       "Origin city/airport code",
		"destination":  "Destination city/airport code",
		"date":         "Date in YYYY-MM-DD format",
		"adults":       "Number of adults (default 2)",
		"wait_seconds": "Seconds to wait for page load (default 8)",
	}
	for name, expectedDesc := range expectedProps {
		prop, exists := properties[name]
		if !exists {
			t.Fatalf("missing property %q", name)
		}
		desc, ok := prop.(map[string]any)["description"].(string)
		if !ok {
			t.Fatalf("property %q missing description", name)
		}
		if desc != expectedDesc {
			t.Fatalf("property %q description mismatch: expected %q, got %q", name, expectedDesc, desc)
		}
	}
}

func TestBrowserTool_DefinitionTypeIsObject(t *testing.T) {
	tool := &CDPBrowserFlightTool{}
	def := tool.Definition()
	params := def.Function.Parameters.(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("parameters type should be 'object', got %v", params["type"])
	}
}

// ==========================================
// TABLE-DRIVEN TESTS
// ==========================================

func TestMockTool_VariousInputs(t *testing.T) {
	tests := []struct {
		name    string
		args    string
		want    string
		wantErr bool
	}{
		{"empty args", "{}", "mock result for echo_tool", false},
		{"json object", `{"key":"val"}`, "mock result for echo_tool", false},
		{"json array", `["a","b"]`, "mock result for echo_tool", false},
		{"json number", `42`, "mock result for echo_tool", false},
		{"json bool", `true`, "mock result for echo_tool", false},
		{"json null", `null`, "mock result for echo_tool", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool := &MockTool{name: "echo_tool"}
			got, err := tool.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Execute() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWAL_VariousMessageRoles(t *testing.T) {
	tests := []string{
		openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
	}
	for _, role := range tests {
		t.Run(role, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			msg := openai.ChatCompletionMessage{Role: role, Content: "test"}
			w.Append(msg)
			loaded, _ := w.LoadAll()
			if len(loaded) != 1 {
				t.Fatalf("expected 1 message, got %d", len(loaded))
			}
			if loaded[0].Role != role {
				t.Fatalf("role mismatch: expected %s, got %s", role, loaded[0].Role)
			}
		})
	}
}

// ==========================================
// TOKEN COUNTING TESTS
// ==========================================

func TestCountTokens_EmptyHistory(t *testing.T) {
	tokens := countTokens(nil)
	if tokens != 0 {
		t.Fatalf("expected 0 tokens for nil history, got %d", tokens)
	}
	tokens = countTokens([]openai.ChatCompletionMessage{})
	if tokens != 0 {
		t.Fatalf("expected 0 tokens for empty history, got %d", tokens)
	}
}

func TestCountTokens_SingleMessage(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Hello, world!"},
	}
	tokens := countTokens(messages)
	if tokens <= 0 {
		t.Fatalf("expected positive token count, got %d", tokens)
	}
	if tokens > 20 {
		t.Fatalf("expected reasonable token count for short message, got %d", tokens)
	}
}

func TestCountTokens_MultipleMessages(t *testing.T) {
	single := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Hello"},
	}
	singleTokens := countTokens(single)

	multi := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Hi there! How can I help you today?"},
	}
	multiTokens := countTokens(multi)

	if multiTokens <= singleTokens {
		t.Fatalf("multi-message tokens (%d) should be > single message tokens (%d)", multiTokens, singleTokens)
	}
}

func TestCountTokens_SpecialCharacters(t *testing.T) {
	cases := []struct {
		name    string
		content string
	}{
		{"unicode", "\u00e9\u00e8\u00ea\u00eb \u4f60\u597d\u4e16\u754c"},
		{"emojis", "\U0001f600\U0001f680\U0001f30d"},
		{"code", "func main() { fmt.Println(\"hello\") }"},
		{"whitespace", "   \t\t\n\n   \t  "},
		{"json", `{"key": "value", "nested": {"a": [1, 2, 3]}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			messages := []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleUser, Content: tc.content},
			}
			tokens := countTokens(messages)
			if tokens <= 0 {
				t.Fatalf("expected positive token count for %q, got %d", tc.name, tokens)
			}
		})
	}
}

func TestCountTokens_LargeContent(t *testing.T) {
	largeContent := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: largeContent},
	}
	tokens := countTokens(messages)
	if tokens < 100 {
		t.Fatalf("expected at least 100 tokens for large content, got %d", tokens)
	}
	if tokens > 5000 {
		t.Fatalf("expected reasonable token count, got %d", tokens)
	}
}

func TestCountTokens_Consistency(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "You are a helpful assistant."},
		{Role: openai.ChatMessageRoleUser, Content: "Tell me about Go programming."},
		{Role: openai.ChatMessageRoleAssistant, Content: "Go is a statically typed language designed at Google."},
	}
	count1 := countTokens(messages)
	count2 := countTokens(messages)
	if count1 != count2 {
		t.Fatalf("token count should be consistent: %d vs %d", count1, count2)
	}
}

func TestCountTokens_EmptyContent(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: ""},
		{Role: openai.ChatMessageRoleAssistant, Content: ""},
	}
	tokens := countTokens(messages)
	if tokens != 8 {
		t.Fatalf("expected 8 tokens (2 messages * 4 overhead), got %d", tokens)
	}
}

// ==========================================
// FIND RECENT START TESTS
// ==========================================

func TestFindRecentStart_AllFit(t *testing.T) {
	messages := []openai.ChatCompletionMessage{}
	for i := range 5 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("short msg %d", i),
		})
	}
	idx := findRecentStart(messages, 10000)
	if idx != 0 {
		t.Fatalf("expected 0 (all fit), got %d", idx)
	}
}

func TestFindRecentStart_SomeExcluded(t *testing.T) {
	messages := []openai.ChatCompletionMessage{}
	for range 20 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", 100),
		})
	}
	idx := findRecentStart(messages, 500)
	if idx == 0 {
		t.Fatal("expected some messages to be excluded, got 0")
	}
	if idx >= len(messages) {
		t.Fatalf("expected idx < %d, got %d", len(messages), idx)
	}
}

func TestFindRecentStart_SingleMessage(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	}
	idx := findRecentStart(messages, 1)
	if idx != 0 {
		t.Fatalf("expected 0 (keep last message even if it exceeds budget), got %d", idx)
	}
}

func TestFindRecentStart_EmptySlice(t *testing.T) {
	idx := findRecentStart(nil, 1000)
	if idx != 0 {
		t.Fatalf("expected 0 for empty slice, got %d", idx)
	}
	idx = findRecentStart([]openai.ChatCompletionMessage{}, 1000)
	if idx != 0 {
		t.Fatalf("expected 0 for empty slice, got %d", idx)
	}
}

func TestFindRecentStart_AllLarge(t *testing.T) {
	messages := []openai.ChatCompletionMessage{}
	for range 5 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("x", 1000),
		})
	}
	idx := findRecentStart(messages, 10)
	if idx != 4 {
		t.Fatalf("expected 4 (keep only last message), got %d", idx)
	}
}

func TestFindRecentStart_PreservesLastMessages(t *testing.T) {
	messages := []openai.ChatCompletionMessage{}
	for i := range 10 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("message_%d_content_here", i),
		})
	}
	idx := findRecentStart(messages, 50)
	recent := messages[idx:]
	if len(recent) == 0 {
		t.Fatal("expected at least one recent message")
	}
	if recent[len(recent)-1].Content != "message_9_content_here" {
		t.Fatalf("last message should be preserved, got %q", recent[len(recent)-1].Content)
	}
}

// ==========================================
// WAL REWRITE TESTS
// ==========================================

func TestWALRewrite_Empty(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old"})

	err := w.Rewrite([]openai.ChatCompletionMessage{})
	if err != nil {
		t.Fatalf("rewrite empty failed: %v", err)
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != 0 {
		t.Fatalf("expected 0 messages after empty rewrite, got %d", len(loaded))
	}
}

func TestWALRewrite_WithMessages(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	err := w.Rewrite(messages)
	if err != nil {
		t.Fatalf("rewrite failed: %v", err)
	}

	loaded, _ := w.LoadAll()
	if len(loaded) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded))
	}
	if loaded[0].Content != "sys" || loaded[1].Content != "hello" || loaded[2].Content != "hi" {
		t.Fatal("message order not preserved")
	}
}

func TestWALRewrite_OverwritesExisting(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old_1"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old_2"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "old_3"})

	newMessages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "new_1"},
		{Role: openai.ChatMessageRoleAssistant, Content: "new_2"},
	}

	w.Rewrite(newMessages)

	loaded, _ := w.LoadAll()
	if len(loaded) != 2 {
		t.Fatalf("expected 2 messages after overwrite, got %d", len(loaded))
	}
	for _, m := range loaded {
		if strings.HasPrefix(m.Content, "old") {
			t.Fatalf("found old message after rewrite: %q", m.Content)
		}
	}
}

func TestWALRewrite_PreservesOrder(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	messages := []openai.ChatCompletionMessage{}
	for i := range 50 {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("ordered_%d", i),
		})
	}

	w.Rewrite(messages)
	loaded, _ := w.LoadAll()

	if len(loaded) != 50 {
		t.Fatalf("expected 50 messages, got %d", len(loaded))
	}
	for i := range 50 {
		expected := fmt.Sprintf("ordered_%d", i)
		if loaded[i].Content != expected {
			t.Fatalf("order mismatch at %d: expected %q, got %q", i, expected, loaded[i].Content)
		}
	}
}

func TestWALRewrite_AfterAppend(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "appended"})
	w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "appended_2"})

	replacement := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "rewritten_sys"},
	}
	w.Rewrite(replacement)

	loaded, _ := w.LoadAll()
	if len(loaded) != 1 {
		t.Fatalf("expected 1 message after rewrite, got %d", len(loaded))
	}
	if loaded[0].Content != "rewritten_sys" {
		t.Fatalf("unexpected content: %q", loaded[0].Content)
	}
}

// ==========================================
// MEMORY TESTS
// ==========================================

func TestMemory_NewMemory(t *testing.T) {
	mem := NewMemory("/tmp/test_memory")
	if mem == nil {
		t.Fatal("NewMemory returned nil")
	}
	if mem.baseDir != "/tmp/test_memory" {
		t.Fatalf("expected baseDir '/tmp/test_memory', got %q", mem.baseDir)
	}
}

func TestMemory_ReadNonExistent(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemory(filepath.Join(dir, "nonexistent"))

	content, err := mem.ReadMemory()
	if err != nil {
		t.Fatalf("expected no error for non-existent memory, got: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty string for non-existent memory, got %q", content)
	}
}

func TestMemory_ReadMemory(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	mem.mu.Lock()
	os.MkdirAll(dir, 0755)
	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("stored fact\n"), 0644)
	mem.mu.Unlock()

	content, err := mem.ReadMemory()
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if content != "stored fact\n" {
		t.Fatalf("unexpected content: %q", content)
	}
}

func TestMemory_FlushCreatesFiles(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "- Important fact about Paris\n- User prefers morning meetings",
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

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system"},
		{Role: openai.ChatMessageRoleUser, Content: "I live in Paris and prefer morning meetings"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Noted! I'll remember that."},
	}

	err := mem.Flush(context.Background(), client, DefaultModel, history)
	if err != nil {
		t.Fatalf("flush error: %v", err)
	}

	mainPath := filepath.Join(dir, "MEMORY.md")
	if _, err := os.Stat(mainPath); os.IsNotExist(err) {
		t.Fatal("MEMORY.md should exist after flush")
	}

	mainContent, _ := os.ReadFile(mainPath)
	if !strings.Contains(string(mainContent), "Important fact") {
		t.Fatal("MEMORY.md should contain flushed facts")
	}

	todayFile := filepath.Join(dir, time.Now().Format("2006-01-02")+".md")
	if _, err := os.Stat(todayFile); os.IsNotExist(err) {
		t.Fatal("dated memory file should exist after flush")
	}
}

func TestMemory_FlushAppends(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "fact_1"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "msg"},
		{Role: openai.ChatMessageRoleAssistant, Content: "reply"},
	}

	mem.Flush(context.Background(), client, DefaultModel, history)
	mem.Flush(context.Background(), client, DefaultModel, history)

	mainContent, _ := os.ReadFile(filepath.Join(dir, "MEMORY.md"))
	occurrences := strings.Count(string(mainContent), "fact_1")
	if occurrences != 2 {
		t.Fatalf("expected 2 occurrences of fact_1 (appended), got %d", occurrences)
	}
}

func TestMemory_FlushTooFewMessages(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("LLM should not be called for < 3 messages")
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "only one"},
	}
	err := mem.Flush(context.Background(), client, DefaultModel, history)
	if err != nil {
		t.Fatalf("flush should succeed with < 3 messages: %v", err)
	}

	_, err = os.Stat(filepath.Join(dir, "MEMORY.md"))
	if !os.IsNotExist(err) {
		t.Fatal("MEMORY.md should not be created for < 3 messages")
	}
}

// ==========================================
// SERIALIZE MESSAGES TESTS
// ==========================================

func TestSerializeMessages_Basic(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi there"},
	}
	result := serializeMessages(messages)
	if !strings.Contains(result, "[user]: hello") {
		t.Fatalf("missing user message in serialization: %q", result)
	}
	if !strings.Contains(result, "[assistant]: hi there") {
		t.Fatalf("missing assistant message in serialization: %q", result)
	}
}

func TestSerializeMessages_Empty(t *testing.T) {
	result := serializeMessages(nil)
	if result != "" {
		t.Fatalf("expected empty string for nil, got %q", result)
	}
	result = serializeMessages([]openai.ChatCompletionMessage{})
	if result != "" {
		t.Fatalf("expected empty string for empty, got %q", result)
	}
}

func TestSerializeMessages_MixedRoles(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "system prompt"},
		{Role: openai.ChatMessageRoleUser, Content: "user input"},
		{Role: openai.ChatMessageRoleAssistant, Content: "assistant reply"},
		{Role: openai.ChatMessageRoleTool, Content: "tool result", Name: "my_tool", ToolCallID: "c1"},
	}
	result := serializeMessages(messages)
	if !strings.Contains(result, "[system]: system prompt") {
		t.Fatal("missing system role")
	}
	if !strings.Contains(result, "[tool]: tool result") {
		t.Fatal("missing tool role")
	}
}

// ==========================================
// MOCK SUMMARIZER TESTS
// ==========================================

type MockSummarizer struct {
	Summary string
	Err     error
	Called  bool
	Calls   int
	mu      sync.Mutex
}

func (m *MockSummarizer) Summarize(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Called = true
	m.Calls++
	return m.Summary, m.Err
}

func (m *MockSummarizer) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.Calls
}

func TestMockSummarizer_CalledWithCorrectArgs(t *testing.T) {
	mock := &MockSummarizer{Summary: "Test summary"}
	agent := &Agent{
		summarizer: mock.Summarize,
	}

	result := agent.generateSummary(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	})

	if !mock.Called {
		t.Fatal("summarizer should have been called")
	}
	if result != "Test summary" {
		t.Fatalf("expected 'Test summary', got %q", result)
	}
}

func TestMockSummarizer_ErrorHandling(t *testing.T) {
	mock := &MockSummarizer{Err: fmt.Errorf("LLM unavailable")}
	agent := &Agent{
		summarizer: mock.Summarize,
	}

	result := agent.generateSummary(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	})

	if !mock.Called {
		t.Fatal("summarizer should have been called even on error")
	}
	if !strings.Contains(result, "Summary generation failed") {
		t.Fatalf("expected failure message, got %q", result)
	}
}

func TestMockSummarizer_EmptyMessages(t *testing.T) {
	mock := &MockSummarizer{Summary: "should not be called"}
	agent := &Agent{
		summarizer: mock.Summarize,
	}

	result := agent.generateSummary(context.Background(), nil)
	if mock.Called {
		t.Fatal("summarizer should NOT be called for empty messages")
	}
	if result != "No previous context." {
		t.Fatalf("expected default message, got %q", result)
	}
}

// ==========================================
// COMPACT HISTORY TESTS
// ==========================================

func createCompactionAgent(t *testing.T, w *WAL, summarizerFn SummarizeFunc) (*Agent, *httptest.Server) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)
	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	if summarizerFn != nil {
		agent.summarizer = summarizerFn
	}

	return agent, server
}

func makeLargeHistory(numMessages int, tokensPerMessage int) []openai.ChatCompletionMessage {
	var messages []openai.ChatCompletionMessage
	for range numMessages {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: strings.Repeat("word ", tokensPerMessage),
		})
	}
	return messages
}

func TestCompactHistory_BelowThreshold(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "should not be called"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.reserveTokens = 100000
	agent.history = append(agent.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "short message",
	})

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mock.Called {
		t.Fatal("summarizer should not be called when below threshold")
	}
	if len(agent.history) != 2 {
		t.Fatalf("history should be unchanged: expected 2, got %d", len(agent.history))
	}
}

func TestCompactHistory_AboveThreshold(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "Compacted summary of old messages"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)

	beforeLen := len(agent.history)
	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !mock.Called {
		t.Fatal("summarizer should have been called")
	}
	if len(agent.history) >= beforeLen {
		t.Fatalf("history should have shrunk: before=%d, after=%d", beforeLen, len(agent.history))
	}
}

func TestCompactHistory_PreservesSystemPrompt(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "summary"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)

	agent.compactHistoryIfNeeded(context.Background())

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Fatalf("first message should be system prompt, got %s", agent.history[0].Role)
	}
	if !strings.Contains(agent.history[0].Content, "autonomous") {
		t.Fatal("system prompt content should be preserved")
	}
}

func TestCompactHistory_PreservesRecentMessages(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "old stuff was here"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	recentMsg := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "MOST RECENT MESSAGE MARKER",
	}
	agent.history = append(agent.history, makeLargeHistory(20, 50)...)
	agent.history = append(agent.history, recentMsg)

	agent.compactHistoryIfNeeded(context.Background())

	found := false
	for _, m := range agent.history {
		if m.Content == "MOST RECENT MESSAGE MARKER" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("recent message should be preserved after compaction")
	}
}

func TestCompactHistory_GeneratesSummary(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "Key fact: user likes pizza"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)

	agent.compactHistoryIfNeeded(context.Background())

	foundSummary := false
	for _, m := range agent.history {
		if strings.Contains(m.Content, "[CONTEXT SUMMARY]") {
			foundSummary = true
			if !strings.Contains(m.Content, "Key fact: user likes pizza") {
				t.Fatal("summary content should be in context summary message")
			}
		}
	}
	if !foundSummary {
		t.Fatal("context summary message should be in history after compaction")
	}
}

func TestCompactHistory_UpdatesWAL(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "summary for WAL"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)
	agent.compactHistoryIfNeeded(context.Background())

	loaded, _ := w.LoadAll()
	if len(loaded) != len(agent.history) {
		t.Fatalf("WAL should match history: WAL=%d, history=%d", len(loaded), len(agent.history))
	}

	foundSummary := false
	for _, m := range loaded {
		if strings.Contains(m.Content, "[CONTEXT SUMMARY]") {
			foundSummary = true
		}
	}
	if !foundSummary {
		t.Fatal("WAL should contain context summary after compaction")
	}
}

func TestCompactHistory_EmptyHistory(t *testing.T) {
	agent := &Agent{
		reserveTokens:    100,
		keepRecentTokens: 50,
		history:          []openai.ChatCompletionMessage{},
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 0 {
		t.Fatalf("history should remain empty, got %d", len(agent.history))
	}
}

func TestCompactHistory_DisabledWhenZero(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "should not be called"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.reserveTokens = 0
	agent.history = append(agent.history, makeLargeHistory(100, 100)...)

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mock.Called {
		t.Fatal("summarizer should not be called when compaction is disabled")
	}
}

func TestCompactHistory_MultipleCompactions(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	callNum := 0
	summarizer := func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		callNum++
		return fmt.Sprintf("summary_%d", callNum), nil
	}

	agent, server := createCompactionAgent(t, w, summarizer)
	defer server.Close()

	agent.history = append(agent.history, makeLargeHistory(30, 80)...)

	agent.compactHistoryIfNeeded(context.Background())
	firstLen := len(agent.history)

	agent.history = append(agent.history, makeLargeHistory(30, 80)...)
	agent.compactHistoryIfNeeded(context.Background())
	secondLen := len(agent.history)

	if firstLen >= 31 {
		t.Fatalf("first compaction should reduce history: %d", firstLen)
	}
	if secondLen >= firstLen+30 {
		t.Fatalf("second compaction should reduce history: first=%d, second=%d", firstLen, secondLen)
	}
	if callNum != 2 {
		t.Fatalf("expected 2 summary calls, got %d", callNum)
	}
}

func TestCompactHistory_WithMemoryFlush(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "test.wal")
	w := &WAL{path: walPath}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "memory facts extracted"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := NewAgent(client, w)
	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.memory = NewMemory(filepath.Join(dir, "memory"))
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		return "compacted", nil
	}

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)
	agent.compactHistoryIfNeeded(context.Background())

	memDir := filepath.Join(dir, "memory")
	if _, err := os.Stat(filepath.Join(memDir, "MEMORY.md")); os.IsNotExist(err) {
		t.Fatal("MEMORY.md should be created after compaction with memory flush")
	}
}

// ==========================================
// GENERATE SUMMARY TESTS
// ==========================================

func TestGenerateSummary_NilClient(t *testing.T) {
	agent := &Agent{
		client:     nil,
		summarizer: nil,
	}
	result := agent.generateSummary(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "test"},
	})
	if !strings.Contains(result, "failed") {
		t.Fatalf("expected failure message for nil client, got %q", result)
	}
}

func TestGenerateSummary_NilClientWithMockSummarizer(t *testing.T) {
	mock := &MockSummarizer{Summary: "mocked summary"}
	agent := &Agent{
		client:     nil,
		summarizer: mock.Summarize,
	}
	result := agent.generateSummary(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "test"},
	})
	if result != "mocked summary" {
		t.Fatalf("expected mocked summary, got %q", result)
	}
}

func TestGenerateSummary_ContextCancellation(t *testing.T) {
	mock := func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}
	agent := &Agent{summarizer: mock}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result := agent.generateSummary(ctx, []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "test"},
	})
	if !strings.Contains(result, "Summary generation failed") {
		t.Fatalf("expected failure for cancelled context, got %q", result)
	}
}

// ==========================================
// DEFAULT SUMMARIZER TESTS
// ==========================================

func TestDefaultSummarizer_WithLLM(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summarized: user asked about flights"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := &Agent{client: client}

	result, err := agent.defaultSummarizer(context.Background(), "Summarize this", []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Tell me about flights"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Summarized: user asked about flights" {
		t.Fatalf("unexpected summary: %q", result)
	}
}

func TestDefaultSummarizer_NilClient(t *testing.T) {
	agent := &Agent{client: nil}
	_, err := agent.defaultSummarizer(context.Background(), "prompt", nil)
	if err == nil {
		t.Fatal("expected error for nil client")
	}
}

func TestDefaultSummarizer_LLMReturnsNoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	config := openai.DefaultConfig("test-key")
	config.BaseURL = server.URL + "/v1"
	client := openai.NewClientWithConfig(config)

	agent := &Agent{client: client}
	_, err := agent.defaultSummarizer(context.Background(), "prompt", []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "test"},
	})
	if err == nil {
		t.Fatal("expected error when LLM returns no choices")
	}
}

// ==========================================
// CHAT WITH COMPACTION TESTS
// ==========================================

func TestChat_TriggersCompaction(t *testing.T) {
	summaryCalled := false
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		summaryCalled = true
		return "compacted", nil
	}

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)

	_, err := agent.Chat(context.Background(), "trigger compaction")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !summaryCalled {
		t.Fatal("compaction should have been triggered during chat")
	}
}

func TestChat_NoCompactionForSmallHistory(t *testing.T) {
	summaryCalled := false
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.reserveTokens = 100000
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		summaryCalled = true
		return "compacted", nil
	}

	agent.Chat(context.Background(), "short message")

	if summaryCalled {
		t.Fatal("compaction should not trigger for small history")
	}
}

func TestChat_CompactionFailureDoesNotBlockChat(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "still works"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.reserveTokens = 100
	agent.keepRecentTokens = 50
	agent.summarizer = func(ctx context.Context, prompt string, messages []openai.ChatCompletionMessage) (string, error) {
		return "", fmt.Errorf("forced compaction failure")
	}

	agent.history = append(agent.history, makeLargeHistory(20, 50)...)

	reply, err := agent.Chat(context.Background(), "test")
	if err != nil {
		t.Fatalf("chat should succeed even if compaction fails: %v", err)
	}
	if reply != "still works" {
		t.Fatalf("unexpected reply: %s", reply)
	}
}

// ==========================================
// TABLE-DRIVEN COMPACTION TESTS
// ==========================================

func TestCompactHistory_Thresholds(t *testing.T) {
	tests := []struct {
		name          string
		reserve       int
		keepRecent    int
		numMessages   int
		tokensPerMsg  int
		expectCompact bool
	}{
		{"well below threshold", 10000, 5000, 5, 5, false},
		{"just above threshold", 100, 50, 20, 50, true},
		{"high threshold no compact", 100000, 50000, 10, 10, false},
		{"zero reserve disabled", 0, 50, 100, 100, false},
		{"zero keepRecent disabled", 100, 0, 100, 100, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			mock := &MockSummarizer{Summary: "test"}
			agent, server := createCompactionAgent(t, w, mock.Summarize)
			defer server.Close()

			agent.reserveTokens = tt.reserve
			agent.keepRecentTokens = tt.keepRecent

			agent.history = append(agent.history, makeLargeHistory(tt.numMessages, tt.tokensPerMsg)...)
			agent.compactHistoryIfNeeded(context.Background())

			if tt.expectCompact && !mock.Called {
				t.Fatal("expected compaction to trigger")
			}
			if !tt.expectCompact && mock.Called {
				t.Fatal("expected no compaction")
			}
		})
	}
}

func TestChat_VariousReplies(t *testing.T) {
	tests := []struct {
		name           string
		llmContent     string
		llmToolCalls   []openai.ToolCall
		expectedReply  string
		expectError    bool
		registeredTool string
	}{
		{
			name:          "simple text",
			llmContent:    "hello there",
			expectedReply: "hello there",
		},
		{
			name:          "empty string",
			llmContent:    "",
			expectedReply: "",
		},
		{
			name:          "multiline",
			llmContent:    "line1\nline2\nline3",
			expectedReply: "line1\nline2\nline3",
		},
		{
			name:          "json in content",
			llmContent:    `{"key": "value"}`,
			expectedReply: `{"key": "value"}`,
		},
		{
			name:          "very long reply",
			llmContent:    strings.Repeat("word ", 1000),
			expectedReply: strings.TrimSpace(strings.Repeat("word ", 1000)),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:      openai.ChatMessageRoleAssistant,
								Content:   tt.llmContent,
								ToolCalls: tt.llmToolCalls,
							},
						},
					},
				}
				json.NewEncoder(w).Encode(resp)
			})
			defer server.Close()

			agent := NewAgent(client, nil)
			if tt.registeredTool != "" {
				agent.RegisterTool(&MockTool{name: tt.registeredTool})
			}

			reply, err := agent.Chat(context.Background(), "test")
			if tt.expectError {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if reply != tt.expectedReply {
				t.Fatalf("reply mismatch:\n  expected: %q\n  got:      %q", tt.expectedReply, reply)
			}
		})
	}
}

// ==========================================
// BUILD SYSTEM PROMPT TESTS
// ==========================================

func TestBuildSystemPrompt_NoMemory(t *testing.T) {
	agent := &Agent{
		baseSystemPrompt: "You are a helpful assistant.",
		memory:           nil,
	}
	prompt := agent.buildSystemPrompt()
	if prompt != "You are a helpful assistant." {
		t.Fatalf("expected base prompt, got %q", prompt)
	}
}

func TestBuildSystemPrompt_EmptyMemory(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	agent := &Agent{
		baseSystemPrompt: "You are a helpful assistant.",
		memory:           mem,
	}
	prompt := agent.buildSystemPrompt()
	if prompt != "You are a helpful assistant." {
		t.Fatalf("expected base prompt when memory is empty, got %q", prompt)
	}
}

func TestBuildSystemPrompt_WithMemory(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("- User likes pizza\n- Lives in Berlin"), 0644)

	agent := &Agent{
		baseSystemPrompt: "You are a helpful assistant.",
		memory:           mem,
	}
	prompt := agent.buildSystemPrompt()
	if !strings.Contains(prompt, "You are a helpful assistant.") {
		t.Fatal("prompt should contain base prompt")
	}
	if !strings.Contains(prompt, "Remembered Context") {
		t.Fatal("prompt should contain Remembered Context header")
	}
	if !strings.Contains(prompt, "User likes pizza") {
		t.Fatal("prompt should contain memory content")
	}
}

func TestRefreshSystemPrompt_UpdatesHistory(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	agent := &Agent{
		baseSystemPrompt: "Base prompt",
		memory:           mem,
		history: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "Base prompt"},
			{Role: openai.ChatMessageRoleUser, Content: "hello"},
		},
	}

	os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("- New memory"), 0644)

	agent.RefreshSystemPrompt()

	if !strings.Contains(agent.history[0].Content, "New memory") {
		t.Fatalf("system prompt should be updated with memory: %q", agent.history[0].Content)
	}
}

func TestRefreshSystemPrompt_NoSystemPrompt(t *testing.T) {
	agent := &Agent{
		baseSystemPrompt: "Base",
		history: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "no system here"},
		},
	}

	agent.RefreshSystemPrompt()

	if len(agent.history) != 1 {
		t.Fatal("history should be unchanged")
	}
}

// ==========================================
// SERIALIZE MESSAGES WITH TOOL CALLS TESTS
// ==========================================

func TestSerializeMessages_ToolCalls(t *testing.T) {
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "search for x"},
		{Role: openai.ChatMessageRoleAssistant, Content: "", ToolCalls: []openai.ToolCall{
			{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "search", Arguments: `{"q":"x"}`}},
		}},
		{Role: openai.ChatMessageRoleTool, Content: "results", Name: "search", ToolCallID: "c1"},
	}
	result := serializeMessages(messages)
	if !strings.Contains(result, "[tool_call:search]") {
		t.Fatalf("tool calls should be serialized: %q", result)
	}
	if !strings.Contains(result, `{"q":"x"}`) {
		t.Fatalf("tool arguments should be serialized: %q", result)
	}
}

// ==========================================
// COMPACTION SKIPS WHEN NOTHING TO COMPACT TESTS
// ==========================================

func TestCompactHistory_SkipsWhenAllFit(t *testing.T) {
	w, cleanup := newTestWAL(t)
	defer cleanup()

	mock := &MockSummarizer{Summary: "should not be called"}
	agent, server := createCompactionAgent(t, w, mock.Summarize)
	defer server.Close()

	agent.reserveTokens = 100000
	agent.keepRecentTokens = 100000

	agent.history = append(agent.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "short message",
	})
	agent.history = append(agent.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: "short reply",
	})

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mock.Called {
		t.Fatal("summarizer should not be called when all messages fit in recent budget")
	}
}

// ==========================================
// EDGE CASE TESTS - TOKEN COUNTING
// ==========================================

func TestCountTokens_EdgeCases(t *testing.T) {
	tests := []struct {
		name           string
		messages       []openai.ChatCompletionMessage
		expectPositive bool
		description    string
	}{
		{"nil slice", nil, false, "nil should return 0"},
		{"empty slice", []openai.ChatCompletionMessage{}, false, "empty slice should return 0"},
		{"single empty message", []openai.ChatCompletionMessage{{Role: "user", Content: ""}}, true, "empty content still has overhead"},
		{"only whitespace", []openai.ChatCompletionMessage{{Role: "user", Content: "   \t\n  "}}, true, "whitespace counts"},
		{"very long single message", []openai.ChatCompletionMessage{{Role: "user", Content: strings.Repeat("a", 100000)}}, true, "large content"},
		{"many small messages", makeMessages(1000, "hi"), true, "many small messages"},
		{"unicode heavy", []openai.ChatCompletionMessage{{Role: "user", Content: strings.Repeat("你好世界", 1000)}}, true, "chinese chars"},
		{"emoji heavy", []openai.ChatCompletionMessage{{Role: "user", Content: strings.Repeat("😀🎉🚀", 1000)}}, true, "emojis"},
		{"mixed roles", []openai.ChatCompletionMessage{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "a"},
			{Role: "tool", Content: "t"},
		}, true, "all role types"},
		{"code content", []openai.ChatCompletionMessage{{Role: "user", Content: "func main() { fmt.Println(\"hello\") }"}}, true, "code"},
		{"json content", []openai.ChatCompletionMessage{{Role: "user", Content: `{"key": "value", "nested": {"a": [1, 2, 3]}}`}}, true, "json"},
		{"markdown content", []openai.ChatCompletionMessage{{Role: "user", Content: "# Header\n\n**bold**\n\n- item 1\n- item 2"}}, true, "markdown"},
		{"null bytes", []openai.ChatCompletionMessage{{Role: "user", Content: "hello\x00world"}}, true, "null bytes in content"},
		{"only newlines", []openai.ChatCompletionMessage{{Role: "user", Content: "\n\n\n\n\n"}}, true, "only newlines"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := countTokens(tt.messages)
			if tt.expectPositive && result <= 0 {
				t.Errorf("%s: expected positive, got %d", tt.description, result)
			}
			if !tt.expectPositive && result != 0 {
				t.Errorf("%s: expected 0, got %d", tt.description, result)
			}
		})
	}
}

func makeMessages(count int, content string) []openai.ChatCompletionMessage {
	msgs := make([]openai.ChatCompletionMessage, count)
	for i := range msgs {
		msgs[i] = openai.ChatCompletionMessage{Role: "user", Content: content}
	}
	return msgs
}

// ==========================================
// EDGE CASE TESTS - FIND RECENT START
// ==========================================

func TestFindRecentStart_EdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		messages      []openai.ChatCompletionMessage
		budget        int
		expectedStart int
		description   string
	}{
		{"nil messages", nil, 100, 0, "nil should return 0"},
		{"empty slice", []openai.ChatCompletionMessage{}, 100, 0, "empty should return 0"},
		{"zero budget", []openai.ChatCompletionMessage{{Role: "user", Content: "hello"}}, 0, 0, "zero budget"},
		{"negative budget", []openai.ChatCompletionMessage{{Role: "user", Content: "hello"}}, -100, 0, "negative budget"},
		{"single message small", []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}, 10000, 0, "single small message"},
		{"single message large", []openai.ChatCompletionMessage{{Role: "user", Content: strings.Repeat("x", 100000)}}, 10, 0, "single large message"},
		{"budget exactly fits", []openai.ChatCompletionMessage{
			{Role: "user", Content: "a"},
			{Role: "user", Content: "b"},
		}, 20, 0, "exact fit"},
		{"budget one less", []openai.ChatCompletionMessage{
			{Role: "user", Content: "a"},
			{Role: "user", Content: "b"},
		}, 10, 0, "one less than needed"},
		{"all messages identical", []openai.ChatCompletionMessage{
			{Role: "user", Content: "same"},
			{Role: "user", Content: "same"},
			{Role: "user", Content: "same"},
			{Role: "user", Content: "same"},
			{Role: "user", Content: "same"},
		}, 20, 1, "identical messages"},
		{"very small budget", []openai.ChatCompletionMessage{
			{Role: "user", Content: "hello world this is a test"},
			{Role: "user", Content: "another message"},
			{Role: "user", Content: "last message"},
		}, 5, 2, "very small budget"},
		{"extremely large budget", []openai.ChatCompletionMessage{
			{Role: "user", Content: "a"},
			{Role: "user", Content: "b"},
		}, 1000000000, 0, "huge budget fits all"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findRecentStart(tt.messages, tt.budget)
			if result != tt.expectedStart {
				t.Errorf("%s: expected %d, got %d", tt.description, tt.expectedStart, result)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - WAL
// ==========================================

func TestWAL_Append_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		message     openai.ChatCompletionMessage
		shouldError bool
		description string
	}{
		{"empty content", openai.ChatCompletionMessage{Role: "user", Content: ""}, false, "empty content should work"},
		{"very long content", openai.ChatCompletionMessage{Role: "user", Content: strings.Repeat("x", 1000000)}, false, "1MB content"},
		{"all role types", openai.ChatCompletionMessage{Role: "tool", Content: "result", Name: "tool_name", ToolCallID: "call_123"}, false, "tool message with all fields"},
		{"unicode content", openai.ChatCompletionMessage{Role: "user", Content: "你好世界 🌍\n\t\r"}, false, "unicode and special chars"},
		{"json in content", openai.ChatCompletionMessage{Role: "user", Content: `{"nested": {"key": "value"}}`}, false, "json content"},
		{"null bytes", openai.ChatCompletionMessage{Role: "user", Content: "hello\x00world"}, false, "null bytes"},
		{"tool calls", openai.ChatCompletionMessage{
			Role:    "assistant",
			Content: "",
			ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "test", Arguments: `{"x": 1}`}},
			},
		}, false, "message with tool calls"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			err := w.Append(tt.message)
			if tt.shouldError && err == nil {
				t.Errorf("%s: expected error, got nil", tt.description)
			}
			if !tt.shouldError && err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}

			if !tt.shouldError {
				loaded, err := w.LoadAll()
				if err != nil {
					t.Errorf("failed to load: %v", err)
				}
				if len(loaded) != 1 {
					t.Errorf("expected 1 message, got %d", len(loaded))
				}
			}
		})
	}
}

func TestWAL_LoadAll_EdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		content       string
		expectedCount int
		description   string
	}{
		{"empty file", "", 0, "empty file"},
		{"only newlines", "\n\n\n\n", 0, "only newlines"},
		{"single valid", `{"role":"user","content":"hi"}`, 1, "single valid line"},
		{"multiple valid", `{"role":"user","content":"a"}
{"role":"assistant","content":"b"}`, 2, "multiple valid lines"},
		{"mixed valid invalid", `{"role":"user","content":"valid"}
not json at all
{"role":"assistant","content":"also valid"}`, 2, "skip invalid"},
		{"trailing newline", `{"role":"user","content":"hi"}` + "\n", 1, "trailing newline ok"},
		{"leading newline", "\n" + `{"role":"user","content":"hi"}`, 1, "leading newline ok"},
		{"empty lines between", `{"role":"user","content":"a"}

{"role":"assistant","content":"b"}`, 2, "empty lines skipped"},
		{"partial json", `{"role":"user","content":"missing end`, 0, "partial json skipped"},
		{"extra commas", `{"role":"user","content":"test",}`, 0, "trailing comma skipped as invalid json"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			if tt.content != "" {
				os.WriteFile(w.path, []byte(tt.content), 0644)
			}

			loaded, err := w.LoadAll()
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}
			if len(loaded) != tt.expectedCount {
				t.Errorf("%s: expected %d, got %d", tt.description, tt.expectedCount, len(loaded))
			}
		})
	}
}

func TestWAL_Rewrite_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		messages    []openai.ChatCompletionMessage
		shouldError bool
		description string
	}{
		{"empty slice", []openai.ChatCompletionMessage{}, false, "empty slice creates empty file"},
		{"nil slice", nil, false, "nil creates empty file"},
		{"single message", []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}, false, "single message"},
		{"thousand messages", makeMessages(1000, "test"), false, "many messages"},
		{"messages with special chars", []openai.ChatCompletionMessage{
			{Role: "user", Content: "hello\nworld\ttab"},
			{Role: "user", Content: "unicode: 你好"},
			{Role: "user", Content: "emoji: 🚀"},
		}, false, "special characters"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			w.Append(openai.ChatCompletionMessage{Role: "user", Content: "old"})

			err := w.Rewrite(tt.messages)
			if tt.shouldError && err == nil {
				t.Errorf("%s: expected error", tt.description)
			}
			if !tt.shouldError && err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}

			if !tt.shouldError {
				loaded, _ := w.LoadAll()
				if len(loaded) != len(tt.messages) {
					t.Errorf("%s: expected %d, got %d", tt.description, len(tt.messages), len(loaded))
				}
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - MEMORY
// ==========================================

func TestMemory_Flush_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		history     []openai.ChatCompletionMessage
		shouldWrite bool
		description string
	}{
		{"nil history", nil, false, "nil should not flush"},
		{"empty history", []openai.ChatCompletionMessage{}, false, "empty should not flush"},
		{"one message", []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}, false, "one message too few"},
		{"two messages", []openai.ChatCompletionMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		}, false, "two messages too few"},
		{"three messages minimum", []openai.ChatCompletionMessage{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		}, true, "three is minimum"},
		{"empty content messages", []openai.ChatCompletionMessage{
			{Role: "system", Content: ""},
			{Role: "user", Content: ""},
			{Role: "assistant", Content: ""},
		}, true, "empty content still flushes"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			mem := NewMemoryWithoutRateLimit(dir)

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "- fact 1\n- fact 2"}},
					},
				}
				json.NewEncoder(w).Encode(resp)
			}))
			defer server.Close()

			config := openai.DefaultConfig("test-key")
			config.BaseURL = server.URL + "/v1"
			client := openai.NewClientWithConfig(config)

			mem.Flush(context.Background(), client, DefaultModel, tt.history)

			_, err := os.Stat(filepath.Join(dir, "MEMORY.md"))
			if tt.shouldWrite && os.IsNotExist(err) {
				t.Errorf("%s: expected file to be created", tt.description)
			}
			if !tt.shouldWrite && !os.IsNotExist(err) {
				t.Errorf("%s: expected no file to be created", tt.description)
			}
		})
	}
}

func TestMemory_ReadMemory_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		setup       func(dir string)
		expected    string
		description string
	}{
		{"no file", func(dir string) {}, "", "no file returns empty"},
		{"empty file", func(dir string) {
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte(""), 0644)
		}, "", "empty file returns empty"},
		{"has content", func(dir string) {
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("test content"), 0644)
		}, "test content", "returns content"},
		{"large file", func(dir string) {
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte(strings.Repeat("x", 100000)), 0644)
		}, strings.Repeat("x", 100000), "large file works"},
		{"unicode content", func(dir string) {
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("你好世界 🌍"), 0644)
		}, "你好世界 🌍", "unicode preserved"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			mem := NewMemory(dir)

			tt.setup(dir)

			result, err := mem.ReadMemory()
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}
			if result != tt.expected {
				t.Errorf("%s: expected %q, got %q", tt.description, tt.expected, result)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - COMPACTION
// ==========================================

func TestCompactHistory_EdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		setupAgent    func(*Agent)
		expectedCount int
		description   string
	}{
		{"empty history", func(a *Agent) {
			a.history = nil
			a.reserveTokens = 100
		}, 0, "nil history doesn't crash"},
		{"only system prompt", func(a *Agent) {
			a.history = []openai.ChatCompletionMessage{{Role: "system", Content: "sys"}}
			a.reserveTokens = 1
		}, 1, "only system prompt"},
		{"exactly at threshold", func(a *Agent) {
			a.history = append(a.history, openai.ChatCompletionMessage{Role: "system", Content: "sys"})
			for range 10 {
				a.history = append(a.history, openai.ChatCompletionMessage{
					Role: "user", Content: "test message here",
				})
			}
			tokens := countTokens(a.history)
			a.reserveTokens = tokens
		}, 9, "at threshold compacts"},
		{"one over threshold", func(a *Agent) {
			a.history = append(a.history, openai.ChatCompletionMessage{Role: "system", Content: "sys"})
			for range 10 {
				a.history = append(a.history, openai.ChatCompletionMessage{
					Role: "user", Content: "test message here",
				})
			}
			tokens := countTokens(a.history)
			a.reserveTokens = tokens - 1
		}, 9, "one over triggers compaction"},
		{"zero reserve tokens", func(a *Agent) {
			a.reserveTokens = 0
			a.history = append(a.history, makeLargeHistory(100, 100)...)
		}, 101, "zero reserve disables"},
		{"negative reserve tokens", func(a *Agent) {
			a.reserveTokens = -100
			a.history = append(a.history, makeLargeHistory(100, 100)...)
		}, 101, "negative reserve disables"},
		{"zero keep recent", func(a *Agent) {
			a.reserveTokens = 100
			a.keepRecentTokens = 0
			a.history = append(a.history, makeLargeHistory(100, 100)...)
		}, 101, "zero keep recent disables"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, cleanup := newTestWAL(t)
			defer cleanup()

			mock := &MockSummarizer{Summary: "compacted"}
			agent, server := createCompactionAgent(t, w, mock.Summarize)
			defer server.Close()

			tt.setupAgent(agent)

			agent.compactHistoryIfNeeded(context.Background())

			if len(agent.history) != tt.expectedCount {
				t.Errorf("%s: expected %d messages, got %d", tt.description, tt.expectedCount, len(agent.history))
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - CHAT
// ==========================================

func TestChat_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		userInput   string
		llmResponse string
		description string
	}{
		{"empty input", "", "ok", "empty input handled"},
		{"whitespace only", "   \t\n  ", "ok", "whitespace input"},
		{"very long input", strings.Repeat("hello ", 10000), "ok", "long input"},
		{"unicode input", "你好世界 🌍 مرحبا", "ok", "unicode"},
		{"json input", `{"key": "value"}`, "ok", "json as input"},
		{"code input", "func main() { fmt.Println(\"hi\") }", "ok", "code as input"},
		{"markdown input", "# Header\n\n**bold**\n\n- item", "ok", "markdown"},
		{"special chars", "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/", "ok", "special chars"},
		{"null byte", "hello\x00world", "ok", "null byte"},
		{"newlines", "line1\nline2\nline3", "ok", "newlines preserved"},
		{"only newlines", "\n\n\n", "ok", "only newlines"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: "assistant", Content: tt.llmResponse}},
					},
				}
				json.NewEncoder(w).Encode(resp)
			})
			defer server.Close()

			agent := NewAgent(client, nil)
			agent.reserveTokens = 100000

			_, err := agent.Chat(context.Background(), tt.userInput)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}
		})
	}
}

func TestChat_ToolExecution_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		toolResult  string
		toolError   error
		description string
	}{
		{"empty result", "", nil, "empty result"},
		{"very large result", strings.Repeat("x", 100000), nil, "large result"},
		{"unicode result", "你好世界 🚀", nil, "unicode"},
		{"error result", "", fmt.Errorf("tool failed"), "tool error"},
		{"json result", `{"status": "ok", "data": [1, 2, 3]}`, nil, "json result"},
		{"multiline result", "line1\nline2\nline3", nil, "multiline"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callCount := 0
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				callCount++
				if callCount == 1 {
					args, _ := json.Marshal(map[string]string{"x": "y"})
					resp := openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{Message: openai.ChatCompletionMessage{
								Role: "assistant",
								ToolCalls: []openai.ToolCall{
									{ID: "call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "test_tool", Arguments: string(args)}},
								},
							}},
						},
					}
					json.NewEncoder(w).Encode(resp)
				} else {
					resp := openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}},
						},
					}
					json.NewEncoder(w).Encode(resp)
				}
			})
			defer server.Close()

			agent := NewAgent(client, nil)
			agent.RegisterTool(&MockTool{
				name: "test_tool",
				executeFn: func(args string) (string, error) {
					return tt.toolResult, tt.toolError
				},
			})
			agent.reserveTokens = 100000

			_, err := agent.Chat(context.Background(), "test")
			if err != nil && tt.toolError == nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - RATE LIMITING
// ==========================================

func TestRateLimiter_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		minDelay    time.Duration
		callCount   int
		description string
	}{
		{"zero delay", 0, 10, "zero delay allows all calls"},
		{"negative delay", -1 * time.Second, 10, "negative delay treated as zero"},
		{"very small delay", time.Nanosecond, 5, "tiny delay"},
		{"large delay", time.Hour, 1, "large delay blocks subsequent"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limiter := NewRateLimiter(tt.minDelay)

			successes := 0
			for i := 0; i < tt.callCount; i++ {
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
				if err := limiter.Wait(ctx); err == nil {
					successes++
				}
				cancel()
			}

			if tt.minDelay <= 0 && successes != tt.callCount {
				t.Errorf("%s: expected %d successes, got %d", tt.description, tt.callCount, successes)
			}
			if tt.minDelay > time.Second && successes > 1 {
				t.Errorf("%s: expected at most 1 success with large delay, got %d", tt.description, successes)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - SERIALIZE MESSAGES
// ==========================================

func TestSerializeMessages_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		messages    []openai.ChatCompletionMessage
		contains    []string
		description string
	}{
		{"nil", nil, nil, "nil returns empty"},
		{"empty", []openai.ChatCompletionMessage{}, nil, "empty returns empty"},
		{"single message", []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}, []string{"[user]: hi"}, "single message"},
		{"tool calls", []openai.ChatCompletionMessage{{
			Role:    "assistant",
			Content: "",
			ToolCalls: []openai.ToolCall{
				{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "test", Arguments: `{"x":1}`}},
			},
		}}, []string{"[tool_call:test]", `{"x":1}`}, "tool calls serialized"},
		{"multiple tool calls", []openai.ChatCompletionMessage{{
			Role:    "assistant",
			Content: "",
			ToolCalls: []openai.ToolCall{
				{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "tool1", Arguments: "{}"}},
				{ID: "c2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "tool2", Arguments: `{"a":1}`}},
			},
		}}, []string{"tool1", "tool2"}, "multiple tool calls"},
		{"special content", []openai.ChatCompletionMessage{
			{Role: "user", Content: "line1\nline2\ttab"},
		}, []string{"line1", "line2", "tab"}, "special chars"},
		{"empty content", []openai.ChatCompletionMessage{{Role: "user", Content: ""}}, []string{"[user]:"}, "empty content"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := serializeMessages(tt.messages)

			if tt.contains == nil && result != "" {
				t.Errorf("%s: expected empty, got %q", tt.description, result)
			}

			for _, s := range tt.contains {
				if !strings.Contains(result, s) {
					t.Errorf("%s: expected to contain %q, got %q", tt.description, s, result)
				}
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - BUILD SYSTEM PROMPT
// ==========================================

func TestBuildSystemPrompt_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		setup       func(*Agent)
		expected    string
		notExpected string
		description string
	}{
		{"nil memory", func(a *Agent) {
			a.memory = nil
			a.baseSystemPrompt = "base"
		}, "base", "Remembered Context", "nil memory no context"},
		{"empty memory", func(a *Agent) {
			a.memory = NewMemory(t.TempDir())
			a.baseSystemPrompt = "base"
		}, "base", "Remembered Context", "empty memory no context"},
		{"memory with content", func(a *Agent) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("- fact 1\n- fact 2"), 0644)
			a.memory = NewMemory(dir)
			a.baseSystemPrompt = "base"
		}, "Remembered Context", "", "memory adds context"},
		{"empty base prompt", func(a *Agent) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte("memory"), 0644)
			a.memory = NewMemory(dir)
			a.baseSystemPrompt = ""
		}, "memory", "", "empty base still shows memory"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := &Agent{}
			tt.setup(agent)

			result := agent.buildSystemPrompt()

			if tt.expected != "" && !strings.Contains(result, tt.expected) {
				t.Errorf("%s: expected to contain %q, got %q", tt.description, tt.expected, result)
			}
			if tt.notExpected != "" && strings.Contains(result, tt.notExpected) {
				t.Errorf("%s: expected not to contain %q, got %q", tt.description, tt.notExpected, result)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - CONCURRENT ACCESS
// ==========================================

func TestAgent_ConcurrentAccess_EdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		numGoroutines int
		operations    int
		description   string
	}{
		{"single goroutine", 1, 10, "baseline"},
		{"few goroutines", 5, 10, "light concurrency"},
		{"many goroutines", 50, 20, "heavy concurrency"},
		{"extreme concurrency", 100, 10, "stress test"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}},
					},
				}
				json.NewEncoder(w).Encode(resp)
			})
			defer server.Close()

			w, cleanup := newTestWAL(t)
			defer cleanup()

			agent := NewAgent(client, w)
			agent.reserveTokens = 100000

			var wg sync.WaitGroup
			errCh := make(chan error, tt.numGoroutines*tt.operations)

			for g := 0; g < tt.numGoroutines; g++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					for i := 0; i < tt.operations; i++ {
						_, err := agent.Chat(context.Background(), fmt.Sprintf("g%d_m%d", id, i))
						if err != nil {
							errCh <- err
						}
					}
				}(g)
			}

			done := make(chan struct{})
			go func() {
				wg.Wait()
				close(done)
			}()

			select {
			case <-done:
			case <-time.After(30 * time.Second):
				t.Fatalf("%s: timeout", tt.description)
			}

			close(errCh)
			errCount := 0
			for err := range errCh {
				t.Errorf("%s: error: %v", tt.description, err)
				errCount++
				if errCount > 5 {
					break
				}
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - CONTEXT CANCELLATION
// ==========================================

func TestContextCancellation_EdgeCases(t *testing.T) {
	tests := []struct {
		name        string
		cancelAfter time.Duration
		description string
	}{
		{"immediate cancel", 0, "cancel before start"},
		{"quick cancel", time.Millisecond, "cancel quickly"},
		{"slow cancel", 100 * time.Millisecond, "cancel after some work"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blockCh := make(chan struct{})
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				<-blockCh
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}},
					},
				}
				json.NewEncoder(w).Encode(resp)
			})
			defer server.Close()

			agent := NewAgent(client, nil)

			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(tt.cancelAfter, cancel)

			_, err := agent.Chat(ctx, "test")

			close(blockCh)

			if err == nil {
				t.Errorf("%s: expected error from cancellation", tt.description)
			}
		})
	}
}

// ==========================================
// EDGE CASE TESTS - TOOL TIMEOUT
// ==========================================

func TestToolTimeout_EdgeCases(t *testing.T) {
	tests := []struct {
		name          string
		toolDelay     time.Duration
		timeout       time.Duration
		shouldContain string
		description   string
	}{
		{"fast tool", 10 * time.Millisecond, time.Second, "", "fast tool succeeds"},
		{"slow tool", 2 * time.Second, 100 * time.Millisecond, "timed out", "slow tool times out"},
		{"zero timeout", 10 * time.Millisecond, 0, "", "zero timeout immediate"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callCount := 0
			server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
				callCount++
				if callCount == 1 {
					args, _ := json.Marshal(map[string]string{"x": "y"})
					resp := openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{Message: openai.ChatCompletionMessage{
								Role: "assistant",
								ToolCalls: []openai.ToolCall{
									{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "slow_tool", Arguments: string(args)}},
								},
							}},
						},
					}
					json.NewEncoder(w).Encode(resp)
				} else {
					resp := openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}},
						},
					}
					json.NewEncoder(w).Encode(resp)
				}
			})
			defer server.Close()

			agent := NewAgent(client, nil)
			agent.SetToolTimeout(tt.timeout)
			agent.RegisterTool(&MockTool{
				name: "slow_tool",
				executeFn: func(args string) (string, error) {
					time.Sleep(tt.toolDelay)
					return "result", nil
				},
			})
			agent.reserveTokens = 100000

			_, err := agent.Chat(context.Background(), "test")

			if err != nil {
				t.Errorf("%s: unexpected error: %v", tt.description, err)
			}

			if tt.shouldContain != "" {
				history := agent.GetHistory()
				found := false
				for _, m := range history {
					if m.Role == "tool" && strings.Contains(m.Content, tt.shouldContain) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("%s: expected tool result to contain %q", tt.description, tt.shouldContain)
				}
			}
		})
	}
}

// ==========================================
// LOOP DETECTOR TESTS
// ==========================================

func TestLoopDetector_NoLoop(t *testing.T) {
	ld := NewLoopDetector()

	isLoop, hint := ld.detectLoop("tool1", `{"a": "b"}`)
	if isLoop {
		t.Fatalf("unexpected loop detected: %s", hint)
	}
	if hint != "" {
		t.Fatalf("unexpected hint: %s", hint)
	}
}

func TestLoopDetector_ExactLoop(t *testing.T) {
	ld := NewLoopDetector()
	args := `{"origin": "HKT", "dest": "FRA"}`

	ld.record("browser", args, "result1")
	ld.record("browser", args, "result2")

	isLoop, _ := ld.detectLoop("browser", args)
	if isLoop {
		t.Fatal("should not detect loop after 2 calls")
	}

	ld.record("browser", args, "result3")

	isLoop, hint := ld.detectLoop("browser", args)
	if !isLoop {
		t.Fatal("expected loop detection after 3 identical calls")
	}
	if !strings.Contains(hint, "LOOP DETECTED") {
		t.Fatalf("hint should contain LOOP DETECTED: %s", hint)
	}
}

func TestLoopDetector_DifferentTools(t *testing.T) {
	ld := NewLoopDetector()
	args := `{"x": "y"}`

	ld.record("tool1", args, "r1")
	ld.record("tool2", args, "r2")
	ld.record("tool1", args, "r3")

	isLoop, _ := ld.detectLoop("tool1", args)
	if isLoop {
		t.Fatal("should not detect loop - calls distributed across different tools")
	}
}

func TestLoopDetector_SimilarArgs(t *testing.T) {
	ld := NewLoopDetector()

	ld.record("browser", `{"query": "flights London Paris"}`, "r1")
	ld.record("browser", `{"query": "flights London Paris"}`, "r2")
	ld.record("browser", `{"query": "flights London Paris"}`, "r3")

	isLoop, hint := ld.detectLoop("browser", `{"query": "flights London Paris"}`)
	if !isLoop {
		t.Fatal("expected loop detection for similar args")
	}
	if !strings.Contains(hint, "LOOP DETECTED") {
		t.Fatalf("hint should mention loop: %s", hint)
	}
}

func TestLoopDetector_FailingTool(t *testing.T) {
	ld := NewLoopDetector()
	args := `{"url": "http://fail.com"}`

	ld.record("browser", args, "Error: timeout")
	ld.record("browser", args, "Error: timeout")
	ld.record("browser", args, "Error: timeout")

	isLoop, hint := ld.detectLoop("browser", args)
	if !isLoop {
		t.Fatal("expected loop detection")
	}
	if !strings.Contains(hint, "different strategy") {
		t.Fatalf("hint should suggest different strategy: %s", hint)
	}
}

func TestSimilarity(t *testing.T) {
	tests := []struct {
		s1, s2   string
		minScore float64
		maxScore float64
	}{
		{"a b c", "a b c", 1.0, 1.0},
		{"a b c", "a b d", 0.5, 0.8},
		{"hello world", "hello", 0.5, 0.8},
		{"", "", 1.0, 1.0},
		{"a", "", 0.0, 0.0},
		{"", "a", 0.0, 0.0},
	}

	for _, tt := range tests {
		result := similarity(tt.s1, tt.s2)
		if result < tt.minScore || result > tt.maxScore+0.01 {
			t.Errorf("similarity(%q, %q) = %f, expected between %f and %f", tt.s1, tt.s2, result, tt.minScore, tt.maxScore)
		}
	}
}

func TestLoopDetector_MaxHistory(t *testing.T) {
	ld := &LoopDetector{
		maxHistory:    5,
		loopThreshold: 3,
	}

	for i := range 10 {
		ld.record("tool", fmt.Sprintf(`{"i": %d}`, i), "result")
	}

	ld.mu.Lock()
	historyLen := len(ld.recentToolCalls)
	ld.mu.Unlock()

	if historyLen > 5 {
		t.Fatalf("history should be limited to maxHistory, got %d", historyLen)
	}
}

func TestToolTimeout_WithContextInterface(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			args, _ := json.Marshal(map[string]string{"x": "y"})
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{
						Role: "assistant",
						ToolCalls: []openai.ToolCall{
							{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "context_tool", Arguments: string(args)}},
						},
					}},
				},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.SetToolTimeout(5 * time.Second)
	agent.RegisterTool(&MockToolWithContext{
		name: "context_tool",
		executeFn: func(ctx context.Context, args string) (string, error) {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(100 * time.Millisecond):
				return "result", nil
			}
		},
	})
	agent.reserveTokens = 100000

	reply, err := agent.Chat(context.Background(), "test")

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if reply != "done" {
		t.Errorf("expected 'done', got %q", reply)
	}
}

type MockToolWithContext struct {
	name      string
	executeFn func(ctx context.Context, args string) (string, error)
}

func (m *MockToolWithContext) Name() string { return m.name }
func (m *MockToolWithContext) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        m.name,
			Description: "mock tool with context",
			Parameters:  map[string]any{"type": "object"},
		},
	}
}
func (m *MockToolWithContext) ExecuteWithContext(ctx context.Context, args string) (string, error) {
	return m.executeFn(ctx, args)
}
func (m *MockToolWithContext) Execute(args string) (string, error) {
	return m.ExecuteWithContext(context.Background(), args)
}

func TestLoopDetector_EmptyHistory(t *testing.T) {
	ld := NewLoopDetector()

	isLoop, hint := ld.detectLoop("any_tool", `{"args": "value"}`)
	if isLoop {
		t.Errorf("unexpected loop with empty history: %s", hint)
	}
}

func TestLoopDetector_DifferentArgTypes(t *testing.T) {
	ld := NewLoopDetector()

	ld.record("tool", `{"num": 123}`, "r1")
	ld.record("tool", `{"num": 123}`, "r2")
	ld.record("tool", `{"num": 123}`, "r3")

	isLoop, _ := ld.detectLoop("tool", `{"num": 456}`)
	if isLoop {
		t.Error("should not detect loop with different numeric args")
	}
}

func TestAgent_NilWAL(t *testing.T) {
	agent := NewAgent(nil, nil)
	if agent.wal != nil {
		t.Error("expected nil WAL")
	}

	agent.appendHistory(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "test",
	})

	if len(agent.history) != 2 {
		t.Errorf("expected 2 messages (system + user), got %d", len(agent.history))
	}
}

func TestScheduler_ZeroInterval(t *testing.T) {
	agent := NewAgent(nil, nil)
	scheduler := NewScheduler(agent, context.Background())

	err := scheduler.Schedule("test", "prompt", 0)
	if err == nil {
		t.Error("expected error for zero interval")
	}
}

func TestScheduler_NegativeInterval(t *testing.T) {
	agent := NewAgent(nil, nil)
	scheduler := NewScheduler(agent, context.Background())

	err := scheduler.Schedule("test", "prompt", -1*time.Second)
	if err == nil {
		t.Error("expected error for negative interval")
	}
}

func TestScheduler_DuplicateID(t *testing.T) {
	agent := NewAgent(nil, nil)
	ctx := t.Context()
	scheduler := NewScheduler(agent, ctx)

	err := scheduler.Schedule("task1", "prompt", time.Hour)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = scheduler.Schedule("task1", "different prompt", time.Hour)
	if err == nil {
		t.Error("expected error for duplicate task ID")
	}
}

func TestScheduler_CancelNonExistent(t *testing.T) {
	agent := NewAgent(nil, nil)
	scheduler := NewScheduler(agent, context.Background())

	err := scheduler.Cancel("nonexistent")
	if err == nil {
		t.Error("expected error for canceling non-existent task")
	}
}

func TestParseDuration_EdgeCases(t *testing.T) {
	tests := []struct {
		input    string
		expected time.Duration
		hasError bool
	}{
		{"1 hour", time.Hour, false},
		{"2 hours", 2 * time.Hour, false},
		{"30 minutes", 30 * time.Minute, false},
		{"1 minute", time.Minute, false},
		{"45 seconds", 45 * time.Second, false},
		{"1 second", time.Second, false},
		{"1 day", 24 * time.Hour, false},
		{"2 days", 48 * time.Hour, false},
		{"1h", time.Hour, false},
		{"30m", 30 * time.Minute, false},
		{"45s", 45 * time.Second, false},
		{"", 0, true},
		{"invalid", 0, true},
		{"1 parsec", 0, true},
	}

	for _, tt := range tests {
		result, err := parseDurationTestHelper(tt.input)
		if tt.hasError {
			if err == nil {
				t.Errorf("expected error for %q", tt.input)
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error for %q: %v", tt.input, err)
			}
			if result != tt.expected {
				t.Errorf("parseDuration(%q) = %v, expected %v", tt.input, result, tt.expected)
			}
		}
	}
}

func parseDurationTestHelper(input string) (time.Duration, error) {
	s := NewScheduler(nil, context.Background())
	return s.ParseDuration(input)
}
