package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/sashabaranov/go-openai"
)

// ==========================================
// FUZZ: WAL Roundtrip
// ==========================================

func FuzzWAL_Roundtrip(f *testing.F) {
	f.Fuzz(func(t *testing.T, role string, content string) {
		if !isValidUTF8(role) || !isValidUTF8(content) {
			return
		}

		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "fuzz.wal")}

		msg := openai.ChatCompletionMessage{Role: role, Content: content}

		if err := w.Append(msg); err != nil {
			return
		}

		loaded, err := w.LoadAll()
		if err != nil {
			t.Fatalf("load error: %v", err)
		}
		if len(loaded) != 1 {
			t.Fatalf("expected 1 message, got %d", len(loaded))
		}
		if loaded[0].Role != role {
			t.Fatalf("role mismatch: expected %q, got %q", role, loaded[0].Role)
		}
		if loaded[0].Content != content {
			t.Fatalf("content mismatch: expected %q, got %q", content, loaded[0].Content)
		}
	})
}

func isValidUTF8(s string) bool {
	for _, r := range s {
		if r == utf8.RuneError {
			return false
		}
	}
	return true
}

// ==========================================
// FUZZ: WAL LoadAll with raw bytes
// ==========================================

func FuzzWAL_LoadRawBytes(f *testing.F) {
	f.Fuzz(func(t *testing.T, data []byte) {
		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "fuzz.wal")}

		os.WriteFile(w.path, data, 0644)

		loaded, err := w.LoadAll()
		if err != nil {
			return
		}

		for _, msg := range loaded {
			if msg.Role == "" {
				t.Errorf("loaded message has empty role")
			}
		}
	})
}

// ==========================================
// FUZZ: WAL Multiple Appends Roundtrip
// ==========================================

func FuzzWAL_MultipleAppends(f *testing.F) {
	f.Add("user", "hello")
	f.Add("assistant", "world")

	f.Fuzz(func(t *testing.T, role string, content string) {
		if !isValidUTF8(role) || !isValidUTF8(content) {
			return
		}

		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "fuzz.wal")}

		w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: "sys"})
		w.Append(openai.ChatCompletionMessage{Role: role, Content: content})
		w.Append(openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: "end"})

		loaded, err := w.LoadAll()
		if err != nil {
			t.Fatalf("load error: %v", err)
		}
		if len(loaded) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(loaded))
		}
		if loaded[1].Role != role {
			t.Fatalf("middle role mismatch: expected %q, got %q", role, loaded[1].Role)
		}
		if loaded[1].Content != content {
			t.Fatalf("middle content mismatch: expected %q, got %q", content, loaded[1].Content)
		}
	})
}

// ==========================================
// FUZZ: WAL Concurrent Append + Load
// ==========================================

func FuzzWAL_ConcurrentAppendLoad(f *testing.F) {
	f.Fuzz(func(t *testing.T, content string) {
		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "fuzz.wal")}

		var wg sync.WaitGroup
		for i := range 20 {
			wg.Add(2)
			go func(idx int) {
				defer wg.Done()
				msg := openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("%s_%d", content, idx),
				}
				w.Append(msg)
			}(i)
			go func() {
				defer wg.Done()
				loaded, err := w.LoadAll()
				if err != nil {
					return
				}
				for _, m := range loaded {
					if m.Role == "" {
						t.Errorf("concurrent load returned message with empty role")
					}
				}
			}()
		}
		wg.Wait()
	})
}

// ==========================================
// FUZZ: Tool JSON Argument Parsing
// ==========================================

func FuzzBrowserTool_ArgParsing(f *testing.F) {
	f.Fuzz(func(t *testing.T, argsJSON string) {
		tool := &BrowserTool{}
		result, err := tool.Execute(argsJSON)
		if err != nil {
			if !strings.Contains(err.Error(), "failed to parse tool args") {
				t.Errorf("unexpected error type for bad args: %v", err)
			}
			return
		}
		if !strings.Contains(result, "BROWSER EXTRACTED") {
			t.Errorf("unexpected result for valid args: %s", result)
		}
	})
}

func FuzzMockTool_ArgParsing(f *testing.F) {
	f.Fuzz(func(t *testing.T, argsJSON string) {
		tool := &MockTool{
			name: "fuzz_tool",
			executeFn: func(args string) (string, error) {
				return fmt.Sprintf("received %d bytes", len(args)), nil
			},
		}
		result, err := tool.Execute(argsJSON)
		if err != nil {
			t.Errorf("mock tool should never error: %v", err)
			return
		}
		if result == "" {
			t.Error("mock tool returned empty result")
		}
	})
}

// ==========================================
// FUZZ: Chat with Fuzzed User Input
// ==========================================

func FuzzChat_UserInput(f *testing.F) {
	f.Fuzz(func(t *testing.T, userInput string) {
		if len(userInput) > 10000 || !isValidUTF8(userInput) {
			return
		}

		callCount := 0
		server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			var req openai.ChatCompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				return
			}

			lastMsg := req.Messages[len(req.Messages)-1]
			if lastMsg.Content != userInput {
				t.Errorf("user input not preserved: expected %q, got %q", userInput, lastMsg.Content)
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
		reply, err := agent.Chat(context.Background(), userInput)
		if err != nil {
			return
		}
		if reply != "ok" {
			t.Errorf("unexpected reply: %s", reply)
		}
	})
}

// ==========================================
// FUZZ: Chat with Fuzzed LLM Responses
// ==========================================

func FuzzChat_FuzzedLLMResponse(f *testing.F) {
	f.Fuzz(func(t *testing.T, content string) {
		if len(content) > 10000 || !isValidUTF8(content) {
			return
		}

		server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: content,
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		})
		defer server.Close()

		agent := NewAgent(client, nil)
		reply, err := agent.Chat(context.Background(), "test")
		if err != nil {
			return
		}
		if reply != content {
			t.Errorf("reply mismatch: expected %q, got %q", content, reply)
		}
	})
}

// ==========================================
// FUZZ: Chat with Fuzzed Tool Names
// ==========================================

func FuzzChat_FuzzedToolName(f *testing.F) {
	f.Fuzz(func(t *testing.T, toolName string) {
		if len(toolName) > 1000 {
			return
		}

		args, _ := json.Marshal(map[string]string{"x": "y"})
		server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_fuzz",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      toolName,
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
		agent.RegisterTool(&MockTool{name: "the_real_tool"})

		reply, err := agent.Chat(context.Background(), "trigger")
		if toolName != "the_real_tool" {
			if err == nil {
				t.Errorf("expected error for unknown tool %q, got reply: %s", toolName, reply)
			}
		} else {
			if err != nil {
				t.Errorf("unexpected error for registered tool: %v", err)
			}
		}
	})
}

// ==========================================
// FUZZ: Chat with Fuzzed Tool Args
// ==========================================

func FuzzChat_FuzzedToolArgs(f *testing.F) {
	f.Fuzz(func(t *testing.T, argsStr string) {
		if len(argsStr) > 10000 {
			return
		}

		callCount := 0
		server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			if callCount == 1 {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_fuzz_args",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "fuzz_args_tool",
											Arguments: argsStr,
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
								Content: "processed",
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
			name: "fuzz_args_tool",
			executeFn: func(args string) (string, error) {
				return fmt.Sprintf("got %d bytes", len(args)), nil
			},
		})

		_, err := agent.Chat(context.Background(), "test")
		if err != nil {
			return
		}
	})
}

// ==========================================
// FUZZ: Agent History with WAL + Chat
// ==========================================

func FuzzAgent_HistoryPersistence(f *testing.F) {
	f.Fuzz(func(t *testing.T, msg1 string, msg2 string) {
		if len(msg1) > 5000 || len(msg2) > 5000 || !isValidUTF8(msg1) || !isValidUTF8(msg2) {
			return
		}

		w, cleanup := newTestWAL(t)
		defer cleanup()

		callCount := 0
		server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
			callCount++
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: fmt.Sprintf("reply_%d", callCount),
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		})
		defer server.Close()

		agent := NewAgent(client, w)
		agent.Chat(context.Background(), msg1)

		loaded1, _ := w.LoadAll()
		found1 := false
		for _, m := range loaded1 {
			if m.Content == msg1 && m.Role == openai.ChatMessageRoleUser {
				found1 = true
			}
		}
		if !found1 {
			t.Error("first message not persisted")
		}

		agent.Chat(context.Background(), msg2)

		loaded2, _ := w.LoadAll()
		found2 := false
		for _, m := range loaded2 {
			if m.Content == msg2 && m.Role == openai.ChatMessageRoleUser {
				found2 = true
			}
		}
		if !found2 {
			t.Error("second message not persisted")
		}
	})
}

// ==========================================
// FUZZ: JSON Unmarshal for ChatCompletionMessage
// ==========================================

func FuzzChatCompletionMessage_JSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, data []byte) {
		var msg openai.ChatCompletionMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return
		}

		out, err := json.Marshal(msg)
		if err != nil {
			t.Errorf("failed to re-marshal valid message: %v", err)
			return
		}

		var msg2 openai.ChatCompletionMessage
		if err := json.Unmarshal(out, &msg2); err != nil {
			t.Errorf("failed to re-unmarshal: %v", err)
			return
		}
		if msg2.Role != msg.Role {
			t.Errorf("role changed after marshal/unmarshal: %q -> %q", msg.Role, msg2.Role)
		}
		if msg2.Content != msg.Content {
			t.Errorf("content changed after marshal/unmarshal: %q -> %q", msg.Content, msg2.Content)
		}
	})
}

// ==========================================
// FUZZ: Tool Definition Serialization
// ==========================================

func FuzzToolDefinition_JSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, toolName string, description string) {
		if len(toolName) > 500 || len(description) > 2000 || !isValidUTF8(toolName) || !isValidUTF8(description) {
			return
		}

		tool := &MockTool{name: toolName, description: description}
		def := tool.Definition()

		data, err := json.Marshal(def)
		if err != nil {
			t.Errorf("failed to marshal tool definition: %v", err)
			return
		}

		var def2 openai.Tool
		if err := json.Unmarshal(data, &def2); err != nil {
			t.Errorf("failed to unmarshal tool definition: %v", err)
			return
		}
		if def2.Function.Name != toolName {
			t.Errorf("name mismatch: expected %q, got %q", toolName, def2.Function.Name)
		}
	})
}

// ==========================================
// FUZZ: Token Counting
// ==========================================

func FuzzCountTokens(f *testing.F) {
	f.Fuzz(func(t *testing.T, content string) {
		if len(content) > 100000 {
			return
		}
		messages := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: content},
		}
		tokens := countTokens(messages)
		if tokens < 0 {
			t.Errorf("negative token count: %d", tokens)
		}
		if len(content) == 0 && tokens != 4 {
			t.Errorf("empty content should be 4 tokens (overhead), got %d", tokens)
		}
	})
}

func FuzzCountTokens_MultipleMessages(f *testing.F) {
	f.Add("hello", "world")
	f.Add("", "")
	f.Add("a", "b")

	f.Fuzz(func(t *testing.T, content1 string, content2 string) {
		if len(content1) > 50000 || len(content2) > 50000 {
			return
		}
		single := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: content1},
		}
		double := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: content1},
			{Role: openai.ChatMessageRoleAssistant, Content: content2},
		}
		singleTokens := countTokens(single)
		doubleTokens := countTokens(double)
		if doubleTokens < singleTokens {
			t.Errorf("double message tokens (%d) should be >= single (%d)", doubleTokens, singleTokens)
		}
	})
}

// ==========================================
// FUZZ: Find Recent Start
// ==========================================

func FuzzFindRecentStart(f *testing.F) {
	f.Fuzz(func(t *testing.T, budget int, numMessages int) {
		if budget < 0 || budget > 100000 || numMessages < 0 || numMessages > 100 {
			return
		}
		messages := []openai.ChatCompletionMessage{}
		for i := range numMessages {
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("message_%d with some content", i),
			})
		}
		idx := findRecentStart(messages, budget)
		if idx < 0 || idx > numMessages {
			t.Errorf("invalid index: %d for %d messages with budget %d", idx, numMessages, budget)
		}
		if numMessages == 0 && idx != 0 {
			t.Errorf("expected 0 for empty, got %d", idx)
		}
	})
}

// ==========================================
// FUZZ: Serialize Messages
// ==========================================

func FuzzSerializeMessages(f *testing.F) {
	f.Fuzz(func(t *testing.T, role string, content string) {
		if len(role) > 100 || len(content) > 50000 || !isValidUTF8(role) || !isValidUTF8(content) {
			return
		}
		messages := []openai.ChatCompletionMessage{
			{Role: role, Content: content},
		}
		result := serializeMessages(messages)
		if !strings.Contains(result, role) {
			t.Errorf("serialized output should contain role %q", role)
		}
		if !strings.Contains(result, content) {
			t.Errorf("serialized output should contain content %q", content)
		}
	})
}

// ==========================================
// FUZZ: WAL Rewrite
// ==========================================

func FuzzWAL_Rewrite(f *testing.F) {
	f.Fuzz(func(t *testing.T, content string) {
		if len(content) > 100000 || !isValidUTF8(content) {
			return
		}
		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "rewrite_fuzz.wal")}

		messages := []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: content},
		}

		if err := w.Rewrite(messages); err != nil {
			return
		}

		loaded, err := w.LoadAll()
		if err != nil {
			t.Fatalf("load error: %v", err)
		}
		if len(loaded) != 1 {
			t.Fatalf("expected 1 message, got %d", len(loaded))
		}
		if loaded[0].Content != content {
			t.Fatalf("content mismatch: expected %q, got %q", content, loaded[0].Content)
		}
	})
}

func FuzzWAL_RewriteMultiple(f *testing.F) {
	f.Fuzz(func(t *testing.T, numMessages int) {
		if numMessages < 0 || numMessages > 500 {
			return
		}
		dir := t.TempDir()
		w := &WAL{path: filepath.Join(dir, "multi_rewrite.wal")}

		messages := []openai.ChatCompletionMessage{}
		for i := range numMessages {
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("msg_%d", i),
			})
		}

		w.Rewrite(messages)
		loaded, _ := w.LoadAll()
		if len(loaded) != numMessages {
			t.Fatalf("expected %d messages, got %d", numMessages, len(loaded))
		}
	})
}

// ==========================================
// FUZZ: Compact History
// ==========================================

func FuzzCompactHistory_TokenCount(f *testing.F) {
	f.Fuzz(func(t *testing.T, numMessages int, tokensPerMessage int) {
		if numMessages < 0 || numMessages > 100 || tokensPerMessage < 0 || tokensPerMessage > 500 {
			return
		}
		messages := []openai.ChatCompletionMessage{}
		for range numMessages {
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleUser,
				Content: strings.Repeat("word ", tokensPerMessage),
			})
		}
		tokens := countTokens(messages)
		if tokens < 0 {
			t.Errorf("negative token count: %d", tokens)
		}
		if numMessages == 0 && tokens != 0 {
			t.Errorf("empty messages should have 0 tokens, got %d", tokens)
		}
	})
}

// ==========================================
// FUZZ: Memory Flush Content
// ==========================================

func FuzzMemory_Serialize(f *testing.F) {
	f.Fuzz(func(t *testing.T, role string, content string) {
		if len(role) > 50 || len(content) > 10000 || !isValidUTF8(role) || !isValidUTF8(content) {
			return
		}
		msg := openai.ChatCompletionMessage{Role: role, Content: content}
		serialized := serializeMessages([]openai.ChatCompletionMessage{msg})
		if !strings.Contains(serialized, "["+role+"]") {
			t.Errorf("serialized should contain [%s]", role)
		}
	})
}
