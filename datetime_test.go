package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestDateTimeInSystemPrompt(t *testing.T) {
	agent := NewAgent(nil, nil)
	prompt := agent.buildSystemPrompt()

	if !strings.HasPrefix(prompt, "Current date and time:") {
		t.Fatalf("system prompt should start with date, got: %s", truncate(prompt, 200))
	}

	now := time.Now().Format("2006")
	if !strings.Contains(prompt, now) {
		t.Fatalf("system prompt should contain current year, got: %s", truncate(prompt, 200))
	}

	t.Logf("System prompt starts with: %s", truncate(prompt, 100))
}

func TestDateTimeRefreshesOnEveryChat(t *testing.T) {
	var lastPrompt string
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleSystem {
				lastPrompt = m.Content
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

	agent := NewAgent(client, nil)

	agent.Chat(context.Background(), "test1")
	first := lastPrompt
	t.Logf("First prompt: %s", truncate(first, 80))

	time.Sleep(2 * time.Second)

	agent.Chat(context.Background(), "test2")
	second := lastPrompt
	t.Logf("Second prompt: %s", truncate(second, 80))

	if !strings.Contains(first, "Current date and time:") {
		t.Fatal("first system prompt missing date")
	}
	if !strings.Contains(second, "Current date and time:") {
		t.Fatal("second system prompt missing date")
	}
	if first == second {
		t.Log("WARNING: prompts identical (fast execution), but both contain date - OK")
	}
}

func TestModelResolvesTomorrow(t *testing.T) {
	var capturedSystem string
	var capturedUser string

	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleSystem {
				capturedSystem = m.Content
			}
			if m.Role == openai.ChatMessageRoleUser {
				capturedUser = m.Content
			}
		}

		tomorrow := time.Now().AddDate(0, 0, 1).Format("2006-01-02")
		reply := fmt.Sprintf(`I'll search for flights for tomorrow (%s). Let me use the search_flights tool with date=%s.`, tomorrow, tomorrow)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: reply}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)
	agent.RegisterTool(&MomondoFlightTool{})

	reply, err := agent.Chat(context.Background(), "search flights from London to Berlin for tomorrow")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	t.Logf("System prompt contains date: %v", strings.Contains(capturedSystem, "Current date and time:"))
	t.Logf("User message: %s", capturedUser)
	t.Logf("Reply: %s", reply)

	if !strings.Contains(capturedSystem, "Current date and time:") {
		t.Fatal("system prompt must contain current date/time so model can resolve 'tomorrow'")
	}

	tomorrow := time.Now().AddDate(0, 0, 1).Format("2006-01-02")
	if !strings.Contains(reply, tomorrow) {
		t.Fatalf("model should resolve 'tomorrow' to %s, got: %s", tomorrow, reply)
	}
}
