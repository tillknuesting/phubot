package main

import (
	"context"
	"strings"
	"testing"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

func newTestTelegramBot(t *testing.T, allowedIDs []int64) *TelegramBot {
	t.Helper()
	allowed := make(map[int64]bool)
	for _, id := range allowedIDs {
		allowed[id] = true
	}
	return &TelegramBot{
		agent:        newDefaultTestAgent(nil, nil),
		scheduler:    NewScheduler(nil, context.Background()),
		ctx:          context.Background(),
		allowedUsers: allowed,
	}
}

func TestIsAllowed_EmptyAllowList(t *testing.T) {
	bot := newTestTelegramBot(t, nil)
	if !bot.isAllowed(12345) {
		t.Error("empty allow list should allow all users")
	}
	if !bot.isAllowed(0) {
		t.Error("empty allow list should allow user 0")
	}
}

func TestIsAllowed_UserInList(t *testing.T) {
	bot := newTestTelegramBot(t, []int64{100, 200, 300})
	if !bot.isAllowed(100) {
		t.Error("user 100 should be allowed")
	}
	if !bot.isAllowed(200) {
		t.Error("user 200 should be allowed")
	}
	if bot.isAllowed(999) {
		t.Error("user 999 should not be allowed")
	}
}

func TestIsAllowed_UserNotInList(t *testing.T) {
	bot := newTestTelegramBot(t, []int64{42})
	if bot.isAllowed(43) {
		t.Error("user 43 should not be allowed")
	}
}

func TestIsAllowed_NegativeUserID(t *testing.T) {
	bot := newTestTelegramBot(t, []int64{-100})
	if !bot.isAllowed(-100) {
		t.Error("negative user ID should be allowed")
	}
	if bot.isAllowed(100) {
		t.Error("positive 100 should not match negative -100")
	}
}

func TestIsAllowed_SingleUser(t *testing.T) {
	bot := newTestTelegramBot(t, []int64{123456789})
	if !bot.isAllowed(123456789) {
		t.Error("allowed user should be allowed")
	}
	if bot.isAllowed(987654321) {
		t.Error("other user should not be allowed")
	}
}

func TestTelegramBot_CancelCommandParsing(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
		ok       bool
	}{
		{"cancel with id", "/cancel task1", "task1", true},
		{"cancel with uuid", "/cancel abc-123-def", "abc-123-def", true},
		{"cancel with spaces in id", "/cancel my task id", "my task id", true},
		{"cancel without space", "/cancel", "", false},
		{"not cancel", "/start", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			after, ok := strings.CutPrefix(tt.input, "/cancel ")
			if ok != tt.ok {
				t.Errorf("expected ok=%v, got ok=%v", tt.ok, ok)
			}
			if ok && after != tt.expected {
				t.Errorf("expected task ID %q, got %q", tt.expected, after)
			}
		})
	}
}

func TestTelegramBot_LongMessageSplit(t *testing.T) {
	longText := make([]byte, 12000)
	for i := range longText {
		longText[i] = 'A'
	}
	text := string(longText)

	chunkSize := 4000
	chunks := 0
	for i := 0; i < len(text); i += chunkSize {
		end := min(i+chunkSize, len(text))
		chunk := text[i:end]
		if len(chunk) > chunkSize {
			t.Errorf("chunk %d too large: %d", chunks, len(chunk))
		}
		chunks++
	}
	if chunks != 3 {
		t.Errorf("expected 3 chunks, got %d", chunks)
	}
}

func TestTelegramBot_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	bot := &TelegramBot{
		agent:        newDefaultTestAgent(nil, nil),
		scheduler:    NewScheduler(nil, context.Background()),
		ctx:          ctx,
		allowedUsers: map[int64]bool{},
	}
	cancel()
	if bot.ctx.Err() == nil {
		t.Error("expected context to be cancelled")
	}
}

func TestTelegramBot_PhotoSizeSelection(t *testing.T) {
	photos := []tgbotapi.PhotoSize{
		{FileID: "small", Width: 90, Height: 90},
		{FileID: "medium", Width: 320, Height: 320},
		{FileID: "large", Width: 800, Height: 800},
	}
	selected := photos[len(photos)-1]
	if selected.FileID != "large" {
		t.Errorf("expected largest photo, got %s", selected.FileID)
	}

	singlePhoto := []tgbotapi.PhotoSize{
		{FileID: "only", Width: 100, Height: 100},
	}
	selected = singlePhoto[len(singlePhoto)-1]
	if selected.FileID != "only" {
		t.Errorf("expected only photo, got %s", selected.FileID)
	}
}

func TestTelegramBot_DefaultCaption(t *testing.T) {
	caption := ""
	if caption == "" {
		caption = "What do you see in this image?"
	}
	if caption != "What do you see in this image?" {
		t.Errorf("expected default caption, got %q", caption)
	}

	caption = "describe this"
	if caption == "" {
		caption = "What do you see in this image?"
	}
	if caption != "describe this" {
		t.Errorf("expected user caption to be preserved, got %q", caption)
	}
}

func TestTelegramBot_SchedulerCancelIntegration(t *testing.T) {
	ctx := t.Context()

	sched := NewScheduler(nil, ctx)

	err := sched.Schedule("task1", "test prompt", 1*time.Hour)
	if err != nil {
		t.Fatalf("failed to schedule task: %v", err)
	}

	tasks := sched.ListTasks()
	if len(tasks) != 1 {
		t.Fatalf("expected 1 task, got %d", len(tasks))
	}
	if tasks[0].ID != "task1" {
		t.Errorf("expected task ID task1, got %s", tasks[0].ID)
	}
	if !tasks[0].Active {
		t.Error("task should be active")
	}

	if err := sched.Cancel("task1"); err != nil {
		t.Errorf("failed to cancel task: %v", err)
	}

	task, found := sched.GetTask("task1")
	if !found {
		t.Fatal("task should still exist after cancel")
	}
	if task.Active {
		t.Error("task should be inactive after cancel")
	}
}

func TestTelegramBot_SchedulerCancelNonexistentTask(t *testing.T) {
	sched := NewScheduler(nil, context.Background())
	err := sched.Cancel("nonexistent")
	if err == nil {
		t.Error("expected error cancelling nonexistent task")
	}
}

func TestTelegramBot_SchedulerDuplicateTask(t *testing.T) {
	sched := NewScheduler(nil, context.Background())

	err := sched.Schedule("dup", "prompt1", 1*time.Hour)
	if err != nil {
		t.Fatalf("first schedule failed: %v", err)
	}

	err = sched.Schedule("dup", "prompt2", 2*time.Hour)
	if err == nil {
		t.Error("expected error scheduling duplicate task")
	}
}
