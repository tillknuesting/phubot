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
// UTILITY FUNCTION TESTS
// ==========================================

func TestTruncate_ShortString(t *testing.T) {
	result := truncate("hello", 10)
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestTruncate_ExactLength(t *testing.T) {
	result := truncate("hello", 5)
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestTruncate_LongString(t *testing.T) {
	result := truncate("hello world", 5)
	if result != "hello..." {
		t.Errorf("expected 'hello...', got %q", result)
	}
}

func TestTruncate_EmptyString(t *testing.T) {
	result := truncate("", 5)
	if result != "" {
		t.Errorf("expected '', got %q", result)
	}
}

func TestTruncate_ZeroMaxLen(t *testing.T) {
	result := truncate("hello", 0)
	if result != "" {
		t.Errorf("expected '', got %q", result)
	}
}

func TestTruncate_OneCharOver(t *testing.T) {
	result := truncate("abcdef", 5)
	if result != "abcde..." {
		t.Errorf("expected 'abcde...', got %q", result)
	}
}

func TestCleanResponse_TrimsWhitespace(t *testing.T) {
	result := cleanResponse("  hello  ")
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestCleanResponse_StripsImEnd(t *testing.T) {
	result := cleanResponse("Hello there<|im_end|>")
	if result != "Hello there" {
		t.Errorf("expected 'Hello there', got %q", result)
	}
}

func TestCleanResponse_StripsEndSequence(t *testing.T) {
	result := cleanResponse("Response text</s>")
	if result != "Response text" {
		t.Errorf("expected 'Response text', got %q", result)
	}
}

func TestCleanResponse_StripsEotId(t *testing.T) {
	result := cleanResponse("Some answer<|eot_id|>")
	if result != "Some answer" {
		t.Errorf("expected 'Some answer', got %q", result)
	}
}

func TestCleanResponse_StripsMultipleSuffixes(t *testing.T) {
	result := cleanResponse("  Answer<|im_end|>  ")
	if result != "Answer" {
		t.Errorf("expected 'Answer', got %q", result)
	}
}

func TestCleanResponse_EmptyString(t *testing.T) {
	result := cleanResponse("")
	if result != "" {
		t.Errorf("expected '', got %q", result)
	}
}

func TestCleanResponse_OnlySuffix(t *testing.T) {
	result := cleanResponse("<|im_end|>")
	if result != "" {
		t.Errorf("expected '', got %q", result)
	}
}

func TestStripMarkdown_Bold(t *testing.T) {
	result := stripMarkdown("This is **bold** text")
	if result != "This is bold text" {
		t.Errorf("expected 'This is bold text', got %q", result)
	}
}

func TestStripMarkdown_Italic(t *testing.T) {
	result := stripMarkdown("This is *italic* text")
	if result != "This is italic text" {
		t.Errorf("expected 'This is italic text', got %q", result)
	}
}

func TestStripMarkdown_Code(t *testing.T) {
	result := stripMarkdown("Run `go test` now")
	if result != "Run go test now" {
		t.Errorf("expected 'Run go test now', got %q", result)
	}
}

func TestStripMarkdown_UnderscoreBold(t *testing.T) {
	result := stripMarkdown("This is __bold__ text")
	if result != "This is bold text" {
		t.Errorf("expected 'This is bold text', got %q", result)
	}
}

func TestStripMarkdown_UnderscoreItalic(t *testing.T) {
	result := stripMarkdown("This is _italic_ text")
	if result != "This is italic text" {
		t.Errorf("expected 'This is italic text', got %q", result)
	}
}

func TestStripMarkdown_MultipleFormats(t *testing.T) {
	result := stripMarkdown("**bold** and *italic* and `code`")
	if result != "bold and italic and code" {
		t.Errorf("expected 'bold and italic and code', got %q", result)
	}
}

func TestStripMarkdown_NoFormatting(t *testing.T) {
	input := "plain text no formatting"
	result := stripMarkdown(input)
	if result != input {
		t.Errorf("expected unchanged, got %q", result)
	}
}

func TestStripMarkdown_Empty(t *testing.T) {
	result := stripMarkdown("")
	if result != "" {
		t.Errorf("expected '', got %q", result)
	}
}

// ==========================================
// countTokens WITH TOOLCALLS TESTS
// ==========================================

func TestCountTokens_ToolCalls_Counted(t *testing.T) {
	msgsWithoutToolCalls := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
	}
	msgsWithToolCalls := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleAssistant,
			Content: "let me check",
			ToolCalls: []openai.ToolCall{
				{
					ID:   "call_1",
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      "browse_web",
						Arguments: `{"query":"flights London Tokyo"}`,
					},
				},
			},
		},
	}

	withoutTC := countTokens(msgsWithoutToolCalls)
	withTC := countTokens(msgsWithToolCalls)

	if withTC <= withoutTC {
		t.Errorf("tool calls should add tokens: without=%d, with=%d", withoutTC, withTC)
	}
}

func TestCountTokens_MultipleToolCalls(t *testing.T) {
	single := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleAssistant,
			Content: "checking",
			ToolCalls: []openai.ToolCall{
				{Function: openai.FunctionCall{Name: "tool_a", Arguments: `{"x":1}`}},
			},
		},
	}
	double := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleAssistant,
			Content: "checking",
			ToolCalls: []openai.ToolCall{
				{Function: openai.FunctionCall{Name: "tool_a", Arguments: `{"x":1}`}},
				{Function: openai.FunctionCall{Name: "tool_b", Arguments: `{"y":2}`}},
			},
		},
	}

	singleTokens := countTokens(single)
	doubleTokens := countTokens(double)

	if doubleTokens <= singleTokens {
		t.Errorf("more tool calls should mean more tokens: single=%d, double=%d", singleTokens, doubleTokens)
	}
}

func TestCountTokens_LargeToolArgs(t *testing.T) {
	largeArgs := makeLongString(5000)
	msgs := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleAssistant,
			Content: "",
			ToolCalls: []openai.ToolCall{
				{Function: openai.FunctionCall{Name: "tool", Arguments: largeArgs}},
			},
		},
	}

	tokens := countTokens(msgs)
	if tokens <= 10 {
		t.Errorf("large tool args should produce significant tokens: got %d", tokens)
	}
}

// ==========================================
// MEMORY ROTATION / CLEANUP TESTS
// ==========================================

func TestMemory_AppendWithRotation_SmallFile(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	memFile := filepath.Join(dir, "MEMORY.md")

	err := mem.appendWithRotation(memFile, "first entry\n")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, err := os.ReadFile(memFile)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if string(data) != "first entry\n" {
		t.Errorf("expected 'first entry\\n', got %q", string(data))
	}
}

func TestMemory_AppendWithRotation_AppendsToExisting(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	memFile := filepath.Join(dir, "MEMORY.md")

	mem.appendWithRotation(memFile, "first\n")
	mem.appendWithRotation(memFile, "second\n")

	data, err := os.ReadFile(memFile)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if !strings.Contains(string(data), "first") || !strings.Contains(string(data), "second") {
		t.Errorf("expected both entries, got %q", string(data))
	}
}

func TestMemory_AppendWithRotation_TriggersRotation(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	mem.maxSize = 50
	memFile := filepath.Join(dir, "MEMORY.md")

	os.WriteFile(memFile, []byte(makeLongString(100)), 0644)

	err := mem.appendWithRotation(memFile, "new entry\n")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	matches, _ := filepath.Glob(filepath.Join(dir, "archive", "MEMORY-*.md"))
	if len(matches) == 0 {
		t.Error("expected archive file to be created when exceeding maxSize")
	}
}

func TestMemory_RotateMemory(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	memFile := filepath.Join(dir, "MEMORY.md")

	content := "original memory content that should be archived"
	os.WriteFile(memFile, []byte(content), 0644)

	err := mem.rotateMemory(memFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	matches, _ := filepath.Glob(filepath.Join(dir, "archive", "MEMORY-*.md"))
	if len(matches) == 0 {
		t.Fatal("expected archive file")
	}

	archiveData, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatalf("failed to read archive: %v", err)
	}
	if string(archiveData) != content {
		t.Errorf("archive content mismatch: got %q", string(archiveData))
	}

	mainData, err := os.ReadFile(memFile)
	if err != nil {
		t.Fatalf("failed to read main file: %v", err)
	}
	if strings.Contains(string(mainData), content) {
		t.Error("main file should have been truncated after rotation")
	}
	if !strings.Contains(string(mainData), "rotated") {
		t.Errorf("main file should contain rotation header, got: %q", string(mainData))
	}
}

func TestMemory_RotateMemory_CreatesSummaryFile(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	memFile := filepath.Join(dir, "MEMORY.md")

	os.WriteFile(memFile, []byte("some content"), 0644)

	err := mem.rotateMemory(memFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	summaryFile := memFile + ".summary"
	if _, err := os.Stat(summaryFile); os.IsNotExist(err) {
		t.Error("expected summary file to be created")
	}
}

func TestMemory_RotateMemory_NonExistentFile(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	memFile := filepath.Join(dir, "MEMORY.md")

	err := mem.rotateMemory(memFile)
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestMemory_CleanupOldArchives_RemovesOld(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	archiveDir := filepath.Join(dir, "archive")
	os.MkdirAll(archiveDir, 0755)

	oldFile := filepath.Join(archiveDir, "MEMORY-2020-01-01-000000.md")
	os.WriteFile(oldFile, []byte("old archive"), 0644)

	oldTime := time.Now().Add(-48 * time.Hour)
	os.Chtimes(oldFile, oldTime, oldTime)

	recentFile := filepath.Join(archiveDir, "MEMORY-2026-03-28-120000.md")
	os.WriteFile(recentFile, []byte("recent archive"), 0644)

	err := mem.CleanupOldArchives(24 * time.Hour)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := os.Stat(oldFile); !os.IsNotExist(err) {
		t.Error("old archive should have been deleted")
	}
	if _, err := os.Stat(recentFile); os.IsNotExist(err) {
		t.Error("recent archive should still exist")
	}
}

func TestMemory_CleanupOldArchives_NoArchiveDir(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)

	err := mem.CleanupOldArchives(24 * time.Hour)
	if err != nil {
		t.Fatalf("should not error when archive dir doesn't exist: %v", err)
	}
}

func TestMemory_CleanupOldArchives_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	archiveDir := filepath.Join(dir, "archive")
	os.MkdirAll(archiveDir, 0755)

	err := mem.CleanupOldArchives(24 * time.Hour)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMemory_CleanupOldArchives_SkipsDirectories(t *testing.T) {
	dir := t.TempDir()
	mem := NewMemoryWithoutRateLimit(dir)
	archiveDir := filepath.Join(dir, "archive")
	os.MkdirAll(archiveDir, 0755)

	subdir := filepath.Join(archiveDir, "subdir")
	os.MkdirAll(subdir, 0755)

	err := mem.CleanupOldArchives(0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := os.Stat(subdir); os.IsNotExist(err) {
		t.Error("subdirectories should not be deleted")
	}
}

// ==========================================
// SCHEDULER TESTS
// ==========================================

func newTestScheduler(t *testing.T, agent *Agent) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return NewScheduler(agent, ctx)
}

func TestScheduler_GetTask_Exists(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	s.Schedule("task1", "hello", time.Hour)

	task, found := s.GetTask("task1")
	if !found {
		t.Fatal("expected task to be found")
	}
	if task.ID != "task1" {
		t.Errorf("expected ID 'task1', got %q", task.ID)
	}
	if task.Prompt != "hello" {
		t.Errorf("expected prompt 'hello', got %q", task.Prompt)
	}
	if !task.Active {
		t.Error("expected task to be active")
	}
}

func TestScheduler_GetTask_NotExists(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	task, found := s.GetTask("nonexistent")
	if found {
		t.Error("expected task not to be found")
	}
	if task != nil {
		t.Error("expected nil task")
	}
}

func TestScheduler_GetTask_ReturnsCopy(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	s.Schedule("task1", "hello", time.Hour)

	task, _ := s.GetTask("task1")
	task.Active = false

	original, _ := s.GetTask("task1")
	if !original.Active {
		t.Error("GetTask should return a copy, modifying it should not affect internal state")
	}
}

func TestScheduler_ListTasks_Empty(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	tasks := s.ListTasks()
	if len(tasks) != 0 {
		t.Errorf("expected 0 tasks, got %d", len(tasks))
	}
}

func TestScheduler_ListTasks_Multiple(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	s.Schedule("task1", "prompt1", time.Hour)
	s.Schedule("task2", "prompt2", 2*time.Hour)

	tasks := s.ListTasks()
	if len(tasks) != 2 {
		t.Fatalf("expected 2 tasks, got %d", len(tasks))
	}

	ids := map[string]bool{}
	for _, task := range tasks {
		ids[task.ID] = true
	}
	if !ids["task1"] || !ids["task2"] {
		t.Error("expected both task1 and task2 in list")
	}
}

func TestScheduler_ListTasks_ReturnsCopies(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	s.Schedule("task1", "hello", time.Hour)
	tasks := s.ListTasks()
	tasks[0].Active = false

	original, _ := s.GetTask("task1")
	if !original.Active {
		t.Error("ListTasks should return copies")
	}
}

func TestScheduler_Cancel_SetsInactive(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	s.Schedule("task1", "hello", time.Hour)
	s.Cancel("task1")

	task, _ := s.GetTask("task1")
	if task.Active {
		t.Error("expected task to be inactive after cancel")
	}
}

func TestScheduler_Cancel_NonExistent(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)

	err := s.Cancel("nonexistent")
	if err == nil {
		t.Error("expected error for cancelling non-existent task")
	}
}

// ==========================================
// SchedulerTool TESTS
// ==========================================

func TestSchedulerTool_Name(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	if tool.Name() != "schedule_task" {
		t.Errorf("expected 'schedule_task', got %q", tool.Name())
	}
}

func TestSchedulerTool_Definition(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	def := tool.Definition()
	if def.Type != openai.ToolTypeFunction {
		t.Error("expected function tool type")
	}
	if def.Function == nil {
		t.Fatal("expected function definition")
	}
	if def.Function.Name != "schedule_task" {
		t.Errorf("expected function name 'schedule_task', got %q", def.Function.Name)
	}
}

func TestSchedulerTool_Execute_Schedule(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	args := `{"action":"schedule","task_id":"t1","prompt":"check prices","interval":"1 hour"}`
	result, err := tool.Execute(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "t1") {
		t.Errorf("expected task_id in result, got: %s", result)
	}

	task, found := s.GetTask("t1")
	if !found {
		t.Fatal("expected task to be scheduled")
	}
	if task.Prompt != "check prices" {
		t.Errorf("expected prompt 'check prices', got %q", task.Prompt)
	}
}

func TestSchedulerTool_Execute_ScheduleMissingFields(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	tests := []struct {
		name string
		args string
	}{
		{"missing task_id", `{"action":"schedule","prompt":"hello","interval":"1 hour"}`},
		{"missing prompt", `{"action":"schedule","task_id":"t1","interval":"1 hour"}`},
		{"missing interval", `{"action":"schedule","task_id":"t1","prompt":"hello"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tool.Execute(tt.args)
			if err == nil {
				t.Error("expected error for missing fields")
			}
		})
	}
}

func TestSchedulerTool_Execute_Cancel(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	s.Schedule("t1", "hello", time.Hour)

	args := `{"action":"cancel","task_id":"t1"}`
	result, err := tool.Execute(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "cancelled") {
		t.Errorf("expected cancel confirmation, got: %s", result)
	}

	task, _ := s.GetTask("t1")
	if task.Active {
		t.Error("expected task to be inactive")
	}
}

func TestSchedulerTool_Execute_CancelMissingID(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	_, err := tool.Execute(`{"action":"cancel"}`)
	if err == nil {
		t.Error("expected error for missing task_id")
	}
}

func TestSchedulerTool_Execute_List(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	s.Schedule("t1", "prompt1", time.Hour)

	args := `{"action":"list"}`
	result, err := tool.Execute(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "t1") {
		t.Errorf("expected task listing, got: %s", result)
	}
}

func TestSchedulerTool_Execute_ListEmpty(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	args := `{"action":"list"}`
	result, err := tool.Execute(args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "No scheduled tasks") {
		t.Errorf("expected 'No scheduled tasks', got: %s", result)
	}
}

func TestSchedulerTool_Execute_UnknownAction(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	_, err := tool.Execute(`{"action":"unknown"}`)
	if err == nil {
		t.Error("expected error for unknown action")
	}
	if !strings.Contains(err.Error(), "unknown action") {
		t.Errorf("expected 'unknown action' error, got: %v", err)
	}
}

func TestSchedulerTool_Execute_MalformedJSON(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	_, err := tool.Execute("not json")
	if err == nil {
		t.Error("expected error for malformed JSON")
	}
}

func TestSchedulerTool_Execute_InvalidInterval(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	s := newTestScheduler(t, agent)
	tool := &SchedulerTool{scheduler: s}

	args := `{"action":"schedule","task_id":"t1","prompt":"hello","interval":"bad interval"}`
	_, err := tool.Execute(args)
	if err == nil {
		t.Error("expected error for invalid interval")
	}
}

func TestSchedulerTool_ImplementsTool(t *testing.T) {
	var _ Tool = &SchedulerTool{}
}

// ==========================================
// ChatWithImage TOOL EXECUTION TESTS
// ==========================================

func TestChatWithImage_ToolCallExecuted(t *testing.T) {
	toolExecuted := false
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		args, _ := json.Marshal(map[string]string{"q": "test"})
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I searched for you.",
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_img_1",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "img_tool",
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

	agent := newDefaultTestAgent(client, nil)
	agent.RegisterTool(&MockTool{
		name: "img_tool",
		params: map[string]any{
			"q": map[string]any{"type": "string"},
		},
		required: []string{"q"},
		executeFn: func(args string) (string, error) {
			toolExecuted = true
			return "tool result from image", nil
		},
	})

	result, err := agent.ChatWithImage(context.Background(), "what do you see?", "dGVzdA==")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !toolExecuted {
		t.Error("expected tool to be executed when ChatWithImage returns tool calls")
	}

	foundAssistant := false
	foundToolResult := false
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleAssistant && len(m.ToolCalls) > 0 {
			foundAssistant = true
		}
		if m.Role == openai.ChatMessageRoleTool && m.Content == "tool result from image" {
			foundToolResult = true
		}
	}
	if !foundAssistant {
		t.Error("expected assistant message with tool calls in history")
	}
	if !foundToolResult {
		t.Error("expected tool result message in history")
	}

	if !strings.Contains(result, "searched") {
		t.Logf("result: %s", result)
	}
}

func TestChatWithImage_NonExistentTool(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "",
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_fake",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "nonexistent",
									Arguments: "{}",
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

	agent := newDefaultTestAgent(client, nil)

	_, err := agent.ChatWithImage(context.Background(), "test", "dGVzdA==")
	if err != nil {
		t.Fatalf("should not return error for non-existent tool: %v", err)
	}

	found := false
	for _, m := range agent.history {
		if m.Role == openai.ChatMessageRoleTool && strings.Contains(m.Content, "does not exist") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected tool error message in history for non-existent tool in ChatWithImage")
	}
}

// ==========================================
// PARSE ALLOWED USERS TESTS
// ==========================================

func TestParseAllowedUsers_Empty(t *testing.T) {
	result := parseAllowedUsers("")
	if result != nil {
		t.Errorf("expected nil for empty string, got %v", result)
	}
}

func TestParseAllowedUsers_SingleUser(t *testing.T) {
	result := parseAllowedUsers("123456")
	if len(result) != 1 || result[0] != 123456 {
		t.Errorf("expected [123456], got %v", result)
	}
}

func TestParseAllowedUsers_MultipleUsers(t *testing.T) {
	result := parseAllowedUsers("123,456,789")
	if len(result) != 3 {
		t.Fatalf("expected 3 IDs, got %d", len(result))
	}
	expected := []int64{123, 456, 789}
	for i, id := range expected {
		if result[i] != id {
			t.Errorf("index %d: expected %d, got %d", i, id, result[i])
		}
	}
}

func TestParseAllowedUsers_WithSpaces(t *testing.T) {
	result := parseAllowedUsers(" 123 , 456 , 789 ")
	if len(result) != 3 {
		t.Fatalf("expected 3 IDs, got %d", len(result))
	}
}

func TestParseAllowedUsers_InvalidEntry(t *testing.T) {
	result := parseAllowedUsers("123,abc,456")
	if len(result) != 2 {
		t.Errorf("expected 2 valid IDs (skipping 'abc'), got %d: %v", len(result), result)
	}
	if result[0] != 123 || result[1] != 456 {
		t.Errorf("expected [123, 456], got %v", result)
	}
}

func TestParseAllowedUsers_AllInvalid(t *testing.T) {
	result := parseAllowedUsers("abc,def,ghi")
	if len(result) != 0 {
		t.Errorf("expected 0 IDs, got %d: %v", len(result), result)
	}
}

func TestParseAllowedUsers_TrailingComma(t *testing.T) {
	result := parseAllowedUsers("123,")
	if len(result) != 1 || result[0] != 123 {
		t.Errorf("expected [123], got %v", result)
	}
}

func TestParseAllowedUsers_NegativeID(t *testing.T) {
	result := parseAllowedUsers("-123")
	if len(result) == 1 && result[0] == -123 {
		return
	}
	if len(result) == 0 {
		return
	}
	t.Errorf("expected 0 or [-123], got %v", result)
}

// ==========================================
// PRUNING BOUNDARY / EDGE CASE TESTS
// ==========================================

func TestPruneToolResults_RatioExactlyAtSoftTrim(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.50,
		HardClearRatio:       0.90,
		SoftTrimMaxChars:     10000,
		SoftTrimHeadChars:    1000,
		SoftTrimTailChars:    1000,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	toolContent := "a"
	var nonToolContent strings.Builder
	nonToolContent.WriteString("a")
	for range 10 {
		nonToolContent.WriteString(" bcdefghij")
	}

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: nonToolContent.String()},
		makeToolResultMessage(toolContent),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != toolContent {
		t.Error("at exact boundary, should not prune if ratio < threshold")
	}
}

func TestPruneToolResults_ToolResultWithEmptyContent(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "run tool"},
		makeToolResultMessage(""),
	}

	result := agent.pruneToolResults(history)
	if len(result) != 3 {
		t.Errorf("expected 3 messages, got %d", len(result))
	}
}

func TestPruneToolResults_ConservativeMode(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "conservative",
		SoftTrimRatio:        0.50,
		HardClearRatio:       0.80,
		SoftTrimMaxChars:     200,
		SoftTrimHeadChars:    50,
		SoftTrimTailChars:    50,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(makeLongString(5000)),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content == makeLongString(5000) {
		t.Error("conservative mode should still prune when ratio exceeds threshold")
	}
}

func TestPruneToolResults_ContentAtSoftTrimMaxChars(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.99,
		SoftTrimMaxChars:     100,
		SoftTrimHeadChars:    40,
		SoftTrimTailChars:    40,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(makeLongString(100)),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != makeLongString(100) {
		t.Error("content exactly at SoftTrimMaxChars should not be trimmed")
	}
}

func TestPruneToolResults_ContentOneOverSoftTrimMaxChars(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.99,
		SoftTrimMaxChars:     100,
		SoftTrimHeadChars:    40,
		SoftTrimTailChars:    40,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(makeLongString(101)),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content == makeLongString(101) {
		t.Error("content over SoftTrimMaxChars should be trimmed")
	}
}

func TestPruneToolResults_ShortContentNotTrimmed(t *testing.T) {
	cfg := PruningConfig{
		Mode:                 "aggressive",
		SoftTrimRatio:        0.05,
		HardClearRatio:       0.99,
		SoftTrimMaxChars:     200,
		SoftTrimHeadChars:    100,
		SoftTrimTailChars:    100,
		HardClearPlaceholder: "[cleared]",
	}
	agent := newTestAgentWithConfig(nil, nil, DefaultContextWindow, cfg)

	shortContent := "ab"
	history := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "s"},
		makeToolResultMessage(shortContent),
	}

	result := agent.pruneToolResults(history)
	if result[1].Content != shortContent {
		t.Error("content shorter than head+tail should not be modified in soft trim")
	}
}

// ==========================================
// COMPACTION EDGE CASE TESTS
// ==========================================

func TestCompactHistoryIfNeeded_Exactly3Messages(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)

	agent.history = []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 3 {
		t.Errorf("expected 3 messages (< 4 minimum), got %d", len(agent.history))
	}
}

func TestCompactHistoryIfNeeded_AllMessagesFit(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Summary."}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := newTestAgentWithConfig(client, nil, 100000, DefaultPruningConfig)

	for i := range 10 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("msg %d", i),
		})
	}

	err := agent.compactHistoryIfNeeded(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agent.history) != 11 {
		t.Errorf("expected 11 messages (no compaction), got %d", len(agent.history))
	}
}

// ==========================================
// ClearHistory EDGE CASES
// ==========================================

func TestClearHistory_NoSystemPrompt_CreatesFresh(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	agent.history = []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	err := agent.ClearHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(agent.history) != 1 {
		t.Fatalf("expected 1 message, got %d", len(agent.history))
	}
}

func TestClearHistory_MultipleSystemPrompts_KeepsFirst(t *testing.T) {
	agent := newDefaultTestAgent(nil, nil)
	agent.history = []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "first system"},
		{Role: openai.ChatMessageRoleUser, Content: "hello"},
		{Role: openai.ChatMessageRoleSystem, Content: "second system"},
		{Role: openai.ChatMessageRoleAssistant, Content: "hi"},
	}

	agent.ClearHistory()

	if len(agent.history) != 1 {
		t.Fatalf("expected 1 message, got %d", len(agent.history))
	}
	if agent.history[0].Content != "first system" {
		t.Errorf("expected first system prompt, got %q", agent.history[0].Content)
	}
}

// ==========================================
// LOOP DETECTOR EDGE CASES
// ==========================================

func TestLoopDetector_SimilarArgsLoop(t *testing.T) {
	ld := NewLoopDetector()

	ld.record("tool_a", `{"query":"flights from London to Tokyo"}`, "result1")
	ld.record("tool_a", `{"query":"flights from London to Tokyo"}`, "result2")
	ld.record("tool_a", `{"query":"flights from London to Tokyo"}`, "result3")

	isLoop, _ := ld.detectLoop("tool_a", `{"query":"flights from London to Tokyo"}`)
	if !isLoop {
		t.Error("expected exact args loop to be detected after 3 identical calls")
	}
}

func TestLoopDetector_DifferentToolsNoLoop(t *testing.T) {
	ld := NewLoopDetector()

	for range 5 {
		ld.record("tool_a", `{"x":1}`, "result")
		isLoop, _ := ld.detectLoop("tool_b", `{"x":1}`)
		if isLoop {
			t.Error("different tools should not trigger loop")
		}
	}
}

func TestLoopDetector_EmptyHistory_CM(t *testing.T) {
	ld := NewLoopDetector()
	isLoop, _ := ld.detectLoop("tool_a", `{"x":1}`)
	if isLoop {
		t.Error("empty history should not trigger loop")
	}
}

// ==========================================
// CHAT: UNREGISTERED TOOL NO LONGER RETURNS ERROR
// ==========================================

func TestChat_UnregisteredTool_AppendsErrorAndContinues(t *testing.T) {
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "final answer"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	})
	defer server.Close()

	agent := NewAgent(client, nil)

	result, err := agent.Chat(context.Background(), "just say hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "final answer" {
		t.Errorf("expected 'final answer', got %q", result)
	}
}

// ==========================================
// INTEGRATION: COMPACTION + PRUNING PIPELINE
// ==========================================

func TestIntegration_CompactionAndPruningPipeline(t *testing.T) {
	callCount := 0
	server, client := createMockLLMServer(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req openai.ChatCompletionRequest
		json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) > 0 {
			lastMsg := req.Messages[len(req.Messages)-1]
			if strings.Contains(lastMsg.Content, "[user]:") || strings.Contains(lastMsg.Content, "[User]:") {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Extracted facts."}},
					},
				}
				json.NewEncoder(w).Encode(resp)
				return
			}
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

	agent := newTestAgentWithConfig(client, wal, 300, DefaultPruningConfig)

	for i := range 20 {
		agent.history = append(agent.history, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("Message %d with enough content to use tokens", i),
		})
	}
	agent.history = append(agent.history, makeToolResultMessage(makeLongString(3000)))

	_, err := agent.Chat(context.Background(), "trigger")
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if agent.history[0].Role != openai.ChatMessageRoleSystem {
		t.Error("system prompt must be preserved")
	}

	stats := agent.GetHistoryStats()
	if stats.MessageCount == 0 {
		t.Error("expected messages in history")
	}
}
