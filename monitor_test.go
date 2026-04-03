package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestNewMonitor(t *testing.T) {
	m := NewMonitor()
	if m == nil {
		t.Fatal("NewMonitor returned nil")
	}
	if m.ch == nil {
		t.Fatal("channel is nil")
	}
	if m.tools == nil {
		t.Fatal("tools map is nil")
	}
	if len(m.events) != 0 {
		t.Fatalf("expected 0 events, got %d", len(m.events))
	}
}

func TestMonitorEmit_UserMsg(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "hello"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(m.events))
	}
	if m.events[0].Type != EventUserMsg {
		t.Errorf("expected EventUserMsg, got %d", m.events[0].Type)
	}
	if m.events[0].Message != "hello" {
		t.Errorf("expected 'hello', got %q", m.events[0].Message)
	}
	if m.stats.totalMessages != 1 {
		t.Errorf("expected 1 total message, got %d", m.stats.totalMessages)
	}
	if m.stats.agentPhase != "THINKING" {
		t.Errorf("expected THINKING phase, got %q", m.stats.agentPhase)
	}
}

func TestMonitorEmit_AssistantMsg(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventAssistantMsg, Message: "hi there"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalMessages != 1 {
		t.Errorf("expected 1 total message, got %d", m.stats.totalMessages)
	}
	if m.stats.agentPhase != "IDLE" {
		t.Errorf("expected IDLE phase, got %q", m.stats.agentPhase)
	}
}

func TestMonitorEmit_ToolCall(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "search_flights", Detail: "BOS->LAX"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalToolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", m.stats.totalToolCalls)
	}
	if m.stats.activeTool != "search_flights" {
		t.Errorf("expected active tool 'search_flights', got %q", m.stats.activeTool)
	}
	if m.stats.agentPhase != "EXECUTING" {
		t.Errorf("expected EXECUTING phase, got %q", m.stats.agentPhase)
	}
	ts, ok := m.tools["search_flights"]
	if !ok {
		t.Fatal("tool state not created")
	}
	if ts.status != "RUNNING" {
		t.Errorf("expected RUNNING, got %q", ts.status)
	}
	if ts.callCount != 1 {
		t.Errorf("expected call count 1, got %d", ts.callCount)
	}
}

func TestMonitorEmit_ToolCallAndResult(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "identify_aircraft"})
	m.Emit(MonitorEvent{
		Type:     EventToolResult,
		Tool:     "identify_aircraft",
		Message:  "Boeing 737-800",
		Duration: 3 * time.Second,
	})

	m.mu.RLock()
	defer m.mu.RUnlock()
	ts := m.tools["identify_aircraft"]
	if ts.status != "DONE" {
		t.Errorf("expected DONE, got %q", ts.status)
	}
	if ts.lastDur != 3*time.Second {
		t.Errorf("expected 3s duration, got %v", ts.lastDur)
	}
	if m.stats.activeTool != "" {
		t.Errorf("expected no active tool after result, got %q", m.stats.activeTool)
	}
	if m.stats.agentPhase != "OBSERVING" {
		t.Errorf("expected OBSERVING phase, got %q", m.stats.agentPhase)
	}
}

func TestMonitorEmit_Error(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventError, Message: "something broke"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalErrors != 1 {
		t.Errorf("expected 1 error, got %d", m.stats.totalErrors)
	}
	if len(m.stats.errors) != 1 || m.stats.errors[0] != "something broke" {
		t.Errorf("expected error stored, got %v", m.stats.errors)
	}
}

func TestMonitorEmit_Compaction(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventCompaction, Message: "50 -> 10 messages"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalCompactions != 1 {
		t.Errorf("expected 1 compaction, got %d", m.stats.totalCompactions)
	}
}

func TestMonitorEmit_EventBuffer(t *testing.T) {
	m := NewMonitor()
	for range 600 {
		m.Emit(MonitorEvent{Type: EventUserMsg, Message: "msg"})
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.events) != 500 {
		t.Errorf("expected 500 events (capped), got %d", len(m.events))
	}
}

func TestMonitorEmit_ChannelDelivery(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "test"})

	select {
	case evt := <-m.ch:
		if evt.Type != EventUserMsg {
			t.Errorf("expected EventUserMsg, got %d", evt.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("event not delivered to channel within 1s")
	}
}

func TestMonitorEmit_ChannelNonBlocking(t *testing.T) {
	m := NewMonitor()
	for range 300 {
		m.Emit(MonitorEvent{Type: EventUserMsg, Message: "flood"})
	}
}

func TestMonitorChan(t *testing.T) {
	m := NewMonitor()
	ch := m.Chan()
	if ch == nil {
		t.Fatal("Chan() returned nil")
	}
	m.Emit(MonitorEvent{Type: EventSystem, Message: "test"})
	select {
	case evt := <-m.ch:
		if evt.Type != EventSystem {
			t.Errorf("expected EventSystem, got %d", evt.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("event not received via internal channel")
	}
}

func TestMonitorStatsMultipleToolCalls(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "search_flights"})
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "search_flights", Duration: time.Second})
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "identify_aircraft"})
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "identify_aircraft", Duration: 2 * time.Second})
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "search_flights"})
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "search_flights", Duration: 1500 * time.Millisecond})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalToolCalls != 3 {
		t.Errorf("expected 3 total tool calls, got %d", m.stats.totalToolCalls)
	}
	sf := m.tools["search_flights"]
	if sf.callCount != 2 {
		t.Errorf("expected search_flights call count 2, got %d", sf.callCount)
	}
	ia := m.tools["identify_aircraft"]
	if ia.callCount != 1 {
		t.Errorf("expected identify_aircraft call count 1, got %d", ia.callCount)
	}
}

func TestMonitorStatsCombined(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "hi"})
	m.Emit(MonitorEvent{Type: EventAssistantMsg, Message: "hello"})
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "t1"})
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "t1"})
	m.Emit(MonitorEvent{Type: EventError, Message: "oops"})
	m.Emit(MonitorEvent{Type: EventCompaction, Message: "compacted"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.totalMessages != 2 {
		t.Errorf("expected 2 messages, got %d", m.stats.totalMessages)
	}
	if m.stats.totalToolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", m.stats.totalToolCalls)
	}
	if m.stats.totalErrors != 1 {
		t.Errorf("expected 1 error, got %d", m.stats.totalErrors)
	}
	if m.stats.totalCompactions != 1 {
		t.Errorf("expected 1 compaction, got %d", m.stats.totalCompactions)
	}
}

func TestMonitorTokenTracking(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "hello world"})
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "t1", Detail: "arg1"})
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "t1", Message: "result data"})

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stats.promptTokens <= 0 {
		t.Errorf("expected prompt tokens > 0, got %d", m.stats.promptTokens)
	}
	if m.stats.completionTokens <= 0 {
		t.Errorf("expected completion tokens > 0, got %d", m.stats.completionTokens)
	}
	if m.stats.estimatedCost <= 0 {
		t.Errorf("expected cost > 0, got %f", m.stats.estimatedCost)
	}
}

func TestMonitorErrorBuffer(t *testing.T) {
	m := NewMonitor()
	for i := range 25 {
		m.Emit(MonitorEvent{Type: EventError, Message: fmt.Sprintf("err%d", i)})
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.stats.errors) != 20 {
		t.Errorf("expected 20 errors (capped), got %d", len(m.stats.errors))
	}
}

func TestTruncStr(t *testing.T) {
	tests := []struct {
		input string
		max   int
		want  string
	}{
		{"hello", 10, "hello"},
		{"hello world", 8, "hello..."},
		{"hi\nthere", 10, "hi there"},
		{"  spaces  ", 10, "spaces"},
		{"", 5, ""},
	}
	for _, tt := range tests {
		got := truncStr(tt.input, tt.max)
		if got != tt.want {
			t.Errorf("truncStr(%q, %d) = %q, want %q", tt.input, tt.max, got, tt.want)
		}
	}
}

func TestSmartTruncate(t *testing.T) {
	result := smartTruncate("line1\nline2\nline3\nline4\nline5", 3, 200)
	if strings.Count(result, "\n") > 3 {
		t.Errorf("expected at most 3 lines, got: %q", result)
	}
	if !strings.Contains(result, "...") {
		t.Errorf("expected truncation marker, got: %q", result)
	}

	short := smartTruncate("hello", 3, 200)
	if short != "hello" {
		t.Errorf("expected 'hello', got %q", short)
	}
}

func TestToolIcon(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"search_flights", "[PLANE]"},
		{"identify_aircraft", "[ACFT]"},
		{"browse_web", "[WEB]"},
		{"scheduler", "[CLK]"},
		{"unknown", "[?]"},
	}
	for _, tt := range tests {
		got := toolIcon(tt.name)
		if got != tt.want {
			t.Errorf("toolIcon(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestBuildCoT_Empty(t *testing.T) {
	m := NewMonitor()
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "Awaiting transmissions") {
		t.Errorf("expected placeholder text, got: %q", content)
	}
}

func TestBuildCoT_UserMsg(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "find flights"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "USER: find flights") {
		t.Errorf("expected user message in log, got: %q", content)
	}
}

func TestBuildCoT_AssistantMsg(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventAssistantMsg, Message: "here are flights"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "BOT: here are flights") {
		t.Errorf("expected assistant message in log, got: %q", content)
	}
}

func TestBuildCoT_ToolCallWithDetail(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "search_flights", Detail: "BOS->LAX"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "CALL: search_flights") {
		t.Errorf("expected tool call in log, got: %q", content)
	}
	if !strings.Contains(content, "BOS->LAX") {
		t.Errorf("expected detail in log, got: %q", content)
	}
}

func TestBuildCoT_ToolResultWithDuration(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "search_flights", Message: "5 results", Duration: 2 * time.Second})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "DONE: search_flights") {
		t.Errorf("expected tool result in log, got: %q", content)
	}
	if !strings.Contains(content, "5 results") {
		t.Errorf("expected result message in log, got: %q", content)
	}
}

func TestBuildCoT_Error(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventError, Message: "timeout"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "ERROR: timeout") {
		t.Errorf("expected error in log, got: %q", content)
	}
}

func TestBuildCoT_Compaction(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventCompaction, Message: "50 -> 10"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "COMPACT: 50 -> 10") {
		t.Errorf("expected compaction in log, got: %q", content)
	}
}

func TestBuildCoT_Progress(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventProgress, Message: "loading page"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "loading page") {
		t.Errorf("expected progress message in log, got: %q", content)
	}
}

func TestBuildCoT_LLMRequestResponse(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventLLMRequest})
	m.Emit(MonitorEvent{Type: EventLLMResponse})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "LLM REQUEST") {
		t.Errorf("expected LLM REQUEST in log, got: %q", content)
	}
	if !strings.Contains(content, "LLM RESPONSE") {
		t.Errorf("expected LLM RESPONSE in log, got: %q", content)
	}
}

func TestBuildCoT_System(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventSystem, Message: "started"})
	model := newMonitorModel(m, false)
	model.monitor.stats.uptime = time.Now()
	content := model.buildCoT()
	if !strings.Contains(content, "SYS: started") {
		t.Errorf("expected system message in log, got: %q", content)
	}
}

func TestMonitorTimestamp(t *testing.T) {
	m := NewMonitor()
	before := time.Now()
	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "test"})
	after := time.Now()

	m.mu.RLock()
	ts := m.events[0].Timestamp
	m.mu.RUnlock()
	if ts.Before(before) || ts.After(after) {
		t.Errorf("timestamp %v not between %v and %v", ts, before, after)
	}
}

func TestMonitorToolStateRunning(t *testing.T) {
	m := NewMonitor()
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "search_flights"})

	m.mu.RLock()
	ts := m.tools["search_flights"]
	m.mu.RUnlock()
	if ts == nil {
		t.Fatal("tool state is nil")
	}
	if ts.status != "RUNNING" {
		t.Errorf("expected RUNNING, got %q", ts.status)
	}
	if ts.lastCall.IsZero() {
		t.Error("lastCall should not be zero")
	}
}

func TestMonitorPhaseTransitions(t *testing.T) {
	m := NewMonitor()

	m.Emit(MonitorEvent{Type: EventUserMsg, Message: "hi"})
	m.mu.RLock()
	if m.stats.agentPhase != "THINKING" {
		t.Errorf("after UserMsg: expected THINKING, got %q", m.stats.agentPhase)
	}
	m.mu.RUnlock()

	m.Emit(MonitorEvent{Type: EventLLMRequest})
	m.mu.RLock()
	if m.stats.agentPhase != "THINKING" {
		t.Errorf("after LLMRequest: expected THINKING, got %q", m.stats.agentPhase)
	}
	m.mu.RUnlock()

	m.Emit(MonitorEvent{Type: EventLLMResponse})
	m.Emit(MonitorEvent{Type: EventToolCall, Tool: "t1"})
	m.mu.RLock()
	if m.stats.agentPhase != "EXECUTING" {
		t.Errorf("after ToolCall: expected EXECUTING, got %q", m.stats.agentPhase)
	}
	m.mu.RUnlock()

	m.Emit(MonitorEvent{Type: EventToolResult, Tool: "t1"})
	m.mu.RLock()
	if m.stats.agentPhase != "OBSERVING" {
		t.Errorf("after ToolResult: expected OBSERVING, got %q", m.stats.agentPhase)
	}
	m.mu.RUnlock()

	m.Emit(MonitorEvent{Type: EventAssistantMsg, Message: "done"})
	m.mu.RLock()
	if m.stats.agentPhase != "IDLE" {
		t.Errorf("after AssistantMsg: expected IDLE, got %q", m.stats.agentPhase)
	}
	m.mu.RUnlock()
}
