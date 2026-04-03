package main

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type EventType int

const (
	EventUserMsg EventType = iota
	EventAssistantMsg
	EventToolCall
	EventToolResult
	EventProgress
	EventError
	EventCompaction
	EventLLMRequest
	EventLLMResponse
	EventSystem
)

type MonitorEvent struct {
	Type      EventType
	Timestamp time.Time
	Tool      string
	Message   string
	Detail    string
	Duration  time.Duration
}

type Monitor struct {
	mu     sync.RWMutex
	ch     chan MonitorEvent
	events []MonitorEvent
	stats  monitorStats
	tools  map[string]*toolState
}

type monitorStats struct {
	totalMessages    int
	totalToolCalls   int
	totalErrors      int
	totalCompactions int
	promptTokens     int
	completionTokens int
	estimatedCost    float64
	uptime           time.Time
	activeTool       string
	activeToolStart  time.Time
	agentPhase       string
	recentFiles      []string
	errors           []string
}

type toolState struct {
	name      string
	callCount int
	lastCall  time.Time
	lastDur   time.Duration
	status    string
}

func NewMonitor() *Monitor {
	return &Monitor{
		ch:     make(chan MonitorEvent, 256),
		events: make([]MonitorEvent, 0, 500),
		tools:  make(map[string]*toolState),
	}
}

func (m *Monitor) Chan() chan<- MonitorEvent { return m.ch }

func countTokenStr(s string) int {
	tke := getTiktokenEncoding()
	if tke == nil {
		return len(s)/4 + 4
	}
	return len(tke.Encode(s, nil, nil)) + 4
}

func (m *Monitor) Emit(evt MonitorEvent) {
	evt.Timestamp = time.Now()

	m.mu.Lock()
	m.events = append(m.events, evt)
	if len(m.events) > 500 {
		m.events = m.events[len(m.events)-500:]
	}

	switch evt.Type {
	case EventToolCall:
		m.stats.totalToolCalls++
		m.stats.activeTool = evt.Tool
		m.stats.activeToolStart = evt.Timestamp
		m.stats.agentPhase = "EXECUTING"
		m.stats.completionTokens += countTokenStr(evt.Detail)
		ts, ok := m.tools[evt.Tool]
		if !ok {
			ts = &toolState{name: evt.Tool}
			m.tools[evt.Tool] = ts
		}
		ts.callCount++
		ts.lastCall = evt.Timestamp
		ts.status = "RUNNING"
	case EventToolResult:
		if ts, ok := m.tools[evt.Tool]; ok {
			ts.status = "DONE"
			ts.lastDur = evt.Duration
		}
		m.stats.activeTool = ""
		m.stats.agentPhase = "OBSERVING"
		m.stats.completionTokens += countTokenStr(evt.Message)
	case EventError:
		m.stats.totalErrors++
		m.stats.errors = append(m.stats.errors, evt.Message)
		if len(m.stats.errors) > 20 {
			m.stats.errors = m.stats.errors[len(m.stats.errors)-20:]
		}
	case EventCompaction:
		m.stats.totalCompactions++
	case EventUserMsg:
		m.stats.totalMessages++
		m.stats.promptTokens += countTokenStr(evt.Message)
		m.stats.agentPhase = "THINKING"
	case EventAssistantMsg:
		m.stats.totalMessages++
		m.stats.completionTokens += countTokenStr(evt.Message)
		m.stats.agentPhase = "IDLE"
	case EventLLMRequest:
		m.stats.agentPhase = "THINKING"
	case EventLLMResponse:
		m.stats.agentPhase = "THINKING"
	case EventProgress:
		m.stats.completionTokens += countTokenStr(evt.Message)
	}

	m.stats.estimatedCost = float64(m.stats.promptTokens)*0.000003 + float64(m.stats.completionTokens)*0.000015
	m.mu.Unlock()

	select {
	case m.ch <- evt:
	default:
	}
}

// ──────────────────────────────────────────────
// CYBERPUNK STYLES
// ──────────────────────────────────────────────

var (
	neonGreen    = lipgloss.Color("#00FF41")
	electricBlue = lipgloss.Color("#00D4FF")
	deepPurple   = lipgloss.Color("#BD93F9")
	starkRed     = lipgloss.Color("#FF5555")
	warmYellow   = lipgloss.Color("#F1FA8C")
	dimGray      = lipgloss.Color("#44475A")
	darkBg       = lipgloss.Color("#21222C")
	paneBg       = lipgloss.Color("#1E1F2B")
	cyanAccent   = lipgloss.Color("#8BE9FD")
	orangeAccent = lipgloss.Color("#FFB86C")

	borderStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(neonGreen).
			Padding(0, 1)

	headerTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(neonGreen).
				Background(darkBg).
				Padding(0, 2)

	headerStatStyle = lipgloss.NewStyle().
			Foreground(electricBlue).
			Background(darkBg).
			Padding(0, 1)

	phaseStyles = map[string]lipgloss.Style{
		"THINKING":  lipgloss.NewStyle().Foreground(deepPurple).Bold(true),
		"EXECUTING": lipgloss.NewStyle().Foreground(neonGreen).Bold(true),
		"OBSERVING": lipgloss.NewStyle().Foreground(electricBlue).Bold(true),
		"IDLE":      lipgloss.NewStyle().Foreground(dimGray),
	}

	userMsgStyle  = lipgloss.NewStyle().Foreground(electricBlue)
	botMsgStyle   = lipgloss.NewStyle().Foreground(deepPurple).Bold(true)
	toolRunStyle  = lipgloss.NewStyle().Foreground(warmYellow)
	toolDoneStyle = lipgloss.NewStyle().Foreground(neonGreen)
	progressStyle = lipgloss.NewStyle().Foreground(orangeAccent)
	errorLogStyle = lipgloss.NewStyle().Foreground(starkRed).Bold(true)
	compactStyle  = lipgloss.NewStyle().Foreground(cyanAccent)
	llmStyle      = lipgloss.NewStyle().Foreground(dimGray)
	systemStyle   = lipgloss.NewStyle().Foreground(dimGray)
	dimText       = lipgloss.NewStyle().Foreground(dimGray)
	metricLabel   = lipgloss.NewStyle().Foreground(dimGray)
	metricValue   = lipgloss.NewStyle().Foreground(electricBlue).Bold(true)
	metricWarn    = lipgloss.NewStyle().Foreground(starkRed).Bold(true)
)

// ──────────────────────────────────────────────
// BUBBLETEA MODEL
// ──────────────────────────────────────────────

type monitorModel struct {
	monitor   *Monitor
	thoughtVP viewport.Model
	spinner   spinner.Model
	progress  progress.Model
	width     int
	height    int
	ready     bool
	lastCoT   string
	lastRight string
	demo      bool
	paused    bool
}

func newMonitorModel(m *Monitor, demo bool) monitorModel {
	s := spinner.New()
	s.Spinner = spinner.MiniDot
	s.Style = lipgloss.NewStyle().Foreground(neonGreen)

	p := progress.New(progress.WithGradient("#00FF41", "#00D4FF"))
	p.Width = 30

	return monitorModel{
		monitor:  m,
		spinner:  s,
		progress: p,
		demo:     demo,
	}
}

func (m monitorModel) Init() tea.Cmd {
	return tea.Batch(
		tea.EnterAltScreen,
		m.spinner.Tick,
		waitForEvent(m.monitor.ch),
		m.demoTick(),
	)
}

func waitForEvent(ch <-chan MonitorEvent) tea.Cmd {
	return func() tea.Msg {
		evt, ok := <-ch
		if !ok {
			return tea.Quit()
		}
		return evt
	}
}

type demoTickMsg time.Time

func (m monitorModel) demoTick() tea.Cmd {
	if !m.demo {
		return nil
	}
	return tea.Tick(800*time.Millisecond, func(t time.Time) tea.Msg {
		return demoTickMsg(t)
	})
}

func (m monitorModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "p":
			m.paused = !m.paused
			return m, nil
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		leftW := max(msg.Width/2-4, 20)
		cotH := max(msg.Height-8, 10)
		if !m.ready {
			m.thoughtVP = viewport.New(leftW, cotH)
			m.thoughtVP.Style = lipgloss.NewStyle()
			m.ready = true
		} else {
			m.thoughtVP.Width = leftW
			m.thoughtVP.Height = cotH
		}
		m.progress.Width = max(msg.Width/2-8, 20)

	case MonitorEvent:
		if !m.paused {
			m.rebuildView()
		}
		return m, tea.Batch(waitForEvent(m.monitor.ch), m.spinner.Tick)

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		if !m.paused {
			m.rebuildExecPane()
		}
		return m, cmd

	case demoTickMsg:
		if m.demo && !m.paused {
			fireDemoEvent(m.monitor)
		}
		return m, m.demoTick()

	case progress.FrameMsg:
		var cmd tea.Cmd
		mm, _ := m.progress.Update(msg)
		m.progress = mm.(progress.Model)
		return m, cmd
	}

	if m.ready {
		var cmd tea.Cmd
		m.thoughtVP, cmd = m.thoughtVP.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m *monitorModel) rebuildView() {
	m.monitor.mu.RLock()
	cot := m.buildCoT()
	m.monitor.mu.RUnlock()
	if cot != m.lastCoT {
		m.lastCoT = cot
		m.thoughtVP.SetContent(cot)
		m.thoughtVP.GotoBottom()
	}
}

func (m *monitorModel) rebuildExecPane() {
}

// ──────────────────────────────────────────────
// VIEW RENDERING
// ──────────────────────────────────────────────

func (m monitorModel) View() string {
	if !m.ready {
		return lipgloss.NewStyle().Foreground(neonGreen).Render("  Initializing PHUBOT AGI-CORE...")
	}

	m.monitor.mu.RLock()
	defer m.monitor.mu.RUnlock()

	header := m.renderHeader()
	left := m.renderLeftPane()
	right := m.renderRightPane()
	footer := m.renderFooter()

	leftStyled := borderStyle.
		Width(max(m.width/2-4, 20)).
		Height(max(m.height-8, 10)).
		BorderForeground(neonGreen).
		Render(left)

	rightStyled := borderStyle.
		Width(max(m.width/2-4, 20)).
		Height(max(m.height-8, 10)).
		BorderForeground(deepPurple).
		Render(right)

	body := lipgloss.JoinHorizontal(lipgloss.Top, leftStyled, rightStyled)
	return lipgloss.JoinVertical(lipgloss.Left, header, body, footer)
}

func (m monitorModel) renderHeader() string {
	w := m.width
	uptime := time.Since(m.monitor.stats.uptime).Round(time.Second)
	uptimeStr := fmt.Sprintf("%02d:%02d:%02d", int(uptime.Hours()), int(uptime.Minutes())%60, int(uptime.Seconds())%60)

	phase := m.monitor.stats.agentPhase
	if phase == "" {
		phase = "IDLE"
	}
	phaseIcon := phaseIcon(phase)
	phaseStr := phaseStyles[phase].Render(fmt.Sprintf(" %s %s ", phaseIcon, phase))

	title := headerTitleStyle.Render(" PHUBOT // AGI-CORE v1.0 ")
	up := headerStatStyle.Render(fmt.Sprintf(" UP %s ", uptimeStr))

	leftPart := lipgloss.JoinHorizontal(lipgloss.Center, title, phaseStr, up)

	cost := m.monitor.stats.estimatedCost
	statsLine := headerStatStyle.Render(fmt.Sprintf(
		" MSG:%d | TOOLS:%d | ERR:%d | COST:$%.3f ",
		m.monitor.stats.totalMessages,
		m.monitor.stats.totalToolCalls,
		m.monitor.stats.totalErrors,
		cost,
	))

	line1 := lipgloss.Place(w, 1, lipgloss.Left, lipgloss.Center, leftPart)
	line2 := lipgloss.Place(w, 1, lipgloss.Center, lipgloss.Center, statsLine)

	return lipgloss.JoinVertical(lipgloss.Left, line1, line2)
}

func phaseIcon(phase string) string {
	switch phase {
	case "THINKING":
		return "\u2699" // ⚙
	case "EXECUTING":
		return "\u26A1" // ⚡
	case "OBSERVING":
		return "\U0001F441" // 👁
	case "IDLE":
		return "\u25CB" // ○
	default:
		return "\u25CB"
	}
}

func (m monitorModel) renderLeftPane() string {
	return m.thoughtVP.View()
}

func (m monitorModel) renderRightPane() string {
	w := max(m.width/2-6, 18)
	execH := max(m.height/2-6, 5)
	metH := max(m.height/2-6, 5)

	exec := m.renderExecPane(w, execH)
	met := m.renderMetricsPane(w, metH)

	execBox := lipgloss.NewStyle().
		Width(w).
		Height(execH).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(warmYellow).
		Padding(0, 1).
		Render(exec)

	metBox := lipgloss.NewStyle().
		Width(w).
		Height(metH).
		Border(lipgloss.RoundedBorder()).
		BorderForeground(electricBlue).
		Padding(0, 1).
		Render(met)

	return lipgloss.JoinVertical(lipgloss.Left, execBox, metBox)
}

func (m monitorModel) renderExecPane(w, h int) string {
	var sb strings.Builder

	sb.WriteString(toolRunStyle.Render("EXECUTION"))
	sb.WriteString("\n")

	active := m.monitor.stats.activeTool
	if active != "" {
		elapsed := time.Since(m.monitor.stats.activeToolStart).Round(time.Millisecond)
		spin := m.spinner.View()
		sb.WriteString(fmt.Sprintf("%s %s", spin, toolRunStyle.Render(active)))
		sb.WriteString("\n")

		detail := ""
		m.monitor.mu.RLock()
		for i := len(m.monitor.events) - 1; i >= 0; i-- {
			if m.monitor.events[i].Type == EventToolCall && m.monitor.events[i].Tool == active {
				detail = m.monitor.events[i].Detail
				break
			}
		}
		m.monitor.mu.RUnlock()
		if detail != "" {
			sb.WriteString(dimText.Render(fmt.Sprintf("  %s", truncStr(detail, w-4))))
			sb.WriteString("\n")
		}

		sb.WriteString(fmt.Sprintf("  %s elapsed", elapsed.Round(time.Second)))
		sb.WriteString("\n")

		pct := min(elapsed.Seconds()/15.0, 0.95)
		m.progress.SetPercent(pct)
		sb.WriteString("  ")
		sb.WriteString(m.progress.View())
		sb.WriteString("\n")
	} else {
		lastTool := ""
		lastDur := time.Duration(0)
		for name, ts := range m.monitor.tools {
			if ts.lastCall.After(m.monitor.stats.activeToolStart) && ts.status == "DONE" {
				if lastTool == "" || m.monitor.tools[lastTool].lastCall.Before(ts.lastCall) {
					lastTool = name
					lastDur = ts.lastDur
				}
			}
		}
		if lastTool != "" {
			sb.WriteString(toolDoneStyle.Render(fmt.Sprintf("LAST: %s", lastTool)))
			sb.WriteString("\n")
			sb.WriteString(dimText.Render(fmt.Sprintf("  completed in %s", lastDur.Round(time.Millisecond))))
			sb.WriteString("\n")
		} else {
			sb.WriteString(dimText.Render("  awaiting commands..."))
			sb.WriteString("\n")
		}
		m.progress.SetPercent(0)
	}

	return sb.String()
}

func (m monitorModel) renderMetricsPane(w, h int) string {
	var sb strings.Builder

	sb.WriteString(metricLabel.Render("CONTEXT & METRICS"))
	sb.WriteString("\n")

	promptT := m.monitor.stats.promptTokens
	complT := m.monitor.stats.completionTokens
	totalT := promptT + complT
	ctxWindow := 128000
	pct := float64(totalT) / float64(ctxWindow) * 100

	tokenStyle := metricValue
	if pct > 80 {
		tokenStyle = metricWarn
	}

	sb.WriteString(fmt.Sprintf("  Tokens: %s / %s (%.1f%%)",
		tokenStyle.Render(fmt.Sprintf("%dk", totalT/1000)),
		metricLabel.Render(fmt.Sprintf("%dk", ctxWindow/1000)),
		pct))
	sb.WriteString("\n")

	sb.WriteString(fmt.Sprintf("  Prompt: %s | Completion: %s",
		metricValue.Render(fmt.Sprintf("%dk", promptT/1000)),
		metricValue.Render(fmt.Sprintf("%dk", complT/1000))))
	sb.WriteString("\n")

	cost := m.monitor.stats.estimatedCost
	sb.WriteString(fmt.Sprintf("  Est. Cost: %s",
		metricValue.Render(fmt.Sprintf("$%.4f", cost))))
	sb.WriteString("\n")
	sb.WriteString("\n")

	if len(m.monitor.stats.errors) > 0 {
		sb.WriteString(errorLogStyle.Render("ERRORS"))
		sb.WriteString("\n")
		start := 0
		if len(m.monitor.stats.errors) > 5 {
			start = len(m.monitor.stats.errors) - 5
		}
		for _, e := range m.monitor.stats.errors[start:] {
			sb.WriteString(errorLogStyle.Render(fmt.Sprintf("  %s", truncStr(e, w-4))))
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func (m monitorModel) renderFooter() string {
	pauseStr := " "
	if m.paused {
		pauseStr = " [PAUSED] "
	}
	help := dimText.Render(fmt.Sprintf("%s[q] QUIT  [p] PAUSE  [\u2191\u2193] SCROLL  PHUBOT MONITOR", pauseStr))
	return lipgloss.Place(m.width, 1, lipgloss.Center, lipgloss.Center, help)
}

// ──────────────────────────────────────────────
// CHAIN OF THOUGHT BUILDER
// ──────────────────────────────────────────────

func (m monitorModel) buildCoT() string {
	events := m.monitor.events
	if len(events) == 0 {
		return dimText.Render("\n  Awaiting transmissions...\n")
	}

	var sb strings.Builder
	start := 0
	if len(events) > 200 {
		start = len(events) - 200
	}

	for _, evt := range events[start:] {
		ts := evt.Timestamp.Format("15:04:05")

		switch evt.Type {
		case EventUserMsg:
			sb.WriteString(userMsgStyle.Render(fmt.Sprintf("[%s] >> USER: %s\n", ts, truncStr(evt.Message, 120))))

		case EventAssistantMsg:
			lines := smartTruncate(evt.Message, 3, 200)
			sb.WriteString(botMsgStyle.Render(fmt.Sprintf("[%s] << BOT: %s\n", ts, lines)))

		case EventToolCall:
			detail := ""
			if evt.Detail != "" {
				detail = dimText.Render(fmt.Sprintf(" (%s)", truncStr(evt.Detail, 60)))
			}
			sb.WriteString(toolRunStyle.Render(fmt.Sprintf("[%s] >> CALL: %s", ts, evt.Tool)))
			sb.WriteString(detail)
			sb.WriteString("\n")

		case EventToolResult:
			dur := ""
			if evt.Duration > 0 {
				dur = fmt.Sprintf(" [%v]", evt.Duration.Round(time.Millisecond))
			}
			sb.WriteString(toolDoneStyle.Render(fmt.Sprintf("[%s] << DONE: %s%s", ts, evt.Tool, dur)))
			if evt.Message != "" {
				sb.WriteString(dimText.Render(fmt.Sprintf(" -- %s", truncStr(evt.Message, 100))))
			}
			sb.WriteString("\n")

		case EventProgress:
			sb.WriteString(progressStyle.Render(fmt.Sprintf("[%s]    ... %s\n", ts, evt.Message)))

		case EventError:
			sb.WriteString(errorLogStyle.Render(fmt.Sprintf("[%s] !! ERROR: %s\n", ts, evt.Message)))

		case EventCompaction:
			sb.WriteString(compactStyle.Render(fmt.Sprintf("[%s] <> COMPACT: %s\n", ts, evt.Message)))

		case EventLLMRequest:
			sb.WriteString(llmStyle.Render(fmt.Sprintf("[%s] >> LLM REQUEST\n", ts)))

		case EventLLMResponse:
			sb.WriteString(llmStyle.Render(fmt.Sprintf("[%s] << LLM RESPONSE\n", ts)))

		case EventSystem:
			sb.WriteString(systemStyle.Render(fmt.Sprintf("[%s] SYS: %s\n", ts, evt.Message)))
		}
	}

	return sb.String()
}

func smartTruncate(s string, maxLines, maxChars int) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\r", "")
	lines := strings.Split(s, "\n")

	totalChars := 0
	var kept []string
	for _, line := range lines {
		totalChars += len(line) + 1
		if len(kept) >= maxLines || totalChars > maxChars {
			kept = append(kept, "...")
			break
		}
		kept = append(kept, line)
	}
	result := strings.Join(kept, "\n")
	if len(result) > maxChars {
		return result[:maxChars-3] + "..."
	}
	return result
}

func truncStr(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	if len(s) > maxLen {
		return s[:maxLen-3] + "..."
	}
	return s
}

func toolIcon(name string) string {
	switch name {
	case "search_flights":
		return "[PLANE]"
	case "identify_aircraft":
		return "[ACFT]"
	case "browse_web":
		return "[WEB]"
	case "scheduler":
		return "[CLK]"
	default:
		return "[?]"
	}
}

// ──────────────────────────────────────────────
// MOCK DEMO GENERATOR
// ──────────────────────────────────────────────

var demoCounter int

func fireDemoEvent(m *Monitor) {
	demoSeq := []MonitorEvent{
		{Type: EventUserMsg, Message: "find me cheap flights from Bangkok to Kuala Lumpur next week"},
		{Type: EventLLMRequest},
		{Type: EventLLMResponse},
		{Type: EventToolCall, Tool: "search_flights", Detail: `{"from":"BKK","to":"KUL","date":"2025-04-15"}`},
		{Type: EventProgress, Message: "opening headless browser..."},
		{Type: EventProgress, Message: "navigating to momondo.de..."},
		{Type: EventProgress, Message: "scraping flight results..."},
		{Type: EventProgress, Message: "found 12 results"},
		{Type: EventToolResult, Tool: "search_flights", Message: "5 flights found, cheapest $89 AirAsia", Duration: 8 * time.Second},
		{Type: EventToolCall, Tool: "identify_aircraft", Detail: `{"from":"BKK","to":"KUL"}`},
		{Type: EventProgress, Message: "searching Google for aircraft type..."},
		{Type: EventToolResult, Tool: "identify_aircraft", Message: "Airbus A320, Boeing 737-800", Duration: 3 * time.Second},
		{Type: EventLLMRequest},
		{Type: EventLLMResponse},
		{Type: EventAssistantMsg, Message: "Here are the cheapest flights from Bangkok to Kuala Lumpur:\n\n1. AirAsia FD $89 - Airbus A320\n2. Thai Lion Air SL $102 - Boeing 737-800\n3. Malaysia Airlines MH $145 - Airbus A330\n\nThe cheapest option is AirAsia at $89."},
		{Type: EventUserMsg, Message: "what about the following week?"},
		{Type: EventLLMRequest},
		{Type: EventLLMResponse},
		{Type: EventToolCall, Tool: "search_flights", Detail: `{"from":"BKK","to":"KUL","date":"2025-04-22"}`},
		{Type: EventProgress, Message: "navigating to momondo.de..."},
		{Type: EventToolResult, Tool: "search_flights", Message: "3 flights found, cheapest $95 Batik Air", Duration: 7 * time.Second},
		{Type: EventLLMRequest},
		{Type: EventLLMResponse},
		{Type: EventAssistantMsg, Message: "For April 22nd, prices are slightly higher:\n\n1. Batik Air $95\n2. AirAsia $108\n3. Thai Lion Air $119"},
		{Type: EventError, Message: "scheduler: task heartbeat timeout"},
		{Type: EventCompaction, Message: "45 -> 12 messages (saved 8k tokens)"},
	}

	if demoCounter >= len(demoSeq) {
		demoCounter = 0
	}
	m.Emit(demoSeq[demoCounter])
	demoCounter++
}

// ──────────────────────────────────────────────
// ENTRY POINT
// ──────────────────────────────────────────────

func RunMonitor(m *Monitor, demo bool) *tea.Program {
	model := newMonitorModel(m, demo)
	model.monitor.stats.uptime = time.Now()
	model.monitor.stats.agentPhase = "IDLE"
	p := tea.NewProgram(model, tea.WithAltScreen())
	go func() {
		_, _ = p.Run()
	}()
	return p
}
