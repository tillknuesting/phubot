package main

import (
	"fmt"
	"strings"
	"sync"
	"time"

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
	uptime           time.Time
	activeTool       string
	activeToolStart  time.Time
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
	case EventError:
		m.stats.totalErrors++
	case EventCompaction:
		m.stats.totalCompactions++
	case EventUserMsg, EventAssistantMsg:
		m.stats.totalMessages++
	}
	m.mu.Unlock()

	select {
	case m.ch <- evt:
	default:
	}
}

type monitorModel struct {
	monitor  *Monitor
	logs     viewport.Model
	width    int
	height   int
	ready    bool
	lastLogs string
}

func newMonitorModel(m *Monitor) monitorModel {
	return monitorModel{monitor: m}
}

func (m monitorModel) Init() tea.Cmd {
	return tea.EnterAltScreen
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

func (m monitorModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		logHeight := max(msg.Height-10, 5)
		if !m.ready {
			m.logs = viewport.New(msg.Width, logHeight)
			m.logs.Style = lipgloss.NewStyle()
			m.ready = true
		} else {
			m.logs.Width = msg.Width
			m.logs.Height = logHeight
		}
	case MonitorEvent:
		m.monitor.mu.RLock()
		logs := m.buildLogContent()
		m.monitor.mu.RUnlock()
		if logs != m.lastLogs {
			m.lastLogs = logs
			m.logs.SetContent(logs)
			m.logs.GotoBottom()
		}
		return m, waitForEvent(m.monitor.ch)
	}

	var cmd tea.Cmd
	m.logs, cmd = m.logs.Update(msg)
	return m, cmd
}

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#00FF00")).
			Background(lipgloss.Color("#1a1a2e"))

	statusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00FF00")).
			Background(lipgloss.Color("#16213e"))

	toolRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFFF00")).
				Bold(true)

	toolDoneStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00FF00"))

	toolIdleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#333333"))

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF0000")).
			Bold(true)

	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00BFFF"))

	assistantStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF00FF"))

	progressStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFA500"))

	systemStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#666666"))

	dimStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#444444"))

	compactionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00CED1"))

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00FF00")).
			Padding(0, 1)
)

func (m monitorModel) View() string {
	if !m.ready {
		return "Initializing..."
	}

	m.monitor.mu.RLock()
	defer m.monitor.mu.RUnlock()

	header := m.renderHeader()
	logs := m.logs.View()
	footer := m.renderFooter()

	return lipgloss.JoinVertical(lipgloss.Left, header, logs, footer)
}

func (m monitorModel) renderHeader() string {
	w := m.width
	uptime := time.Since(m.monitor.stats.uptime).Round(time.Second)
	uptimeStr := fmt.Sprintf("%02d:%02d:%02d", int(uptime.Hours()), int(uptime.Minutes())%60, int(uptime.Seconds())%60)

	title := titleStyle.Render(fmt.Sprintf(" PHUBOT v1.0 -- SYSTEM MONITOR -- UPTIME %s ", uptimeStr))
	title = lipgloss.Place(w, 1, lipgloss.Center, lipgloss.Center, title)

	statsLine := fmt.Sprintf(
		" MSG: %d  |  TOOLS: %d  |  ERRORS: %d  |  COMPACT: %d ",
		m.monitor.stats.totalMessages,
		m.monitor.stats.totalToolCalls,
		m.monitor.stats.totalErrors,
		m.monitor.stats.totalCompactions,
	)
	statsBar := statusStyle.Render(statsLine)
	statsBar = lipgloss.Place(w, 1, lipgloss.Center, lipgloss.Center, statsBar)

	toolPanel := m.renderToolPanel(w)

	return lipgloss.JoinVertical(lipgloss.Left, title, statsBar, toolPanel)
}

func (m monitorModel) renderToolPanel(w int) string {
	tools := []string{"search_flights", "identify_aircraft", "browse_web", "scheduler"}
	panelW := w/len(tools) - 2

	var boxes []string
	for _, name := range tools {
		ts, ok := m.monitor.tools[name]
		status := "IDLE"
		style := toolIdleStyle
		info := "---"

		if ok {
			elapsed := time.Since(ts.lastCall)
			if ts.status == "RUNNING" {
				status = ">> RUN"
				style = toolRunningStyle
				info = elapsed.Round(time.Second).String()
			} else {
				status = "DONE"
				style = toolDoneStyle
				if ts.lastDur > 0 {
					info = fmt.Sprintf("%dx %s", ts.callCount, ts.lastDur.Round(time.Second))
				} else {
					info = fmt.Sprintf("%dx", ts.callCount)
				}
			}
		}

		if m.monitor.stats.activeTool == name {
			status = ">> RUN"
			style = toolRunningStyle
		}

		icon := toolIcon(name)
		line1 := style.Render(fmt.Sprintf(" %s %s", icon, truncStr(name, 16)))
		line2 := style.Render(fmt.Sprintf(" %s", status))
		line3 := dimStyle.Render(fmt.Sprintf(" %s", info))

		content := lipgloss.JoinVertical(lipgloss.Left, line1, line2, line3)
		box := lipgloss.NewStyle().
			Width(panelW).
			Height(3).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00FF00")).
			Render(content)

		boxes = append(boxes, box)
	}

	return lipgloss.JoinHorizontal(lipgloss.Top, boxes...)
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

func (m monitorModel) renderFooter() string {
	helpText := dimStyle.Render(" [q] QUIT  |  [up/down] SCROLL  |  PHUBOT MONITOR ")
	return lipgloss.Place(m.width, 1, lipgloss.Center, lipgloss.Center, helpText)
}

func (m monitorModel) buildLogContent() string {
	events := m.monitor.events
	if len(events) == 0 {
		return dimStyle.Render("\n  Awaiting transmissions...\n")
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
			sb.WriteString(userStyle.Render(fmt.Sprintf("[%s] >> USER: %s\n", ts, truncStr(evt.Message, 120))))

		case EventAssistantMsg:
			sb.WriteString(assistantStyle.Render(fmt.Sprintf("[%s] << BOT: %s\n", ts, truncStr(evt.Message, 120))))

		case EventToolCall:
			detail := ""
			if evt.Detail != "" {
				detail = dimStyle.Render(fmt.Sprintf(" (%s)", truncStr(evt.Detail, 60)))
			}
			sb.WriteString(toolRunningStyle.Render(fmt.Sprintf("[%s] >> CALL: %s", ts, evt.Tool)))
			sb.WriteString(detail)
			sb.WriteString("\n")

		case EventToolResult:
			dur := ""
			if evt.Duration > 0 {
				dur = fmt.Sprintf(" [%v]", evt.Duration.Round(time.Millisecond))
			}
			sb.WriteString(toolDoneStyle.Render(fmt.Sprintf("[%s] << DONE: %s%s", ts, evt.Tool, dur)))
			if evt.Message != "" {
				sb.WriteString(dimStyle.Render(fmt.Sprintf(" -- %s", truncStr(evt.Message, 100))))
			}
			sb.WriteString("\n")

		case EventProgress:
			sb.WriteString(progressStyle.Render(fmt.Sprintf("[%s]    ... %s\n", ts, evt.Message)))

		case EventError:
			sb.WriteString(errorStyle.Render(fmt.Sprintf("[%s] !! ERROR: %s\n", ts, evt.Message)))

		case EventCompaction:
			sb.WriteString(compactionStyle.Render(fmt.Sprintf("[%s] <> COMPACT: %s\n", ts, evt.Message)))

		case EventLLMRequest:
			sb.WriteString(systemStyle.Render(fmt.Sprintf("[%s] >> LLM REQUEST\n", ts)))

		case EventLLMResponse:
			sb.WriteString(systemStyle.Render(fmt.Sprintf("[%s] << LLM RESPONSE\n", ts)))

		case EventSystem:
			sb.WriteString(systemStyle.Render(fmt.Sprintf("[%s] SYS: %s\n", ts, evt.Message)))
		}
	}

	return sb.String()
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

func RunMonitor(m *Monitor) *tea.Program {
	model := newMonitorModel(m)
	model.monitor.stats.uptime = time.Now()
	p := tea.NewProgram(model, tea.WithAltScreen())
	go func() {
		_, _ = p.Run()
	}()
	return p
}
