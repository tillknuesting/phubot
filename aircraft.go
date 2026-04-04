package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/chromedp"
	"github.com/sashabaranov/go-openai"
)

type AircraftEntry struct {
	Airline      string `json:"airline"`
	AircraftType string `json:"aircraft_type"`
	FlightNumber string `json:"flight_number,omitempty"`
	Source       string `json:"source"`
}

type AircraftResult struct {
	Route         string          `json:"route"`
	Origin        string          `json:"origin"`
	Destination   string          `json:"destination"`
	Source        string          `json:"source"`
	AircraftTypes []string        `json:"aircraft_types"`
	Entries       []AircraftEntry `json:"entries"`
}

type aircraftCacheEntry struct {
	result   AircraftResult
	cachedAt time.Time
}

type IdentifyAircraftTool struct {
	progressCb func(string)
	cache      map[string]aircraftCacheEntry
	cacheMu    sync.RWMutex
	cacheTTL   time.Duration
	headless   bool
}

func NewIdentifyAircraftTool(headless bool) *IdentifyAircraftTool {
	return &IdentifyAircraftTool{
		cache:    make(map[string]aircraftCacheEntry),
		cacheTTL: 24 * time.Hour,
		headless: headless,
	}
}

func (t *IdentifyAircraftTool) SetProgressCallback(cb func(string)) {
	t.progressCb = cb
}

func (t *IdentifyAircraftTool) progress(msg string) {
	if t.progressCb != nil {
		t.progressCb(msg)
	}
}

func (t *IdentifyAircraftTool) Name() string { return "identify_aircraft" }

func (t *IdentifyAircraftTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: "Identify likely aircraft types used on a flight route. Searches flight tracking sites via Google to find real equipment data for the route. Returns airline, aircraft type, and source.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"origin":      map[string]any{"type": "string", "description": "Origin airport IATA code (e.g. HKT, FRA, MUC)"},
					"destination": map[string]any{"type": "string", "description": "Destination airport IATA code (e.g. KUL, JFK, BCN)"},
					"airline":     map[string]any{"type": "string", "description": "Optional airline name to filter results (e.g. 'AirAsia', 'Lufthansa')"},
				},
				"required": []string{"origin", "destination"},
			},
		},
	}
}

func (t *IdentifyAircraftTool) Execute(args string) (string, error) {
	return t.ExecuteWithContext(context.Background(), args)
}

func (t *IdentifyAircraftTool) ExecuteWithContext(ctx context.Context, args string) (string, error) {
	log.Printf("[Aircraft] ExecuteWithContext called: %s", args)
	lookupStart := time.Now()

	var params struct {
		Origin      string `json:"origin"`
		Destination string `json:"destination"`
		Airline     string `json:"airline"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		log.Printf("[Aircraft] Failed to parse args: %v", err)
		return "", fmt.Errorf("failed to parse args: %v", err)
	}

	log.Printf("[Aircraft] Looking up route: %s → %s (airline filter: %q)", params.Origin, params.Destination, params.Airline)

	if params.Origin == "" || params.Destination == "" {
		return "", fmt.Errorf("origin and destination are required")
	}

	origin := strings.ToUpper(strings.TrimSpace(params.Origin))
	dest := strings.ToUpper(strings.TrimSpace(params.Destination))

	cacheKey := fmt.Sprintf("%s-%s", origin, dest)
	if cached, ok := t.getCached(cacheKey); ok {
		age := time.Since(t.getCacheEntryAge(cacheKey))
		log.Printf("[Aircraft] Cache HIT for %s (age: %v)", cacheKey, age.Round(time.Second))
		t.progress(fmt.Sprintf("Using cached aircraft data for %s -> %s (%v old)", origin, dest, age.Round(time.Hour)))
		return formatAircraftResult(cached, params.Airline), nil
	}

	log.Printf("[Aircraft] Cache MISS for %s, performing live lookup", cacheKey)
	t.progress(fmt.Sprintf("Looking up aircraft types for %s -> %s...", origin, dest))

	result := t.lookupRoute(ctx, origin, dest)

	if len(result.Entries) > 0 {
		t.setCache(cacheKey, result)
		log.Printf("[Aircraft] Lookup completed in %v: found %d entries, %d unique types for %s→%s", time.Since(lookupStart).Round(time.Millisecond), len(result.Entries), len(result.AircraftTypes), origin, dest)
		t.progress(fmt.Sprintf("Found %d aircraft types for %s -> %s", len(result.AircraftTypes), origin, dest))
	} else {
		log.Printf("[Aircraft] Lookup completed in %v: no results for %s→%s", time.Since(lookupStart).Round(time.Millisecond), origin, dest)
		t.progress(fmt.Sprintf("No aircraft data found for %s -> %s", origin, dest))
	}

	return formatAircraftResult(result, params.Airline), nil
}

func (t *IdentifyAircraftTool) lookupRoute(ctx context.Context, origin, dest string) AircraftResult {
	log.Printf("[Aircraft] lookupRoute: %s → %s", origin, dest)
	result := AircraftResult{
		Origin:      origin,
		Destination: dest,
		Route:       fmt.Sprintf("%s-%s", origin, dest),
	}

	queries := []string{
		fmt.Sprintf("aircraft type %s to %s route plane type", origin, dest),
		fmt.Sprintf("what plane flies %s %s aircraft", origin, dest),
	}

	for _, q := range queries {
		log.Printf("[Aircraft] Trying query: %q", q)
		entries := t.googleSearchAircraft(ctx, q)
		if len(entries) > 0 {
			result.Source = "Google"
			result.Entries = entries
			result.AircraftTypes = extractUniqueAircraftTypes(entries)
			log.Printf("[Aircraft] Found %d entries with query %q", len(entries), q)
			return result
		}
		log.Printf("[Aircraft] No entries from query %q, trying next", q)
	}

	log.Printf("[Aircraft] All queries exhausted, no results for %s→%s", origin, dest)
	result.Source = "none"
	return result
}

func (t *IdentifyAircraftTool) googleSearchAircraft(ctx context.Context, query string) []AircraftEntry {
	url := fmt.Sprintf("https://www.google.com/search?q=%s", strings.ReplaceAll(query, " ", "+"))
	log.Printf("[Aircraft] Google search: %s", url)
	t.progress(fmt.Sprintf("Searching: %s...", query))

	pageText, err := t.browsePage(ctx, url, 6)
	if err != nil {
		log.Printf("[Aircraft] Google search failed: %v", err)
		return nil
	}

	entries := parseAircraftFromText(pageText)
	log.Printf("[Aircraft] Google parsed %d entries from %d chars", len(entries), len(pageText))
	return entries
}

func (t *IdentifyAircraftTool) browsePage(ctx context.Context, url string, waitSecs int) (string, error) {
	log.Printf("[Aircraft] browsePage: %s (wait: %ds, headless: %v)", url, waitSecs, t.headless)
	browseStart := time.Now()

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", t.headless),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.Flag("disable-extensions", false),
		chromedp.Flag("disable-images", false),
		chromedp.Flag("disable-web-security", false),
		chromedp.Flag("disable-infobars", true),
		chromedp.Flag("enable-automation", false),
		chromedp.Flag("no-first-run", true),
		chromedp.Flag("no-default-browser-check", true),
		chromedp.Flag("disable-popup-blocking", false),
		chromedp.Flag("disable-background-timer-throttling", true),
		chromedp.Flag("disable-renderer-backgrounding", true),
		chromedp.Flag("disable-backgrounding-occluded-windows", true),
		chromedp.Flag("disable-ipc-flooding-protection", true),
		chromedp.Flag("disable-component-update", true),
		chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
		chromedp.Flag("window-size", "1920,1080"),
		chromedp.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
		chromedp.WindowSize(1920, 1080),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	browserCtx, cancel := chromedp.NewContext(allocCtx, chromedp.WithLogf(log.Printf))
	defer cancel()

	timeoutCtx, cancelTimeout := context.WithTimeout(browserCtx, time.Duration(waitSecs+15)*time.Second)
	defer cancelTimeout()

	err := chromedp.Run(timeoutCtx,
		chromedp.Navigate(url),
		chromedp.Sleep(2*time.Second),
		chromedp.Evaluate(`
			Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
			Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
			Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'de']});
			window.chrome = {runtime: {}};
		`, nil),
	)
	if err != nil {
		log.Printf("[Aircraft] Navigation to %s failed after %v: %v", url, time.Since(browseStart), err)
		return "", fmt.Errorf("navigation failed: %v", err)
	}

	log.Printf("[Aircraft] Page navigated, waiting %ds for content to load", waitSecs)

	chromedp.Run(timeoutCtx, chromedp.Sleep(time.Duration(waitSecs)*time.Second))

	var pageText string
	if err := chromedp.Run(timeoutCtx, chromedp.Evaluate(`document.body.innerText.substring(0, 8000)`, &pageText)); err != nil {
		log.Printf("[Aircraft] Text extraction failed: %v", err)
		return "", fmt.Errorf("extract text failed: %v", err)
	}

	log.Printf("[Aircraft] browsePage completed in %v, extracted %d chars from %s", time.Since(browseStart).Round(time.Millisecond), len(pageText), url)
	return pageText, nil
}

func (t *IdentifyAircraftTool) getCached(key string) (AircraftResult, bool) {
	t.cacheMu.RLock()
	defer t.cacheMu.RUnlock()
	entry, ok := t.cache[key]
	if !ok {
		return AircraftResult{}, false
	}
	if time.Since(entry.cachedAt) > t.cacheTTL {
		return AircraftResult{}, false
	}
	return entry.result, true
}

func (t *IdentifyAircraftTool) setCache(key string, result AircraftResult) {
	t.cacheMu.Lock()
	defer t.cacheMu.Unlock()
	t.cache[key] = aircraftCacheEntry{
		result:   result,
		cachedAt: time.Now(),
	}
}

func (t *IdentifyAircraftTool) getCacheEntryAge(key string) time.Time {
	t.cacheMu.RLock()
	defer t.cacheMu.RUnlock()
	entry, ok := t.cache[key]
	if !ok {
		return time.Time{}
	}
	return entry.cachedAt
}

func extractUniqueAircraftTypes(entries []AircraftEntry) []string {
	seen := make(map[string]bool)
	var types []string
	for _, e := range entries {
		normalized := normalizeAircraftType(e.AircraftType)
		if normalized != "" && !seen[normalized] {
			seen[normalized] = true
			types = append(types, normalized)
		}
	}
	return types
}

func normalizeAircraftType(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	s = strings.ReplaceAll(s, "\u00a0", " ")
	if len(s) > 60 {
		s = s[:60]
	}
	return s
}

func formatAircraftResult(result AircraftResult, airlineFilter string) string {
	if len(result.Entries) == 0 {
		return fmt.Sprintf("No aircraft data found for route %s -> %s.", result.Origin, result.Destination)
	}

	entries := result.Entries
	if airlineFilter != "" {
		filtered := filterEntriesByAirline(entries, airlineFilter)
		if len(filtered) > 0 {
			entries = filtered
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Aircraft types for %s -> %s (source: %s)\n\n", result.Origin, result.Destination, result.Source))

	for i, e := range entries {
		if i >= 10 {
			sb.WriteString(fmt.Sprintf("... and %d more\n", len(entries)-10))
			break
		}
		sb.WriteString(fmt.Sprintf("%d. %s - %s", i+1, e.Airline, e.AircraftType))
		if e.FlightNumber != "" {
			sb.WriteString(fmt.Sprintf(" (%s)", e.FlightNumber))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

func filterEntriesByAirline(entries []AircraftEntry, airline string) []AircraftEntry {
	lower := strings.ToLower(airline)
	var filtered []AircraftEntry
	for _, e := range entries {
		if strings.Contains(strings.ToLower(e.Airline), lower) {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

var (
	reAircraftType = regexp.MustCompile(`(?i)\b(Boeing\s*\d{3,4}(?:-\d{1,4})?(?:\s*(?:MAX|MAX\d*|ER|LR|F|BCF))?\b|Airbus\s*A\d{3}(?:-\d{3})?(?:neo)?\b|Embraer\s*(?:E-?\d{2,3}(?:-\d{3})?|ERJ-\d{3})\b|Bombardier\s*(?:CRJ-?\d{2,3}|Dash\s*8-?\d{3}|CS\d{3})\b|ATR\s*\d{2,3}(?:-\d{3})?\b|De Havilland\s*Dash\s*8-?\d{3}\b)`)
	reICAOAircraft = regexp.MustCompile(`\b(B738|B737|B739|B73G|B744|B748|B752|B753|B763|B764|B772|B773|B77W|B77L|B788|B789|A319|A320|A321|A332|A333|A343|A346|A359|A35K|A388|E170|E175|E190|E195|E290|E295|AT72|AT76|DH8D|CRJ9|CRJ7|CS300|BCS1|BCS3)\b`)
)

var icaoToName = map[string]string{
	"B738": "Boeing 737-800", "B737": "Boeing 737-700", "B739": "Boeing 737-900",
	"B73G": "Boeing 737-700", "B744": "Boeing 747-400", "B748": "Boeing 747-8",
	"B752": "Boeing 757-200", "B753": "Boeing 757-300", "B763": "Boeing 767-300",
	"B764": "Boeing 767-400", "B772": "Boeing 777-200", "B773": "Boeing 777-300",
	"B77W": "Boeing 777-300ER", "B77L": "Boeing 777-200LR",
	"B788": "Boeing 787-8", "B789": "Boeing 787-9",
	"A319": "Airbus A319", "A320": "Airbus A320", "A321": "Airbus A321",
	"A332": "Airbus A330-200", "A333": "Airbus A330-300",
	"A343": "Airbus A340-300", "A346": "Airbus A340-600",
	"A359": "Airbus A350-900", "A35K": "Airbus A350-1000", "A388": "Airbus A380",
	"E170": "Embraer E170", "E175": "Embraer E175",
	"E190": "Embraer E190", "E195": "Embraer E195",
	"E290": "Embraer E290", "E295": "Embraer E295",
	"AT72": "ATR 72-200", "AT76": "ATR 72-600",
	"DH8D": "De Havilland Dash 8-400",
	"CRJ9": "Bombardier CRJ-900", "CRJ7": "Bombardier CRJ-700",
	"CS300": "Airbus A220-300",
	"BCS1":  "Airbus A220-100", "BCS3": "Airbus A220-300",
}

func parseAircraftFromText(body string) []AircraftEntry {
	parseStart := time.Now()
	body = strings.ReplaceAll(body, "\u00a0", " ")
	var entries []AircraftEntry
	seen := make(map[string]bool)

	matches := reAircraftType.FindAllString(body, -1)
	log.Printf("[Aircraft] Regex 'aircraft_type' found %d matches in %d chars", len(matches), len(body))
	for _, m := range matches {
		normalized := normalizeAircraftType(m)
		if normalized != "" && !seen[normalized] {
			seen[normalized] = true
			airline := extractAirlineNearAircraft(body, m)
			entries = append(entries, AircraftEntry{
				Airline:      airline,
				AircraftType: normalized,
				Source:       "Google",
			})
		}
	}

	icaoMatches := reICAOAircraft.FindAllString(body, -1)
	log.Printf("[Aircraft] Regex 'ICAO_aircraft' found %d matches in %d chars", len(icaoMatches), len(body))
	for _, code := range icaoMatches {
		name, ok := icaoToName[code]
		if !ok {
			continue
		}
		if !seen[name] {
			seen[name] = true
			airline := extractAirlineNearAircraft(body, code)
			entries = append(entries, AircraftEntry{
				Airline:      airline,
				AircraftType: name,
				Source:       "Google",
			})
		}
	}

	log.Printf("[Aircraft] parseAircraftFromText: %d total entries from %d chars in %v", len(entries), len(body), time.Since(parseStart).Round(time.Millisecond))
	return entries
}

func ParseFlightAwareAircraft(body string) []AircraftEntry {
	return parseAircraftFromText(body)
}

func ParseFlightradar24Aircraft(body string) []AircraftEntry {
	return parseAircraftFromText(body)
}

func extractAirlineNearAircraft(body, aircraft string) string {
	idx := strings.Index(body, aircraft)
	if idx < 0 {
		return ""
	}

	start := max(idx-80, 0)
	before := body[start:idx]

	lines := strings.Split(before, "\n")
	if len(lines) > 0 {
		candidate := strings.TrimSpace(lines[len(lines)-1])
		if len(candidate) > 2 && len(candidate) < 50 && !isOnlyNumbers(candidate) {
			return candidate
		}
	}

	if len(lines) > 1 {
		candidate := strings.TrimSpace(lines[len(lines)-2])
		if len(candidate) > 2 && len(candidate) < 50 && !isOnlyNumbers(candidate) {
			return candidate
		}
	}

	return ""
}

func isOnlyNumbers(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && c != ' ' && c != '-' {
			return false
		}
	}
	return true
}
