package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/chromedp"
	"github.com/sashabaranov/go-openai"
)

type Flight struct {
	Airline      string  `json:"airline"`
	Departure    string  `json:"departure"`
	Arrival      string  `json:"arrival"`
	Duration     string  `json:"duration"`
	Stops        string  `json:"stops"`
	Price        string  `json:"price"`
	PriceRaw     string  `json:"price_raw"`
	DepartureDay string  `json:"departure_day,omitempty"`
	Origin       string  `json:"origin"`
	Destination  string  `json:"destination"`
	Layover      string  `json:"layover,omitempty"`
	DurationMin  int     `json:"duration_min"`
	PriceFloat   float64 `json:"price_float"`
	Total        string  `json:"total"`
	TotalFloat   float64 `json:"total_float"`
	FareClass    string  `json:"fare_class,omitempty"`
	Label        string  `json:"label,omitempty"`
	Aircraft     string  `json:"aircraft,omitempty"`
}

type FlightSearchResult struct {
	Query    string   `json:"query"`
	URL      string   `json:"url"`
	Count    int      `json:"count"`
	Flights  []Flight `json:"flights"`
	Cheapest string   `json:"cheapest,omitempty"`
	Fastest  string   `json:"fastest,omitempty"`
}

type cacheEntry struct {
	result      string
	cachedAt    time.Time
	origin      string
	dest        string
	date        string
	flightCount int
}

type MomondoFlightTool struct {
	progressCb func(string)
	cache      map[string]cacheEntry
	cacheMu    sync.RWMutex
	cacheTTL   time.Duration
	headless   bool
}

func NewMomondoFlightTool(headless bool) *MomondoFlightTool {
	return &MomondoFlightTool{
		cache:    make(map[string]cacheEntry),
		cacheTTL: 60 * time.Minute,
		headless: headless,
	}
}

func (t *MomondoFlightTool) SetProgressCallback(cb func(string)) {
	t.progressCb = cb
}

func (t *MomondoFlightTool) progress(msg string) {
	if t.progressCb != nil {
		t.progressCb(msg)
	}
}

func (t *MomondoFlightTool) cacheKey(origin, dest, date string) string {
	return fmt.Sprintf("%s-%s-%s", origin, dest, date)
}

func (t *MomondoFlightTool) getCached(key string) (string, bool) {
	t.cacheMu.RLock()
	defer t.cacheMu.RUnlock()
	entry, ok := t.cache[key]
	if !ok {
		return "", false
	}
	if time.Since(entry.cachedAt) > t.cacheTTL {
		return "", false
	}
	return entry.result, true
}

func (t *MomondoFlightTool) setCache(key, result, origin, dest, date string, flightCount int) {
	t.cacheMu.Lock()
	defer t.cacheMu.Unlock()
	t.cache[key] = cacheEntry{
		result:      result,
		cachedAt:    time.Now(),
		origin:      origin,
		dest:        dest,
		date:        date,
		flightCount: flightCount,
	}
}

func (t *MomondoFlightTool) getCacheEntryAge(key string) time.Time {
	t.cacheMu.RLock()
	defer t.cacheMu.RUnlock()
	entry, ok := t.cache[key]
	if !ok {
		return time.Time{}
	}
	return entry.cachedAt
}

func (t *MomondoFlightTool) Name() string { return "search_flights" }

func (t *MomondoFlightTool) Definition() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name(),
			Description: "Search for flights on Momondo.de with structured results. Returns parsed flight data (airline, times, duration, stops, price). Uses Chrome browser automation.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"origin":      map[string]any{"type": "string", "description": "Origin airport IATA code (e.g. HKT, FRA, MUC)"},
					"destination": map[string]any{"type": "string", "description": "Destination airport IATA code (e.g. KUL, JFK, BCN)"},
					"date":        map[string]any{"type": "string", "description": "Departure date in YYYY-MM-DD format"},
					"adults":      map[string]any{"type": "string", "description": "Number of adults (default: 2)"},
					"children":    map[string]any{"type": "string", "description": "Children ages, e.g. '10-10' for two 10-year-olds (default: none)"},
					"sort":        map[string]any{"type": "string", "description": "Sort order: 'price' (cheapest first, default) or 'duration' (fastest first)"},
					"stops":       map[string]any{"type": "string", "description": "Filter by stops: '0' for direct flights only, '1' for max 1 stop, '2' for max 2 stops (default: show all)"},
				},
				"required": []string{"origin", "destination", "date"},
			},
		},
	}
}

func (t *MomondoFlightTool) Execute(args string) (string, error) {
	return t.ExecuteWithContext(context.Background(), args)
}

func (t *MomondoFlightTool) ExecuteWithContext(ctx context.Context, args string) (string, error) {
	log.Printf("[Momondo] ExecuteWithContext called: %s", args)
	searchStart := time.Now()

	var params struct {
		Origin      string `json:"origin"`
		Destination string `json:"destination"`
		Date        string `json:"date"`
		Adults      string `json:"adults"`
		Children    string `json:"children"`
		Sort        string `json:"sort"`
		Stops       string `json:"stops"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		log.Printf("[Momondo] Failed to parse args: %v", err)
		return "", fmt.Errorf("failed to parse args: %v", err)
	}

	log.Printf("[Momondo] Search request: %s → %s, date=%s, adults=%s, children=%s, sort=%s, stops=%s", params.Origin, params.Destination, params.Date, params.Adults, params.Children, params.Sort, params.Stops)

	if params.Origin == "" || params.Destination == "" || params.Date == "" {
		return "", fmt.Errorf("origin, destination, and date are required")
	}

	adults := 2
	if params.Adults != "" {
		if v, err := strconv.Atoi(params.Adults); err == nil && v > 0 {
			adults = v
		}
	}

	cacheKey := t.cacheKey(params.Origin, params.Destination, params.Date)
	if cached, ok := t.getCached(cacheKey); ok {
		age := time.Since(t.getCacheEntryAge(cacheKey))
		log.Printf("[Momondo] Cache HIT for %s (age: %v)", cacheKey, age.Round(time.Second))
		t.progress(fmt.Sprintf("📦 Using cached results for %s → %s on %s (%v old)", params.Origin, params.Destination, params.Date, age.Round(time.Minute)))
		return cached, nil
	}

	log.Printf("[Momondo] Cache MISS for %s, performing live search", cacheKey)

	sortBy := "best"
	if params.Sort == "price" || params.Sort == "cheapest" {
		sortBy = "price"
	} else if params.Sort == "duration" || params.Sort == "fastest" {
		sortBy = "duration"
	}

	targetURL := buildMomondoURL(params.Origin, params.Destination, params.Date, "", adults, params.Children, sortBy, params.Stops)

	stopsLabel := "all"
	if params.Stops == "0" {
		stopsLabel = "direct only"
	} else if params.Stops != "" {
		stopsLabel = "max " + params.Stops + " stop(s)"
	}
	log.Printf("[Momondo] Searching: %s", targetURL)
	log.Printf("[Momondo] Params: origin=%s, dest=%s, date=%s, adults=%d, sort=%s, stops=%s", params.Origin, params.Destination, params.Date, adults, sortBy, stopsLabel)
	t.progress(fmt.Sprintf("🔍 Searching %s → %s on %s (%d adults, %s)...", params.Origin, params.Destination, params.Date, adults, stopsLabel))

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
		chromedp.Flag("start-maximized", true),
		chromedp.UserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
		chromedp.WindowSize(1920, 1080),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	browserCtx, cancel := chromedp.NewContext(allocCtx, chromedp.WithLogf(log.Printf))
	defer cancel()

	timeoutCtx, cancelTimeout := context.WithTimeout(browserCtx, 120*time.Second)
	defer cancelTimeout()

	err := chromedp.Run(timeoutCtx,
		chromedp.Navigate(targetURL),
		chromedp.Sleep(2*time.Second),
		chromedp.Evaluate(`
			Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
			Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
			Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en', 'de']});
			window.chrome = {runtime: {}};
		`, nil),
	)
	if err != nil {
		log.Printf("[Momondo] Navigation to %s failed after %v: %v", targetURL, time.Since(searchStart), err)
		return "", fmt.Errorf("navigation failed: %v", err)
	}

	log.Printf("[Momondo] Page loaded, starting poll for flight results (headless: %v)", t.headless)
	t.progress("🌐 Page loaded, waiting for flight results...")

	var bodyText string
	log.Printf("[Momondo] Starting page poll (up to 30 rounds)")
	for i := range 30 {
		time.Sleep(2 * time.Second)

		if err := chromedp.Run(timeoutCtx, chromedp.Evaluate(`document.body.innerText`, &bodyText)); err != nil {
			log.Printf("[Momondo] Poll %d error: %v", i+1, err)
			continue
		}

		hasPrices := strings.Contains(bodyText, "€") && strings.Contains(bodyText, "/Person")
		log.Printf("[Momondo] Poll %d: prices=%v len=%d", i+1, hasPrices, len(bodyText))

		if i%3 == 0 {
			t.progress(fmt.Sprintf("⏳ Waiting for results... (poll %d/30, prices found: %v)", i+1, hasPrices))
		}

		if hasPrices && i >= 3 {
			t.progress("📜 Scrolling for more flight results...")
			scrollAndWait(timeoutCtx)
			scrollAndWait(timeoutCtx)
			chromedp.Run(timeoutCtx, chromedp.Evaluate(`document.body.innerText`, &bodyText))
			break
		}

		if strings.Contains(bodyText, "Leider keine Ergebnisse") {
			t.progress("❌ No results found on Momondo")
			break
		}
	}

	if bodyText == "" {
		return "No page content retrieved from Momondo.", nil
	}

	flights := ParseMomondoFlights(bodyText)
	log.Printf("[Momondo] Parsed %d flights from body text (%d chars)", len(flights), len(bodyText))

	if len(flights) > 0 {
		t.progress(fmt.Sprintf("✈️ Found %d flights, preparing results...", len(flights)))
	}

	if len(flights) == 0 {
		preview := truncate(bodyText, 4000)
		return fmt.Sprintf("No structured flight results found. Page preview:\n%s", preview), nil
	}

	if len(flights) > 5 {
		log.Printf("[Momondo] Trimming %d flights to top 5", len(flights))
		flights = flights[:5]
	}

	result := FlightSearchResult{
		Query:   fmt.Sprintf("%s → %s on %s (%d adults)", params.Origin, params.Destination, params.Date, adults),
		URL:     targetURL,
		Count:   len(flights),
		Flights: flights,
	}

	if len(result.Flights) > 0 {
		f := result.Flights[0]
		result.Cheapest = fmt.Sprintf("%s - %s → %s | %s | %s (%s)", f.Airline, f.Origin, f.Destination, f.Duration, f.Total, f.Stops)
	}

	for _, f := range result.Flights {
		if f.DurationMin > 0 {
			result.Fastest = fmt.Sprintf("%s - %s → %s | %s | %s (%s)", f.Airline, f.Origin, f.Destination, f.Duration, f.Total, f.Stops)
			break
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Found %d flights for %s\n", result.Count, result.Query))
	if result.Cheapest != "" {
		sb.WriteString(fmt.Sprintf("Cheapest: %s\n", result.Cheapest))
	}
	if result.Fastest != "" {
		sb.WriteString(fmt.Sprintf("Fastest:  %s\n", result.Fastest))
	}
	sb.WriteString("\n")
	for i, f := range result.Flights {
		priceDisplay := f.Price
		if f.Total != "" {
			priceDisplay = f.Total
		}
		sb.WriteString(fmt.Sprintf("%d. %s | %s-%s | %s → %s | %s | %s | %s",
			i+1, f.Airline, f.Departure, f.Arrival, f.Origin, f.Destination, f.Duration, f.Stops, priceDisplay))
		if f.Layover != "" {
			sb.WriteString(fmt.Sprintf(" | layover: %s", f.Layover))
		}
		if f.Label != "" {
			sb.WriteString(fmt.Sprintf(" [%s]", f.Label))
		}
		sb.WriteString("\n")
	}

	log.Printf("[Momondo] Scraped %d flights for %s→%s", result.Count, params.Origin, params.Destination)

	entry := PriceEntry{
		Timestamp:   time.Now(),
		Origin:      params.Origin,
		Destination: params.Destination,
		Date:        params.Date,
		Adults:      adults,
		Count:       result.Count,
	}
	if len(result.Flights) > 0 {
		entry.Cheapest = &result.Flights[0]
	}
	for _, f := range result.Flights {
		if f.DurationMin > 0 {
			entry.Fastest = &f
			break
		}
	}
	if err := AppendPriceHistory(entry); err != nil {
		log.Printf("[Momondo] Failed to save price history: %v", err)
	} else {
		log.Printf("[Momondo] Price history saved: %s→%s %s, %d flights", entry.Origin, entry.Destination, entry.Date, entry.Count)
	}

	t.setCache(cacheKey, sb.String(), params.Origin, params.Destination, params.Date, result.Count)

	log.Printf("[Momondo] Search completed in %v: %d flights found for %s→%s on %s", time.Since(searchStart).Round(time.Millisecond), result.Count, params.Origin, params.Destination, params.Date)
	t.progress(fmt.Sprintf("✅ Done! Found %d flights, cheapest: %s", result.Count, result.Cheapest))

	return sb.String(), nil
}

func buildMomondoURL(origin, dest, date, returnDate string, adults int, children, sortBy, stops string) string {
	path := fmt.Sprintf("/flight-search/%s-%s/%s/%dadults", origin, dest, date, adults)
	if returnDate != "" {
		path += fmt.Sprintf("/%s", returnDate)
	}
	if children != "" {
		path += fmt.Sprintf("/children-%s", children)
	}

	sortParam := "bestflight_a"
	switch sortBy {
	case "price", "cheapest":
		sortParam = "price_a"
	case "duration", "fastest":
		sortParam = "duration_a"
	}

	url := fmt.Sprintf("https://www.momondo.de%s?sort=%s", path, sortParam)
	if stops != "" {
		url += fmt.Sprintf("&stops=%s", stops)
	}
	return url
}

func scrollAndWait(ctx context.Context) {
	log.Printf("[Momondo] Starting page scroll (30 scrolls, ~18s total)")
	scrollStart := time.Now()
	chromedp.Run(ctx,
		chromedp.Evaluate(`
			for (let i = 0; i < 30; i++) {
				setTimeout(() => window.scrollBy(0, 800), i * 400);
			}
		`, nil),
		chromedp.Sleep(6*time.Second),
	)
	log.Printf("[Momondo] Page scroll completed in %v", time.Since(scrollStart).Round(time.Millisecond))
}

func ParseMomondoFlights(body string) []Flight {
	parseStart := time.Now()
	body = strings.ReplaceAll(body, "\u00a0", " ")

	var flights []Flight
	lines := strings.Split(body, "\n")
	log.Printf("[Momondo] ParseMomondoFlights: processing %d lines (%d chars)", len(lines), len(body))

	reTime := regexp.MustCompile(`^\d{1,2}:\d{2}\s*[–-]\s*\d{1,2}:\d{2}(?:\+\d)?$`)
	reDurationShort := regexp.MustCompile(`^\d{1,2}:\d{2}\s+Std\.$`)
	reDurationLong := regexp.MustCompile(`^\d+\s+Std\.\s+\d+\s+Min\.$`)
	rePrice := regexp.MustCompile(`^[\d.,]+\s+€$`)
	reTotal := regexp.MustCompile(`^Gesamt:\s*[\d.,]+\s+€$`)
	reStops := regexp.MustCompile(`^(Nonstop|\d\s+Stopp|(\d+)\s+Stopps?)$`)

	isDuration := func(s string) bool {
		return reDurationShort.MatchString(s) || reDurationLong.MatchString(s)
	}

	labelKeywords := []string{"Beste Option", "Günstigste Verbindung", "Günstigste Option", "Schnellste Option", "Weitere Optionen"}

	noisePrefixes := []string{"Zum ", "Nächstes", "Zu den", "Speichern", "Teilen", "Auswählen", "Beobachte", "Tolle Angebote", "Buchen Sie", "Eigenständiger Transfer", "Durch KI"}
	noiseExact := map[string]bool{
		"Werbung": true, "": true, "0": true, "1": true, "2": true, "3": true, "4": true, "5": true,
		"Idealer Zeitpunkt dank Preisbeobachtung": true, "Intelligente Filter": true,
	}

	isNoise := func(s string) bool {
		if noiseExact[s] {
			return true
		}
		for _, prefix := range noisePrefixes {
			if strings.HasPrefix(s, prefix) {
				return true
			}
		}
		return false
	}

	for i := range lines {
		line := strings.TrimSpace(lines[i])
		if !reTime.MatchString(line) {
			continue
		}

		parts := regexp.MustCompile(`\s*[–-]\s*`).Split(line, 2)
		if len(parts) != 2 {
			continue
		}
		dep, arr := parts[0], parts[1]

		if i+3 >= len(lines) {
			continue
		}
		origin := strings.TrimSpace(lines[i+1])
		dash := strings.TrimSpace(lines[i+2])
		if dash != "-" {
			continue
		}
		dest := strings.TrimSpace(lines[i+3])

		label := ""
		if i > 0 {
			prev := strings.TrimSpace(lines[i-1])
			for _, kw := range labelKeywords {
				if strings.Contains(prev, kw) {
					label = kw
					break
				}
			}
		}

		j := i + 4
		stops := ""
		layover := ""
		for j < len(lines) {
			l := strings.TrimSpace(lines[j])
			if reStops.MatchString(l) {
				stops = l
				j++
				continue
			}
			if isDuration(l) {
				break
			}
			if stops != "" && !strings.HasPrefix(l, "Gesamt:") && l != "/Person" && !rePrice.MatchString(l) {
				if layover != "" {
					layover += " "
				}
				layover += l
				j++
				continue
			}
			break
		}

		if j >= len(lines) {
			continue
		}

		duration := ""
		if isDuration(strings.TrimSpace(lines[j])) {
			duration = strings.TrimSpace(lines[j])
			j++
		}

		airline := ""
		if j < len(lines) {
			airline = strings.TrimSpace(lines[j])
			j++
		}

		var priceLines []string
		for j < len(lines) {
			l := strings.TrimSpace(lines[j])
			if l == "/Person" {
				j++
				break
			}
			priceLines = append(priceLines, l)
			j++
		}

		var price, total, fareClass string
		for _, pl := range priceLines {
			pl = strings.TrimSpace(pl)
			if rePrice.MatchString(pl) && price == "" {
				price = pl
				continue
			}
			if reTotal.MatchString(pl) {
				total = pl
				continue
			}
			if isNoise(pl) {
				continue
			}
			if fareClass == "" {
				fareClass = pl
			}
		}

		if j < len(lines) {
			l := strings.TrimSpace(lines[j])
			if reTotal.MatchString(l) {
				total = l
			}
		}

		if price == "" && total == "" {
			continue
		}

		priceRaw := price
		if priceRaw == "" {
			priceRaw = total
		}

		flights = append(flights, Flight{
			Departure:   dep,
			Arrival:     arr,
			Origin:      origin,
			Destination: dest,
			Stops:       stops,
			Layover:     strings.TrimSpace(layover),
			Duration:    duration,
			DurationMin: DurMinutes(duration),
			Airline:     airline,
			Price:       price,
			PriceRaw:    priceRaw,
			PriceFloat:  ExtractPrice(price),
			Total:       total,
			TotalFloat:  ExtractPrice(total),
			FareClass:   fareClass,
			Label:       label,
		})
	}

	sort.Slice(flights, func(i, j int) bool {
		fi := flights[i].TotalFloat
		fj := flights[j].TotalFloat
		if fi == 0 {
			fi = flights[i].PriceFloat
		}
		if fj == 0 {
			fj = flights[j].PriceFloat
		}
		if fi != fj {
			return fi < fj
		}
		return flights[i].DurationMin < flights[j].DurationMin
	})

	log.Printf("[Momondo] ParseMomondoFlights: found %d flights in %v", len(flights), time.Since(parseStart).Round(time.Millisecond))
	if len(flights) == 0 {
		preview := body
		if len(preview) > 500 {
			preview = preview[:500]
		}
		log.Printf("[Momondo] ParseMomondoFlights: no flights found, first 500 chars of body: %q", preview)
	}

	return flights
}

func ExtractPrice(s string) float64 {
	s = strings.TrimPrefix(s, "Gesamt:")
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "€")
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, ".", "")
	s = strings.ReplaceAll(s, ",", ".")
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}

func DurMinutes(s string) int {
	s = strings.TrimSpace(s)

	if m := regexp.MustCompile(`^(\d{1,2}):(\d{2})\s+Std\.$`).FindStringSubmatch(s); m != nil {
		h, _ := strconv.Atoi(m[1])
		mm, _ := strconv.Atoi(m[2])
		return h*60 + mm
	}
	if m := regexp.MustCompile(`^(\d+)\s+Std\.\s+(\d+)\s+Min\.$`).FindStringSubmatch(s); m != nil {
		h, _ := strconv.Atoi(m[1])
		mm, _ := strconv.Atoi(m[2])
		return h*60 + mm
	}
	return 0
}

type PriceEntry struct {
	Timestamp   time.Time `json:"timestamp"`
	TaskID      string    `json:"task_id,omitempty"`
	Origin      string    `json:"origin"`
	Destination string    `json:"destination"`
	Date        string    `json:"date"`
	Adults      int       `json:"adults"`
	Count       int       `json:"count"`
	Cheapest    *Flight   `json:"cheapest,omitempty"`
	Fastest     *Flight   `json:"fastest,omitempty"`
}

func AppendPriceHistory(entry PriceEntry) error {
	dir := ".phubot"
	os.MkdirAll(dir, 0755)

	path := filepath.Join(dir, "price_history.jsonl")

	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("marshal price entry: %w", err)
	}

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("open price history: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write price entry: %w", err)
	}

	log.Printf("[Momondo] Price history appended: %s→%s %s, %d flights, cheapest=%v", entry.Origin, entry.Destination, entry.Date, entry.Count, func() string {
		if entry.Cheapest != nil {
			return entry.Cheapest.Total
		}
		return "n/a"
	}())
	return nil
}

func LoadPriceHistory() ([]PriceEntry, error) {
	path := filepath.Join(".phubot", "price_history.jsonl")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var entries []PriceEntry
	skipped := 0
	for line := range strings.SplitSeq(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var entry PriceEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			skipped++
			continue
		}
		entries = append(entries, entry)
	}
	log.Printf("[Momondo] Loaded price history: %d entries, %d skipped", len(entries), skipped)
	return entries, nil
}
