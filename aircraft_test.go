package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestIdentifyAircraftTool_Name(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	if tool.Name() != "identify_aircraft" {
		t.Errorf("expected 'identify_aircraft', got %q", tool.Name())
	}
}

func TestIdentifyAircraftTool_Definition(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	def := tool.Definition()
	if def.Type != "function" {
		t.Error("expected function type")
	}
	if def.Function.Name != "identify_aircraft" {
		t.Errorf("expected identify_aircraft, got %q", def.Function.Name)
	}
	if def.Function.Parameters == nil {
		t.Error("expected parameters to be defined")
	}
}

func TestIdentifyAircraftTool_Execute_MissingParams(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	_, err := tool.Execute(`{}`)
	if err == nil {
		t.Fatal("expected error for missing params")
	}
	if !strings.Contains(err.Error(), "required") {
		t.Errorf("expected 'required' in error, got: %v", err)
	}
}

func TestIdentifyAircraftTool_Execute_InvalidJSON(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	_, err := tool.Execute(`not json`)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestIdentifyAircraftTool_ImplementsTool(t *testing.T) {
	var _ Tool = NewIdentifyAircraftTool(true)
}

func TestIdentifyAircraftTool_ImplementsToolWithProgress(t *testing.T) {
	var _ ToolWithProgress = NewIdentifyAircraftTool(true)
}

func TestIdentifyAircraftTool_SetProgressCallback(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	called := false
	tool.SetProgressCallback(func(msg string) {
		called = true
	})
	tool.progress("test")
	if !called {
		t.Error("expected progress callback to be called")
	}
}

func TestIdentifyAircraftTool_ProgressNoCallback(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	tool.progress("test")
}

func TestParseFlightAwareAircraft_FullNames(t *testing.T) {
	body := `FlightAware Routes: HKT to KUL

AK 701  AirAsia           Boeing 737-800    Daily
AK 705  AirAsia           Airbus A320        Mon Wed Fri
FD 301  Thai AirAsia      Airbus A320neo     Daily
MH 741  Malaysia Airlines Boeing 737-800    Daily
`

	entries := ParseFlightAwareAircraft(body)
	if len(entries) == 0 {
		t.Fatal("expected to parse aircraft entries")
	}

	found := make(map[string]bool)
	for _, e := range entries {
		found[e.AircraftType] = true
	}

	if !found["Boeing 737-800"] {
		t.Error("expected Boeing 737-800")
	}
	if !found["Airbus A320"] {
		t.Error("expected Airbus A320")
	}
	if !found["Airbus A320neo"] {
		t.Error("expected Airbus A320neo")
	}
}

func TestParseFlightAwareAircraft_ICAOCodes(t *testing.T) {
	body := `FlightAware Routes: FRA to JFK

LH 400  Lufthansa     B744  Daily
LH 710  Lufthansa     A346  Daily
BA 117  British Airways  B77W     Daily
`

	entries := ParseFlightAwareAircraft(body)
	if len(entries) == 0 {
		t.Fatal("expected to parse aircraft entries from ICAO codes")
	}

	found := make(map[string]bool)
	for _, e := range entries {
		found[e.AircraftType] = true
	}

	if !found["Boeing 747-400"] {
		t.Errorf("expected Boeing 747-400, got: %v", found)
	}
	if !found["Airbus A340-600"] {
		t.Errorf("expected Airbus A340-600, got: %v", found)
	}
	if !found["Boeing 777-300ER"] {
		t.Errorf("expected Boeing 777-300ER, got: %v", found)
	}
}

func TestParseFlightAwareAircraft_EmptyBody(t *testing.T) {
	entries := ParseFlightAwareAircraft("")
	if len(entries) != 0 {
		t.Errorf("expected 0 entries, got %d", len(entries))
	}
}

func TestParseFlightAwareAircraft_NoAircraft(t *testing.T) {
	body := "Welcome to FlightAware. Please log in to continue."
	entries := ParseFlightAwareAircraft(body)
	if len(entries) != 0 {
		t.Errorf("expected 0 entries for no aircraft text, got %d", len(entries))
	}
}

func TestParseFlightradar24Aircraft_FullNames(t *testing.T) {
	body := `Flightradar24 Flight history for route HKT-KUL

AirAsia AK701    Boeing 737-800    08:30    On time
AirAsia AK705    Airbus A320        12:15    On time
Thai AirAsia FD301  Airbus A320neo  14:00    On time
Malaysia Airlines MH741  Boeing 737-800  16:30  Landed
`

	entries := ParseFlightradar24Aircraft(body)
	if len(entries) == 0 {
		t.Fatal("expected to parse aircraft entries")
	}

	found := make(map[string]bool)
	for _, e := range entries {
		found[e.AircraftType] = true
	}

	if !found["Boeing 737-800"] {
		t.Error("expected Boeing 737-800")
	}
	if !found["Airbus A320"] {
		t.Error("expected Airbus A320")
	}
	if !found["Airbus A320neo"] {
		t.Error("expected Airbus A320neo")
	}
}

func TestParseFlightradar24Aircraft_ICAOCodes(t *testing.T) {
	body := `FR24 Data
SQ 321  Singapore Airlines  B77W  Daily
TG 921  Thai Airways        B789  Daily
EK 1    Emirates            A388  Daily
`

	entries := ParseFlightradar24Aircraft(body)
	if len(entries) == 0 {
		t.Fatal("expected to parse ICAO codes")
	}

	found := make(map[string]bool)
	for _, e := range entries {
		found[e.AircraftType] = true
	}

	if !found["Boeing 777-300ER"] {
		t.Errorf("expected Boeing 777-300ER, got: %v", found)
	}
	if !found["Boeing 787-9"] {
		t.Errorf("expected Boeing 787-9, got: %v", found)
	}
	if !found["Airbus A380"] {
		t.Errorf("expected Airbus A380, got: %v", found)
	}
}

func TestParseFlightradar24Aircraft_EmptyBody(t *testing.T) {
	entries := ParseFlightradar24Aircraft("")
	if len(entries) != 0 {
		t.Errorf("expected 0 entries, got %d", len(entries))
	}
}

func TestNormalizeAircraftType(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Boeing 737-800", "Boeing 737-800"},
		{"  Airbus A320  ", "Airbus A320"},
		{"", ""},
		{"  ", ""},
		{strings.Repeat("x", 70), strings.Repeat("x", 60)},
	}

	for _, tt := range tests {
		result := normalizeAircraftType(tt.input)
		if result != tt.expected {
			t.Errorf("normalizeAircraftType(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestExtractUniqueAircraftTypes(t *testing.T) {
	entries := []AircraftEntry{
		{AircraftType: "Boeing 737-800"},
		{AircraftType: "Airbus A320"},
		{AircraftType: "Boeing 737-800"},
		{AircraftType: "Airbus A320neo"},
	}

	types := extractUniqueAircraftTypes(entries)
	if len(types) != 3 {
		t.Fatalf("expected 3 unique types, got %d: %v", len(types), types)
	}

	seen := make(map[string]bool)
	for _, typ := range types {
		if seen[typ] {
			t.Errorf("duplicate type: %s", typ)
		}
		seen[typ] = true
	}
}

func TestExtractUniqueAircraftTypes_Empty(t *testing.T) {
	types := extractUniqueAircraftTypes(nil)
	if len(types) != 0 {
		t.Errorf("expected 0 types for nil, got %d", len(types))
	}
}

func TestFormatAircraftResult_NoEntries(t *testing.T) {
	result := AircraftResult{Origin: "HKT", Destination: "KUL"}
	s := formatAircraftResult(result, "")
	if !strings.Contains(s, "No aircraft data found") {
		t.Errorf("expected 'no data' message, got: %s", s)
	}
}

func TestFormatAircraftResult_WithEntries(t *testing.T) {
	result := AircraftResult{
		Origin:      "HKT",
		Destination: "KUL",
		Source:      "FlightAware",
		Entries: []AircraftEntry{
			{Airline: "AirAsia", AircraftType: "Boeing 737-800", FlightNumber: "AK701"},
			{Airline: "AirAsia", AircraftType: "Airbus A320"},
		},
		AircraftTypes: []string{"Boeing 737-800", "Airbus A320"},
	}

	s := formatAircraftResult(result, "")
	if !strings.Contains(s, "Boeing 737-800") {
		t.Error("expected Boeing 737-800 in output")
	}
	if !strings.Contains(s, "AK701") {
		t.Error("expected flight number AK701 in output")
	}
	if !strings.Contains(s, "FlightAware") {
		t.Error("expected source FlightAware in output")
	}
}

func TestFormatAircraftResult_AirlineFilter(t *testing.T) {
	result := AircraftResult{
		Origin:      "FRA",
		Destination: "JFK",
		Source:      "FlightAware",
		Entries: []AircraftEntry{
			{Airline: "Lufthansa", AircraftType: "Boeing 747-400"},
			{Airline: "Singapore Airlines", AircraftType: "Airbus A380"},
		},
	}

	s := formatAircraftResult(result, "Lufthansa")
	if strings.Contains(s, "Singapore Airlines") {
		t.Error("expected Singapore Airlines to be filtered out")
	}
	if !strings.Contains(s, "Lufthansa") {
		t.Error("expected Lufthansa to remain")
	}
}

func TestFormatAircraftResult_AirlineFilterFallback(t *testing.T) {
	result := AircraftResult{
		Origin:      "FRA",
		Destination: "JFK",
		Source:      "FlightAware",
		Entries: []AircraftEntry{
			{Airline: "Lufthansa", AircraftType: "Boeing 747-400"},
		},
	}

	s := formatAircraftResult(result, "NonExistent")
	if !strings.Contains(s, "Lufthansa") {
		t.Error("expected fallback to all entries when filter matches nothing")
	}
}

func TestFormatAircraftResult_Truncation(t *testing.T) {
	var entries []AircraftEntry
	for i := range 15 {
		entries = append(entries, AircraftEntry{
			Airline:      "Airline",
			AircraftType: "Boeing 737-800",
		})
		entries[i].FlightNumber = fmt.Sprintf("FL%d", i)
	}

	result := AircraftResult{
		Origin:      "HKT",
		Destination: "KUL",
		Source:      "test",
		Entries:     entries,
	}

	s := formatAircraftResult(result, "")
	if !strings.Contains(s, "... and 5 more") {
		t.Error("expected truncation message for >10 entries")
	}
}

func TestIdentifyAircraftTool_CacheHit(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	key := "HKT-KUL"
	result := AircraftResult{
		Origin:        "HKT",
		Destination:   "KUL",
		Source:        "test",
		AircraftTypes: []string{"Boeing 737-800"},
		Entries: []AircraftEntry{
			{Airline: "AirAsia", AircraftType: "Boeing 737-800"},
		},
	}
	tool.setCache(key, result)

	cached, ok := tool.getCached(key)
	if !ok {
		t.Fatal("expected cache hit")
	}
	if cached.Origin != "HKT" {
		t.Errorf("expected HKT, got %s", cached.Origin)
	}
}

func TestIdentifyAircraftTool_CacheMiss(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	_, ok := tool.getCached("NONEXISTENT")
	if ok {
		t.Error("expected cache miss")
	}
}

func TestIdentifyAircraftTool_CacheExpiry(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	tool.cacheTTL = 1

	key := "HKT-KUL"
	tool.setCache(key, AircraftResult{Origin: "HKT"})
	_, ok := tool.getCached(key)
	if ok {
		t.Error("expected cache to be expired")
	}
}

func TestICAOToName_Mappings(t *testing.T) {
	tests := []struct {
		code string
		name string
	}{
		{"B738", "Boeing 737-800"},
		{"A320", "Airbus A320"},
		{"A321", "Airbus A321"},
		{"B77W", "Boeing 777-300ER"},
		{"A388", "Airbus A380"},
		{"E190", "Embraer E190"},
		{"AT76", "ATR 72-600"},
		{"DH8D", "De Havilland Dash 8-400"},
	}

	for _, tt := range tests {
		name, ok := icaoToName[tt.code]
		if !ok {
			t.Errorf("no mapping for %s", tt.code)
			continue
		}
		if name != tt.name {
			t.Errorf("icaoToName[%s] = %q, want %q", tt.code, name, tt.name)
		}
	}
}

func TestExtractAirlineNearAircraft(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		aircraft string
		want     string
	}{
		{
			name:     "airline before aircraft",
			body:     "AK 701\nAirAsia\nBoeing 737-800\nDaily",
			aircraft: "Boeing 737-800",
			want:     "AirAsia",
		},
		{
			name:     "no airline",
			body:     "Boeing 737-800 is a popular aircraft",
			aircraft: "Boeing 737-800",
			want:     "",
		},
		{
			name:     "aircraft not found",
			body:     "Some random text",
			aircraft: "Boeing 737-800",
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractAirlineNearAircraft(tt.body, tt.aircraft)
			if got != tt.want {
				t.Errorf("extractAirlineNearAircraft() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestIsOnlyNumbers(t *testing.T) {
	tests := []struct {
		input string
		want  bool
	}{
		{"123", true},
		{"12 34", true},
		{"", true},
		{"ABC", false},
		{"12A", false},
		{"AirAsia", false},
	}

	for _, tt := range tests {
		result := isOnlyNumbers(tt.input)
		if result != tt.want {
			t.Errorf("isOnlyNumbers(%q) = %v, want %v", tt.input, result, tt.want)
		}
	}
}

func TestFilterEntriesByAirline(t *testing.T) {
	entries := []AircraftEntry{
		{Airline: "AirAsia"},
		{Airline: "Thai AirAsia"},
		{Airline: "Lufthansa"},
	}

	filtered := filterEntriesByAirline(entries, "airasia")
	if len(filtered) != 2 {
		t.Fatalf("expected 2 AirAsia entries, got %d", len(filtered))
	}
	for _, e := range filtered {
		if !strings.Contains(strings.ToLower(e.Airline), "airasia") {
			t.Errorf("unexpected entry: %s", e.Airline)
		}
	}
}

func TestIdentifyAircraftTool_DefinitionPropertiesHaveDescriptions(t *testing.T) {
	tool := NewIdentifyAircraftTool(true)
	def := tool.Definition()
	params, ok := def.Function.Parameters.(map[string]any)
	if !ok {
		t.Fatal("parameters is not a map")
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties is not a map")
	}
	for name, prop := range props {
		propMap, ok := prop.(map[string]any)
		if !ok {
			t.Errorf("property %q is not a map", name)
			continue
		}
		if desc, ok := propMap["description"].(string); !ok || desc == "" {
			t.Errorf("property %q missing description", name)
		}
	}
}

func TestAircraftResult_JSONRoundTrip(t *testing.T) {
	result := AircraftResult{
		Route:         "HKT-KUL",
		Origin:        "HKT",
		Destination:   "KUL",
		Source:        "FlightAware",
		AircraftTypes: []string{"Boeing 737-800", "Airbus A320"},
		Entries: []AircraftEntry{
			{Airline: "AirAsia", AircraftType: "Boeing 737-800", FlightNumber: "AK701", Source: "FlightAware"},
		},
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}

	var decoded AircraftResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Route != result.Route {
		t.Errorf("route mismatch: %q vs %q", decoded.Route, result.Route)
	}
	if len(decoded.AircraftTypes) != len(result.AircraftTypes) {
		t.Errorf("aircraft types count mismatch: %d vs %d", len(decoded.AircraftTypes), len(result.AircraftTypes))
	}
	if len(decoded.Entries) != len(result.Entries) {
		t.Fatalf("entries count mismatch: %d vs %d", len(decoded.Entries), len(result.Entries))
	}
	if decoded.Entries[0].Airline != "AirAsia" {
		t.Errorf("airline mismatch: %q", decoded.Entries[0].Airline)
	}
}
