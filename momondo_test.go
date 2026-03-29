package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestExtractPrice_SimpleEuro(t *testing.T) {
	result := ExtractPrice("59 €")
	if result != 59.0 {
		t.Fatalf("expected 59.0, got %f", result)
	}
}

func TestExtractPrice_GermanThousands(t *testing.T) {
	result := ExtractPrice("1.992 €")
	if result != 1992.0 {
		t.Fatalf("expected 1992.0, got %f", result)
	}
}

func TestExtractPrice_GermanDecimal(t *testing.T) {
	result := ExtractPrice("59,50 €")
	if result != 59.5 {
		t.Fatalf("expected 59.5, got %f", result)
	}
}

func TestExtractPrice_ThousandsAndDecimal(t *testing.T) {
	result := ExtractPrice("1.234,56 €")
	if result != 1234.56 {
		t.Fatalf("expected 1234.56, got %f", result)
	}
}

func TestExtractPrice_GesamtPrefix(t *testing.T) {
	result := ExtractPrice("Gesamt: 1.992 €")
	if result != 1992.0 {
		t.Fatalf("expected 1992.0, got %f", result)
	}
}

func TestExtractPrice_GesamtDecimal(t *testing.T) {
	result := ExtractPrice("Gesamt: 1.234,50 €")
	if result != 1234.5 {
		t.Fatalf("expected 1234.5, got %f", result)
	}
}

func TestExtractPrice_Empty(t *testing.T) {
	result := ExtractPrice("")
	if result != 0 {
		t.Fatalf("expected 0, got %f", result)
	}
}

func TestExtractPrice_NoNumber(t *testing.T) {
	result := ExtractPrice("€")
	if result != 0 {
		t.Fatalf("expected 0, got %f", result)
	}
}

func TestDurMinutes_ShortFormat(t *testing.T) {
	result := DurMinutes("1:30 Std.")
	if result != 90 {
		t.Fatalf("expected 90, got %d", result)
	}
}

func TestDurMinutes_LongFormat(t *testing.T) {
	result := DurMinutes("12 Std. 30 Min.")
	if result != 750 {
		t.Fatalf("expected 750, got %d", result)
	}
}

func TestDurMinutes_Zero(t *testing.T) {
	result := DurMinutes("0:00 Std.")
	if result != 0 {
		t.Fatalf("expected 0, got %d", result)
	}
}

func TestDurMinutes_Empty(t *testing.T) {
	result := DurMinutes("")
	if result != 0 {
		t.Fatalf("expected 0, got %d", result)
	}
}

func TestDurMinutes_InvalidFormat(t *testing.T) {
	result := DurMinutes("about 2 hours")
	if result != 0 {
		t.Fatalf("expected 0, got %d", result)
	}
}

func TestBuildMomondoURL_Basic(t *testing.T) {
	url := buildMomondoURL("HKT", "FRA", "2026-05-01", "", 2, "", "price")
	expected := "https://www.momondo.de/flight-search/HKT-FRA/2026-05-01/2adults?sort=price_a"
	if url != expected {
		t.Fatalf("expected %q, got %q", expected, url)
	}
}

func TestBuildMomondoURL_WithReturn(t *testing.T) {
	url := buildMomondoURL("HKT", "FRA", "2026-05-01", "2026-05-15", 2, "", "best")
	if url != "https://www.momondo.de/flight-search/HKT-FRA/2026-05-01/2adults/2026-05-15?sort=bestflight_a" {
		t.Fatalf("unexpected URL: %s", url)
	}
}

func TestBuildMomondoURL_WithChildren(t *testing.T) {
	url := buildMomondoURL("HKT", "FRA", "2026-05-01", "", 2, "10-10", "duration")
	if url != "https://www.momondo.de/flight-search/HKT-FRA/2026-05-01/2adults/children-10-10?sort=duration_a" {
		t.Fatalf("unexpected URL: %s", url)
	}
}

func TestBuildMomondoURL_DefaultSort(t *testing.T) {
	url := buildMomondoURL("HKT", "KUL", "2026-06-01", "", 1, "", "")
	if url != "https://www.momondo.de/flight-search/HKT-KUL/2026-06-01/1adults?sort=bestflight_a" {
		t.Fatalf("unexpected URL: %s", url)
	}
}

func TestParseMomondoFlights_SingleFlight(t *testing.T) {
	body := `19:10 – 21:40
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
Batik Air
59 €
/Person
Gesamt: 118 €`

	flights := ParseMomondoFlights(body)
	if len(flights) != 1 {
		t.Fatalf("expected 1 flight, got %d", len(flights))
	}

	f := flights[0]
	if f.Departure != "19:10" {
		t.Errorf("departure: expected '19:10', got %q", f.Departure)
	}
	if f.Arrival != "21:40" {
		t.Errorf("arrival: expected '21:40', got %q", f.Arrival)
	}
	if f.Origin != "Phuket International" {
		t.Errorf("origin: expected 'Phuket International', got %q", f.Origin)
	}
	if f.Destination != "Kuala Lumpur International" {
		t.Errorf("destination: expected 'Kuala Lumpur International', got %q", f.Destination)
	}
	if f.Stops != "Nonstop" {
		t.Errorf("stops: expected 'Nonstop', got %q", f.Stops)
	}
	if f.Duration != "1:30 Std." {
		t.Errorf("duration: expected '1:30 Std.', got %q", f.Duration)
	}
	if f.DurationMin != 90 {
		t.Errorf("duration_min: expected 90, got %d", f.DurationMin)
	}
	if f.Airline != "Batik Air" {
		t.Errorf("airline: expected 'Batik Air', got %q", f.Airline)
	}
	if f.Price != "59 €" {
		t.Errorf("price: expected '59 €', got %q", f.Price)
	}
	if f.PriceFloat != 59.0 {
		t.Errorf("price_float: expected 59.0, got %f", f.PriceFloat)
	}
	if f.Total != "Gesamt: 118 €" {
		t.Errorf("total: expected 'Gesamt: 118 €', got %q", f.Total)
	}
	if f.TotalFloat != 118.0 {
		t.Errorf("total_float: expected 118.0, got %f", f.TotalFloat)
	}
}

func TestParseMomondoFlights_MultipleFlights(t *testing.T) {
	body := `Beste Option
10:00 – 12:30
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
AirAsia
45 €
/Person
Gesamt: 90 €
19:10 – 21:40
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
Batik Air
59 €
/Person
Gesamt: 118 €`

	flights := ParseMomondoFlights(body)
	if len(flights) != 2 {
		t.Fatalf("expected 2 flights, got %d", len(flights))
	}

	if flights[0].TotalFloat > flights[1].TotalFloat {
		t.Errorf("flights should be sorted by price: first=%f, second=%f", flights[0].TotalFloat, flights[1].TotalFloat)
	}
}

func TestParseMomondoFlights_WithStop(t *testing.T) {
	body := `08:00 – 16:30
Phuket International
-
Frankfurt am Main
1 Stopp
Singapur (SIN)2:45 Std.
12 Std. 30 Min.
Singapore Airlines
450 €
/Person
Gesamt: 900 €`

	flights := ParseMomondoFlights(body)
	if len(flights) != 1 {
		t.Fatalf("expected 1 flight, got %d", len(flights))
	}

	f := flights[0]
	if f.Stops != "1 Stopp" {
		t.Errorf("stops: expected '1 Stopp', got %q", f.Stops)
	}
	if f.Duration != "12 Std. 30 Min." {
		t.Errorf("duration: expected '12 Std. 30 Min.', got %q", f.Duration)
	}
	if f.DurationMin != 750 {
		t.Errorf("duration_min: expected 750, got %d", f.DurationMin)
	}
	if f.Layover == "" {
		t.Error("expected layover info")
	}
}

func TestParseMomondoFlights_EmptyBody(t *testing.T) {
	flights := ParseMomondoFlights("")
	if len(flights) != 0 {
		t.Fatalf("expected 0 flights, got %d", len(flights))
	}
}

func TestParseMomondoFlights_NoPrices(t *testing.T) {
	body := `19:10 – 21:40
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
Batik Air`

	flights := ParseMomondoFlights(body)
	if len(flights) != 0 {
		t.Fatalf("expected 0 flights without prices, got %d", len(flights))
	}
}

func TestParseMomondoFlights_NoFlightsAtAll(t *testing.T) {
	body := `Leider keine Ergebnisse für Ihre Suche gefunden.
Bitte versuchen Sie es mit anderen Daten.`

	flights := ParseMomondoFlights(body)
	if len(flights) != 0 {
		t.Fatalf("expected 0 flights, got %d", len(flights))
	}
}

func TestParseMomondoFlights_SortedByPrice(t *testing.T) {
	body := `19:10 – 21:40
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
Batik Air
59 €
/Person
Gesamt: 118 €
10:00 – 12:30
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
AirAsia
45 €
/Person
Gesamt: 90 €`

	flights := ParseMomondoFlights(body)
	if len(flights) != 2 {
		t.Fatalf("expected 2 flights, got %d", len(flights))
	}
	if flights[0].TotalFloat != 90.0 {
		t.Errorf("first flight should be cheapest (90), got %f", flights[0].TotalFloat)
	}
	if flights[1].TotalFloat != 118.0 {
		t.Errorf("second flight should be 118, got %f", flights[1].TotalFloat)
	}
}

func TestParseMomondoFlights_LabelDetection(t *testing.T) {
	body := `Günstigste Option
19:10 – 21:40
Phuket International
-
Kuala Lumpur International
Nonstop
1:30 Std.
Batik Air
59 €
/Person
Gesamt: 118 €`

	flights := ParseMomondoFlights(body)
	if len(flights) != 1 {
		t.Fatalf("expected 1 flight, got %d", len(flights))
	}
	if flights[0].Label != "Günstigste Option" {
		t.Errorf("label: expected 'Günstigste Option', got %q", flights[0].Label)
	}
}

func TestMomondoFlightTool_Name(t *testing.T) {
	tool := &MomondoFlightTool{}
	if tool.Name() != "search_flights" {
		t.Fatalf("expected 'search_flights', got %q", tool.Name())
	}
}

func TestMomondoFlightTool_Definition(t *testing.T) {
	tool := &MomondoFlightTool{}
	def := tool.Definition()

	if def.Function == nil {
		t.Fatal("expected function definition")
	}
	if def.Function.Name != "search_flights" {
		t.Fatalf("expected 'search_flights', got %q", def.Function.Name)
	}

	params := def.Function.Parameters.(map[string]any)
	properties := params["properties"].(map[string]any)

	requiredFields := []string{"origin", "destination", "date"}
	for _, field := range requiredFields {
		if _, exists := properties[field]; !exists {
			t.Fatalf("missing required property %q", field)
		}
	}
}

func TestMomondoFlightTool_Execute_MissingRequired(t *testing.T) {
	tool := &MomondoFlightTool{}
	_, err := tool.Execute(`{"origin": "HKT"}`)
	if err == nil {
		t.Fatal("expected error for missing destination and date")
	}
}

func TestMomondoFlightTool_Execute_MalformedJSON(t *testing.T) {
	tool := &MomondoFlightTool{}
	_, err := tool.Execute("not json")
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestMomondoFlightTool_ImplementsTool(t *testing.T) {
	var _ Tool = &MomondoFlightTool{}
}

func TestAppendPriceHistory_CreatesFile(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(origDir)

	entry := PriceEntry{
		Timestamp:   time.Now().Truncate(time.Millisecond),
		Origin:      "HKT",
		Destination: "FRA",
		Date:        "2026-05-01",
		Adults:      2,
		Count:       5,
	}

	if err := AppendPriceHistory(entry); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	path := filepath.Join(".phubot", "price_history.jsonl")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("expected price_history.jsonl to be created")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}

	var loaded PriceEntry
	if err := json.Unmarshal(data[:len(data)-1], &loaded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if loaded.Origin != "HKT" || loaded.Destination != "FRA" || loaded.Count != 5 {
		t.Fatalf("entry mismatch: %+v", loaded)
	}
}

func TestAppendPriceHistory_AppendsMultiple(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(origDir)

	for i := range 3 {
		entry := PriceEntry{
			Timestamp:   time.Now().Truncate(time.Millisecond),
			Origin:      "HKT",
			Destination: "FRA",
			Date:        "2026-05-01",
			Adults:      2,
			Count:       i + 1,
		}
		if err := AppendPriceHistory(entry); err != nil {
			t.Fatalf("append %d failed: %v", i, err)
		}
	}

	entries, err := LoadPriceHistory()
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}
	if entries[0].Count != 1 || entries[1].Count != 2 || entries[2].Count != 3 {
		t.Fatalf("order wrong: %+v", entries)
	}
}

func TestLoadPriceHistory_NoFile(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(origDir)

	entries, err := LoadPriceHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected 0 entries, got %d", len(entries))
	}
}

func TestLoadPriceHistory_SkipsBadLines(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	os.Chdir(dir)
	defer os.Chdir(origDir)

	os.MkdirAll(".phubot", 0755)
	path := filepath.Join(".phubot", "price_history.jsonl")
	content := `{"timestamp":"2026-01-01T00:00:00Z","origin":"HKT","destination":"FRA","count":1}
bad line
{"timestamp":"2026-01-02T00:00:00Z","origin":"HKT","destination":"KUL","count":2}

`
	os.WriteFile(path, []byte(content), 0644)

	entries, err := LoadPriceHistory()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 valid entries, got %d", len(entries))
	}
	if entries[0].Destination != "FRA" || entries[1].Destination != "KUL" {
		t.Fatalf("entries wrong: %+v", entries)
	}
}
