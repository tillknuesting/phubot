package main

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
)

func processMemoryRSS(name string) (uint64, error) {
	cmd := exec.Command("sh", "-c", "ps aux")
	out, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	var total uint64
	for line := range strings.SplitSeq(string(out), "\n") {
		if !strings.Contains(line, name) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		var rss float64
		fmt.Sscanf(fields[5], "%f", &rss)
		total += uint64(rss * 1024)
	}
	return total, nil
}

func memStats() (alloc, sys uint64) {
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	return m.HeapAlloc, m.Sys
}

func mb(b uint64) string {
	return fmt.Sprintf("%.1f MB", float64(b)/1024/1024)
}

func TestBrowserMemoryUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory benchmark in short mode")
	}

	printMem := func(label string) {
		goAlloc, goSys := memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		t.Logf("%-35s  Go alloc: %s  Go sys: %s  Chrome: %s", label, mb(goAlloc), mb(goSys), mb(total))
	}

	printMem("[baseline] before Chrome")

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", true),
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
		chromedp.WindowSize(1920, 1080),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	printMem("[after alloc] allocator created")

	browserCtx, cancel := chromedp.NewContext(allocCtx)
	defer cancel()

	printMem("[after context] browser context created")

	timeoutCtx, cancelTimeout := context.WithTimeout(browserCtx, 60*time.Second)
	defer cancelTimeout()

	err := chromedp.Run(timeoutCtx, chromedp.Navigate("about:blank"))
	if err != nil {
		t.Fatalf("navigate to about:blank: %v", err)
	}
	printMem("[about:blank] empty tab loaded")

	err = chromedp.Run(timeoutCtx, chromedp.Navigate("https://www.google.com"))
	if err != nil {
		t.Fatalf("navigate to google: %v", err)
	}
	chromedp.Run(timeoutCtx, chromedp.Sleep(3*time.Second))
	printMem("[google.com] static page loaded")

	err = chromedp.Run(timeoutCtx, chromedp.Navigate("https://www.momondo.de"))
	if err != nil {
		t.Fatalf("navigate to momondo: %v", err)
	}
	chromedp.Run(timeoutCtx, chromedp.Sleep(5*time.Second))
	printMem("[momondo.de] JS-heavy page loaded")

	chromedp.Run(timeoutCtx, chromedp.Sleep(10*time.Second))
	printMem("[momondo.de] +10s settle")

	printMem("[before cancel] about to tear down")
	cancel()
	cancelTimeout()

	runtime.GC()
	time.Sleep(2 * time.Second)
	printMem("[after cancel] teardown complete")
}

func TestBrowserMemoryImagesDisabled(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory benchmark in short mode")
	}

	t.Log("=== disable-images vs default ===")

	printMem := func(label string) {
		goAlloc, _ := memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		t.Logf("%-40s  Go alloc: %s  Chrome: %s", label, mb(goAlloc), mb(total))
	}

	testPage := func(name, url string, disableImages bool) {
		opts := append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.Flag("headless", true),
			chromedp.Flag("disable-images", disableImages),
			chromedp.Flag("disable-blink-features", "AutomationControlled"),
			chromedp.Flag("disable-infobars", true),
			chromedp.Flag("no-first-run", true),
			chromedp.Flag("no-default-browser-check", true),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("disable-component-update", true),
			chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
			chromedp.Flag("window-size", "1920,1080"),
			chromedp.WindowSize(1920, 1080),
		)

		allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
		defer cancel()

		browserCtx, cancel2 := chromedp.NewContext(allocCtx)
		defer cancel2()

		timeoutCtx, cancel3 := context.WithTimeout(browserCtx, 30*time.Second)
		defer cancel3()

		chromedp.Run(timeoutCtx, chromedp.Navigate(url))
		chromedp.Run(timeoutCtx, chromedp.Sleep(5*time.Second))

		printMem(fmt.Sprintf("[%s] images=%v", name, disableImages))
	}

	testPage("momondo.de", "https://www.momondo.de", false)
	time.Sleep(1 * time.Second)
	testPage("momondo.de", "https://www.momondo.de", true)
	time.Sleep(1 * time.Second)
	testPage("google.com", "https://www.google.com", false)
	time.Sleep(1 * time.Second)
	testPage("google.com", "https://www.google.com", true)
}

func TestBrowserMemorySizeComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory benchmark in short mode")
	}

	t.Log("=== window size: 1920x1080 vs 800x600 ===")

	printMem := func(label string) {
		goAlloc, _ := memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		t.Logf("%-40s  Go alloc: %s  Chrome: %s", label, mb(goAlloc), mb(total))
	}

	sizes := []struct {
		name string
		w, h int
	}{
		{"1920x1080", 1920, 1080},
		{"1280x720", 1280, 720},
		{"800x600", 800, 600},
	}

	for _, s := range sizes {
		opts := append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.Flag("headless", true),
			chromedp.Flag("disable-extensions", false),
			chromedp.Flag("disable-images", false),
			chromedp.Flag("disable-infobars", true),
			chromedp.Flag("no-first-run", true),
			chromedp.Flag("no-default-browser-check", true),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("disable-component-update", true),
			chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
			chromedp.Flag("window-size", fmt.Sprintf("%d,%d", s.w, s.h)),
			chromedp.WindowSize(s.w, s.h),
		)

		allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
		defer cancel()

		browserCtx, cancel2 := chromedp.NewContext(allocCtx)
		defer cancel2()

		timeoutCtx, cancel3 := context.WithTimeout(browserCtx, 30*time.Second)
		defer cancel3()

		chromedp.Run(timeoutCtx, chromedp.Navigate("https://www.momondo.de"))
		chromedp.Run(timeoutCtx, chromedp.Sleep(5*time.Second))

		printMem(fmt.Sprintf("[%s] momondo.de", s.name))
		time.Sleep(1 * time.Second)
	}
}

func TestBrowserMemoryNavigationLeaks(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory benchmark in short mode")
	}

	t.Log("=== memory leak check: 10 consecutive navigations ===")

	printMem := func(label string) {
		goAlloc, _ := memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		t.Logf("%-40s  Go alloc: %s  Chrome: %s", label, mb(goAlloc), mb(total))
	}

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", true),
		chromedp.Flag("disable-images", true),
		chromedp.Flag("disable-infobars", true),
		chromedp.Flag("no-first-run", true),
		chromedp.Flag("no-default-browser-check", true),
		chromedp.Flag("disable-background-timer-throttling", true),
		chromedp.Flag("disable-renderer-backgrounding", true),
		chromedp.Flag("disable-component-update", true),
		chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
		chromedp.Flag("window-size", "1920,1080"),
		chromedp.WindowSize(1920, 1080),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	browserCtx, cancel2 := chromedp.NewContext(allocCtx)
	defer cancel2()

	timeoutCtx, cancel3 := context.WithTimeout(browserCtx, 120*time.Second)
	defer cancel3()

	urls := []string{
		"https://www.google.com",
		"https://en.wikipedia.org/wiki/Main_Page",
		"https://news.ycombinator.com",
		"https://www.reddit.com",
		"https://github.com",
		"https://www.momondo.de",
		"https://www.amazon.com",
		"https://www.youtube.com",
		"https://www.bbc.com",
		"https://www.momondo.de",
	}

	var readings []uint64
	for i, url := range urls {
		shortURL := url
		if len(shortURL) > 35 {
			shortURL = shortURL[:32] + "..."
		}
		chromedp.Run(timeoutCtx, chromedp.Navigate(url))
		chromedp.Run(timeoutCtx, chromedp.Sleep(3*time.Second))

		_, _ = memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		readings = append(readings, total)

		printMem(fmt.Sprintf("nav %2d: %-35s", i+1, shortURL))
	}

	if len(readings) >= 2 {
		first := float64(readings[0]) / 1024 / 1024
		last := float64(readings[len(readings)-1]) / 1024 / 1024
		peak := float64(readings[0])
		for _, r := range readings[1:] {
			if float64(r) > peak {
				peak = float64(r)
			}
		}
		growth := last - first
		t.Logf("=== Summary: first=%.0f MB  last=%.0f MB  peak=%.0f MB  growth=%.0f MB over %d navigations ===",
			first, last, peak/1024/1024, growth, len(readings))
	}
}

func TestBrowserMemoryMultipleTabs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory benchmark in short mode")
	}

	t.Log("=== memory scaling: 1, 2, 3, 4 concurrent tabs ===")

	printMem := func(label string) {
		goAlloc, _ := memStats()
		chromeRSS, _ := processMemoryRSS("chrome")
		chromeHelperRSS, _ := processMemoryRSS("Chromium")
		total := chromeRSS + chromeHelperRSS
		t.Logf("%-40s  Go alloc: %s  Chrome: %s", label, mb(goAlloc), mb(total))
	}

	opts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", true),
		chromedp.Flag("disable-images", true),
		chromedp.Flag("disable-infobars", true),
		chromedp.Flag("no-first-run", true),
		chromedp.Flag("no-default-browser-check", true),
		chromedp.Flag("disable-background-timer-throttling", true),
		chromedp.Flag("disable-renderer-backgrounding", true),
		chromedp.Flag("disable-component-update", true),
		chromedp.Flag("disable-features", "IsolateOrigins,site-per-process"),
		chromedp.Flag("window-size", "1920,1080"),
		chromedp.WindowSize(1920, 1080),
	)

	allocCtx, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()

	urls := []string{
		"https://www.google.com",
		"https://news.ycombinator.com",
		"https://www.momondo.de",
		"https://en.wikipedia.org/wiki/Main_Page",
	}

	for n := 1; n <= len(urls); n++ {
		var cancels []context.CancelFunc
		for i := 0; i < n; i++ {
			tabCtx, tabCancel := chromedp.NewContext(allocCtx)
			cancels = append(cancels, tabCancel)
			timeoutCtx, cancelTimeout := context.WithTimeout(tabCtx, 20*time.Second)

			chromedp.Run(timeoutCtx, chromedp.Navigate(urls[i]))
			chromedp.Run(timeoutCtx, chromedp.Sleep(2*time.Second))
			cancelTimeout()
		}

		printMem(fmt.Sprintf("%d tab(s) open", n))
		time.Sleep(1 * time.Second)

		for _, c := range cancels {
			c()
		}
		runtime.GC()
		time.Sleep(1 * time.Second)
	}
}
