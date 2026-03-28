package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadConfig_FileNotFound(t *testing.T) {
	cfg, path, err := findAndLoadConfig("")
	if err != nil {
		t.Fatalf("expected no error for auto-discovery with no file, got: %v", err)
	}
	if path != "" {
		t.Errorf("expected empty path, got %q", path)
	}
	if cfg.LLM.BaseURL != "http://127.0.0.1:1234/v1" {
		t.Errorf("expected default base URL, got %q", cfg.LLM.BaseURL)
	}
	if cfg.LLM.Model != "qwen3.5-9b-mlx" {
		t.Errorf("expected default model, got %q", cfg.LLM.Model)
	}
}

func TestLoadConfig_ValidFile(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")

	content := `{
		"llm": {
			"base_url": "http://localhost:9999/v1",
			"api_key": "sk-test-key",
			"model": "my-model"
		},
		"telegram": {
			"token": "123456:ABC-DEF",
			"allowed_users": "111,222"
		},
		"agent": {
			"context_window": 8000,
			"tool_timeout": "10s",
			"reserve_ratio": 0.5,
			"keep_recent_ratio": 0.7,
			"pruning_mode": "off"
		}
	}`
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	if cfg.LLM.BaseURL != "http://localhost:9999/v1" {
		t.Errorf("expected custom base URL, got %q", cfg.LLM.BaseURL)
	}
	if cfg.LLM.APIKey != "sk-test-key" {
		t.Errorf("expected custom API key, got %q", cfg.LLM.APIKey)
	}
	if cfg.LLM.Model != "my-model" {
		t.Errorf("expected custom model, got %q", cfg.LLM.Model)
	}
	if cfg.Telegram.Token != "123456:ABC-DEF" {
		t.Errorf("expected telegram token, got %q", cfg.Telegram.Token)
	}
	if cfg.Telegram.AllowedUsers != "111,222" {
		t.Errorf("expected allowed users '111,222', got %q", cfg.Telegram.AllowedUsers)
	}
	if cfg.Agent.ContextWindow != 8000 {
		t.Errorf("expected context window 8000, got %d", cfg.Agent.ContextWindow)
	}
	if cfg.Agent.ToolTimeout.ToDuration() != 10*time.Second {
		t.Errorf("expected tool timeout 10s, got %v", cfg.Agent.ToolTimeout)
	}
	if cfg.Agent.ReserveRatio != 0.5 {
		t.Errorf("expected reserve ratio 0.5, got %f", cfg.Agent.ReserveRatio)
	}
}

func TestLoadConfig_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte("{invalid}"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := loadConfig(cfgPath)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoadConfig_PartialOverride(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")

	content := `{"llm": {"model": "custom-model"}}`
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.LLM.Model != "custom-model" {
		t.Errorf("expected custom model, got %q", cfg.LLM.Model)
	}
	if cfg.LLM.BaseURL != "http://127.0.0.1:1234/v1" {
		t.Errorf("expected default base URL for unset field, got %q", cfg.LLM.BaseURL)
	}
	if cfg.Agent.ContextWindow != 40000 {
		t.Errorf("expected default context window, got %d", cfg.Agent.ContextWindow)
	}
}

func TestLoadConfig_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.LLM.Model != "qwen3.5-9b-mlx" {
		t.Errorf("expected default model, got %q", cfg.LLM.Model)
	}
}

func TestConfig_EnvOverrides(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")

	content := `{
		"llm": {"api_key": "from-file"},
		"telegram": {"token": "from-file-token"}
	}`
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	os.Setenv("LM_STUDIO_API_KEY", "from-env")
	defer os.Unsetenv("LM_STUDIO_API_KEY")

	os.Setenv("TELEGRAM_TOKEN", "from-env-token")
	defer os.Unsetenv("TELEGRAM_TOKEN")

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.LLM.APIKey != "from-env" {
		t.Errorf("env should override file, got %q", cfg.LLM.APIKey)
	}
	if cfg.Telegram.Token != "from-env-token" {
		t.Errorf("env should override file, got %q", cfg.Telegram.Token)
	}
}

func TestConfig_EnvURL(t *testing.T) {
	os.Setenv("LM_STUDIO_URL", "http://custom:8080/v1")
	defer os.Unsetenv("LM_STUDIO_URL")

	cfg, err := loadConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.BaseURL != "http://custom:8080/v1" {
		t.Errorf("expected env URL, got %q", cfg.LLM.BaseURL)
	}
}

func TestConfig_EnvAllowedUsers(t *testing.T) {
	os.Setenv("ALLOWED_USERS", "100,200,300")
	defer os.Unsetenv("ALLOWED_USERS")

	cfg, err := loadConfig("")
	if err != nil {
		t.Fatal(err)
	}

	ids := cfg.AllowedUserIDs()
	if len(ids) != 3 {
		t.Fatalf("expected 3 user IDs, got %d", len(ids))
	}
	if ids[0] != 100 || ids[1] != 200 || ids[2] != 300 {
		t.Errorf("expected [100,200,300], got %v", ids)
	}
}

func TestConfig_AllowedUserIDs_Empty(t *testing.T) {
	cfg := &Config{}
	ids := cfg.AllowedUserIDs()
	if len(ids) != 0 {
		t.Errorf("expected empty slice, got %v", ids)
	}
}

func TestConfig_AllowedUserIDs_FromFile(t *testing.T) {
	cfg := &Config{
		Telegram: TelegramConfig{
			AllowedUsers: "42, 99",
		},
	}
	ids := cfg.AllowedUserIDs()
	if len(ids) != 2 || ids[0] != 42 || ids[1] != 99 {
		t.Errorf("expected [42, 99], got %v", ids)
	}
}

func TestConfig_PruningConfig_Off(t *testing.T) {
	cfg := &Config{Agent: AgentConfig{PruningMode: "off"}}
	pc := cfg.PruningConfig()
	if pc.Mode != "off" {
		t.Errorf("expected off mode, got %q", pc.Mode)
	}
}

func TestConfig_PruningConfig_Conservative(t *testing.T) {
	cfg := &Config{Agent: AgentConfig{PruningMode: "conservative"}}
	pc := cfg.PruningConfig()
	if pc.Mode != "conservative" {
		t.Errorf("expected conservative mode, got %q", pc.Mode)
	}
	if pc.SoftTrimRatio != 0.30 {
		t.Errorf("expected 0.30, got %f", pc.SoftTrimRatio)
	}
}

func TestConfig_PruningConfig_Aggressive(t *testing.T) {
	cfg := &Config{Agent: AgentConfig{PruningMode: "aggressive"}}
	pc := cfg.PruningConfig()
	if pc.Mode != "aggressive" {
		t.Errorf("expected aggressive mode, got %q", pc.Mode)
	}
	if pc.SoftTrimRatio != 0.20 {
		t.Errorf("expected 0.20, got %f", pc.SoftTrimRatio)
	}
}

func TestConfig_PruningConfig_DefaultFallback(t *testing.T) {
	cfg := &Config{Agent: AgentConfig{PruningMode: "unknown"}}
	pc := cfg.PruningConfig()
	if pc.Mode != "aggressive" {
		t.Errorf("unknown mode should fall back to aggressive, got %q", pc.Mode)
	}
}

func TestFindAndLoadConfig_ExplicitPath(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "myconfig.json")
	content := `{"llm": {"model": "explicit-test"}}`
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, path, err := findAndLoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	if path != cfgPath {
		t.Errorf("expected path %q, got %q", cfgPath, path)
	}
	if cfg.LLM.Model != "explicit-test" {
		t.Errorf("expected explicit-test model, got %q", cfg.LLM.Model)
	}
}

func TestFindAndLoadConfig_NoFile(t *testing.T) {
	cfg, path, err := findAndLoadConfig("")
	if err != nil {
		t.Fatal(err)
	}
	if path != "" {
		t.Errorf("expected empty path when no config found, got %q", path)
	}
	if cfg.LLM.Model != "qwen3.5-9b-mlx" {
		t.Errorf("expected default model, got %q", cfg.LLM.Model)
	}
}

func TestFindAndLoadConfig_ExplicitNotFound(t *testing.T) {
	_, _, err := findAndLoadConfig("/nonexistent/path/config.json")
	if err == nil {
		t.Error("expected error for explicit path that doesn't exist")
	}
}

func TestWriteExampleConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "example.json")

	if err := writeExampleConfig(cfgPath); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("example config should be valid JSON: %v", err)
	}

	if cfg.Telegram.Token != "7722275236:AAEkiziu5qMSRSGHyMtJJLAk0vl9jorb3c4" {
		t.Errorf("expected example telegram token, got %q", cfg.Telegram.Token)
	}
	if cfg.LLM.APIKey != "sk-your-api-key-here" {
		t.Errorf("expected placeholder API key, got %q", cfg.LLM.APIKey)
	}
}

func TestConfigUsage(t *testing.T) {
	usage := configUsage()
	if usage == "" {
		t.Error("expected non-empty usage string")
	}
	if len(usage) < 100 {
		t.Error("usage string seems too short")
	}
}

func TestConfig_DefaultsMatchConstants(t *testing.T) {
	cfg := defaultConfig

	if cfg.Agent.ContextWindow != DefaultContextWindow {
		t.Errorf("context window mismatch: %d vs %d", cfg.Agent.ContextWindow, DefaultContextWindow)
	}
	if cfg.Agent.ToolTimeout.ToDuration() != DefaultToolTimeout {
		t.Errorf("tool timeout mismatch: %v vs %v", cfg.Agent.ToolTimeout, DefaultToolTimeout)
	}
	if cfg.Agent.ReserveRatio != ReserveTokensRatio {
		t.Errorf("reserve ratio mismatch: %f vs %f", cfg.Agent.ReserveRatio, ReserveTokensRatio)
	}
	if cfg.Agent.KeepRecentRatio != KeepRecentTokensRatio {
		t.Errorf("keep recent ratio mismatch: %f vs %f", cfg.Agent.KeepRecentRatio, KeepRecentTokensRatio)
	}
	if cfg.WAL.Dir != WALDir {
		t.Errorf("WAL dir mismatch: %q vs %q", cfg.WAL.Dir, WALDir)
	}
	if cfg.WAL.File != WALFile {
		t.Errorf("WAL file mismatch: %q vs %q", cfg.WAL.File, WALFile)
	}
	if cfg.WAL.MaxSize != WALMaxSize {
		t.Errorf("WAL max size mismatch: %d vs %d", cfg.WAL.MaxSize, WALMaxSize)
	}
	if cfg.Memory.MaxSize != MemoryMaxSize {
		t.Errorf("memory max size mismatch: %d vs %d", cfg.Memory.MaxSize, MemoryMaxSize)
	}
	if cfg.Memory.FlushMinDelay.ToDuration() != CompactionMinDelay {
		t.Errorf("flush min delay mismatch: %v vs %v", cfg.Memory.FlushMinDelay, CompactionMinDelay)
	}
}

func TestConfig_DurationParsing(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	content := `{
		"agent": {"tool_timeout": "1m30s"},
		"memory": {"flush_min_delay": "10s"}
	}`
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Agent.ToolTimeout.ToDuration() != 90*time.Second {
		t.Errorf("expected 90s, got %v", cfg.Agent.ToolTimeout)
	}
	if cfg.Memory.FlushMinDelay.ToDuration() != 10*time.Second {
		t.Errorf("expected 10s, got %v", cfg.Memory.FlushMinDelay)
	}
}
