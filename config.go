package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type Duration time.Duration

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(data []byte) error {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	switch val := v.(type) {
	case float64:
		*d = Duration(time.Duration(val))
		return nil
	case string:
		parsed, err := time.ParseDuration(val)
		if err != nil {
			return fmt.Errorf("invalid duration %q: %w", val, err)
		}
		*d = Duration(parsed)
		return nil
	default:
		return fmt.Errorf("cannot unmarshal %T into Duration", val)
	}
}

func (d Duration) ToDuration() time.Duration {
	return time.Duration(d)
}

type Config struct {
	LLM      LLMConfig      `json:"llm"`
	Telegram TelegramConfig `json:"telegram"`
	Agent    AgentConfig    `json:"agent"`
	Memory   MemoryConfig   `json:"memory"`
	WAL      WALConfig      `json:"wal"`
	Models   []ModelConfig  `json:"models"`
}

type ModelConfig struct {
	Name    string `json:"name"`
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
	Model   string `json:"model"`
}

type LLMConfig struct {
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
	Model   string `json:"model"`
}

type TelegramConfig struct {
	Token        string `json:"token"`
	AllowedUsers string `json:"allowed_users"`
}

type AgentConfig struct {
	ContextWindow   int      `json:"context_window"`
	ToolTimeout     Duration `json:"tool_timeout"`
	ReserveRatio    float64  `json:"reserve_ratio"`
	KeepRecentRatio float64  `json:"keep_recent_ratio"`
	PruningMode     string   `json:"pruning_mode"`
}

type MemoryConfig struct {
	MaxSize       int      `json:"max_size"`
	FlushMinDelay Duration `json:"flush_min_delay"`
}

type WALConfig struct {
	Dir     string `json:"dir"`
	File    string `json:"file"`
	MaxSize int    `json:"max_size"`
}

var defaultConfig = Config{
	LLM: LLMConfig{
		BaseURL: "http://127.0.0.1:1234/v1",
		APIKey:  "",
		Model:   "qwen3.5-9b-mlx",
	},
	Telegram: TelegramConfig{
		Token:        "",
		AllowedUsers: "",
	},
	Agent: AgentConfig{
		ContextWindow:   40000,
		ToolTimeout:     Duration(30 * time.Second),
		ReserveRatio:    0.70,
		KeepRecentRatio: 0.85,
		PruningMode:     "aggressive",
	},
	Memory: MemoryConfig{
		MaxSize:       100 * 1024,
		FlushMinDelay: Duration(5 * time.Second),
	},
	WAL: WALConfig{
		Dir:     ".phubot",
		File:    "history.wal",
		MaxSize: 5 * 1024 * 1024,
	},
	Models: []ModelConfig{
		{
			Name:    "local",
			BaseURL: "http://127.0.0.1:1234/v1",
			APIKey:  "",
			Model:   "qwen3.5-9b-mlx",
		},
		{
			Name:    "glm5-turbo",
			BaseURL: "https://api.z.ai/api/coding/paas/v4",
			APIKey:  "",
			Model:   "glm-5.1",
		},
	},
}

func loadConfig(path string) (*Config, error) {
	cfg := defaultConfig

	if path == "" {
		cfg.applyEnvOverrides()
		return &cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("config file not found: %s", path)
		}
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	cfg.applyEnvOverrides()

	return &cfg, nil
}

func (c *Config) applyEnvOverrides() {
	if v := os.Getenv("LM_STUDIO_API_KEY"); v != "" {
		c.LLM.APIKey = v
	}
	if v := os.Getenv("LM_STUDIO_URL"); v != "" {
		c.LLM.BaseURL = v
	}
	if v := os.Getenv("TELEGRAM_TOKEN"); v != "" {
		c.Telegram.Token = v
	}
	if v := os.Getenv("ALLOWED_USERS"); v != "" {
		c.Telegram.AllowedUsers = v
	}
}

func configPaths() []string {
	wd, _ := os.Getwd()
	return []string{
		filepath.Join(wd, "config.json"),
		filepath.Join(wd, ".phubot", "config.json"),
		filepath.Join(os.Getenv("HOME"), ".config", "phubot", "config.json"),
	}
}

func findAndLoadConfig(explicitPath string) (*Config, string, error) {
	if explicitPath != "" {
		cfg, err := loadConfig(explicitPath)
		if err != nil {
			return nil, "", err
		}
		return cfg, explicitPath, nil
	}

	for _, p := range configPaths() {
		if _, err := os.Stat(p); err == nil {
			cfg, err := loadConfig(p)
			if err != nil {
				return nil, "", err
			}
			return cfg, p, nil
		}
	}

	cfg, err := loadConfig("")
	if err != nil {
		return nil, "", err
	}
	return cfg, "", nil
}

func (c *Config) AllowedUserIDs() []int64 {
	return parseAllowedUsers(c.Telegram.AllowedUsers)
}

func (c *Config) PruningConfig() PruningConfig {
	switch c.Agent.PruningMode {
	case "off":
		return PruningConfig{Mode: "off"}
	case "conservative":
		return PruningConfig{
			Mode:                 "conservative",
			SoftTrimRatio:        0.30,
			HardClearRatio:       0.50,
			SoftTrimMaxChars:     5000,
			SoftTrimHeadChars:    2000,
			SoftTrimTailChars:    2000,
			HardClearPlaceholder: "[tool result cleared]",
		}
	default:
		return DefaultPruningConfig
	}
}

func writeExampleConfig(path string) error {
	example := defaultConfig
	example.Telegram.Token = "7722275236:AAEkiziu5qMSRSGHyMtJJLAk0vl9jorb3c4"
	example.Telegram.AllowedUsers = "123456789,987654321"
	example.LLM.APIKey = "sk-your-api-key-here"

	data, err := json.MarshalIndent(example, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func configUsage() string {
	paths := configPaths()
	return fmt.Sprintf(`Config file search order:
  1. -config flag (explicit path)
  2. %s
  3. %s
  4. %s

Run 'phubot -init' to create an example config.json in the current directory.

Environment variables override config file:
  LM_STUDIO_API_KEY    LLM API key
  LM_STUDIO_URL        LLM base URL
  TELEGRAM_TOKEN       Telegram bot token
  ALLOWED_USERS        Comma-separated Telegram user IDs`,
		paths[0], paths[1], paths[2])
}
