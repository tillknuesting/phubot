package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

type TelegramBot struct {
	bot          *tgbotapi.BotAPI
	agent        *Agent
	scheduler    *Scheduler
	ctx          context.Context
	allowedUsers map[int64]bool
	recentMsgs   map[int]time.Time
	recentMsgsMu sync.Mutex
}

func NewTelegramBot(token string, agent *Agent, scheduler *Scheduler, ctx context.Context, allowedUserIDs []int64) (*TelegramBot, error) {
	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		return nil, fmt.Errorf("failed to create Telegram bot: %w", err)
	}

	allowed := make(map[int64]bool)
	for _, id := range allowedUserIDs {
		allowed[id] = true
	}

	log.Printf("[Telegram] Authorized on account %s", bot.Self.UserName)

	return &TelegramBot{
		bot:          bot,
		agent:        agent,
		scheduler:    scheduler,
		ctx:          ctx,
		allowedUsers: allowed,
		recentMsgs:   make(map[int]time.Time),
	}, nil
}

func (t *TelegramBot) isAllowed(userID int64) bool {
	if len(t.allowedUsers) == 0 {
		return true
	}
	return t.allowedUsers[userID]
}

func (t *TelegramBot) isDuplicate(msgID int) bool {
	t.recentMsgsMu.Lock()
	defer t.recentMsgsMu.Unlock()

	now := time.Now()
	if last, ok := t.recentMsgs[msgID]; ok && now.Sub(last) < 5*time.Second {
		return true
	}
	t.recentMsgs[msgID] = now

	if len(t.recentMsgs) > 1000 {
		cutoff := now.Add(-10 * time.Second)
		for id, ts := range t.recentMsgs {
			if ts.Before(cutoff) {
				delete(t.recentMsgs, id)
			}
		}
	}

	return false
}

func (t *TelegramBot) Run() error {
	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60

	updates := t.bot.GetUpdatesChan(u)

	for {
		select {
		case <-t.ctx.Done():
			return nil
		case update := <-updates:
			if update.Message == nil {
				continue
			}

			if !t.isAllowed(update.Message.From.ID) {
				log.Printf("[Telegram] Unauthorized access from user %d (%s)",
					update.Message.From.ID, update.Message.From.UserName)
				t.sendMessage(update.Message.Chat.ID, "Unauthorized")
				continue
			}

			if t.isDuplicate(update.Message.MessageID) {
				log.Printf("[Telegram] Duplicate message %d from %s, skipping",
					update.Message.MessageID, update.Message.From.UserName)
				continue
			}

			go t.handleMessage(update.Message)
		}
	}
}

func (t *TelegramBot) handleMessage(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID
	text := strings.TrimSpace(msg.Text)

	text = stripBotSuffix(text)

	if msg.Photo != nil && len(msg.Photo) > 0 {
		photo := msg.Photo[len(msg.Photo)-1]
		log.Printf("[Telegram] [%s] Photo received: %dx%d, FileID: %s", msg.From.UserName, photo.Width, photo.Height, photo.FileID)

		statusMsgID := t.sendMessage(chatID, "📸 Processing image...")

		done := make(chan struct{})
		go func() {
			ticker := time.NewTicker(4 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					t.sendTyping(chatID)
				}
			}
		}()

		t.editMessage(chatID, statusMsgID, "📥 Downloading image...")
		imageBase64, err := t.downloadPhotoAsBase64(photo.FileID)

		if err != nil {
			close(done)
			log.Printf("[Telegram] Photo download error: %v", err)
			t.editMessage(chatID, statusMsgID, fmt.Sprintf("❌ Failed to download photo: %v", err))
			return
		}

		log.Printf("[Telegram] Image downloaded, base64 length: %d", len(imageBase64))

		caption := msg.Caption
		if caption == "" {
			caption = "What do you see in this image?"
		}

		t.editMessage(chatID, statusMsgID, "🤖 Analyzing image with vision model...")
		reply, err := t.agent.ChatWithImage(t.ctx, caption, imageBase64)
		close(done)

		if err != nil {
			log.Printf("[Telegram] Vision error: %v", err)
			t.editMessage(chatID, statusMsgID, fmt.Sprintf("❌ Vision error: %v\n\nNote: Your LM Studio model may not support vision. Make sure to load a vision-capable model (e.g., llava, qwen-vl).", err))
			return
		}

		if reply == "" {
			t.editMessage(chatID, statusMsgID, "The model returned an empty response. It may not support vision.")
			return
		}

		if len(reply) > 4000 {
			t.editMessage(chatID, statusMsgID, reply[:4000])
			for i := 4000; i < len(reply); i += 4000 {
				end := min(i+4000, len(reply))
				t.sendMessage(chatID, reply[i:end])
			}
		} else {
			t.editMessage(chatID, statusMsgID, reply)
		}
		return
	}

	if text == "" {
		return
	}

	log.Printf("[Telegram] [%s] %s", msg.From.UserName, text)

	if text == "/start" {
		t.sendMessage(chatID, "👋 Hello! I'm your personal assistant. Ask me anything or use these commands:\n\n/model - List/switch models\n/tasks - List scheduled tasks\n/cancel <id> - Cancel a task\n/help - Show this message")
		return
	}

	if text == "/help" {
		t.sendMessage(chatID, "Commands:\n/model - List available models\n/model set <name> - Switch model\n/tasks - List scheduled tasks\n/cancel <id> - Cancel a scheduled task\n\nYou can ask me to do things periodically, e.g.:\n\"Check flight prices from HKT to FRA every 2 hours\"")
		return
	}

	if text == "/model" || text == "/model list" {
		models := t.agent.ListModels()
		active := t.agent.ActiveModelName()
		var sb strings.Builder
		sb.WriteString("Available models:\n\n")
		for _, m := range models {
			if m.Name == active {
				sb.WriteString(fmt.Sprintf("  * %s (%s @ %s) [active]\n", m.Name, m.Model, m.BaseURL))
			} else {
				sb.WriteString(fmt.Sprintf("  %s (%s @ %s)\n", m.Name, m.Model, m.BaseURL))
			}
		}
		sb.WriteString("\nUse /model set <name> to switch.")
		t.sendMessage(chatID, sb.String())
		return
	}

	if after, ok := strings.CutPrefix(text, "/model set "); ok {
		name := strings.TrimSpace(after)
		if name == "" {
			t.sendMessage(chatID, "Usage: /model set <name>")
			return
		}
		if err := t.agent.SwitchModel(name); err != nil {
			t.sendMessage(chatID, fmt.Sprintf("Error: %v", err))
		} else {
			t.sendMessage(chatID, fmt.Sprintf("Switched to model %q.", name))
		}
		return
	}

	if strings.HasPrefix(text, "/model") {
		t.sendMessage(chatID, "Usage:\n/model - List models\n/model set <name> - Switch model")
		return
	}

	if text == "/tasks" {
		tasks := t.scheduler.ListTasks()
		if len(tasks) == 0 {
			t.sendMessage(chatID, "No scheduled tasks.")
		} else {
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("📋 *Scheduled Tasks (%d):*\n\n", len(tasks)))
			for _, task := range tasks {
				status := "✅ active"
				if !task.Active {
					status = "❌ inactive"
				}
				sb.WriteString(fmt.Sprintf("*%s*\n  `%s`\n  Every %v | %s | Runs: %d\n\n",
					task.ID, task.Prompt, task.Interval, status, task.RunCount))
			}
			t.sendMarkdown(chatID, sb.String())
		}
		return
	}

	if after, ok := strings.CutPrefix(text, "/cancel "); ok {
		taskID := after
		if err := t.scheduler.Cancel(taskID); err != nil {
			t.sendMessage(chatID, fmt.Sprintf("❌ Error: %v", err))
		} else {
			t.sendMessage(chatID, fmt.Sprintf("✅ Task `%s` cancelled.", taskID))
		}
		return
	}

	statusMsgID := t.sendMessage(chatID, "🤔 Thinking...")

	var lastProgress time.Time
	progressCb := func(msg string) {
		if time.Since(lastProgress) < 2*time.Second {
			return
		}
		lastProgress = time.Now()
		t.editMessage(chatID, statusMsgID, msg)
	}
	t.agent.SetProgressCallback(progressCb)

	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(4 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				t.sendTyping(chatID)
			}
		}
	}()

	reply, err := t.agent.Chat(t.ctx, text)
	close(done)
	t.agent.SetProgressCallback(nil)

	if err != nil {
		log.Printf("[Telegram] Error: %v", err)
		t.editMessage(chatID, statusMsgID, fmt.Sprintf("❌ Error: %v", err))
		return
	}

	if len(reply) > 4000 {
		t.editMessage(chatID, statusMsgID, reply[:4000])
		for i := 4000; i < len(reply); i += 4000 {
			end := min(i+4000, len(reply))
			t.sendMessage(chatID, reply[i:end])
		}
	} else {
		t.editMessage(chatID, statusMsgID, reply)
	}

	t.agent.CompactInBackground(t.ctx)
}

func (t *TelegramBot) sendMessage(chatID int64, text string) int {
	msg := tgbotapi.NewMessage(chatID, text)
	sent, err := t.bot.Send(msg)
	if err != nil {
		log.Printf("[Telegram] Failed to send message: %v", err)
		return 0
	}
	return sent.MessageID
}

func (t *TelegramBot) editMessage(chatID int64, messageID int, text string) {
	msg := tgbotapi.NewEditMessageText(chatID, messageID, text)
	if _, err := t.bot.Send(msg); err != nil {
		log.Printf("[Telegram] Failed to edit message: %v", err)
	}
}

func (t *TelegramBot) sendMarkdown(chatID int64, text string) {
	msg := tgbotapi.NewMessage(chatID, text)
	msg.ParseMode = "Markdown"
	if _, err := t.bot.Send(msg); err != nil {
		msg.ParseMode = ""
		if _, err2 := t.bot.Send(msg); err2 != nil {
			log.Printf("[Telegram] Failed to send message: %v", err2)
		}
	}
}

func (t *TelegramBot) sendTyping(chatID int64) {
	action := tgbotapi.NewChatAction(chatID, tgbotapi.ChatTyping)
	t.bot.Send(action)
}

func (t *TelegramBot) downloadPhotoAsBase64(fileID string) (string, error) {
	fileURL, err := t.bot.GetFileDirectURL(fileID)
	if err != nil {
		return "", fmt.Errorf("failed to get file URL: %w", err)
	}

	resp, err := http.Get(fileURL)
	if err != nil {
		return "", fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("bad status: %s", resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	return base64.StdEncoding.EncodeToString(data), nil
}

func parseAllowedUsers(env string) []int64 {
	if env == "" {
		return nil
	}

	var ids []int64
	for s := range strings.SplitSeq(env, ",") {
		s = strings.TrimSpace(s)
		if id, err := strconv.ParseInt(s, 10, 64); err == nil {
			ids = append(ids, id)
		}
	}
	return ids
}

func stripBotSuffix(text string) string {
	if idx := strings.Index(text, "@"); idx > 0 && (strings.HasPrefix(text, "/")) {
		cmd := text[:idx]
		rest := text[idx:]
		if botEnd := strings.Index(rest, " "); botEnd > 0 {
			return cmd + rest[botEnd:]
		}
		return cmd
	}
	return text
}
