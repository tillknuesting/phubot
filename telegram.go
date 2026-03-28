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
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

type TelegramBot struct {
	bot          *tgbotapi.BotAPI
	agent        *Agent
	scheduler    *Scheduler
	ctx          context.Context
	allowedUsers map[int64]bool
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
	}, nil
}

func (t *TelegramBot) isAllowed(userID int64) bool {
	if len(t.allowedUsers) == 0 {
		return true
	}
	return t.allowedUsers[userID]
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

			go t.handleMessage(update.Message)
		}
	}
}

func (t *TelegramBot) handleMessage(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID
	text := strings.TrimSpace(msg.Text)

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
		t.sendMessage(chatID, "👋 Hello! I'm your personal assistant. Ask me anything or use these commands:\n\n/tasks - List scheduled tasks\n/cancel <id> - Cancel a task\n/help - Show this message")
		return
	}

	if text == "/help" {
		t.sendMessage(chatID, "Commands:\n/tasks - List scheduled tasks\n/cancel <id> - Cancel a scheduled task\n\nYou can ask me to do things periodically, e.g.:\n\"Check flight prices from HKT to FRA every 2 hours\"")
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
