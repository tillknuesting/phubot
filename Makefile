.PHONY: all check test race vet fix fmt clean build install run help quick

# Default target
all: check

# Run all quality checks (built-in only)
check: fmt vet test race
	@echo "✅ All quality checks passed!"

# Quick check (fast)
quick: fmt vet test
	@echo "✅ Quick checks passed!"

# Format code
fmt:
	@echo "📝 Formatting code..."
	@gofmt -s -w .
	@go mod tidy
	@echo "   ✓ Formatted"

# Static analysis with go vet
vet:
	@echo "🔍 Running go vet..."
	@go vet ./...
	@echo "   ✓ No issues found"

# Modernize code with go fix
fix:
	@echo "🔧 Modernizing code..."
	@go fix ./...
	@echo "   ✓ Modernized"

# Run tests with race detector
race:
	@echo "🧪 Running tests with race detector..."
	@go test -race -cover ./...
	@echo "   ✓ Tests passed (no races detected)"

# Run tests without race detector (faster)
test:
	@echo "🧪 Running tests..."
	@go test -v -cover ./...

# Build binary
build:
	@echo "🏗️  Building binary..."
	@go build -o phubot .
	@echo "   ✓ Built phubot"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@go mod download
	@go mod verify
	@echo "   ✓ Dependencies installed"

# Run the bot (CLI mode)
run:
	@go run .

# Run with Telegram
run-telegram:
	@go run . -telegram $(TELEGRAM_TOKEN)

# Full cleanup and verification
tidy: fix fmt check
	@echo "✅ Code is clean and modern!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	@rm -f phubot
	@rm -f *.test *.out
	@echo "   ✓ Cleaned"

# Help
help:
	@echo "Phubot - Personal AI Assistant"
	@echo ""
	@echo "Quality Targets:"
	@echo "  make check        - Run all quality checks (fmt, vet, test, race)"
	@echo "  make quick        - Fast checks only (fmt, vet, test)"
	@echo "  make tidy         - Fix + format + check (full cleanup)"
	@echo ""
	@echo "Individual Checks:"
	@echo "  make fmt          - Format code with gofmt"
	@echo "  make vet          - Run go vet (static analysis)"
	@echo "  make fix          - Modernize code with go fix"
	@echo "  make race         - Run tests with race detector"
	@echo "  make test         - Run tests (fast, no race)"
	@echo ""
	@echo "Build & Run:"
	@echo "  make build        - Build binary"
	@echo "  make run          - Run in CLI mode"
	@echo "  make run-telegram - Run with Telegram"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make help         - Show this help"
