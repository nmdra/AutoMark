.PHONY: start stop run test logs clean pull-model init-db

# Start Ollama with CPU profile (default)
start:
	docker compose --profile cpu up -d

# Start Ollama with GPU profile
start-gpu:
	docker compose --profile gpu up -d

# Stop all services
stop:
	docker compose --profile cpu --profile gpu down

# Pull the required model
pull-model:
	docker exec ctse-ollama ollama pull phi4-mini:3.8b-q4_0

# Run the auto-grader pipeline
run:
	uv run python -m ctse_mas.graph

# Run the test suite
test:
	uv run pytest tests/ -v

# Show Ollama service logs
logs:
	docker compose logs -f ollama

# Initialise the SQLite student database
init-db:
	uv run python -c "from ctse_mas.tools.db_manager import init_db; init_db('data/students.db'); print('Database initialised at data/students.db')"

# Remove containers, volumes and cached outputs
clean:
	docker compose --profile cpu --profile gpu down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f agent_trace.log
	rm -f output/feedback_report.md
