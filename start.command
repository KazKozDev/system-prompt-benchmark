#!/bin/zsh

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

print_banner() {
  echo "========================================="
  echo "System Prompt Benchmark"
  echo "macOS Launcher"
  echo "========================================="
  echo ""
}

pause_on_error() {
  echo ""
  read -r "?Press Enter to close this window..."
}

find_free_port() {
  local port
  for port in {8501..8510}; do
    if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$port"
      return 0
    fi
  done
  return 1
}

wait_for_server() {
  local port="$1"
  local attempt
  for attempt in {1..60}; do
    if curl -fsS "http://127.0.0.1:${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

print_banner

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed."
  echo "Install Python 3.12+ and run this file again."
  pause_on_error
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating local virtual environment in .venv ..."
  if ! python3 -m venv .venv; then
    echo "Error: failed to create .venv"
    pause_on_error
    exit 1
  fi
fi

source ".venv/bin/activate"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing project dependencies into .venv ..."
  if ! python -m pip install --upgrade pip; then
    echo "Error: failed to upgrade pip"
    pause_on_error
    exit 1
  fi
  if ! python -m pip install -r requirements.txt; then
    echo "Error: failed to install requirements.txt"
    pause_on_error
    exit 1
  fi
fi

PORT="$(find_free_port)"
if [[ -z "$PORT" ]]; then
  echo "Error: could not find a free port in range 8501-8510"
  pause_on_error
  exit 1
fi

URL="http://127.0.0.1:${PORT}"

echo "Using Python: $(command -v python)"
echo "Using Streamlit port: ${PORT}"
echo "Project directory: $SCRIPT_DIR"
echo ""
echo "Starting Streamlit ..."

python -m streamlit run app.py \
  --server.port="$PORT" \
  --server.address=127.0.0.1 \
  --browser.gatherUsageStats=false &

STREAMLIT_PID=$!

cleanup() {
  if kill -0 "$STREAMLIT_PID" >/dev/null 2>&1; then
    kill "$STREAMLIT_PID" >/dev/null 2>&1
  fi
}

trap cleanup EXIT INT TERM

if wait_for_server "$PORT"; then
  echo "Opening browser: $URL"
  open "$URL"
else
  echo "Warning: Streamlit did not become ready in time."
  echo "You can try opening $URL manually."
fi

echo ""
echo "The app is running. Press Ctrl+C in this window to stop it."
echo ""

wait "$STREAMLIT_PID"
EXIT_CODE=$?

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo ""
  echo "Streamlit exited with code $EXIT_CODE"
  pause_on_error
fi

exit "$EXIT_CODE"