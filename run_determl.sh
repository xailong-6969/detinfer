#!/usr/bin/env bash
# ============================================================================
# run_determl.sh -- Standalone setup & run script for determl
#
# Inspired by https://github.com/gensyn-ai/rl-swarm run_rl_swarm.sh. One script that:
#   1. Creates a Python virtual environment
#   2. Installs determl + all dependencies
#   3. Auto-detects GPU/CPU
#   4. Prompts for a HuggingFace model name
#   5. Launches determl in interactive mode
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN="\033[32m"
BLUE="\033[34m"
RED="\033[31m"
RESET="\033[0m"

echo_green() { echo -e "${GREEN}$1${RESET}"; }
echo_blue()  { echo -e "${BLUE}$1${RESET}"; }
echo_red()   { echo -e "${RED}$1${RESET}"; }

# -- Banner --
echo -e "\033[38;5;39m"
cat << "EOF"
     _      _                  _
  __| | ___| |_ ___ _ __ _ __ | |
 / _` |/ _ \ __/ _ \ '__| '_ \| |
| (_| |  __/ ||  __/ |  | | | | |
 \__,_|\___|\__\___|_|  |_| |_|_|

  Deterministic ML Inference Engine v2

EOF
echo -e "${RESET}"

# -- Cleanup handler --
cleanup() {
    echo_green ">> Shutting down determl..."
    kill -- -$$ 2>/dev/null || true
    exit 0
}
trap cleanup EXIT

# ============================================================================
# Step 1: Python virtual environment
# ============================================================================
VENV_DIR="$ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo_green ">> Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo_green ">> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ============================================================================
# Step 2: Install determl
# ============================================================================
echo_green ">> Installing determl..."
pip install --upgrade pip
pip install -e "$ROOT[transformers,dev]"

echo_green ">> Installation complete!"

# ============================================================================
# Step 3: Detect GPU/CPU
# ============================================================================
echo_green ">> Detecting hardware..."

GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$GPU_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo_green ">> GPU detected: $GPU_NAME"
    DEVICE="cuda"
else
    echo_blue ">> No GPU detected. Using CPU."
    DEVICE="cpu"
fi

# ============================================================================
# Step 4: Prompt for model name
# ============================================================================
echo ""
echo -en "${GREEN}>> Enter the model name in huggingface repo/name format"
echo -en " (e.g., Qwen/Qwen2.5-Coder-0.5B-Instruct),"
echo -e  " or press [Enter] for default:${RESET}"
read -p "> " MODEL_NAME

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-0.5B-Instruct"
    echo_green ">> Using default model: $MODEL_NAME"
else
    echo_green ">> Using model: $MODEL_NAME"
fi

# ============================================================================
# Step 5: Prompt for seed
# ============================================================================
echo -en "${GREEN}>> Enter random seed (default: 42): ${RESET}"
read -p "" SEED

if [ -z "$SEED" ]; then
    SEED=42
fi
echo_green ">> Using seed: $SEED"

# ============================================================================
# Step 6: Main menu loop
# ============================================================================
while true; do
    echo ""
    echo_green ">> What would you like to do?"
    echo "   1) run          - Interactive deterministic inference"
    echo "   2) scan         - Scan model for non-deterministic ops"
    echo "   3) verify       - Verify model produces deterministic output"
    echo "   4) compare      - Before vs after determl comparison"
    echo "   5) benchmark    - Full determinism benchmark (auto-scales)"
    echo "   6) export       - Export inference proof (for cross-GPU verify)"
    echo "   7) cross-verify - Verify a proof from another machine"
    echo "   8) info         - Show environment information"
    echo "   9) exit         - Exit determl"
    echo ""
    echo -en "${GREEN}>> Choose [1-9] (default: 1): ${RESET}"
    read -p "" MODE_CHOICE

    case "${MODE_CHOICE:-1}" in
        1|run)
            echo_green ">> Starting interactive mode..."
            determl run "$MODEL_NAME" --seed "$SEED" --device "$DEVICE"
            ;;
        2|scan)
            echo_green ">> Scanning model..."
            determl scan "$MODEL_NAME" --seed "$SEED" --device "$DEVICE"
            ;;
        3|verify)
            echo_green ">> Verifying determinism..."
            determl verify "$MODEL_NAME" --seed "$SEED" --device "$DEVICE"
            ;;
        4|compare)
            echo_green ">> Running before/after comparison..."
            determl compare "$MODEL_NAME" --seed "$SEED" --device "$DEVICE"
            ;;
        5|benchmark)
            echo_green ">> Running benchmark..."
            determl benchmark "$MODEL_NAME" --seed "$SEED" --device "$DEVICE"
            ;;
        6|export)
            echo_green ">> Exporting proof..."
            determl export "$MODEL_NAME" --seed "$SEED" --device "$DEVICE" -o proof.json
            ;;
        7|cross-verify)
            echo -en "${GREEN}>> Enter proof file path: ${RESET}"
            read -p "" PROOF_FILE
            determl cross-verify "${PROOF_FILE:-proof.json}"
            ;;
        8|info)
            determl info
            ;;
        9|exit|quit|q)
            echo_green ">> Done!"
            break
            ;;
        *)
            echo_red ">> Invalid choice. Please select 1-9."
            ;;
    esac
done
