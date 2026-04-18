#!/usr/bin/env bash
# Idempotent bootstrap for RunPod (or any cloud GPU pod).
# Usage: bash /workspace/pod_bootstrap.sh
# Safe to re-run after every pod rebuild — skips work already done.

set -euo pipefail

WORKSPACE=/workspace
FORK_REPO="https://github.com/sandyhu533/sglang.git"
UPSTREAM_REPO="https://github.com/sgl-project/sglang.git"
DEV_BRANCH="scheduler/ttft-asymmetric-prefill"
REPO_DIR="$WORKSPACE/sglang"
VENV_DIR="$WORKSPACE/venv"
OMZ_DIR="$WORKSPACE/.oh-my-zsh"
SHELLRC="$WORKSPACE/.shellrc"
NPM_PREFIX="$WORKSPACE/.npm-global"
MARKER="# === cloudgpu bootstrap ==="

log() { echo -e "\033[1;34m[bootstrap]\033[0m $*"; }

# ---------- 1. apt packages (zsh + tools + runtime libs) ----------
log "1/8 apt install zsh + tools + libnuma"
# Skip apt entirely when every required tool is already on PATH — on a re-run
# within the same pod this saves ~10-20s of `apt-get update` phone-home.
# On a fresh pod (/ is overlay FS) these are all missing, so the branch fires.
APT_TOOLS=(zsh git curl tmux htop gpg)
MISSING=()
for t in "${APT_TOOLS[@]}"; do command -v "$t" >/dev/null 2>&1 || MISSING+=("$t"); done
# libnuma is a shared library, not a binary — probe via ldconfig. sgl_kernel's
# common_ops.so dlopens libnuma.so.1 during sglang startup and fails silently
# otherwise, falling back to a less-capable variant (breaks first-time import).
NEED_NUMA=0
ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1' || NEED_NUMA=1
if [ "${#MISSING[@]}" -gt 0 ] || [ "$NEED_NUMA" = "1" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq zsh git curl tmux htop ca-certificates gnupg libnuma1
fi

# ---------- 2. shared shell config (sourced by both bash & zsh) ----------
log "2/8 write $SHELLRC"
cat > "$SHELLRC" <<'EOF'
# Shared by bash and zsh. Persisted on /workspace.
# Keep every cache on the persistent network volume so / (20GB overlay) stays clean
# and pod rebuilds don't have to re-download models / wheels / compiled kernels.
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv
export TORCH_HOME=/workspace/.cache/torch
export TRITON_CACHE_DIR=/workspace/.cache/triton
export XDG_CACHE_HOME=/workspace/.cache
export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
export SGLANG_JIT_CACHE_DIR=/workspace/.cache/sglang_jit
export PATH=/workspace/.npm-global/bin:$PATH
cd /workspace 2>/dev/null || true
if [ -f /workspace/venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /workspace/venv/bin/activate
fi
EOF

# ---------- 3. oh-my-zsh + plugins (on /workspace, one-time) ----------
log "3/8 oh-my-zsh + plugins"
if [ ! -d "$OMZ_DIR" ]; then
    RUNZSH=no CHSH=no KEEP_ZSHRC=yes ZSH="$OMZ_DIR" \
        sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi
OMZ_PLUGINS="$OMZ_DIR/custom/plugins"
mkdir -p "$OMZ_PLUGINS"
[ -d "$OMZ_PLUGINS/zsh-autosuggestions" ] || \
    git clone --quiet https://github.com/zsh-users/zsh-autosuggestions "$OMZ_PLUGINS/zsh-autosuggestions"
[ -d "$OMZ_PLUGINS/zsh-syntax-highlighting" ] || \
    git clone --quiet https://github.com/zsh-users/zsh-syntax-highlighting "$OMZ_PLUGINS/zsh-syntax-highlighting"

# ---------- 4. ~/.zshrc (written every run — points to /workspace) ----------
log "4/8 write ~/.zshrc"
cat > ~/.zshrc <<EOF
$MARKER
export ZSH="$OMZ_DIR"
ZSH_THEME="robbyrussell"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source \$ZSH/oh-my-zsh.sh
[ -f $SHELLRC ] && source $SHELLRC
EOF

# ---------- 5. ~/.bashrc source shared config ----------
log "5/8 update ~/.bashrc"
if ! grep -qF "$MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

$MARKER
[ -f $SHELLRC ] && source $SHELLRC
EOF
fi

# ---------- 6. set default shell to zsh ----------
log "6/8 chsh to zsh"
if [ "$(getent passwd "$(id -un)" | cut -d: -f7)" != "/bin/zsh" ]; then
    chsh -s /bin/zsh
fi

# ---------- 7. venv + repo + sglang ----------
log "7/9 caches, venv, repo, sglang"
mkdir -p "$WORKSPACE/.cache/"{huggingface,pip,uv,torch,triton,flashinfer,sglang_jit}
export HF_HOME=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip
export UV_CACHE_DIR=/workspace/.cache/uv
export TRITON_CACHE_DIR=/workspace/.cache/triton
export XDG_CACHE_HOME=/workspace/.cache
export FLASHINFER_WORKSPACE_BASE=/workspace/.cache/flashinfer
export SGLANG_JIT_CACHE_DIR=/workspace/.cache/sglang_jit

FRESH_VENV=0
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python3 -m venv "$VENV_DIR"
    FRESH_VENV=1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
# Only upgrade pip/wheel/setuptools once per venv — PyPI phone-home takes
# ~5-10s each even when nothing changes, and the venv lives on /workspace so
# it persists across pod rebuilds.
if [ "$FRESH_VENV" = "1" ]; then
    pip install --quiet --upgrade pip wheel setuptools
fi
command -v uv >/dev/null 2>&1 || pip install --quiet uv

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$FORK_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git remote | grep -q '^upstream$' || git remote add upstream "$UPSTREAM_REPO"
# Fetch is opt-out via SKIP_FETCH=1 for quick re-runs that don't need remote state.
if [ "${SKIP_FETCH:-0}" != "1" ]; then
    git fetch --all --quiet
fi

if git show-ref --verify --quiet "refs/heads/$DEV_BRANCH"; then
    git checkout "$DEV_BRANCH"
    git pull --ff-only origin "$DEV_BRANCH" 2>/dev/null || true
elif git show-ref --verify --quiet "refs/remotes/origin/$DEV_BRANCH"; then
    git checkout -b "$DEV_BRANCH" "origin/$DEV_BRANCH"
else
    git checkout -b "$DEV_BRANCH"
fi

# ---------- 8. install sglang (editable) ----------
log "8/9 install sglang"
# Editable install drops sglang into $VENV/lib/.../site-packages as a pointer to
# /workspace/sglang/python, so local edits on the dev branch are picked up
# without reinstall. uv pip is ~5-10x faster than pip for this dependency set.
if ! python -c "import sglang" >/dev/null 2>&1; then
    uv pip install -e "$REPO_DIR/python"
fi

# ---------- 9. Node.js + Claude Code (persisted to /workspace) ----------
log "9/9 node + claude code"
# Node.js 20.x via NodeSource (apt package — reinstalled each pod rebuild, ~30s)
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null
    apt-get install -y -qq nodejs
fi

# Global npm packages go to /workspace so claude-code persists across rebuilds
mkdir -p "$NPM_PREFIX"
npm config set prefix "$NPM_PREFIX" >/dev/null
export PATH="$NPM_PREFIX/bin:$PATH"

# Install (or no-op if already on /workspace)
if [ ! -x "$NPM_PREFIX/bin/claude" ]; then
    npm install -g --silent @anthropic-ai/claude-code
fi

# Persist ~/.claude (auth + settings + history) by symlinking to /workspace
mkdir -p /workspace/.claude
[ -e /workspace/.claude.json ] || echo '{}' > /workspace/.claude.json
if [ ! -L ~/.claude ]; then
    [ -d ~/.claude ] && rm -rf ~/.claude
    ln -s /workspace/.claude ~/.claude
fi
if [ ! -L ~/.claude.json ]; then
    [ -f ~/.claude.json ] && rm -f ~/.claude.json
    ln -s /workspace/.claude.json ~/.claude.json
fi

echo
log "done!"
echo "  shell:  $(getent passwd "$(id -un)" | cut -d: -f7)  (exit + re-ssh to enter zsh)"
echo "  repo:   $REPO_DIR ($(git branch --show-current))"
echo "  venv:   $VENV_DIR"
echo "  sglang: $(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))' 2>/dev/null || echo 'not installed')"
echo "  claude: $(command -v claude || echo 'not on PATH yet — re-source ~/.zshrc')"
echo
echo "Next steps (manual):"
echo "  claude                                         # first time: run /login"
