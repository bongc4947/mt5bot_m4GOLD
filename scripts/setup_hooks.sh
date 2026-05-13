#!/usr/bin/env bash
# scripts/setup_hooks.sh — one-time setup so `git pull` prints a stamp.
#
# Run this ONCE per local clone:
#     bash scripts/setup_hooks.sh
#
# What it does:
#   - Tells git to use this repo's .githooks/ instead of .git/hooks/
#     (.git/hooks isn't versioned; .githooks/ is — so the hook ships
#      with the repo and updates with `git pull`).
#   - Marks the hook scripts executable.
#
# After this, every `git pull` (and `git checkout`) prints:
#     remote / branch / HEAD short SHA / commit date / commit subject
#
# Idempotent: safe to re-run.

set -e
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/post-merge .githooks/post-checkout .githooks/_print_stamp.sh 2>/dev/null || true

echo "OK — hooks installed (core.hooksPath=$(git config core.hooksPath))."
echo "Try it: \`git pull\` or \`git checkout master\` — you should see a stamp."
