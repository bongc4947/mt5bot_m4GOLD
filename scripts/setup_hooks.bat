@echo off
:: scripts/setup_hooks.bat — one-time setup on Windows.
:: Run this ONCE per local clone:
::     scripts\setup_hooks.bat
::
:: After this, every `git pull` prints repo state. Idempotent.
git config core.hooksPath .githooks
echo OK -- hooks installed.
echo Try it: `git pull` -- you should see a stamp.
