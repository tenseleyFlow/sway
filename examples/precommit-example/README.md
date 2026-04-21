# Pre-commit gate example

A copy-paste recipe for plugging `sway gate` into any repo's `git
commit` flow via [pre-commit.com](https://pre-commit.com).

## What this gives you

On every `git commit` that touches `sway.yaml`, a `.dlm` document, or
an `adapter/` directory, pre-commit runs `sway gate` against your
spec. A FAIL verdict aborts the commit with the terminal banner
you'd see from `sway gate` directly. No more "oops, I broke the
adapter and didn't notice until the next manual `sway run`."

## Which variant

sway ships two hooks. Pick whichever matches your install posture:

| Hook | When to use | Cost |
|---|---|---|
| `sway-gate` | sway is already on your `PATH` (`pip install 'dlm-sway[hf]'`) | Milliseconds of hook overhead + the actual gate runtime |
| `sway-gate-isolated` | fresh venv, no existing sway install | First run installs sway + torch + transformers (~5 GB, ~2 min). Subsequent runs reuse the cached venv. |

This example shows `sway-gate` as the default and `sway-gate-isolated`
commented out. Swap them in `.pre-commit-config.yaml` if you want the
isolated variant.

## Setup (4 steps)

1. Install pre-commit: `pipx install pre-commit` (or `pip install
   pre-commit` if you don't use pipx).
2. Copy this directory's `.pre-commit-config.yaml` into your repo
   root. If you already have one, merge the `- repo:` block.
3. Edit `sway.yaml` in your repo to point at your real base model
   and adapter paths.
4. Install the hook: `pre-commit install`.

That's it. The next `git commit` that matches the hook's file filter
will run `sway gate` before the commit finalizes.

## Rev pinning

The example's `.pre-commit-config.yaml` pins to a specific commit
SHA because sway hasn't tagged a v0.1.0 release yet. Pinning to
`HEAD` would silently drift your gate's behavior on every `pre-commit
autoupdate`; pinning to a SHA is the honest pre-release pattern.

Bump the SHA deliberately when you want to pick up new probes or
bug fixes. After sway publishes v0.1.0 you'll switch to
`rev: v0.1.0` and forget about SHAs.

## Scope discipline

The hook **only gates** — exits non-zero on FAIL, zero on PASS. It
does not emit JSON / markdown reports; those are for `sway run`
(called ad-hoc or in a separate CI job). Keeping the hook focused
means `git commit` stays fast and the gate's verdict stays the
headline signal.

## Trying it locally before committing

If you want to see the hook fire before integrating it into a real
repo:

```bash
# From your repo's root, after copying the config in and editing
# the spec paths:
pre-commit run sway-gate --all-files
```

That runs the hook against every tracked file and surfaces the
full terminal report. A subsequent `git commit` will run the hook
only against staged files — same logic, narrower scope.
