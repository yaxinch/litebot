---
name: memory
description: Two-layer memory system with grep-based recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/HISTORY.jsonl` — Canonical structured episodic history. Relevant Top-K entries are automatically retrieved for each query.
- `memory/HISTORY.md` — Human-readable, append-only mirror and legacy log. It is not loaded wholesale into context. Each entry starts with `[YYYY-MM-DD HH:MM]`.

## Search Past Events

Automatic retrieval normally supplies relevant history. For supplemental manual investigation, choose the search method based on file size:

- Small `memory/HISTORY.md`: use `read_file`, then search in-memory
- Large or long-lived `memory/HISTORY.md`: use the `exec` tool for targeted search

Examples:
- **Linux/macOS:** `grep -i "keyword" memory/HISTORY.md`
- **Windows:** `findstr /i "keyword" memory\HISTORY.md`
- **Cross-platform Python:** `python -c "from pathlib import Path; text = Path('memory/HISTORY.md').read_text(encoding='utf-8'); print('\n'.join([l for l in text.splitlines() if 'keyword' in l.lower()][-20:]))"`

Prefer targeted command-line search for large history files.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically consolidated into structured episodic entries and mirrored to HISTORY.md when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
