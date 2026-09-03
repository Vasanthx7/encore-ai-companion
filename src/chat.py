"""CLI chat loop.

Wires the seams together into the interactive companion:
  - a persona picker (personas/*.yaml) -> pinned system spine
  - Claude (via AWS Bedrock) reached through the llm.py seam
  - every turn persisted to SQLite; working memory rebuilt from the DB, so
    conversation survives a process restart.

The presentation layer uses `rich` (colored per-persona header panel, a spinner
while the model generates, markdown-rendered replies). None of it touches the
memory pipeline — engine.process_turn() is the same call the demo and eval use.

Run:  python -m src.chat        (from the project root)
Commands:  /memory  /dump  /quit
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from src import engine, llm, persona, store

SHOW_MEMORY_OPS = True  # print a dim indicator when memories are stored/recalled

console = Console()

# A distinct accent colour per shipped persona; unknown personas fall back.
PERSONA_COLORS = {"kai": "magenta", "nova": "cyan", "sage": "green", "milo": "yellow"}
DEFAULT_COLOR = "bright_blue"


def _accent(slug: str) -> str:
    return PERSONA_COLORS.get(slug, DEFAULT_COLOR)


def _short(text: str, limit: int = 64) -> str:
    """First sentence of a concept blurb, trimmed to `limit` chars."""
    text = " ".join((text or "").split())
    cut = text.split(". ")[0]
    if len(cut) > limit:
        cut = cut[: limit - 1].rstrip() + "…"
    return cut


def choose_persona() -> dict | None:
    """Let the user pick a companion from the persona library.

    Returns the chosen persona meta ({slug, name, concept, path}) or None to fall
    back to whatever config.PERSONA_PATH already points at. If COMPANION_PERSONA is
    set in the environment the choice is pinned and honoured silently. With 0 or 1
    personas available the menu is skipped.
    """
    if os.getenv("COMPANION_PERSONA"):
        return None  # explicitly pinned; leave config.PERSONA_PATH as-is
    options = persona.list_personas()
    if len(options) <= 1:
        return None

    default = options[0]  # Kai — used on empty input / Ctrl-C
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right")  # number
    grid.add_column()                 # name
    grid.add_column()                 # one-line concept
    grid.add_column()                 # default tag
    for i, o in enumerate(options, 1):
        color = _accent(o["slug"])
        tag = "[green]● default[/]" if o is default else ""
        grid.add_row(
            f"[dim]{i}[/]",
            f"[bold {color}]{o['name']}[/]",
            f"[dim]{escape(_short(o['concept'], 52))}[/]",
            tag,
        )
    console.print()
    console.print(
        Panel(
            grid,
            title="[bold]Choose your companion[/]",
            border_style="bright_black",
            expand=False,
            padding=(1, 2),
        )
    )
    while True:
        try:
            raw = console.input(
                f"[dim]1-{len(options)}, or enter for[/] [bold]{default['name']}[/] "
                f"[bold]›[/] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return default
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        for o in options:  # also accept a name or slug
            if raw.lower() in (o["slug"].lower(), o["name"].lower()):
                return o
        console.print("  [yellow](didn't catch that — pick a number or a name)[/]")


def _print_header(name: str, concept: str, color: str, existing: int) -> None:
    body = Text()
    body.append(_short(concept, 78), style="italic")
    body.append(f"\n\nmodel  {config.CHAT_MODEL}", style="dim")
    body.append(f"\ndb     {config.DB_PATH}", style="dim")
    if existing:
        body.append(f"\nresumed session — {existing} prior turns", style="dim")
    console.print(
        Panel(
            body,
            title=f"[bold {color}]{name}[/]",
            border_style=color,
            expand=False,
            padding=(1, 2),
        )
    )
    console.print("[dim]commands  /memory  /dump  /quit[/]\n")


def cmd_memory(conn, color: str) -> None:
    """Grouped view of active memory — user facts and the persona's own opinions."""
    mems = store.active_memories(conn)
    by_kind: dict[str, list] = {}
    for m in mems:
        by_kind.setdefault(m["kind"], []).append(m)
    console.print(f"\n[bold]active memory[/] [dim]({len(mems)})[/]")
    for kind in config.ALL_KINDS:
        rows = by_kind.get(kind, [])
        if not rows:
            continue
        tag = color if kind.startswith("persona") else "green"
        console.print(f"  [bold {tag}]{kind}[/] [dim]({len(rows)})[/]")
        for m in rows:
            console.print(
                f"    [dim]{m['id']:>3}[/] {escape(m['subject'])}: {escape(m['content'])}"
            )
    retired = [m for m in store.all_memories(conn) if m["status"] != "active"]
    if retired:
        console.print(f"  [dim]retired ({len(retired)})[/]")
        for m in retired:
            console.print(
                f"    [dim]{m['id']:>3} ({m['status']}) {escape(m['content'])}[/]"
            )
    console.print()


def cmd_dump(conn) -> None:
    turns = store.recent_turns(conn, 1000)
    mems = store.all_memories(conn)
    console.print(f"\n[dim]--- {len(turns)} turns in DB ---[/]")
    for t in turns:
        console.print(f"  [dim]{t['id']:>3} {t['role']:>9}[/] {escape(t['content'][:80])}")
    console.print(f"[dim]--- {len(mems)} memories in DB ---[/]")
    for m in mems:
        console.print(
            f"  [dim]{m['id']:>3} ({m['status']}/{m['kind']})[/] {escape(m['content'][:80])}"
        )
    console.print()


def _force_utf8() -> None:
    """Windows consoles default to cp1252 and crash on emoji the model emits.
    Reconfigure stdio to UTF-8 (replace on the rare unencodable char)."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _force_utf8()

    # Pick a companion before anything touches the DB: switching persona also
    # switches the memory store (unless COMPANION_DB is pinned), so each companion
    # keeps its own facts and its own improvised opinions — no cross-contamination.
    chosen = choose_persona()
    if chosen:
        config.PERSONA_PATH = chosen["path"]
        if not os.getenv("COMPANION_DB"):
            config.DB_PATH = config.ROOT / f"companion_{chosen['slug']}.db"

    conn = store.connect()
    store.init_db(conn)

    p = persona.load()
    spine = persona.render_spine(p)
    name = p["identity"]["name"]
    color = _accent(config.PERSONA_PATH.stem)
    concept = p["identity"].get("concept", "")

    seeded = persona.seed_canon(conn, p)  # idempotent; re-seeds only if yaml changed
    decayed = store.expire_past_episodic(
        conn, datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    ok, detail = llm.health()
    if not ok:
        console.print(f"[bold red][!] Cannot reach Claude on Bedrock ({config.CHAT_MODEL} @ {config.LLM_AWS_REGION})[/]")
        console.print(f"    [dim]{escape(detail)}[/]")
        console.print("    [dim]Check COMPANION_LLM_API_KEY in .env and COMPANION_AWS_REGION.[/]")
        return 1

    existing = store.turn_count(conn)
    _print_header(name, concept, color, existing)
    if seeded:
        console.print(f"[dim]· seeded {seeded} persona-canon memories[/]")
    if decayed:
        console.print(f"[dim]· decayed {decayed} past-dated event memories[/]")
    if seeded or decayed:
        console.print()

    if existing == 0:
        with console.status(f"[{color}]{name} is thinking…[/]", spinner="dots"):
            opener = engine.start_conversation(conn, spine)
        console.print(f"[bold {color}]{name.lower()}[/] [bold]›[/]")
        console.print(Padding(Markdown(opener.reply), (0, 0, 1, 2)))

    while True:
        try:
            user_text = console.input("[bold green]you[/] [bold]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim](take care)[/]")
            break
        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            console.print("[dim](take care)[/]")
            break
        if user_text == "/dump":
            cmd_dump(conn)
            continue
        if user_text == "/memory":
            cmd_memory(conn, color)
            continue

        try:
            with console.status(f"[{color}]{name} is thinking…[/]", spinner="dots"):
                res = engine.process_turn(conn, spine, user_text)
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red][!] turn failed:[/] {escape(str(e))}")
            continue

        ing = res.ingest
        if SHOW_MEMORY_OPS and (ing.changed or res.recalled):
            bits = []
            if ing.inserted:
                bits.append(f"+{len(ing.inserted)}")
            if ing.superseded:
                bits.append(f"~{len(ing.superseded)} retired")
            if ing.refined:
                bits.append(f"✎{len(ing.refined)} refined")
            bits.append(f"recalled {len(res.recalled)}")
            console.print(f"[dim]· {' · '.join(bits)}[/]")
            for note in ing.notes:
                console.print(f"[dim]  ↳ {escape(note)}[/]")

        console.print(f"[bold {color}]{name.lower()}[/] [bold]›[/]")
        console.print(Padding(Markdown(res.reply), (0, 0, 1, 2)))

        if SHOW_MEMORY_OPS and res.stated.get("flagged"):
            for fl in res.stated["flagged"]:
                console.print(f"[yellow]! persona-consistency flag:[/] [dim]{escape(fl)}[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
