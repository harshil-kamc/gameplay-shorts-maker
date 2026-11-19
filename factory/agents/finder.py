"""Agent 1 — FINDER.

Downloads a gameplay video, transcribes it, and asks the AI to pick the most
clip-worthy moments. Results are stored as 'candidate' clips for review.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from .. import db, insights, llm, notify, skills
from ..config import WORK, cfg
from ..utils import media

console = Console()


def _extract_frames(video: Path, times: list[float], out_dir: Path) -> list[Path]:
    frames = []
    for i, t in enumerate(times):
        dest = out_dir / f"frame_{i}_{max(0.0, t):.2f}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}",
             "-i", str(video), "-frames:v", "1", "-q:v", "2", str(dest)],
            check=False, capture_output=True, timeout=30,
        )
        if dest.exists() and dest.stat().st_size > 0:
            frames.append(dest)
    return frames


def _visual_gameplay_gate(video: Path, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    kept = []
    for c in candidates:
        start, end = float(c["start"]), float(c["end"])
        times = [start + 0.4, (start + end) / 2, max(start + 0.4, end - 0.4)]
        with tempfile.TemporaryDirectory(dir=str(WORK)) as td:
            frames = _extract_frames(video, times, Path(td))
            if not frames:
                console.print(f"  [yellow]visual gate: no frames for "
                              f"{start:.0f}-{end:.0f}s — dropped[/]")
                continue
            verdict = llm.call_vision(
                "finder",
                "Look at these frames from one candidate clip. Answer only GAMEPLAY "
                "or NOT_GAMEPLAY. GAMEPLAY means visible player-controlled action: "
                "moving character/vehicle, combat, HUD changes, driving, shooting, "
                "interacting, or a camera following play. NOT_GAMEPLAY means cutscene, "
                "scripted dialogue, cinematic, static shot, menu, inventory, loading "
                "screen, talking head, or face-cam with no gameplay visible.",
                frames,
                max_tokens=16,
            ).strip().upper()
        if "NOT_GAMEPLAY" in verdict or verdict.startswith("NOT") or "CUTSCENE" in verdict:
            console.print(f"  [yellow]visual gate: dropped {start:.0f}-{end:.0f}s "
                          f"(cutscene/dialogue)[/]")
            continue
        if "GAMEPLAY" not in verdict:
            console.print(f"  [yellow]visual gate: dropped {start:.0f}-{end:.0f}s "
                          f"(unclear: {verdict[:24] or 'no vision'})[/]")
            continue
        kept.append(c)
    console.print(f"  [green]✓ visual gameplay gate kept "
                  f"{len(kept)}/{len(candidates)} candidate(s)[/]")
    return kept


@dataclass
class Candidate:
    """A scored clip-worthy moment returned by the headless find_candidates()
    API; the interactive flow stores the same fields as DB rows instead."""
    start_s: float
    end_s: float
    hook: str
    score: float
    reasoning: str
    caption: str = ""
    hashtags: list = field(default_factory=list)


def find_candidates(source_url: str, max_clips: int = 5,
                    niche: str | None = None) -> list[Candidate]:
    """Stateless clip discovery for headless/programmatic callers.

    Same engine as `find()` — download, transcribe, AI-score — but returns
    ranked Candidate objects instead of writing DB rows or asking a human to
    review. The caller decides which clips to accept.
    """
    video, title, channel = media.download(source_url)
    audio = media.extract_audio(video)
    transcript = media.transcribe(audio)
    if niche:
        # bias selection toward the buyer's niche without touching config
        title = f"{title}  [focus niche: {niche}]"
    scored = _visual_gameplay_gate(video, _score_with_claude(title, transcript))[: max(1, int(max_clips))]
    return [
        Candidate(
            start_s=float(c["start"]),
            end_s=float(c["end"]),
            hook=c.get("title", ""),
            score=float(c.get("score", 0)),
            reasoning=c.get("reason", ""),
            caption=c.get("caption", ""),
            hashtags=list(c.get("hashtags", []) or []),
        )
        for c in scored
    ]


def _transcript_for_prompt(transcript: list[dict]) -> str:
    """Compact, timestamped transcript so Claude can reference real times."""
    lines = []
    for seg in transcript:
        lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
    return "\n".join(lines)


# Keep each scoring prompt comfortably inside the model's context. A 3-hour
# gameplay stream transcribes to ~4,000+ segments — one giant prompt silently fails,
# so long streams are scored in chunks and the best moments merged.
_CHUNK_CHARS = 60_000


def _chunks(transcript: list[dict]) -> list[list[dict]]:
    """Split the transcript into pieces whose prompt text stays under budget."""
    out, cur, size = [], [], 0
    for seg in transcript:
        line = len(seg.get("text", "")) + 16
        if cur and size + line > _CHUNK_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append(seg)
        size += line
    if cur:
        out.append(cur)
    return out


def find(url: str) -> int:
    """Run the finder for one gameplay URL or local file. Returns candidate count."""
    console.print(f"[bold cyan]FINDER[/] downloading {url}")
    video, title, channel = media.download(url)
    console.print(f"  ↳ {title}" + (f"  [dim]({channel})[/]" if channel else ""))

    console.print("[bold cyan]FINDER[/] extracting audio + transcribing "
                  f"(whisper={cfg.get('finder.whisper_model')})…")
    audio = media.extract_audio(video)
    transcript = media.transcribe(audio)
    console.print(f"  ↳ {len(transcript)} segments")

    source_id = db.upsert_source(url, title, video, transcript, channel=channel)

    console.print("[bold cyan]FINDER[/] asking AI to score gameplay moments…")
    candidates = _visual_gameplay_gate(video, _score_with_claude(title, transcript))

    n = 0
    for c in candidates:
        db.add_clip(source_id, c["start"], c["end"], c["title"],
                    c["reason"], c["score"], c["caption"], c.get("hashtags", []))
        n += 1
    console.print(f"[green]✓ {n} candidate clips saved.[/] Run "
                  "[bold]python run.py review[/] to approve.")
    if n == 0:
        # A whole episode yielding nothing (even after the relaxed fallback) means
        # no posts that day — alert now instead of discovering it days later.
        notify.notify("Finder found 0 clips",
                      f"'{title[:70]}' produced no candidates — today's queue may be "
                      "empty. Check the source or the selection brief.")
    return n


# While niche_lock is set, mechanically DROP candidates that clearly belong to a
# DIFFERENT topic — the Finder kept clipping the boxing guest (Whittaker: 2-9
# views) despite being told to stay in lane. Coaching wasn't enough; this is the
# hard gate. Each niche has an OFF list (other-topic red flags) + a KEEP lexicon
# (this-niche terms); a clip is dropped only if it hits OFF and misses KEEP.
_OFF_NICHE = {
    "football": re.compile(
        r"\b(boxing|boxer|heavyweight|cruiserweight|welterweight|flyweight|"
        r"knockout|\bko\b|ufc|mma|octagon|sparring|ringwalk|ring walk|"
        r"title fight|undisputed|prizefight|jab|southpaw)\b", re.I),
    # money channel: drop clips that are really about sport/football or pure
    # health/relationships with no money angle (DOAC etc. mix topics)
    "money": re.compile(
        r"\b(football|soccer|penalty|midfield|boxing|ufc|world cup|premier "
        r"league|goalkeeper|striker|workout|reps|calories|protein)\b", re.I),
}
_NICHE_KEEP = {
    "football": re.compile(
        r"\b(football|soccer|goal|keeper|striker|midfield|defender|winger|"
        r"penalty|premier league|world cup|england|pitch|manager|transfer|squad|"
        r"dressing room|gaffer|nations league|champions league|var|offside|"
        r"clean sheet)\b", re.I),
    "money": re.compile(
        r"\b(money|cash|dollar|debt|income|salary|invest|investing|business|"
        r"entrepreneur|millionaire|billionaire|rich|wealth|wealthy|broke|budget|"
        r"savings?|credit|loan|mortgage|profit|revenue|startup|finance|financial|"
        r"bank|retire|retirement|portfolio|stocks?|crypto|side hustle|net worth|"
        r"bankrupt|paycheck|expenses?|afford|price|cost|\$|percent|interest)\b",
        re.I),
}


def _niche_ok(clip: dict) -> bool:
    """False if `clip` is clearly off-niche while niche_lock is active."""
    lock = cfg.get("finder.niche_lock")
    if not lock:
        return True
    off, keep = _OFF_NICHE.get(lock), _NICHE_KEEP.get(lock)
    if not off or not keep:
        return True
    text = f"{clip.get('title', '')} {clip.get('reason', '')} " \
           f"{clip.get('caption', '')}"
    if off.search(text) and not keep.search(text):
        console.print(f"  [yellow]niche-lock: dropped off-{lock} clip "
                      f"'{str(clip.get('title', ''))[:40]}'[/]")
        return False
    return True


def _score_with_claude(title: str, transcript: list[dict]) -> list[dict]:
    f = cfg.finder
    schema = {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number", "description": "start time in seconds"},
                        "end": {"type": "number", "description": "end time in seconds"},
                        "title": {"type": "string", "description": "punchy hook/title"},
                        "reason": {"type": "string", "description": "why this will perform"},
                        "score": {"type": "number", "description": "0-100 virality estimate"},
                        "caption": {"type": "string", "description": "post caption"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["start", "end", "title", "reason", "score", "caption"],
                },
            }
        },
        "required": ["clips"],
    }

    skill_block = skills.load(cfg.get("skills.finder", []))
    pieces = _chunks(transcript)
    max_cand = f.get("max_candidates", 8)
    per_chunk = max_cand if len(pieces) == 1 else max(3, max_cand // len(pieces) + 1)
    if len(pieces) > 1:
        console.print(f"  [dim]long episode → scoring {len(pieces)} chunks[/]")

    # The fallback brief for when the strict pass finds nothing. The free model is
    # flaky and the strict drama-only brief can make it return an EMPTY list on a
    # whole episode (a football-pundit show reads as "merely interesting"). A day
    # with zero clips is the worst outcome, so we loosen up rather than post nothing.
    relaxed_brief = (
        "Pick the 3 most engaging, self-contained AAA PC GAMEPLAY moments for a "
        "short-form clip — the bits most likely to stop someone scrolling: a huge "
        "crash, a chaotic chase, a boss attempt, a funny in-game failure, a wild "
        "stunt, a surprising discovery, or a big win against the odds. Favor visible "
        "action and stakes, but do NOT return an empty list — always return your "
        "best 3, each at least 30 seconds long.")

    def _mk_prompt(pi: int, piece: list[dict], brief: str, insist: bool) -> str:
        span = ""
        if len(pieces) > 1:
            span = (f"\n(This is part {pi + 1}/{len(pieces)} of the episode, "
                    f"covering {piece[0]['start']:.0f}s–{piece[-1]['end']:.0f}s.)")
        must = ("\n- IMPORTANT: return your best picks — do NOT return an empty list."
                if insist else "")
        return f"""You are a viral short-form video editor specializing in AAA PC GAMEPLAY. The gameplay video is titled "{title}".{span}

{skill_block}
Selection brief:
{brief}

Constraints:
- This is GAMEPLAY, not a podcast or interview. Pick moments where the VISIBLE
  game action is the story: combat, chases, crashes, stunts, boss fights, close
  calls, glitches, funny in-game beats, big wins/fails, and cinematic set pieces.
- Return at most {per_chunk} clips.{must}
- HARD LENGTH RULE: every clip must be at least {f.get('clip_min_seconds', 18)}s and
  at most {f.get('clip_max_seconds', 45)}s. Target exactly 30s. Any clip shorter
  than {f.get('clip_min_seconds', 30)}s will be discarded. State your length
  reasoning in the clip's `reason`.
- Start at the moment the action or payoff LANDS: visible motion, impact, damage,
  a kill, a crash, a reaction, or a clear stake within the first 1.5s.
- Use REAL timestamps from the transcript. Snap to natural action beats and
  sentence boundaries. One self-contained moment per clip; end on a payoff or
  comment-triggering hot take.
- REJECT CUTSCENES: scripted cinematic/dialogue scenes are NOT gameplay, even if
  the transcript is dramatic. Pick only playable-camera moments with visible
  player-controlled action or a gameplay HUD.
- AVOID menus, inventory management, idle walking, loading screens, long dialogue,
  and creator face-cam with no gameplay visible. Face-cam is only useful when the
  gameplay action is visible and central at the same time.
- CAPTIONS: write 1-2 GTA 6 gameplay description sentences, then end with ONE
  forced-choice question. Do not mention cutscenes, dialogue, podcasts, or
  interviews. Trending GTA 6 hashtags are appended automatically.

What's worked before (learn from this):
{insights.learnings()}

What's trending right now (lean into these when a clip fits naturally):
{insights.trends()}

Transcript (timestamped):
{_transcript_for_prompt(piece)}

Call submit_clips with your picks, best first."""

    def _score_pass(brief: str, insist: bool, attempts: int = 2) -> list[dict]:
        out: list[dict] = []
        for pi, piece in enumerate(pieces):
            prompt = _mk_prompt(pi, piece, brief, insist)
            got: list[dict] = []
            for _ in range(attempts):        # the free model intermittently returns
                try:                         # an empty tool call — one miss must not
                    result = llm.call_tool("finder", prompt, "submit_clips",  # zero
                                           schema, max_tokens=4000)           # out
                    got = (result or {}).get("clips", [])
                    if got:
                        break
                except Exception as ex:  # noqa: BLE001 - a failed try shouldn't kill the run
                    console.print(f"  [yellow]chunk {pi + 1} attempt failed: {ex}[/]")
            out.extend(got)
            if len(pieces) > 1:
                console.print(f"  [dim]chunk {pi + 1}/{len(pieces)}: "
                              f"{len(got)} candidates[/]")
        return out

    min_len = float(f.get("clip_min_seconds", 18))
    max_len = float(f.get("clip_max_seconds", 90))

    def _length(c: dict) -> float:
        try:
            return float(c["end"]) - float(c["start"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _key(c: dict) -> tuple[float, float]:
        return round(float(c["start"]), 1), round(float(c["end"]), 1)

    def _split_length(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        keep, short = [], []
        for c in candidates:
            length = _length(c)
            if min_len <= length <= max_len:
                keep.append(c)
            elif 0 < length < min_len:
                short.append(c)
        if short:
            console.print(f"  [yellow]dropped {len(short)} clip(s) shorter than "
                          f"{min_len:.0f}s[/]")
        return keep, short

    def _expand(c: dict) -> dict | None:
        target = min(min_len + 2, max_len)
        start, end = float(c["start"]), float(c["end"])
        if end - start >= target:
            return c
        overlap = [i for i, s in enumerate(transcript)
                   if s["end"] >= start and s["start"] <= end]
        if not overlap:
            return None
        lo, hi = overlap[0], overlap[-1]
        while end - start < target and (lo > 0 or hi < len(transcript) - 1):
            before = start - transcript[lo - 1]["end"] if lo > 0 else float("inf")
            after = (transcript[hi + 1]["start"] - end
                     if hi + 1 < len(transcript) else float("inf"))
            if lo > 0 and before <= after:
                lo -= 1
                start = transcript[lo]["start"]
            elif hi + 1 < len(transcript):
                hi += 1
                end = transcript[hi]["end"]
            else:
                break
        if end - start < min_len:
            return None
        fixed = dict(c)
        fixed["start"] = round(start, 2)
        fixed["end"] = round(end, 2)
        fixed["reason"] = (str(c.get("reason", "")).strip()
                           + f" Extended to {end - start:.0f}s to meet the "
                             "minimum length.").strip()
        return fixed

    strict_brief = f.get("selection_brief", "")
    clips, short_pool = _split_length(_score_pass(strict_brief, insist=False))
    clips = [c for c in clips if _niche_ok(c)]
    short_pool = [c for c in short_pool if _niche_ok(c)]

    target = min(max(4, int(f.get("auto_approve_top", 4))), max_cand)
    retry_brief = (strict_brief + "\n\nCRITICAL: previous picks were rejected for "
                   f"being under {min_len:.0f}s. Return only clips "
                   f"{min_len:.0f}-{max_len:.0f}s long.")
    for _ in range(2):
        if len(clips) >= target:
            break
        extra, extra_short = _split_length(_score_pass(retry_brief, insist=True))
        short_pool.extend(c for c in extra_short if _niche_ok(c))
        seen = {_key(c) for c in clips}
        for c in extra:
            if _niche_ok(c) and _key(c) not in seen:
                clips.append(c)
                seen.add(_key(c))

    if not clips:
        console.print("  [yellow]0 valid clips on the strict pass — retrying with "
                      "a relaxed gameplay brief so the day isn't empty[/]")
        relaxed, relaxed_short = _split_length(_score_pass(relaxed_brief, insist=True))
        short_pool.extend(c for c in relaxed_short if _niche_ok(c))
        seen = {_key(c) for c in clips}
        for c in relaxed:
            if _niche_ok(c) and _key(c) not in seen:
                clips.append(c)
                seen.add(_key(c))

    if len(clips) < target:
        seen = {_key(c) for c in clips}
        expanded = 0
        for c in short_pool:
            fixed = _expand(c)
            if fixed and _key(fixed) not in seen and min_len <= _length(fixed) <= max_len:
                clips.append(fixed)
                seen.add(_key(fixed))
                expanded += 1
                if len(clips) >= target:
                    break
        if expanded:
            console.print(f"  [yellow]extended {expanded} short candidate(s) to "
                          f"meet the {min_len:.0f}s minimum[/]")

    clips.sort(key=lambda c: -float(c.get("score", 0)))
    return clips[:max_cand]
