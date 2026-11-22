#!/usr/bin/env python3
"""
split_takes.py - Split voice-actor recordings into one file per read.

Recording sessions often produce files that hold a spoken slate
("line one") followed by several reads of the same line. This script:

  1. Finds each read in each recording, using a silence threshold that
     adapts to the noise floor of each file.
  2. Removes the slate at the start of the recording.
  3. Joins pieces of one read that a pause split in two, and drops
     short stray sounds between reads.
  4. Writes each read as its own .wav file, in the original sample rate
     and bit depth, named  <Character>_L<line>_Take<nn>_Try<n>.wav
  5. Writes manifest.csv, with one row per output file (source file,
     start time, length, and final script name if one is given).

Optional: with --script and --transcribe, it transcribes each recording
offline (PocketSphinx) and matches it to the closest script line, so a
mapping file is not needed.

------------------------------------------------------------------------
INSTALL (Python 3.9 or later)
    pip install numpy soundfile
    pip install pocketsphinx rapidfuzz       # only for --transcribe

INPUT FILES
  mapping.csv  (optional, recommended) - tells the script what each
  recording is. Columns:
      source,character,line,mode
      Track 1_002.wav,Witch,1,split
      Track 3_020.wav,Witch,5,whole   <- monologue: keep in one piece
      Track 4_035.wav,Crow,1,sounds      <- non-speech: keep every sound
      Track 1_001.wav,,,                  <- blank character: skip file
  In bulk runs, put the path from the input folder in "source" when two
  sessions have files with the same name (e.g. ActorA/Raw/Track 1_002.wav).
  mode: split (default) | whole | sounds

  script.csv  (optional) - the script. Columns:
      character,line,text,filename
      Witch,1,"You will not leave this forest.",A_Witch_Warning_1.wav

USAGE
  # Bulk: give several folders, or one parent folder. Every .wav below it
  # is processed, and the output keeps one folder per session, e.g.
  #   Sessions/ActorA/Raw/...  ->  OUT/ActorA/...
  python split_takes.py Sessions OUT_FOLDER --mapping mapping.csv
  python split_takes.py ActorA/Raw ActorB/Raw OUT_FOLDER --mapping mapping.csv

  # With a mapping file (most reliable):
  python split_takes.py RAW_FOLDER OUT_FOLDER --mapping mapping.csv --script script.csv

  # Let the script identify lines itself (rougher; check the manifest):
  python split_takes.py RAW_FOLDER OUT_FOLDER --script script.csv --transcribe

  # No mapping or script: files are named <source>_Try<n>.wav
  python split_takes.py RAW_FOLDER OUT_FOLDER

TUNING (defaults tuned on recordings from a voice-over booth)
  --min-gap 0.6      seconds of silence that separate two reads
  --threshold 0.3    0-1; where between noise floor and peak "speech" starts
  --pad 0.2          seconds of room tone kept before each read
  --tail 0.35        seconds kept after each read
  --keep-slate       do not remove the slate

The original recordings are never changed.
------------------------------------------------------------------------
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.02  # analysis step in seconds


# ---------------------------------------------------------------- detection
def find_segments(x, sr, min_gap, threshold, min_len=0.3):
    """Return (start, end) times of sound regions, using a per-file threshold."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    hop = int(sr * FRAME)
    n = len(mono) // hop
    if n == 0:
        return []
    frames = mono[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    if peak - floor < 6:  # no real signal in this file (silence or hum)
        return []
    active = np.flatnonzero(db > floor + threshold * (peak - floor))
    segs, start, prev = [], active[0], active[0]
    for i in active[1:]:
        if (i - prev) * FRAME > min_gap:
            segs.append((start * FRAME, (prev + 1) * FRAME))
            start = i
        prev = i
    segs.append((start * FRAME, (prev + 1) * FRAME))
    return [s for s in segs if s[1] - s[0] >= min_len]


def remove_slate(segs):
    """A short first sound near the start of the file is the spoken slate."""
    if len(segs) > 1 and segs[0][0] < 2.0 and segs[0][1] - segs[0][0] < 1.3:
        return segs[1:], segs[0]
    return segs, None


def tidy_reads(segs):
    """Join pieces of one read split by a pause; drop short stray sounds."""
    if not segs:
        return segs
    segs = [list(s) for s in segs if s[1] - s[0] >= 0.8] or [list(s) for s in segs]
    lengths = sorted(b - a for a, b in segs)
    med = lengths[len(lengths) // 2]
    merged = True
    while merged:
        merged = False
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            short = min(a[1] - a[0], b[1] - b[0]) < 0.7 * med
            if b[0] - a[1] < 1.0 and short and (b[1] - a[0]) <= 1.5 * med:
                segs[i] = [a[0], b[1]]
                segs.pop(i + 1)
                merged = True
                break
    return [tuple(s) for s in segs if s[1] - s[0] >= 0.4 * med]


# ------------------------------------------------------------ transcription
_decoder = None


def transcribe(x, sr):
    """Rough offline transcript (PocketSphinx). Used only to identify lines."""
    global _decoder
    from pocketsphinx import Decoder
    if _decoder is None:
        _decoder = Decoder(samprate=16000)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    y = np.interp(np.arange(0, len(mono), sr / 16000), np.arange(len(mono)), mono)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    _decoder.start_utt()
    _decoder.process_raw(pcm, full_utt=True)
    _decoder.end_utt()
    hyp = _decoder.hyp()
    return hyp.hypstr if hyp else ""


def best_line(text, script):
    from rapidfuzz import fuzz
    best, score = None, 0
    for row in script:
        s = fuzz.partial_ratio(text, row["text"].lower())
        if s > score:
            best, score = row, s
    return best, score


# --------------------------------------------------------------------- main
def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "", name or "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+",
                   help="one or more folders or .wav files, then the output folder last")
    p.add_argument("--mapping", type=Path, help="CSV: source,character,line,mode")
    p.add_argument("--script", type=Path, help="CSV: character,line,text,filename")
    p.add_argument("--transcribe", action="store_true", help="identify lines by speech recognition")
    p.add_argument("--min-gap", type=float, default=0.6)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--pad", type=float, default=0.2)
    p.add_argument("--tail", type=float, default=0.35)
    p.add_argument("--keep-slate", action="store_true")
    a = p.parse_args()
    if len(a.inputs) < 2:
        p.error("give at least one input and an output folder")
    *sources, a.out = a.inputs

    # Collect (session name, path relative to its input, file).
    jobs = []
    plain = [s.name for s in sources]
    def label(src):  # name a session; use the parent folder if names repeat (e.g. two "Raw")
        return src.parent.name if plain.count(src.name) > 1 or src.name.lower() == "raw" else src.name
    for src in sources:
        if src.is_file() and src.suffix.lower() == ".wav":
            jobs.append((src.parent.name, src.name, src))
            continue
        if not src.is_dir():
            sys.exit(f"Not found: {src}")
        wavs = sorted(w for w in src.rglob("*") if w.suffix.lower() == ".wav")
        subs = {w.relative_to(src).parts[0] for w in wavs if len(w.relative_to(src).parts) > 1}
        for w in wavs:
            rel = w.relative_to(src)
            # Parent folder with one subfolder per session: name sessions by subfolder.
            session = rel.parts[0] if len(sources) == 1 and len(subs) > 1 and len(rel.parts) > 1 else label(src)
            if a.out.resolve() in w.resolve().parents:
                continue  # never re-process our own output
            jobs.append((session, rel.as_posix(), w))
    if not jobs:
        sys.exit("No .wav files found in the inputs")
    sessions = {s for s, _, _ in jobs}
    multi = len(sessions) > 1

    mapping = {}
    if a.mapping:
        with open(a.mapping, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                mapping[r["source"].strip()] = r
    script = []
    if a.script:
        with open(a.script, newline="", encoding="utf-8-sig") as f:
            script = list(csv.DictReader(f))
    names = {(r["character"].strip(), str(r["line"]).strip()): r.get("filename", "") for r in script}
    if a.transcribe and not script:
        sys.exit("--transcribe needs --script to match against")

    a.out.mkdir(parents=True, exist_ok=True)
    take_no, rows = {}, []

    for session, rel, f in jobs:
        x, sr = sf.read(f, always_2d=False)
        info = sf.info(f)
        dur = len(x) / sr
        segs = find_segments(x, sr, a.min_gap, a.threshold)
        if not segs:
            rows.append([session, rel, "", "", "", "", "", "", "", "empty or silent; skipped"])
            print(f"{rel}: silent, skipped")
            continue

        m = mapping.get(rel) or mapping.get(f"{session}/{rel}") or mapping.get(f.name)
        character, line, mode, note = "", "", "split", ""
        if m is not None:
            character, line = m.get("character", "").strip(), str(m.get("line", "")).strip()
            mode = (m.get("mode") or "split").strip().lower()
            if not character:
                rows.append([session, rel, "", "", "", "", "", "", "", "skipped by mapping"])
                continue
        elif a.transcribe:
            row, score = best_line(transcribe(x, sr), script)
            if row and score >= 60:
                character, line = row["character"].strip(), str(row["line"]).strip()
                note = f"auto-matched (score {score:.0f}); check"
            else:
                note = "no confident script match"

        slate = None
        if not a.keep_slate:
            segs, slate = remove_slate(segs)
        if mode == "whole":
            reads = [(segs[0][0], segs[-1][1])]
        elif mode == "sounds":
            reads = segs
        else:
            reads = tidy_reads(segs)

        if character:
            key = (session, character, line)
            take_no[key] = take_no.get(key, 0) + 1
            base = f"{safe(character)}_L{int(line):02d}_Take{take_no[key]:02d}"
            folder = a.out / session / safe(character) if multi else a.out / safe(character)
        else:
            base = f.stem.replace(" ", "_")
            folder = a.out / session / "Unmatched" if multi else a.out / "Unmatched"
        folder.mkdir(parents=True, exist_ok=True)

        for i, (s, e) in enumerate(reads):
            lo = (reads[i - 1][1] if i else (slate[1] if slate else 0)) + (0.05 if i or slate else 0)
            hi = reads[i + 1][0] - 0.05 if i + 1 < len(reads) else dur
            start, end = max(lo, s - a.pad), min(hi, e + a.tail)
            name = f"{base}_Try{i + 1}.wav"
            sf.write(folder / name, x[int(start * sr): int(end * sr)], sr, subtype=info.subtype)
            rows.append([session, rel, character, line, f"{start:.2f}", f"{end - start:.2f}",
                         (folder / name).relative_to(a.out).as_posix(), names.get((character, line), ""), "", note])
        print(f"{rel}: {len(reads)} read(s) -> {base}")

    with open(a.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "source", "character", "line", "start_s", "length_s",
                    "output_file", "final_name", "keep", "note"])
        w.writerows(rows)
    print(f"\nDone. {sum(1 for r in rows if r[6])} files from {len(sessions)} session(s). "
          f"See {a.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
