#!/usr/bin/env python3
"""
voice-take-splitter - turn a voice-over session into named, one-read-per-file clips.

A recording session gives long files. Each file holds a spoken slate
("line five") and then one or more reads of a script line. This tool:

  1. Finds each read in each recording (silence detection that adapts to
     the noise floor of each file).
  2. Reads the script: the same PDF the actors got (one page per character,
     a table with #, LINE and FILE NAME), or a CSV.
     Identifies which script line each recording is, with no help:
       - it listens to the slate ("line five") with a recognizer that only
         knows the words "line" and numbers, and
       - it transcribes the reads and compares them to the script text, and
       - it follows the script order through the session, so a "line one"
         after the last Witch line is taken as the first Crow line, and
       - it works out which characters are in the session, and listens
         to the slates again using only their line numbers.
  3. Writes each read as its own .wav, named
         <Character>_L<line>_Take<nn>_Try<n>.wav
     in the original sample rate and bit depth.
  4. Writes manifest.csv (open it in Excel). Mark the keepers in the
     "keep" column, then run the "final" step: it copies each keeper into
     a Final folder under the file name the game expects.

All speech recognition runs offline on your computer (PocketSphinx).
The original recordings are never changed.

------------------------------------------------------------------------
INSTALL (Python 3.9 or later)
    pip install numpy soundfile pocketsphinx rapidfuzz pdfplumber

EASIEST: run with no arguments (or double-click run_windows.bat / run_mac.command)
    python split_takes.py
A small window opens. Pick the recordings folder, the script (PDF or CSV)
and an output folder, then press Split. Later, press Make Final folder.

COMMAND LINE
  Split (identify lines automatically from the script):
    python split_takes.py RECORDINGS_FOLDER OUT_FOLDER --script All_actors_lines.pdf

  Bulk: a parent folder with one subfolder per actor or session, or
  several folders. Output keeps one folder per session.
    python split_takes.py SESSIONS_FOLDER OUT_FOLDER --script script.csv
    python split_takes.py ActorA/Raw ActorB/Raw OUT_FOLDER --script script.csv

  Copy the keepers (rows with keep = yes / x / 1 in manifest.csv):
    python split_takes.py final OUT_FOLDER

SCRIPT AS PDF: the tool reads each page's title as the character name and the
table columns #, LINE and FILE NAME. Lines that say N/A are skipped. File
names with a wrong or missing extension (.waw, ,wav, none) get ".wav".

SCRIPT AS CSV (in the order the actors read it)
    character,line,text,filename,mode
    Witch,1,"You will not leave this forest.",A_Witch_Warning_1.wav,
    Witch,3,"Long ago I lived in a village... (a long monologue)",A_Witch_Memory.wav,whole
    Crow,1,"[single caw]",A_Crow_Caw.wav,sounds
  mode (optional): split = one file per read (default)
                   whole = keep the take in one piece (monologues)
                   sounds = keep every sound (non-speech lines)
  If mode is empty (always, for a PDF): text in [brackets] or of 3 words or
  fewer -> sounds; more than 40 words or with // marks -> whole; else split.

OPTIONAL MAPPING FILE (to override the automatic identification)
    source,character,line,mode
    Track 2_015.wav,Witch,4,split
    Track 1_001.wav,,,               <- empty character: skip this file

TUNING
  --min-gap 0.6     seconds of silence that separate two reads
  --threshold 0.3   0-1; where between noise floor and peak sound starts
  --pad 0.2         seconds kept before each read
  --tail 0.35       seconds kept after each read
  --keep-slate      keep the slate in the first read
------------------------------------------------------------------------
"""
import argparse
import csv
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.02  # analysis step in seconds
NUMBERS = ("zero one two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    if n < 20:
        return NUMBERS[n]
    return TENS[n // 10] + ("" if n % 10 == 0 else " " + NUMBERS[n % 10])


# --------------------------------------------------------------- audio
def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def find_segments(x, sr, min_gap, threshold, min_len=0.3):
    """(start, end) times of sound regions, with a threshold set per file."""
    mono = to_mono(x)
    hop = int(sr * FRAME)
    n = len(mono) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    if peak - floor < 6:  # silence or steady hum only
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
    """Join parts of one read that a pause split; drop short stray sounds."""
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
            if (b[0] - a[1] < 1.0 and min(a[1] - a[0], b[1] - b[0]) < 0.7 * med
                    and (b[1] - a[0]) <= 1.5 * med):
                segs[i] = [a[0], b[1]]
                segs.pop(i + 1)
                merged = True
                break
    return [tuple(s) for s in segs if s[1] - s[0] >= 0.4 * med]


def pcm16k(x, sr, start=0.0, end=None):
    mono = to_mono(x)
    seg = mono[int(start * sr): int((end if end is not None else len(mono) / sr) * sr)]
    if len(seg) == 0:
        return b""
    y = np.interp(np.arange(0, len(seg), sr / 16000), np.arange(len(seg)), seg)
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


# ------------------------------------------------------- recognition
class Listener:
    """Offline recognition: one general decoder, one that only knows slates."""

    def __init__(self, max_line):
        from pocketsphinx import Decoder
        self.general = Decoder(samprate=16000)
        self.set_range(max_line)

    def set_range(self, max_line):
        """Slate recognizer that knows only 'line' and the numbers 1..max_line."""
        from pocketsphinx import Decoder
        max_line = max(max_line, 1)
        words = " | ".join(number_words(n) for n in range(1, max_line + 1))
        gram = tempfile.NamedTemporaryFile("w", suffix=".gram", delete=False)
        gram.write(f"#JSGF V1.0;\ngrammar slate;\npublic <s> = [line] ({words});\n")
        gram.close()
        self.slater = Decoder(samprate=16000, jsgf=gram.name)
        self.lookup = {number_words(n): n for n in range(1, max_line + 1)}
        self.max_line = max_line

    @staticmethod
    def _run(dec, pcm):
        if not pcm:
            return ""
        dec.start_utt()
        dec.process_raw(pcm, full_utt=True)
        dec.end_utt()
        h = dec.hyp()
        return h.hypstr if h else ""

    def text(self, x, sr, start, end):
        return self._run(self.general, pcm16k(x, sr, start, end))

    @staticmethod
    def slate_audio(x, sr, seg):
        start, end = seg
        return pcm16k(x, sr, max(0, start - 0.15), min(end + 0.15, start + 1.8))

    def slate(self, pcm):
        """Line number spoken at the start of the take, or None."""
        heard = self._run(self.slater, pcm).replace("line", "").strip()
        return self.lookup.get(heard)


def identify(takes, script):
    """Best path through the script for the whole session (Viterbi).

    takes: list of dicts with 'slate' (int or None) and 'texts' (list of str).
    Returns one (script index, confident) per take.
    """
    from rapidfuzz import fuzz
    S = len(script)
    E = np.zeros((len(takes), S))
    for t, tk in enumerate(takes):
        for s, row in enumerate(script):
            ref = row["text"].lower()
            score = max((fuzz.partial_ratio(tx, ref) for tx in tk["texts"] if tx), default=0) / 100
            bonus = 0.0
            if tk["slate"] is not None:
                bonus = 1.0 if tk["slate"] == row["line_no"] else -0.4
            E[t, s] = 3 * max(0.0, score - 0.5) + bonus
    i = np.arange(S)
    trans = np.where(i[None, :] == i[:, None], 0.0,
                     np.where(i[None, :] == i[:, None] + 1, -0.1,
                              np.where(i[None, :] > i[:, None], -0.6, -1.5)))
    V, back = E[0].copy(), []
    for t in range(1, len(takes)):
        m = V[:, None] + trans
        back.append(m.argmax(0))
        V = m.max(0) + E[t]
    path = [int(V.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    # Characters in this session: those the path uses for 3 or more takes
    # (or every character used, for very short sessions).
    from collections import Counter
    used = Counter(script[s]["character"] for s in path)
    cast = {c for c, n in used.items() if n >= 3} or set(used)
    in_cast = np.array([script[s]["character"] in cast for s in range(S)])
    out = []
    for t, s in enumerate(path):
        top = E[t][in_cast].max()  # compare only with lines of the session's characters
        sure = E[t, s] >= top - 0.3 and E[t, s] > 0.4
        if takes[t]["slate"] is not None and takes[t]["slate"] != script[s]["line_no"]:
            sure = False  # the slate said another line number
        if script[s]["character"] not in cast:
            sure = False  # a character that is almost absent from this session
        out.append((s, sure))
    return out


# --------------------------------------------------------------- files
def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "", name or "")


def read_pdf_script(path):
    """Script rows from a voice-actor PDF: one page (or more) per character,
    a title with the character name, and a table with #, LINE and FILE NAME."""
    import pdfplumber
    rows, character = [], None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False, extra_attrs=["size"])
            if not words:
                continue
            hdr_hash = next((w for w in words if w["text"] == "#"), None)
            hdr_line = next((w for w in words if w["text"] == "LINE"), None)
            hdr_file = next((w for w in words if w["text"] == "FILE"), None)
            if not (hdr_hash and hdr_line and hdr_file):
                continue
            title = [w for w in words if w["bottom"] <= hdr_hash["top"] - 2]
            if title:  # largest text above the table header is the character name
                big = max(w["size"] for w in title)
                top = min(w["top"] for w in title if abs(w["size"] - big) < 0.5)
                name = " ".join(w["text"] for w in sorted(title, key=lambda w: w["x0"])
                                if abs(w["size"] - big) < 0.5 and abs(w["top"] - top) < 3)
                character = name or character
            x_line, x_file = hdr_line["x0"] - 2, hdr_file["x0"] - 2
            body = [w for w in words if w["top"] > hdr_hash["bottom"] + 1]
            footer_y = max((w["top"] for w in body if w["text"] == "Page"), default=None)
            if footer_y is not None:
                body = [w for w in body if w["top"] < footer_y - 2]
            # Each row starts at its number; a word belongs to the last row that
            # starts at or above it (with 4 pt tolerance for cells set a bit higher).
            starts = sorted((w["top"], int(w["text"])) for w in body
                            if w["x0"] < x_line and re.fullmatch(r"\d+", w["text"]))
            page_rows = [{"character": character, "line": n, "text": [], "file": []} for _, n in starts]
            rows.extend(page_rows)
            for w in body:
                if w["x0"] < x_line or not starts:
                    continue
                idx = None
                for k, (top, _) in enumerate(starts):
                    if w["top"] >= top - 4:
                        idx = k
                if idx is not None:
                    r = page_rows[idx]
                    (r["file"] if w["x0"] >= x_file else r["text"]).append(w)
    out = []
    for r in rows:
        text = " ".join(w["text"] for w in sorted(r["text"], key=lambda w: (round(w["top"]), w["x0"])))
        fname = "".join(w["text"] for w in sorted(r["file"], key=lambda w: (round(w["top"]), w["x0"])))
        text = text.strip().strip('"').strip()
        if not text or text.upper() in ("N/A", "NA", "-"):
            continue
        out.append({"character": r["character"], "line": str(r["line"]), "text": text, "filename": fname, "mode": ""})
    return out


def fix_wav_name(name):
    """Normalise the extension: 'x.waw', 'x,wav' and 'x' all become 'x.wav'."""
    name = (name or "").strip()
    if not name:
        return name, False
    fixed = re.sub(r"[.,]?(wav|waw|wv|wa)$", "", name, flags=re.I) + ".wav"
    return fixed, fixed != name


def load_script(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        rows = read_pdf_script(path)
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if (r.get("character") or "").strip()]
    if not rows:
        raise SystemExit(f"No script lines found in {path.name}.")
    for r in rows:
        r["character"] = r["character"].strip()
        r["line_no"] = int(str(r["line"]).strip())
        text = (r.get("text") or "").strip()
        mode = (r.get("mode") or "").strip().lower()
        if not mode:
            if re.fullmatch(r"\[.*\]", text.strip('"')):
                mode = "sounds"
            elif len(text.split()) > 40 or "//" in text:
                mode = "whole"
            elif len(text.split()) <= 3:
                mode = "sounds"  # very short lines and noises: keep every sound
            else:
                mode = "split"
        r["mode"] = mode
        r["filename"], r["name_fixed"] = fix_wav_name(r.get("filename"))
    return rows


def collect(sources, out):
    plain = [s.name for s in sources]

    def label(src):
        return src.parent.name if plain.count(src.name) > 1 or src.name.lower() == "raw" else src.name

    jobs = []
    for src in sources:
        if src.is_file() and src.suffix.lower() == ".wav":
            jobs.append((src.parent.name, src.name, src))
            continue
        if not src.is_dir():
            sys.exit(f"Not found: {src}")
        wavs = sorted(w for w in src.rglob("*") if w.suffix.lower() == ".wav")
        subs = {w.relative_to(src).parts[0] for w in wavs if len(w.relative_to(src).parts) > 1}
        for w in wavs:
            if out.resolve() in w.resolve().parents:
                continue
            rel = w.relative_to(src)
            session = rel.parts[0] if len(sources) == 1 and len(subs) > 1 and len(rel.parts) > 1 else label(src)
            jobs.append((session, rel.as_posix(), w))
    return jobs


def split(sources, out, script_path=None, mapping_path=None, min_gap=0.6, threshold=0.3,
          pad=0.2, tail=0.35, keep_slate=False, log=print):
    out = Path(out)
    jobs = collect([Path(s) for s in sources], out)
    if not jobs:
        raise SystemExit("No .wav files found.")
    script = load_script(script_path) if script_path else []
    mapping = {}
    if mapping_path:
        with open(mapping_path, newline="", encoding="utf-8-sig") as f:
            mapping = {r["source"].strip(): r for r in csv.DictReader(f)}
    listener = Listener(max((r["line_no"] for r in script), default=12)) if script else None
    out.mkdir(parents=True, exist_ok=True)
    multi = len({s for s, _, _ in jobs}) > 1

    # Pass 1: find sound regions, and listen (only when a script is given).
    takes = []
    for session, rel, f in jobs:
        x, sr = sf.read(f, always_2d=False)
        segs = find_segments(x, sr, min_gap, threshold)
        tk = {"session": session, "rel": rel, "file": f, "segs": segs, "slate": None, "texts": []}
        if segs and listener:
            tk["slate_pcm"] = listener.slate_audio(x, sr, segs[0])
            tk["slate"] = listener.slate(tk["slate_pcm"])
            long_segs = [s for s in segs if s[1] - s[0] > 1.0] or segs
            tk["texts"] = [listener.text(x, sr, s, e) for s, e in long_segs[:4]]
        takes.append(tk)
        log(f"listened: {rel}" if listener else f"read: {rel}")

    # Pass 2: decide which script line each take is (per session).
    for session in dict.fromkeys(t["session"] for t in takes):
        group = [t for t in takes if t["session"] == session and t["segs"]]
        if script and group:
            result = identify(group, script)
            # Listen to the slates again, knowing only the line numbers of the
            # characters in this session (fewer numbers, fewer mistakes).
            from collections import Counter
            used = Counter(script[i]["character"] for i, _ in result)
            cast = {c for c, n in used.items() if n >= 3} or set(used)
            cast_rows = [r for r in script if r["character"] in cast]
            if len(cast_rows) < len(script):
                top_line = max(r["line_no"] for r in cast_rows)
                if top_line < listener.max_line:
                    listener.set_range(top_line)
                    for tk in group:
                        tk["slate"] = listener.slate(tk["slate_pcm"])
                    listener.set_range(max(r["line_no"] for r in script))
                result = identify(group, cast_rows)
                log("characters in " + session + ": " + ", ".join(sorted(cast)))
            else:
                cast_rows = script
            for tk, (idx, sure) in zip(group, result):
                tk["row"], tk["sure"] = cast_rows[idx], sure

    # Pass 3: apply mapping overrides, cut and write.
    rows, take_no, written = [], {}, 0
    for tk in takes:
        session, rel, f = tk["session"], tk["rel"], tk["file"]
        base_row = [session, rel]
        if not tk["segs"]:
            rows.append(base_row + [""] * 7 + ["empty or silent; skipped"])
            continue
        row, note = tk.get("row"), ("" if tk.get("sure", True) else "check: low-confidence match")
        character = row["character"] if row else ""
        line = str(row["line_no"]) if row else ""
        mode = row["mode"] if row else "split"
        final = row["filename"] if row else ""
        if row and row.get("name_fixed") and not note:
            note = "file name extension fixed from script"
        m = mapping.get(rel) or mapping.get(f"{session}/{rel}") or mapping.get(f.name)
        if m is not None:
            character, line = (m.get("character") or "").strip(), str(m.get("line") or "").strip()
            if not character:
                rows.append(base_row + [""] * 7 + ["skipped by mapping"])
                continue
            hit = next((r for r in script if r["character"] == character and str(r["line_no"]) == line), None)
            mode = (m.get("mode") or (hit["mode"] if hit else "split")).strip().lower()
            final, note = (hit["filename"] if hit else ""), "set by mapping"

        x, sr = sf.read(f, always_2d=False)
        info = sf.info(f)
        dur = len(x) / sr
        segs, slate = (tk["segs"], None) if keep_slate else remove_slate(tk["segs"])
        if not segs:
            rows.append(base_row + [""] * 7 + ["slate only; skipped"])
            continue
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
            folder = out / session / safe(character) if multi else out / safe(character)
        else:
            base = f.stem.replace(" ", "_")
            folder = out / session / "Unmatched" if multi else out / "Unmatched"
            note = note or "no script given or no match"
        folder.mkdir(parents=True, exist_ok=True)
        for i, (s, e) in enumerate(reads):
            lo = (reads[i - 1][1] if i else (slate[1] if slate else 0)) + (0.05 if i or slate else 0)
            hi = reads[i + 1][0] - 0.05 if i + 1 < len(reads) else dur
            start, end = max(lo, s - pad), min(hi, e + tail)
            name = f"{base}_Try{i + 1}.wav"
            sf.write(folder / name, x[int(start * sr): int(end * sr)], sr, subtype=info.subtype)
            rows.append(base_row + [character, line, f"{start:.2f}", f"{end - start:.2f}",
                                    (folder / name).relative_to(out).as_posix(), final, "", note])
            written += 1
        log(f"{rel}: {len(reads)} read(s) -> {base}" + (f"   [{note}]" if note else ""))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "source", "character", "line", "start_s", "length_s",
                    "output_file", "final_name", "keep", "note"])
        w.writerows(rows)
    checks = sum(1 for r in rows if r[9].startswith("check"))
    log(f"\nDone: {written} files. Open {out / 'manifest.csv'}, put 'yes' in the keep column "
        f"for each keeper, then run the final step."
        + (f"\n{checks} file(s) are marked 'check' in the note column: listen to those first." if checks else ""))
    return out / "manifest.csv"


def finalize(out, log=print):
    """Copy every row marked keep into OUT/Final (per session), named for the game."""
    out = Path(out)
    man = out / "manifest.csv"
    if not man.exists():
        raise SystemExit(f"No manifest.csv in {out}. Run the split step first.")
    with open(man, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if (r.get("keep") or "").strip().lower() in ("yes", "y", "x", "1", "keep", "true")]
    if not rows:
        raise SystemExit("No rows are marked keep. Put 'yes' in the keep column of manifest.csv.")
    multi = len({r["session"] for r in rows}) > 1
    used, copied = {}, 0
    for r in rows:
        src = out / r["output_file"]
        if not src.exists():
            log(f"missing, skipped: {r['output_file']}")
            continue
        name = r["final_name"] or Path(r["output_file"]).name
        if not name.lower().endswith(".wav"):
            name += ".wav"
        dest_dir = out / "Final" / r["session"] if multi else out / "Final"
        dest_dir.mkdir(parents=True, exist_ok=True)
        key = (dest_dir, name.lower())
        used[key] = used.get(key, 0) + 1
        if used[key] > 1:  # two keepers for one line: keep both, mark the second
            stem, ext = name.rsplit(".", 1)
            name = f"{stem}_ALT{used[key] - 1}.{ext}"
            log(f"note: more than one keeper for {stem}; saved this one as {name}")
        shutil.copy2(src, dest_dir / name)
        copied += 1
    log(f"\nDone: {copied} file(s) copied to {out / 'Final'}")


# ------------------------------------------------------------------ GUI
def gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("Voice take splitter")
    rec, scr, outv = tk.StringVar(), tk.StringVar(), tk.StringVar()

    def pick(var, kind):
        p = (filedialog.askdirectory() if kind == "dir"
             else filedialog.askopenfilename(filetypes=[("Script", "*.pdf *.csv"), ("All files", "*.*")]))
        if p:
            var.set(p)

    for r, (lab, var, kind) in enumerate([("Recordings folder", rec, "dir"),
                                          ("Script (.pdf or .csv)", scr, "file"),
                                          ("Output folder", outv, "dir")]):
        tk.Label(root, text=lab).grid(row=r, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(root, textvariable=var, width=60).grid(row=r, column=1, padx=4)
        tk.Button(root, text="Browse", command=lambda v=var, k=kind: pick(v, k)).grid(row=r, column=2, padx=8)

    logbox = scrolledtext.ScrolledText(root, width=95, height=18)
    logbox.grid(row=4, column=0, columnspan=3, padx=8, pady=8)

    def log(msg):
        root.after(0, lambda: (logbox.insert("end", msg + "\n"), logbox.see("end")))

    def run(fn):
        def work():
            try:
                fn()
            except SystemExit as e:
                log(f"Stopped: {e}")
            except Exception as e:  # show any error in the window
                log(f"Error: {e}")
            finally:
                root.after(0, lambda: [b.config(state="normal") for b in buttons])
        for b in buttons:
            b.config(state="disabled")
        threading.Thread(target=work, daemon=True).start()

    def do_split():
        if not rec.get() or not outv.get():
            messagebox.showinfo("Missing", "Pick the recordings folder and an output folder.")
            return
        run(lambda: split([rec.get()], outv.get(), scr.get() or None, log=log))

    def do_final():
        if not outv.get():
            messagebox.showinfo("Missing", "Pick the output folder you used for the split.")
            return
        run(lambda: finalize(outv.get(), log=log))

    buttons = [tk.Button(root, text="1. Split and name", width=22, command=do_split),
               tk.Button(root, text="2. Make Final folder", width=22, command=do_final)]
    buttons[0].grid(row=3, column=1, sticky="w", padx=4)
    buttons[1].grid(row=3, column=1, sticky="e", padx=4)
    log("1. Pick the folder with the session's .wav files, the script (the actors' PDF, or a .csv)\n"
        "   and an output folder.\n"
        "2. Press 'Split and name'. Then open manifest.csv in the output folder and type 'yes'\n"
        "   in the keep column for each take you want.\n"
        "3. Press 'Make Final folder'. The keepers are copied there with the game's file names.\n")
    root.mainloop()


# ------------------------------------------------------------------ main
def main():
    if len(sys.argv) == 1:
        return gui()
    if sys.argv[1] == "final":
        if len(sys.argv) != 3:
            sys.exit("usage: python split_takes.py final OUT_FOLDER")
        return finalize(sys.argv[2])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+", help="input folder(s) or .wav files, then the output folder")
    p.add_argument("--script", type=Path, help="the actors' script PDF, or a CSV: character,line,text,filename[,mode]")
    p.add_argument("--mapping", type=Path, help="optional CSV to override identification")
    p.add_argument("--min-gap", type=float, default=0.6)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--pad", type=float, default=0.2)
    p.add_argument("--tail", type=float, default=0.35)
    p.add_argument("--keep-slate", action="store_true")
    a = p.parse_args()
    if len(a.inputs) < 2:
        p.error("give at least one input and an output folder")
    *sources, out = a.inputs
    split(sources, out, a.script, a.mapping, a.min_gap, a.threshold, a.pad, a.tail, a.keep_slate)


if __name__ == "__main__":
    main()
