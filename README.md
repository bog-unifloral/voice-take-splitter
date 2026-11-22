# voice-take-splitter

A Python script that splits voice-actor recordings into one file per read.

Voice-over sessions often give long files. Each file has a spoken slate ("line one") and then several reads of the same line. This script finds each read, removes the slate, and writes each read to its own `.wav` file with a clear name:

```
Witch_L01_Take02_Try3.wav   ->  character Witch, script line 1, 2nd recording of that line, 3rd read in it
```

It also writes `manifest.csv`. The manifest has one row per output file: the source recording, the start time, the length, the final script file name, and an empty `keep` column for the review.

The script does not change the original recordings.

## How it works

1. **Find the reads.** The script measures the loudness of each file in 20 ms steps. It sets a threshold from the noise floor of that file, so it works in quiet and in noisy rooms. A silence longer than `--min-gap` separates two reads.
2. **Remove the slate.** A short sound at the start of the file is the slate. The script removes it.
3. **Clean up.** The script joins two parts of one read that a pause split. It removes short stray sounds between reads (for example "okay" or a breath).
4. **Write the files.** The script keeps the original sample rate and bit depth. It keeps a short margin of room tone before and after each read.

## Install

Python 3.9 or later.

```
pip install numpy soundfile
pip install pocketsphinx rapidfuzz     # optional, only for --transcribe
```

## Use

**One session, with a mapping file (most reliable):**

```
python split_takes.py RAW_FOLDER OUT_FOLDER --mapping mapping.csv --script script.csv
```

**Bulk.** Give one parent folder with one subfolder for each actor or session, or give several folders. The output has one folder for each session.

```
python split_takes.py Sessions OUT_FOLDER --mapping mapping.csv
python split_takes.py ActorA/Raw ActorB/Raw OUT_FOLDER --mapping mapping.csv
```

**Identify the lines automatically.** The script transcribes each recording offline and matches it to the nearest script line. The result is a first pass, so check the `note` column in the manifest.

```
python split_takes.py RAW_FOLDER OUT_FOLDER --script script.csv --transcribe
```

**No mapping and no script.** The files get the names `<source>_Try<n>.wav`.

## Input files

See `example_mapping.csv` and `example_script.csv`.

`mapping.csv` tells the script what each recording contains:

| column | value |
|---|---|
| `source` | file name of the recording (in bulk runs, use the path from the input folder if two sessions have the same file names) |
| `character` | character name; leave empty to skip the file |
| `line` | script line number |
| `mode` | `split` (default): one file for each read. `whole`: keep the recording in one piece, for monologues. `sounds`: keep each sound, for non-speech lines such as animal sounds |

`script.csv` gives the script:

| column | value |
|---|---|
| `character`, `line` | same as in the mapping |
| `text` | the line, used by `--transcribe` |
| `filename` | the final name for the game, written to the manifest |

## Settings

| option | default | meaning |
|---|---|---|
| `--min-gap` | 0.6 | seconds of silence between two reads |
| `--threshold` | 0.3 | 0 to 1: the level between noise floor and peak where sound starts |
| `--pad` | 0.2 | seconds kept before each read |
| `--tail` | 0.35 | seconds kept after each read |
| `--keep-slate` | off | do not remove the slate |

## Limits

- Listen to the output before you choose takes. The script finds reads by sound level, not by meaning. Sometimes it splits one read with a long pause, or keeps two fast reads in one file. Change `--min-gap` for that session, or cut the file by hand.
- `--transcribe` uses PocketSphinx, a small offline recognizer. It identifies most speech lines but it is not accurate. A mapping file is more reliable.
