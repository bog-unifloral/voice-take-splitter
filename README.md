# voice-take-splitter

A tool for voice-over sessions. It cuts each recording into one file per read, finds which script line each recording is, and names the files. You then choose the best take of each line, and the tool copies those takes into a `Final` folder with the file names the game uses.

All speech recognition runs offline on your computer. The tool does not change the original recordings.

## What you need

- The session folder with the `.wav` recordings.
- The script as a `.csv` file (see [The script file](#the-script-file)). You can make it in Excel with **File > Save As > CSV**.
- Python 3.9 or later, from [python.org](https://www.python.org/downloads/). On Windows, tick **Add python.exe to PATH** during setup.

## Use it (no typing)

1. Download this repository: **Code > Download ZIP**, then unzip it.
2. Double-click **`run_windows.bat`** (Windows) or **`run_mac.command`** (Mac). The first time, it installs the packages it needs. This takes about a minute.
3. In the window, pick:
   - **Recordings folder**: the folder with the session's `.wav` files. It can also be a folder with one subfolder for each actor.
   - **Script (.csv)**: the script file.
   - **Output folder**: an empty folder for the results.
4. Click **1. Split and name**. A session of about 50 recordings takes 2 to 3 minutes.
5. Open `manifest.csv` in the output folder. Each row is one read. Listen to the files and type `yes` in the **keep** column for the best take of each line. Start with the rows that have **check** in the **note** column: the tool was not sure about them.
6. Click **2. Make Final folder**. The tool copies each keeper into `Final` and gives it the file name from the script.

## What the tool does

1. **Finds the reads.** It measures the loudness of each recording and sets a silence level for that file, so it works in quiet and in noisy rooms.
2. **Identifies the line.** It uses three clues:
   - the slate ("line five"), with a recognizer that knows only the word "line" and numbers,
   - a transcript of the reads, compared to the script text,
   - the script order: takes usually follow the script, so a "line one" after the last Witch line is the first Crow line.
3. **Cuts and names.** It removes the slate, joins two parts of one read that a pause split, and removes short stray sounds. Each read becomes one file:

   ```
   Witch_L03_Take02_Try1.wav  =  Witch, script line 3, 2nd recording of that line, 1st read in it
   ```
4. **Writes `manifest.csv`.** One row for each file: source recording, start time, length, final file name, keep, and note.

In a test on one real session (50 recordings, two characters, slates in most takes), the tool identified 48 recordings correctly. It marked the other 2 as **check**. Results change with each session, so listen before you choose.

## The script file

One row for each line, **in the order the actor reads them**:

| column | meaning |
|---|---|
| `character` | character name |
| `line` | line number (the number the actor says in the slate) |
| `text` | the words of the line |
| `filename` | the file name the game uses |
| `mode` | optional: `split` (one file for each read), `whole` (keep the take in one piece, for monologues), `sounds` (keep each sound, for animal sounds and similar) |

If `mode` is empty: text in `[brackets]` becomes `sounds`, text of more than 40 words becomes `whole`, and all other text becomes `split`. See `example_script.csv`.

## Correct a wrong match

If the tool puts a recording on the wrong line, make a `mapping.csv` with only that recording, and run the split again with `--mapping mapping.csv`:

```
source,character,line,mode
Track 2_015.wav,Witch,2,split
Track 1_001.wav,,,              <- empty character: skip this file
```

## Command line

```
python split_takes.py RECORDINGS OUT --script script.csv              # split and name
python split_takes.py SESSIONS OUT --script script.csv                # one subfolder for each actor
python split_takes.py ActorA/Raw ActorB/Raw OUT --script script.csv   # several folders
python split_takes.py RECORDINGS OUT --script script.csv --mapping mapping.csv
python split_takes.py final OUT                                       # copy the keepers
```

| option | default | meaning |
|---|---|---|
| `--min-gap` | 0.6 | seconds of silence between two reads |
| `--threshold` | 0.3 | 0 to 1: the level between noise floor and peak where sound starts |
| `--pad` | 0.2 | seconds kept before each read |
| `--tail` | 0.35 | seconds kept after each read |
| `--keep-slate` | off | do not remove the slate |

## Tips for the recording session

- Ask the actor to say "line" and the number before each line, in the same numbers as the script.
- Record the lines in script order when you can.
- Leave about one second of silence between reads.

## Limits

- The tool finds reads by sound level. Sometimes it splits one read at a long pause, or keeps two fast reads in one file. Change `--min-gap`, or cut that file by hand.
- Lines with only animal sounds or noises are hard to identify when they are recorded out of order. The tool marks them **check**.
- PocketSphinx is a small offline recognizer. It is good enough to match lines to a known script. Do not use it for accurate transcripts.
