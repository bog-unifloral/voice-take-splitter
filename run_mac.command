#!/bin/bash
# Double-click this file to open the Voice take splitter window.
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Install it from https://www.python.org/downloads/ and run this again."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
echo "Checking the required packages (first run takes a minute)..."
python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt
python3 split_takes.py
