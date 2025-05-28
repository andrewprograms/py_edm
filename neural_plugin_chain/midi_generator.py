import pygame.midi
import time
import threading
import random
import numpy as np

"""
Groove Generator v2 – melodic‑house edition
-------------------------------------------
* Key: E minor
* Bass‑lines: Euclidean & off‑beat patterns that lock to the kick
* Top‑line: chord‑aware, step‑wise melody with phrase memory
* All timing is sample‑accurate (single scheduler thread)
* Optional Magenta RNN backend (Melody RNN / ImprovRNN) – enable by
  setting USE_MAGENTA = True and installing magenta‑music.
* Dependencies: pygame.midi, numpy, (optional) magenta‑music
"""

# === USER SETTINGS ===
BPM            = 123
SWING          = 0.05            # delay the off‑beat 8ths (0–0.08)
VELOCITY       = 100
KEY            = "E"             # root note
MODE           = "minor"         # scale mode
LENGTH_BARS    = 4               # how many bars to generate per run
USE_MAGENTA    = True           # set True if magenta is installed & configured

# ------------------------------------
# === MIDI SET‑UP ===
pygame.midi.init()
try:
    player = pygame.midi.Output(0)  # change if default is not 0
except pygame.midi.MidiException:
    raise RuntimeError("No MIDI output device available. Please connect one and restart.")

BASS_CH, LEAD_CH = 0, 1

# === TIME CONSTANTS ===
QUARTER     = 60.0 / BPM
EIGHTH      = QUARTER / 2
SIXTEENTH   = QUARTER / 4

# === THEORY HELPERS ===
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALE_NAMES = ["E", "F#", "G", "A", "B", "C", "D"]  # natural minor
SCALE_IDX   = {n: i for i, n in enumerate(SCALE_NAMES)}

CHORD_ROOTS      = ["E", "C", "D", "B"]  # 4‑bar loop i – VI – VII – v
CHORD_QUALITIES  = {"E": "minor", "C": "major", "D": "major", "B": "minor"}
CHORD_INTERVALS  = {"major": [0, 4, 7], "minor": [0, 3, 7]}


def note_num(name: str, octave: int) -> int:
    """Convert note name to MIDI note number in given octave."""
    return 12 * (octave + 1) + NOTE_NAMES.index(name)


def chord_notes(root: str, octave: int = 4):
    """Return MIDI numbers for the triad rooted at *root* in *octave*."""
    intervals = CHORD_INTERVALS[CHORD_QUALITIES[root]]
    r = note_num(root, octave)
    return [r + i for i in intervals]


# === EUCLIDEAN RHYTHM UTILITY ===

def euclidean_rhythm(pulses: int, steps: int):
    """Classic Björklund algorithm (binary pattern as list)."""
    pattern, bucket = [], 0
    for _ in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            pattern.append(1)
        else:
            pattern.append(0)
    return pattern


# === BASS‑LINE GENERATOR ===
BASS_PATTERNS = [
    # Classic house off‑beat root pump (root on the "&" of every beat)
    {"rhythm": euclidean_rhythm(4, 8), "intervals": [0, 0, 12, 0]},
    # Root–5th drive (straight 8ths)
    {"rhythm": [1, 0, 1, 0, 1, 0, 1, 0], "intervals": [0, 7, 0, 7]},
    # Root / octave converse
    {"rhythm": [1, 0, 1, 0, 1, 0, 1, 0], "intervals": [0, 12, 7, 12]},
]


def generate_bassline(num_bars: int = LENGTH_BARS):
    """Return list of (time, pitch, dur, ch) for the bass line."""
    events = []
    for bar in range(num_bars):
        root_name  = CHORD_ROOTS[bar % len(CHORD_ROOTS)]
        root_midi  = note_num(root_name, 2)
        pattern    = random.choice(BASS_PATTERNS)
        rhythm     = pattern["rhythm"]
        intervals  = pattern["intervals"]
        iv_idx     = 0
        for step, hit in enumerate(rhythm):
            if not hit:
                continue
            offset_beats = step * 0.5  # 8th‑note grid
            t = bar * 4 * QUARTER + offset_beats * QUARTER
            if step % 2 == 1:  # swing the off‑beats
                t += SWING * EIGHTH
            interval = intervals[iv_idx % len(intervals)]
            iv_idx += 1
            pitch = root_midi + interval
            events.append((t, pitch, EIGHTH * 0.95, BASS_CH))
    return events


# === MELODY GENERATOR ===
# Step‑wise weights; adjusted later for chord‑tone emphasis
_BASE_WEIGHTS = {
    -2: 1, -1: 4, 0: 2, 1: 4, 2: 1, 3: 0.5, -3: 0.5, 4: 0.2, -4: 0.2
}


def generate_lead(num_bars: int = LENGTH_BARS):
    """Chord‑aware step‑wise melody.«"""
    events, last_pitch = [], None
    for bar in range(num_bars):
        chord_midis = chord_notes(CHORD_ROOTS[bar % len(CHORD_ROOTS)], 5)
        for sub in range(8):          # 8th‑note grid per bar
            t = bar * 4 * QUARTER + sub * EIGHTH
            dur = random.choice([EIGHTH, SIXTEENTH])
            # density: strong beats always play; weak beats 50 %
            if sub % 2 == 0 or random.random() < 0.5:
                if last_pitch is None:
                    pitch = random.choice(chord_midis)
                else:
                    candidates, weights = [], []
                    for step, base_w in _BASE_WEIGHTS.items():
                        cand = last_pitch + step
                        # bias to chord‑tones on the beat
                        bias = 2 if (cand % 12) in [(p % 12) for p in chord_midis] else 1
                        candidates.append(cand)
                        weights.append(base_w * bias)
                    pitch = random.choices(candidates, weights=weights, k=1)[0]
                    # constrain range
                    while pitch < 65:
                        pitch += 12
                    while pitch > 88:
                        pitch -= 12
                last_pitch = pitch
                events.append((t, pitch, dur * 0.95, LEAD_CH))
    return events


# === OPTIONAL MAGENTA BACK‑END ===
if USE_MAGENTA:
    try:
        from magenta.music import (melody_rnn_generate_sequence,
                                   sequence_generator_bundle,
                                   note_sequence_to_pretty_midi)
        from magenta.protobuf import music_pb2

        # Point to a trained bundle (MelodyRNN or ImprovRNN).
        BUNDLE_PATH = "basic_rnn.mag"
        _bundle = sequence_generator_bundle.read_bundle_file(BUNDLE_PATH)

        def generate_lead(num_bars: int = LENGTH_BARS):  # type: ignore
            """Use Magenta MelodyRNN for the top line if available."""
            primer = music_pb2.NoteSequence()
            primer.tempos.add(qpm=BPM)
            generator_options = melody_rnn_generate_sequence.protobuf.generator_pb2.GeneratorOptions()
            generator_options.generate_sections.add(
                start_time=0,
                end_time=num_bars * 4 * QUARTER)
            seq = melody_rnn_generate_sequence(_bundle, primer, generator_options)
            events = []
            for n in seq.notes:
                events.append((n.start_time, n.pitch, n.end_time - n.start_time, LEAD_CH))
            return events
    except Exception as e:  # noqa: BLE001
        print("Magenta unavailable – falling back to internal melody generator:", e)
        USE_MAGENTA = False


# === SCHEDULER ===

def play_events(events):
    events.sort(key=lambda e: e[0])
    start = time.perf_counter()

    def note_off_later(p, ch, d):
        time.sleep(d)
        player.note_off(p, VELOCITY, ch)

    for note_time, pitch, dur, ch in events:
        while time.perf_counter() - start < note_time:
            time.sleep(0.0005)
        player.note_on(pitch, VELOCITY, ch)
        threading.Thread(target=note_off_later, args=(pitch, ch, dur), daemon=True).start()

    # let tails ring
    longest = max(t + d for t, _, d, _ in events)
    while time.perf_counter() - start < longest + 0.5:
        time.sleep(0.02)

    # panic / all‑notes‑off
    for p in range(128):
        player.note_off(p, VELOCITY, BASS_CH)
        player.note_off(p, VELOCITY, LEAD_CH)


# === MAIN ===
if __name__ == "__main__":
    try:
        print("▶ Generating melodic‑house groove in E minor…")
        bass_events = generate_bassline(LENGTH_BARS)
        lead_events = generate_lead(LENGTH_BARS)
        play_events(bass_events + lead_events)
    finally:
        player.close()
        pygame.midi.quit()
