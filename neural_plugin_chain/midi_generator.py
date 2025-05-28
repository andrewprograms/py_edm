import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import random
from pathlib import Path

import pygame.midi

import mido

# ────────────────────────────────────────────────────────────────────────────────
# User switches
# ────────────────────────────────────────────────────────────────────────────────
BPM                = 123
SWING              = 0.05        # 0‑0.08
LENGTH_BARS        = 4
USE_MAGENTA_STUDIO = True        # hand‑off to Magenta Studio CLI
#   Hard‑coded full path to Generate.exe provided by the user
print('MEGENTA_GEN' in os.environ)
print(os.environ.get('MEGENTA_GEN'))
_MAGENTA_DIR = os.environ.get('MEGENTA_GEN')
print("magenta dir: ", _MAGENTA_DIR)
MAGENTA_BIN        = os.path.join(_MAGENTA_DIR , "Generate.exe")
KEEP_TEMP          = False       # keep exported MIDI for debugging

# ────────────────────────────────────────────────────────────────────────────────
# MIDI I/O setup
# ────────────────────────────────────────────────────────────────────────────────
pygame.midi.init()
try:
    player = pygame.midi.Output(0)
except pygame.midi.MidiException:
    raise RuntimeError("No MIDI output device. Connect one and restart.")

BASS_CH, LEAD_CH = 0, 1
VELOCITY = 100

# Time constants
QUARTER   = 60.0 / BPM
EIGHTH    = QUARTER / 2
SIXTEENTH = QUARTER / 4

# ────────────────────────────────────────────────────────────────────────────────
# Music theory helpers
# ────────────────────────────────────────────────────────────────────────────────
NOTE_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
SCALE_NAMES = ["E","F#","G","A","B","C","D"]  # natural minor
SCALE_IDX   = {n:i for i,n in enumerate(SCALE_NAMES)}
CHORD_ROOTS = ["E","C","D","B"]                 # i‑VI‑VII‑v
QUALITIES   = {"E":"minor","C":"major","D":"major","B":"minor"}
INTERVALS   = {"major":[0,4,7], "minor":[0,3,7]}

def note_num(name:str, octave:int)->int:
    return 12*(octave+1)+NOTE_NAMES.index(name)

def chord_notes(root:str, octave:int=4):
    return [note_num(root, octave)+i for i in INTERVALS[QUALITIES[root]]]

# ────────────────────────────────────────────────────────────────────────────────
# Bass‑line generator (Euclidean pulse + octave/fifth fills)
# ────────────────────────────────────────────────────────────────────────────────

def euclid(p:int, n:int):
    bucket, pat = 0, []
    for _ in range(n):
        bucket += p
        if bucket >= n:
            bucket -= n; pat.append(1)
        else:
            pat.append(0)
    return pat

BASS_PATTERNS = [
    {"rhythm": euclid(4,8), "intervals": [0,0,12,0]},
    {"rhythm": [1,0,1,0,1,0,1,0], "intervals": [0,7,0,7]},
    {"rhythm": [1,0,1,0,1,0,1,0], "intervals": [0,12,7,12]},
]

def gen_bass(bars:int=LENGTH_BARS):
    events=[]
    for bar in range(bars):
        root = CHORD_ROOTS[bar%len(CHORD_ROOTS)]
        root_m = note_num(root, 2)
        pat = random.choice(BASS_PATTERNS)
        iv_i = 0
        for step, hit in enumerate(pat["rhythm"]):
            if not hit:
                continue
            t = bar*4*QUARTER + step*EIGHTH
            if step % 2:
                t += SWING * EIGHTH
            interval = pat["intervals"][iv_i % len(pat["intervals"])]
            iv_i += 1
            events.append((t, root_m + interval, EIGHTH*0.95, BASS_CH))
    return events

# ────────────────────────────────────────────────────────────────────────────────
# Melody generator (internal fallback – chord‑aware step walker)
# ────────────────────────────────────────────────────────────────────────────────
_BASE_W = {-2:1,-1:4,0:2,1:4,2:1,3:0.5,-3:0.5,4:0.2,-4:0.2}

def gen_lead_internal(bars:int=LENGTH_BARS):
    events, last = [], None
    for bar in range(bars):
        chord = chord_notes(CHORD_ROOTS[bar%len(CHORD_ROOTS)], 5)
        for sub in range(8):
            t = bar*4*QUARTER + sub*EIGHTH
            dur = random.choice([EIGHTH, SIXTEENTH])
            if sub % 2 == 0 or random.random() < 0.5:
                if last is None:
                    pitch = random.choice(chord)
                else:
                    cand, w = [], []
                    for step, bw in _BASE_W.items():
                        p = last + step
                        bias = 2 if (p % 12) in [c % 12 for c in chord] else 1
                        cand.append(p); w.append(bw * bias)
                    pitch = random.choices(cand, weights=w, k=1)[0]
                    while pitch < 65: pitch += 12
                    while pitch > 88: pitch -= 12
                last = pitch
                events.append((t, pitch, dur*0.95, LEAD_CH))
    return events

# ────────────────────────────────────────────────────────────────────────────────
# Magenta Studio hand‑off helpers
# ────────────────────────────────────────────────────────────────────────────────

def events_to_midi(events, path:str):
    if mido is None:
        raise RuntimeError("mido not installed – cannot export MIDI.")
    mid = mido.MidiFile(type=1)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    last_tick = 0
    for t,p,d,ch in sorted(events, key=lambda e:e[0]):
        on_tick = int(mido.second2tick(t, mid.ticks_per_beat, BPM))
        tr.append(mido.Message('note_on', note=p, velocity=VELOCITY, channel=ch, time=on_tick-last_tick))
        off_tick = int(mido.second2tick(d, mid.ticks_per_beat, BPM))
        tr.append(mido.Message('note_off', note=p, velocity=0, channel=ch, time=off_tick))
        last_tick = on_tick + off_tick
    mid.save(path)


def midi_to_events(path:str, ch_map:dict):
    if mido is None:
        raise RuntimeError("mido not installed – cannot parse MIDI.")
    mid = mido.MidiFile(path)
    abs_ticks = 0
    events = []
    for msg in mid:
        abs_ticks += msg.time
        if msg.type in ('note_on', 'note_off') and msg.velocity > 0:
            sec = mido.tick2second(abs_ticks, mid.ticks_per_beat, BPM)
            ch  = ch_map.get(msg.channel, LEAD_CH)
            events.append((sec, msg.note, SIXTEENTH*0.95, ch))
    return events


def call_magenta_generate(seed:Path, bars:int):
    """Invoke the hard‑coded Generate.exe and return first generated MIDI."""
    exe = Path(MAGENTA_BIN)
    if not exe.exists():
        print(f"Generate.exe not found at {exe} – falling back to internal melody generator.")
        return None

    out_dir = Path(tempfile.mkdtemp(prefix="magenta_out_"))
    cmd = [str(exe),
           "--input", str(seed),
           "--output", str(out_dir),
           "--length", str(bars*4),
           "--temperature", "0.5",
           "--num_outputs", "4"]

    run_kwargs = dict(check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # exe is an .exe on Windows – no shell=True needed
        subprocess.run(cmd, **run_kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Generate.exe failed ({e}) – falling back to internal generator.")
        return None
    except PermissionError:
        print("Permission denied launching Generate.exe – unblock the file or run as admin")
        return None

    files = sorted(out_dir.glob("*.mid"))
    if not files:
        print("Generate.exe produced no MIDI – falling back.")
        return None
    return files[0]

# ────────────────────────────────────────────────────────────────────────────────
# Scheduler – sample‑accurate, daemonized note‑off
# ────────────────────────────────────────────────────────────────────────────────

def play(events):
    events = sorted(events, key=lambda e:e[0])
    start = time.perf_counter()

    def note_off(pitch, ch, dur):
        time.sleep(dur)
        player.note_off(pitch, VELOCITY, ch)

    for t,p,d,ch in events:
        while time.perf_counter() - start < t:
            time.sleep(0.0005)
        player.note_on(p, VELOCITY, ch)
        threading.Thread(target=note_off, args=(p,ch,d), daemon=True).start()

    longest = max(t+d for t,_,d,_ in events)
    while time.perf_counter() - start < longest + 0.5:
        time.sleep(0.02)
    for n in range(128):
        player.note_off(n, VELOCITY, BASS_CH)
        player.note_off(n, VELOCITY, LEAD_CH)

# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        bass = gen_bass(LENGTH_BARS)

        if USE_MAGENTA_STUDIO and mido is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                seed_path = Path(tmpdir) / "seed.mid"
                events_to_midi(gen_lead_internal(LENGTH_BARS), seed_path)
                gen_path = call_magenta_generate(seed_path, LENGTH_BARS)
                if gen_path is not None:
                    lead = midi_to_events(gen_path, {0: LEAD_CH})
                    if KEEP_TEMP:
                        Path("mag_seed.mid").write_bytes(seed_path.read_bytes())
                        Path("mag_gen.mid").write_bytes(gen_path.read_bytes())
                else:
                    lead = gen_lead_internal(LENGTH_BARS)
        else:
            lead = gen_lead_internal(LENGTH_BARS)

        play(bass + lead)
    finally:
        player.close()
        pygame.midi.quit()
