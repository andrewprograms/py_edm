#!/usr/bin/env python3
"""
Groove Generator v4.4 – Transformer Edition (no TensorFlow)
===========================================================
* Generates a Euclidean-groove bass line + Transformer melody.
* Streams everything to the first MIDI OUT port.

Dependencies (PyPI only):
    pip install torch transformers miditok symusic pygame mido
"""

import random
import tempfile
import threading
import time
from pathlib import Path

import pygame.midi

# ─────────────────────────────────────────────────────────────────────────────
# Optional: export / import MIDI files
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mido
except ImportError:
    mido = None
    print("mido not installed – MIDI file I/O disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Transformer + tokenizer stack
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM
    from miditok import REMI, TokenizerConfig
    from symusic import Score
except ImportError:
    torch = AutoModelForCausalLM = REMI = None
    print("Install torch transformers miditok symusic for Transformer generation.")

# ─────────────────────────────────────────────────────────────────────────────
# User switches
# ─────────────────────────────────────────────────────────────────────────────
BPM             = 123
SWING           = 0           # 0–0.08 (16-note swing)
LENGTH_BARS     = 4
USE_MODEL       = True
MODEL_ID        = "Natooz/Maestro-REMI-bpe20k"   # Hugging-Face repo
KEEP_TEMP_MIDI  = False          # keep temp files?

# ─────────────────────────────────────────────────────────────────────────────
# MIDI output
# ─────────────────────────────────────────────────────────────────────────────
pygame.midi.init()
try:
    player = pygame.midi.Output(0)
except pygame.midi.MidiException:
    raise RuntimeError("No MIDI output device detected – connect one and retry.")

BASS_CH, LEAD_CH = 0, 1
VELOCITY          = 100

QUARTER   = 60.0 / BPM
EIGHTH    = QUARTER / 2
SIXTEENTH = QUARTER / 4

# ─────────────────────────────────────────────────────────────────────────────
# Music-theory helpers
# ─────────────────────────────────────────────────────────────────────────────
NOTE_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CHORD_ROOTS = ["E","C","D","B"]                 # i-VI-VII-v
QUALITIES   = {"E":"minor","C":"major","D":"major","B":"minor"}
INTERVALS   = {"major":[0,4,7], "minor":[0,3,7]}

def note_num(name: str, octave: int) -> int:
    return 12 * (octave + 1) + NOTE_NAMES.index(name)

def chord_notes(root: str, octave: int = 4):
    return [note_num(root, octave) + i for i in INTERVALS[QUALITIES[root]]]

# ─────────────────────────────────────────────────────────────────────────────
# Bass-line generator (Euclidean rhythm + 5th/octave fills)
# ─────────────────────────────────────────────────────────────────────────────
def euclid(p: int, n: int):
    bucket, pattern = 0, []
    for _ in range(n):
        bucket += p
        if bucket >= n:
            bucket -= n
            pattern.append(1)
        else:
            pattern.append(0)
    return pattern

BASS_PATTERNS = [
    {"rhythm": euclid(4, 8),       "intervals": [0, 0, 12, 0]},
    {"rhythm": [1,0,1,0,1,0,1,0], "intervals": [0, 7, 0, 7]},
    {"rhythm": [1,0,1,0,1,0,1,0], "intervals": [0, 12, 7, 12]},
]

def gen_bass(bars: int = LENGTH_BARS):
    events = []
    for bar in range(bars):
        root = CHORD_ROOTS[bar % len(CHORD_ROOTS)]
        root_m = note_num(root, 2)
        pat = random.choice(BASS_PATTERNS)
        iv_i = 0
        for step, hit in enumerate(pat["rhythm"]):
            if not hit:
                continue
            t = bar * 4 * QUARTER + step * EIGHTH
            if step % 2:
                t += SWING * EIGHTH
            interval = pat["intervals"][iv_i % len(pat["intervals"])]
            iv_i += 1
            events.append((t, root_m + interval, EIGHTH * 0.95, BASS_CH))
    return events

# ─────────────────────────────────────────────────────────────────────────────
# Melody fallback (chord-aware random walk)
# ─────────────────────────────────────────────────────────────────────────────
_BASE_W = {-2:1,-1:4,0:2,1:4,2:1,3:0.5,-3:0.5,4:0.2,-4:0.2}

def gen_lead_internal(bars: int = LENGTH_BARS):
    events, last = [], None
    for bar in range(bars):
        chord = chord_notes(CHORD_ROOTS[bar % len(CHORD_ROOTS)], 5)
        for sub in range(8):
            t = bar * 4 * QUARTER + sub * EIGHTH
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
                events.append((t, pitch, dur * 0.95, LEAD_CH))
    return events

# ─────────────────────────────────────────────────────────────────────────────
# MIDI helpers (mido) – non-negative delta times guaranteed
# ─────────────────────────────────────────────────────────────────────────────
def events_to_midi(events, path: str):
    if mido is None:
        raise RuntimeError("mido not installed – cannot export MIDI.")

    mid = mido.MidiFile(type=1)
    tr  = mido.MidiTrack()
    mid.tracks.append(tr)

    abs_msgs = []                        # (abs_tick, mido.Message)

    for t, pitch, dur, ch in events:
        on_tick  = int(mido.second2tick(t,        mid.ticks_per_beat, BPM))
        off_tick = int(mido.second2tick(t + dur,  mid.ticks_per_beat, BPM))

        abs_msgs.append((on_tick,
                         mido.Message("note_on",  note=pitch,
                                      velocity=VELOCITY, channel=ch, time=0)))
        abs_msgs.append((off_tick,
                         mido.Message("note_off", note=pitch,
                                      velocity=0,        channel=ch, time=0)))

    abs_msgs.sort(key=lambda x: x[0])

    prev = 0
    for tick, msg in abs_msgs:
        msg.time = tick - prev
        prev = tick
        tr.append(msg)

    mid.save(path)

def midi_to_events(path: str, ch_map: dict):
    if mido is None:
        raise RuntimeError("mido not installed – cannot parse MIDI.")
    mid = mido.MidiFile(path)
    abs_ticks = 0
    events = []
    for msg in mid:
        abs_ticks += msg.time
        if msg.type in ("note_on", "note_off") and msg.velocity > 0:
            sec = mido.tick2second(abs_ticks, mid.ticks_per_beat, BPM)
            ch  = ch_map.get(msg.channel, LEAD_CH)
            events.append((sec, msg.note, SIXTEENTH * 0.95, ch))
    return events

# ─────────────────────────────────────────────────────────────────────────────
# Transformer helpers
# ─────────────────────────────────────────────────────────────────────────────
_tokenizer_cache = None
_model_cache     = None
_device          = None

def load_transformer():
    global _tokenizer_cache, _model_cache, _device
    if _tokenizer_cache is not None:
        return _tokenizer_cache, _model_cache, _device
    if AutoModelForCausalLM is None:
        raise ImportError("Install torch transformers miditok symusic to use the model.")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{MODEL_ID}' on {_device} …")
    _model_cache = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", trust_remote_code=True
    ).to(_device).eval()

    try:
        _tokenizer_cache = REMI.from_pretrained(MODEL_ID)
    except Exception:
        _tokenizer_cache = REMI(TokenizerConfig())
    return _tokenizer_cache, _model_cache, _device

def _find_pad_id(tok: REMI):
    """Return a valid pad-token ID or None."""
    if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
        return tok.pad_token_id
    for name in ("PAD_None", "PAD", "PAD_Velocity"):
        if name in tok.vocab:
            return tok[name]
    return None

def transformer_generate(seed_path: Path, bars: int):
    try:
        tok, mdl, dev = load_transformer()
    except Exception as exc:
        print(f"Transformer init failed: {exc}")
        return None

    seed_tokens = tok.midi_to_tokens(seed_path)[0].ids

    max_pos = getattr(mdl.config, "n_positions",
                      getattr(mdl.config, "max_position_embeddings", 1024))
    if len(seed_tokens) >= max_pos:
        keep = max_pos // 2
        print(f"Seed too long ({len(seed_tokens)} tokens) – keeping last {keep}.")
        seed_tokens = seed_tokens[-keep:]

    room = max_pos - len(seed_tokens) - 1
    if room < 32:
        print("Not enough room for generation – using internal melody.")
        return None
    gen_tok_count = min(256, room)

    input_ids = torch.tensor(seed_tokens, device=dev).unsqueeze(0)
    pad_id = _find_pad_id(tok)

    gen_kwargs = dict(
        max_new_tokens=gen_tok_count,
        temperature=0.9,
        top_k=0,
        top_p=0.95,
        do_sample=True,
    )
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    with torch.no_grad():
        gen_ids = mdl.generate(input_ids, **gen_kwargs)[0].cpu().tolist()

    full_ids = seed_tokens + gen_ids

    try:
        score: Score = tok(full_ids, programs=[(0, False)])
        out_mid = seed_path.with_suffix(".gen.mid")
        score.dump_midi(out_mid)
    except Exception as exc:
        print(f"Token-to-MIDI decode failed: {exc}")
        return None

    return out_mid

# ─────────────────────────────────────────────────────────────────────────────
# Lead orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def make_lead(bars: int = LENGTH_BARS):
    bass_events = gen_bass(bars)

    seed_path = Path(tempfile.mktemp(prefix="groove_seed_", suffix=".mid"))
    events_to_midi(bass_events, seed_path)

    if USE_MODEL:
        gen_mid = transformer_generate(seed_path, bars)
        if gen_mid and gen_mid.exists():
            lead_events = midi_to_events(gen_mid, {0: LEAD_CH, 1: LEAD_CH})
            if not KEEP_TEMP_MIDI:
                seed_path.unlink(missing_ok=True)
                gen_mid.unlink(missing_ok=True)
            return bass_events + lead_events
        print("Model generation failed – falling back to internal melody.")

    return bass_events + gen_lead_internal(bars)

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler – sample-accurate playback
# ─────────────────────────────────────────────────────────────────────────────
def play(events):
    events = sorted(events, key=lambda e: e[0])
    start = time.perf_counter()

    def off_thread(pitch, ch, delay):
        time.sleep(delay)
        player.note_off(pitch, ch)

    try:
        for t, pitch, dur, ch in events:
            now = time.perf_counter() - start
            wait = t - now
            if wait > 0:
                time.sleep(wait)
            player.note_on(pitch, VELOCITY, ch)
            threading.Thread(target=off_thread,
                             args=(pitch, ch, dur),
                             daemon=True).start()
    finally:
        for ch in (BASS_CH, LEAD_CH):
            for n in range(128):
                player.note_off(n, ch)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    events = make_lead()
    play(events)
