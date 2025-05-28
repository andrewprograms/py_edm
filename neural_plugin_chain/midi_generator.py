"""
Live E-minor bassline + melody (saw lead) at 123 BPM.
Requires: pip install numpy sounddevice
"""
import time, threading, numpy as np, sounddevice as sd

BPM = 123
SR  = 48_000           # sample rate
VELOCITY = 0.4         # master volume 0-1
QUARTER = 60 / BPM
EIGHTH  = QUARTER / 2

SCALE = ['E','F#','G','A','B','C','D']
ROOTS = ['E','G','A','B']
NOTE_NUM = {n:i for i,n in enumerate(
   ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])}

def hz(note, octv):
    n = 12*(octv+1)+NOTE_NUM[note]-69
    return 440*2**(n/12)

def saw(freq, dur):
    t = np.linspace(0, dur, int(SR*dur), False)
    return (2*(t*freq%1)-1) * 0.6           # raw saw

def adsr(sig, atk=.01, dec=.05, sus=.7, rel=.05):
    n = len(sig); env = np.ones(n)
    a = int(atk*SR); d = int(dec*SR); r = int(rel*SR)
    env[:a] = np.linspace(0,1,a)
    env[a:a+d] = np.linspace(1,sus,d)
    env[-r:] = np.linspace(sus,0,r)
    return sig*env

def gen_events():
    ev=[]
    # 4-bar off-beat bass
    for bar in range(4):
        for beat in range(4):
            t0 = (bar*4+beat)*QUARTER+EIGHTH
            note = np.random.choice(ROOTS)
            ev.append((t0,hz(note,2),EIGHTH*0.95,'bass'))
    # 16-beat melody
    t=0
    while t<16*QUARTER:
        dur = QUARTER*np.random.choice([.5,1.0])
        note = np.random.choice(SCALE)
        ev.append((t, hz(note,5), dur,'lead'))
        t+=dur
    return sorted(ev,key=lambda x:x[0])

def render():
    ev = gen_events()
    longest = max(t+d for t,_,d,_ in ev)+1
    buf = np.zeros(int(SR*longest))
    for t,f,d,kind in ev:
        wave = saw(f,d)
        wave = adsr(wave) * (0.6 if kind=='lead' else 0.4)
        i = int(t*SR)
        buf[i:i+len(wave)] += wave
    return np.clip(buf*VELOCITY,-1,1)

def stream():
    audio = render()
    sd.play(audio, SR)
    sd.wait()

if __name__ == "__main__":
    print("🔊 Playing live E-minor groove… Ctrl-C to stop.")
    audio = render()        # build the waveform
    sd.play(audio, SR)      # start playback
    sd.wait()               # block until it finishes