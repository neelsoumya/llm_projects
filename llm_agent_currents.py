'''
LLM-driven ProtoLife simulation with wave dynamics and agent-based interactions. 

This code simulates a 2D environment where agents (ProtoLife) interact with wave dynamics,
sense local energy, move, reproduce, and die based on their genome and environmental conditions.


Compact Python demo that treats a 2D superconducting fluid as a wave medium and lets small, localized oscillatory regions (proto-life) emerge, emit pulses, move toward coherent oscillations and sometimes reproduce.

A scalar potential field V(x,y) evolves with a discrete 2-D wave-like update (toy wave equation with damping).

Tiny noise + agent emissions seed localized oscillatory patterns.

ProtoLife objects sense a smoothed local oscillatory energy, inject sinusoidal Gaussian pulses into the field, move up local energy gradients, and reproduce if they've accumulated enough energy.

When oscillatory energy concentrates and persists it looks like a self-sustaining “organism” in the current field — a simple ALife metaphor for your superconducting-helium currents idea.

highly exploratory / sci-fi and numerically permissive (low damping, relatively strong emissions). That produced visible instabilities and occasional numeric overflows (bright artifacts) — visually striking for sci-fi, but not physically realistic.

ncrease damping (e.g. DAMP = 0.08) — waves die out faster unless sustained by agents.

Lower agent emission strength (reduce genome emission range, e.g. strength in [0.2, 1.0]).

Clamp the field after each step: V_next = np.clip(V_next, -CLIP, CLIP) (CLIP ≈ 5–10) to avoid overflow.

Reduce grid size / time steps for faster iterations while developing.

Add stochastic death or metabolic cost so agents cannot pump unlimited energy.

'''

# use scipy's uniform_filter if available, otherwise fall back to a safe (but slower) box-mean implementation.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
from PIL import Image

# Parameters
NX, NY = 128, 128
TIME_STEPS = 200
C = 1.0
DAMP = 0.04
FORCE_NOISE = 1e-4
AGENT_DETECTION_THRESHOLD = 0.02
AGENT_REPRODUCE_ENERGY = 1.2
MAX_AGENTS = 60
CAPTURE_EVERY = 3

# Utilities
def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
        4 * Z
    )

def gauss_bump(shape, x0, y0, sigma, amplitude=1.0):
    ny, nx = shape
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# smoothing: try scipy, else fallback
try:
    from scipy.ndimage import uniform_filter
    def smooth_energy_field(arr, k):
        return uniform_filter(arr, size=k, mode='wrap')
except Exception as e:
    def smooth_energy_field(arr, k):
        pad = k//2
        arr_p = np.pad(arr, pad, mode='wrap')
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                block = arr_p[i:i+k, j:j+k]
                out[i,j] = block.mean()
        return out

# ProtoLife class
class ProtoLife:
    def __init__(self, x, y, genome=None):
        self.x = float(x)
        self.y = float(y)
        if genome is None:
            self.genome = [
                random.uniform(0.02, 0.12),
                random.uniform(0.5, 1.8),
                random.uniform(0.0, 0.9),
                random.uniform(0.015, 0.05)
            ]
        else:
            self.genome = genome
        self.energy = 0.9 + random.random() * 0.3
        self.age = 0

    def emit(self, t):
        freq, strength, mobility, thresh = self.genome
        phase = 2 * np.pi * freq * t
        amp = strength * (0.5 + 0.5 * np.sin(phase)) * (self.energy)
        return gauss_bump((NY, NX), self.x, self.y, sigma=2.0, amplitude=amp)

    def sense_local_energy(self, energy_field):
        ix = int(round(self.x)) % NX
        iy = int(round(self.y)) % NY
        local = energy_field[max(0, iy-2):iy+3, max(0, ix-2):ix+3]
        return local.mean()

    def step(self, energy_field):
        sense = self.sense_local_energy(energy_field)
        self.energy += 0.06 * sense
        self.energy *= 0.995
        self.energy -= 0.002
        self.age += 1

    def try_move(self, energy_field):
        ix = int(round(self.x)) % NX
        iy = int(round(self.y)) % NY
        best_dx, best_dy = 0, 0
        best_val = -1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx_ = (ix + dx) % NX
                ny_ = (iy + dy) % NY
                val = energy_field[ny_, nx_]
                if val > best_val:
                    best_val = val
                    best_dx, best_dy = dx, dy
        mob = self.genome[2]
        self.x = (self.x + best_dx * mob) % NX
        self.y = (self.y + best_dy * mob) % NY

    def can_reproduce(self):
        return self.energy > AGENT_REPRODUCE_ENERGY and random.random() < 0.12

    def reproduce(self):
        child_genome = [g + random.uniform(-0.02, 0.02) for g in self.genome]
        child = ProtoLife(self.x + random.uniform(-2, 2), self.y + random.uniform(-2, 2), genome=child_genome)
        self.energy *= 0.55
        child.energy = self.energy * 0.6
        return child

    def alive(self):
        return (self.energy > 0.05) and (self.age < 1200)

# Initialize
V_prev = np.random.normal(scale=1e-4, size=(NY, NX))
V = np.random.normal(scale=1e-4, size=(NY, NX))
agents = [ProtoLife(random.uniform(0, NX), random.uniform(0, NY)) for _ in range(6)]

frames = []
pop_history = []

import numpy as np
import matplotlib.pyplot as plt

def fig_to_rgb_array(fig):
    """
    Return an (H, W, 3) uint8 RGB numpy array for a Matplotlib Figure,
    trying several backend-compatible methods.
    """
    canvas = fig.canvas

    # Ensure the canvas has been drawn so buffer/size are correct
    fig.canvas.draw()

    # Obtain width & height
    try:
        w, h = canvas.get_width_height()
    except Exception:
        # fallback
        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)

    # 1) Preferred: tostring_rgb (present on Agg)
    if hasattr(canvas, "tostring_rgb"):
        buf = canvas.tostring_rgb()
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

    # 2) Mac backend: tostring_argb (ARGB order) -> convert to RGB
    if hasattr(canvas, "tostring_argb"):
        buf = canvas.tostring_argb()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # arr is ARGB -> reorder to RGB
        rgb = arr[:, :, [1, 2, 3]]
        return rgb

    # 3) buffer_rgba (returns raw RGBA bytes) -> drop alpha
    if hasattr(canvas, "buffer_rgba"):
        buf = canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, :3]

    # 4) print_to_buffer (returns bytes, (w,h)) — many backends implement this
    if hasattr(canvas, "print_to_buffer"):
        try:
            buf, (w2, h2) = canvas.print_to_buffer()
            # Some implementations give ARGB32, some give RGBA; we try to detect layout.
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h2, w2, 4)
            # Heuristic: if alpha channel seems to be first (ARGB), convert; else assume RGBA.
            # Check mean intensity of channel 0 (alpha) vs channel 3 (alpha candidate)
            c0_mean = arr[:, :, 0].mean()
            c3_mean = arr[:, :, 3].mean()
            if c0_mean > c3_mean * 1.1:
                # likely ARGB -> convert to RGB
                rgb = arr[:, :, [1, 2, 3]]
                return rgb
            else:
                # assume RGBA -> drop alpha
                return arr[:, :, :3]
        except Exception:
            pass

    # If we reach here, give an informative error
    raise RuntimeError("Could not extract RGB image from Matplotlib Figure canvas (no supported method).")

def render_frame(V, agents):
    """
    Render V + agents into an RGB numpy array (H, W, 3).
    Returns a uint8 numpy array suitable for Image.fromarray().
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(V, origin='lower', interpolation='bilinear')
    if agents:
        xs = [a.x % V.shape[1] for a in agents]
        ys = [a.y % V.shape[0] for a in agents]
        sizes = [max(8, 30 * a.energy) for a in agents]
        ax.scatter(xs, ys, s=sizes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, V.shape[1])
    ax.set_ylim(0, V.shape[0])

    # draw and convert
    rgb = fig_to_rgb_array(fig)
    plt.close(fig)
    return rgb


# Simulation loop
for t in range(TIME_STEPS):
    lap = laplacian(V)
    V_next = 2*V - V_prev + (C**2) * lap - DAMP * V
    V_next += np.random.normal(scale=FORCE_NOISE, size=V.shape)

    for ag in agents:
        V_next += ag.emit(t)

    vel = V_next - V_prev
    energy_field = vel**2 + V_next**2
    smooth_energy = smooth_energy_field(energy_field, 5)

    new_agents = []
    for ag in agents:
        ag.step(smooth_energy)
        ag.try_move(smooth_energy)
        if ag.can_reproduce() and len(agents) + len(new_agents) < MAX_AGENTS:
            child = ag.reproduce()
            new_agents.append(child)
    agents.extend(new_agents)

    # spontaneous births
    for _ in range(3):
        rx = random.randrange(NX)
        ry = random.randrange(NY)
        if smooth_energy[ry, rx] > AGENT_DETECTION_THRESHOLD and len(agents) < MAX_AGENTS:
            close = False
            for a in agents:
                if (abs(a.x - rx) < 4) and (abs(a.y - ry) < 4):
                    close = True
                    break
            if not close:
                agents.append(ProtoLife(rx+random.random(), ry+random.random()))

    agents = [a for a in agents if a.alive()]
    mask = smooth_energy > 0.9
    V_next[mask] *= 0.4
    V_prev = V.copy()
    V = V_next

    pop_history.append(len(agents))

    if (t % CAPTURE_EVERY) == 0:
        frame = render_frame(V, agents)
        frames.append(frame)

# Save GIF
pil_frames = [Image.fromarray(f) for f in frames]
gif_path = "~/soumya_cam_mac/code/llm_projects/superconducting_life.gif"
pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=80, loop=0)

# show final frame
display_img = Image.fromarray(frames[-1])
display_img

print("Saved GIF to:", gif_path)


# --- Stable-tuned snippet (paste into a notebook) ---
DAMP = 0.09
FORCE_NOISE = 5e-5
MAX_AGENTS = 40
TIME_STEPS = 180
C = 1.0
CLIP = 8.0  # clamp field to avoid blowups

# reduce agent emission strengths when initializing genome
# e.g. use strength in [0.2, 1.0] inside ProtoLife.__init__

# in the main loop, after computing V_next:
V_next = 2*V - V_prev + (C**2) * lap - DAMP * V
V_next += np.random.normal(scale=FORCE_NOISE, size=V.shape)
# agents emit as before but with reduced strength
#...
# clamp:
V_next = np.clip(V_next, -CLIP, CLIP)

#stable / slow” (calm, physically plausible-looking waves),

#“ecological” (lots of births & deaths, population plots),

#or “spectacular” (high-energy, visually dramatic — like the one I saved).

#richer agent genomes (signaling, energy budgets, cooperation).