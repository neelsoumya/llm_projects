# ALife Simulation Framework: Re-envisioning Life via Generative AI and Sci-Fi

import random
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION FROM LLM WORLD GENERATOR ---
world_config = {
    'name': 'Methanora',
    'temperature': 94,  # Kelvin
    'medium': 'liquid methane',
    'gravity': 0.7,  # relative to Earth
    'communication': 'pressure waves',
    'metabolism': 'chemical oscillations'
}

# --- SIMULATION WORLD ---
class World:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.config = config
        self.field = np.random.rand(height, width) * config['temperature']

# --- ALIEN AGENT ---
class Agent:
    def __init__(self, x, y, genome=None):
        self.x = x
        self.y = y
        self.energy = 1.0
        self.genome = genome or [random.random() for _ in range(5)]  # simple float genome
        self.age = 0

    def sense(self, world):
        # Sense environment locally
        env_value = world.field[self.y % world.height][self.x % world.width]
        return env_value

    def act(self, world):
        env = self.sense(world)
        self.energy += 0.1 * (1.0 - abs(env - world.config['temperature']) / 100)
        self.age += 1
        if random.random() < 0.1:
            self.move()

    def move(self):
        self.x += random.choice([-1, 0, 1])
        self.y += random.choice([-1, 0, 1])

    def can_reproduce(self):
        return self.energy > 1.5

    def reproduce(self):
        child_genome = [g + random.uniform(-0.05, 0.05) for g in self.genome]
        child = Agent(self.x, self.y, genome=child_genome)
        self.energy -= 0.5
        return child

# --- SIMULATION LOOP ---
def simulate(steps=100):
    world = World(50, 50, world_config)
    agents = [Agent(25, 25) for _ in range(10)]

    for step in range(steps):
        new_agents = []
        for agent in agents:
            agent.act(world)
            if agent.can_reproduce():
                new_agents.append(agent.reproduce())
        agents.extend(new_agents)
        agents = [a for a in agents if a.age < 100]  # natural death

        if step % 10 == 0:
            visualize(world, agents, step)

    return agents

# --- VISUALIZATION ---
def visualize(world, agents, step):
    plt.figure(figsize=(5, 5))
    plt.imshow(world.field, cmap='Blues', alpha=0.6)
    xs = [agent.x % world.width for agent in agents]
    ys = [agent.y % world.height for agent in agents]
    plt.scatter(xs, ys, color='red')
    plt.title(f"Step {step} - Population: {len(agents)}")
    plt.axis('off')
    plt.show()

# --- EXAMPLE PROMPT TO LLM (FOR EXPANSION) ---
example_prompt = "Design a lifeform that communicates via pressure waves in liquid methane. Describe its social behavior."

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Simulation parameters
WIDTH, HEIGHT = 100, 100
NUM_AGENTS = 5
TIME_STEPS = 300
SOUND_SPEED = 3  # wave expands by this much each step

# Methanoglot agent
class Methanoglot:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def move(self):
        dx, dy = random.randint(-1,1), random.randint(-1,1)
        self.x = np.clip(self.x + dx, 0, WIDTH - 1)
        self.y = np.clip(self.y + dy, 0, HEIGHT - 1)

    def emit_wave(self, t):
        return {
            'x': self.x,
            'y': self.y,
            'start_time': t,
            'radius': 0,
            'source': self.id
        }

# Initialize agents
agents = [Methanoglot(i, random.randint(0, WIDTH), random.randint(0, HEIGHT)) for i in range(NUM_AGENTS)]
all_wavefronts = []

# Set up plotting
fig, ax = plt.subplots()
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
agent_dots, = ax.plot([], [], 'bo')
wave_circles = []

def init():
    return [agent_dots]

def update(t):
    ax.clear()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_title(f"Time Step {t}")

    # Agents move and emit waves
    positions = []
    for agent in agents:
        agent.move()
        positions.append((agent.x, agent.y))
        if t % 5 == 0:  # emit wave every 5 steps
            wave = agent.emit_wave(t)
            all_wavefronts.append(wave)

    # Update and draw wavefronts
    still_active = []
    for wave in all_wavefronts:
        age = t - wave['start_time']
        if age >= 0:
            wave['radius'] = SOUND_SPEED * age
            circle = plt.Circle((wave['x'], wave['y']), wave['radius'], color='cyan', fill=False, alpha=0.4)
            ax.add_patch(circle)
            if wave['radius'] < WIDTH:
                still_active.append(wave)
    all_wavefronts[:] = still_active

    # Draw agents
    if positions:
        xs, ys = zip(*positions)
        ax.plot(xs, ys, 'bo')

    return []

#from google.colab import drive
#drive.mount('/content/drive')
#import os
#os.chdir('/content/drive/My Drive/Colab Notebooks')

# Run the simulation



# --- RUN THE SIMULATION ---
if __name__ == "__main__":

    ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, init_func=init, blit=False, interval=300)
    ani.save('wavefront_animation.gif', writer='pillow', fps=5)
    plt.show()

    # final_population = simulate(steps=100)
    #print(f"Final population size: {len(final_population)}")
    # Example: Extract evolved genome stats
    # avg_genome = np.mean([a.genome for a in final_population], axis=0)
    #print(f"Average genome: {avg_genome}")
