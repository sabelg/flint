"""Console variables: named settings a developer changes while the game runs.

Each one has a `CVAR_` constant, used at every call site, and a line in the
table in `cvar_entry`. Code reads `cvars.on(CVAR_DEBUG_DRAW)`, which is a list
lookup, so reading one on a hot path costs nothing worth noticing; names are
only for people, through the console and `SD_CVARS`.

They are presentation. `NativeRuntime` owns them and nothing inside a tick may
read them, so no console variable can change the simulation, a replay or a
save. Settings a player owns stay in the settings menu. The one that reaches
the simulation, physics.serial, is copied in between ticks and only changes
which thread solves what: the solver gives the same bits either way
(tests/test_solver.mojo).

`SD_CVARS="debug.draw=1,debug.lights=1"` sets them at start. A flag written
alone, as in `SD_CVARS=debug.draw`, turns on.
"""
from flint.core.memory import MEM_TAGS, mem_tag_key, default_budget_mib

comptime CVAR_DEBUG_DRAW: Int = 0
comptime CVAR_DEBUG_PLAYER: Int = 1
comptime CVAR_DEBUG_ENEMIES: Int = 2
comptime CVAR_DEBUG_BODIES: Int = 3
comptime CVAR_DEBUG_COLLISION: Int = 4
comptime CVAR_DEBUG_CHUNKS: Int = 5
comptime CVAR_DEBUG_LIGHTS: Int = 6
comptime CVAR_DEBUG_CAMERA: Int = 7
comptime CVAR_DEBUG_LABELS: Int = 8
comptime CVAR_DEBUG_RANGE: Int = 9
comptime CVAR_LOG_LEVEL: Int = 10
comptime CVAR_LOG_ECHO: Int = 11
comptime CVAR_GPU_FENCE_MS: Int = 12
comptime CVAR_GPU_HANG_LIMIT: Int = 13
comptime CVAR_MEM_OVERLAY: Int = 14
comptime CVAR_MEM_DUMP: Int = 15
comptime CVAR_DEBUG_AREAS: Int = 16
comptime CVAR_TERRAIN_GPU_MESH: Int = 17
comptime CVAR_TERRAIN_MESH_BUDGET: Int = 18
comptime CVAR_DEBUG_CONTACTS: Int = 19  # the physics debug views (package 2.11)
comptime CVAR_DEBUG_NORMALS: Int = 20
comptime CVAR_DEBUG_IMPULSES: Int = 21
comptime CVAR_DEBUG_ISLANDS: Int = 22
comptime CVAR_DEBUG_COLOURS: Int = 23
comptime CVAR_DEBUG_SLEEP: Int = 24
comptime CVAR_DEBUG_AABBS: Int = 25
comptime CVAR_DEBUG_COM: Int = 26
comptime CVAR_PHYSICS_SERIAL: Int = 27  # the solver's forced serial path (package 2.2)
comptime CVAR_DEBUG_JOINTS: Int = 28
comptime CVAR_DEBUG_FORCES: Int = 29  # field volumes, water and ropes (package 2.12)
comptime CVAR_DEBUG_WIND: Int = 30    # the wind field as arrows, and the vorticles
comptime CVAR_DIG_PREVIEW: Int = 31   # what the next punch or tear would take (package 2.6)
comptime CVAR_TERRAIN_RESET: Int = 32  # rebuild the layer's terrain from its brushes, once
comptime CVAR_TERRAIN_RESET_HERE: Int = 33  # put the layer's own terrain back within this many metres of Maroon, once
comptime CVAR_PARTICLES_GPU: Int = 34  # particles simulated and drawn on the GPU (package 2.7)
comptime CVAR_PARTICLES_STATS: Int = 35  # the GPU particle counters on the debug HUD
comptime CVAR_DEBUG_GROUND: Int = 36  # loose ground and cave-ins (package 2.8)
comptime CVAR_GROUND_DEMO: Int = 37  # build loose-ground test pieces ahead of Maroon, once
comptime CVAR_DEBUG_FIELDS: Int = 38  # field blocks, burning cells, element states (package 2.9)
comptime CVAR_FIELDS_DEMO: Int = 39  # build field test pieces ahead of Maroon, once
comptime CVAR_CAMERA_DITHER: Int = 40  # occluders thin out around Maroon (package 3.2)
comptime CVAR_SKIN_METHOD: Int = 41  # how skinned meshes deform (package 3.4)
comptime CVAR_SKIN_GPU: Int = 42     # skin on the GPU when a GPU backend draws
comptime CVAR_SKILLS_DEMO: Int = 43   # every bear skill owned and test pieces ahead of Maroon, once (package 3.8)
comptime CVAR_FORMS_DEMO: Int = 44    # every form unlocked and test pieces ahead of Maroon, once (package 3.9)
# Package 2.13: the effects library, its comfort scales (3.15's options set
# them) and its demo and debug view.
comptime CVAR_FX: Int = 45
comptime CVAR_FX_VIGNETTE: Int = 46
comptime CVAR_FX_LOW_HEALTH: Int = 47
comptime CVAR_FX_SPEED_LINES: Int = 48
comptime CVAR_FX_DROPLETS: Int = 49
comptime CVAR_FX_DEMO: Int = 50
comptime CVAR_DEBUG_FX: Int = 51
# Package 2.14: movers and layer mechanics.
comptime CVAR_MECH_DEMO: Int = 52
comptime CVAR_DEBUG_MECH: Int = 53
comptime CVAR_LOGIC_DEMO: Int = 54  # package 3.13
comptime CVAR_MEM_BUDGET: Int = 55  # mem.budget.<tag>, one per memory tag, in tag order
comptime CVAR_COUNT: Int = CVAR_MEM_BUDGET + MEM_TAGS
comptime CVAR_KEEP: Float32 = -1  # the safety variables' default: keep what start-up chose


@fieldwise_init
struct CVarEntry(Copyable, Movable):
    var name: String
    var default: Float32
    var flag: Bool
    var help: String


def cvar_entry(index: Int) -> CVarEntry:
    if index == CVAR_DEBUG_DRAW:
        return CVarEntry("debug.draw", 0, True, "all world debug drawing")
    if index == CVAR_DEBUG_PLAYER:
        return CVarEntry("debug.player", 1, True, "player collision capsule, ground state and rays, ledge, ride, safe ring")
    if index == CVAR_DEBUG_ENEMIES:
        return CVarEntry("debug.enemies", 1, True, "enemy boxes and states, projectiles")
    if index == CVAR_DEBUG_BODIES:
        return CVarEntry("debug.bodies", 1, True, "physics bodies and their contact spheres")
    if index == CVAR_DEBUG_COLLISION:
        return CVarEntry("debug.collision", 0, True, "static collision and prop boxes")
    if index == CVAR_DEBUG_CHUNKS:
        return CVarEntry("debug.chunks", 0, True, "voxel chunk bounds, dirty ones red")
    if index == CVAR_DEBUG_LIGHTS:
        return CVarEntry("debug.lights", 0, True, "selected lights and their reach")
    if index == CVAR_DEBUG_CAMERA:
        return CVarEntry("debug.camera", 0, True, "camera target and boom")
    if index == CVAR_DEBUG_LABELS:
        return CVarEntry("debug.labels", 1, True, "text beside what is drawn")
    if index == CVAR_DEBUG_RANGE:
        return CVarEntry("debug.range", 40, False, "metres from the camera to draw")
    if index == CVAR_LOG_LEVEL:
        return CVarEntry("log.level", CVAR_KEEP, False, "log entries kept: 0 trace .. 5 fatal, 6 off; -1 keeps SD_LOG's")
    if index == CVAR_LOG_ECHO:
        return CVarEntry("log.echo", CVAR_KEEP, False, "log entries also printed, from this level up; -1 keeps warn")
    if index == CVAR_GPU_FENCE_MS:
        return CVarEntry("gpu.fence.ms", CVAR_KEEP, False, "milliseconds before a GPU frame counts as hung; -1 keeps 2000")
    if index == CVAR_GPU_HANG_LIMIT:
        return CVarEntry("gpu.hang.limit", CVAR_KEEP, False, "amdgpu hangs before moving to Vulkan; -1 keeps 3")
    if index == CVAR_MEM_OVERLAY:
        return CVarEntry("mem.overlay", 0, True, "the memory page: tags, pools, arenas (F4)")
    if index == CVAR_MEM_DUMP:
        return CVarEntry("mem.dump", 0, True, "write memory.json to the state folder once, then turn off")
    if index == CVAR_DEBUG_AREAS:
        return CVarEntry("debug.areas", 0, True, "the layer's fog, camera, audio and trigger areas")
    if index == CVAR_TERRAIN_GPU_MESH:
        return CVarEntry("terrain.gpu_mesh", 1, True, "mesh terrain on the GPU when a GPU backend draws (package 1.3)")
    if index == CVAR_TERRAIN_MESH_BUDGET:
        return CVarEntry("terrain.mesh_budget", 8, False, "dirty chunks remeshed per frame; the rest wait")
    if index == CVAR_DEBUG_CONTACTS:
        return CVarEntry("debug.physics.contacts", 0, True, "body contact points: terrain yellow, bodies magenta, static grey")
    if index == CVAR_DEBUG_NORMALS:
        return CVarEntry("debug.physics.normals", 0, True, "contact normals, out of what was touched")
    if index == CVAR_DEBUG_IMPULSES:
        return CVarEntry("debug.physics.impulses", 0, True, "contact normal impulses as lines, longer is harder")
    if index == CVAR_DEBUG_ISLANDS:
        return CVarEntry("debug.physics.islands", 0, True, "awake bodies coloured by the touching group they are in")
    if index == CVAR_DEBUG_COLOURS:
        return CVarEntry("debug.physics.colours", 0, True, "contacts coloured by a graph colouring (no colour twice on a body)")
    if index == CVAR_DEBUG_SLEEP:
        return CVarEntry("debug.physics.sleep", 0, True, "bodies asleep blue, awake green, with ticks at rest")
    if index == CVAR_DEBUG_AABBS:
        return CVarEntry("debug.physics.aabbs", 0, True, "bodies' world boxes, as the broadphase sees them")
    if index == CVAR_DEBUG_COM:
        return CVarEntry("debug.physics.com", 0, True, "centres of mass and principal axes of inertia")
    if index == CVAR_PHYSICS_SERIAL:
        return CVarEntry("physics.serial", 0, True, "solve every contact on one worker, in colour order (the same result; for checking the parallel path)")
    if index == CVAR_DEBUG_JOINTS:
        return CVarEntry("debug.physics.joints", 0, True, "joints white (broken red), motors orange to their target; hinge, slide and twist axes cyan")
    if index == CVAR_DEBUG_FORCES:
        return CVarEntry("debug.physics.forces", 0, True, "field volumes by target, water surfaces blue, ropes, bodies' field pushes")
    if index == CVAR_DEBUG_WIND:
        return CVarEntry("debug.physics.wind", 0, True, "the wind around Maroon as arrows (longer is faster), vorticles as spheres of their reach")
    if index == CVAR_DIG_PREVIEW:
        return CVarEntry("dig.preview", 1, True, "outline what the next punch would break (white), what would fall after it (orange) and what a tear would take (cyan)")
    if index == CVAR_TERRAIN_RESET:
        return CVarEntry("terrain.reset", 0, True, "rebuild this layer's terrain from its brushes, journal and wear cleared, once; not while recording")
    if index == CVAR_TERRAIN_RESET_HERE:
        return CVarEntry("terrain.reset_here", 0, False, "put the layer's own terrain back within this many metres of Maroon (journaled), once; not while recording")
    if index == CVAR_PARTICLES_GPU:
        return CVarEntry("particles.gpu", 1, True, "simulate and draw particles on the GPU when a GPU backend draws; off: the CPU's simpler particles (package 2.7)")
    if index == CVAR_PARTICLES_STATS:
        return CVarEntry("particles.stats", 0, True, "count GPU particle contacts and splatted pixels, and print the counters on the debug HUD")
    if index == CVAR_DEBUG_GROUND:
        return CVarEntry("debug.ground", 0, True, "loose ground: waiting cave-ins red with their ticks left, awake settle boxes yellow, supports green (broken red), noises, the shedding level")
    if index == CVAR_GROUND_DEMO:
        return CVarEntry("ground.demo", 0, True, "build a sand block, a soil shelf on a post and a timber-held ceiling ahead of Maroon (journaled), once; not while recording")
    if index == CVAR_DEBUG_FIELDS:
        return CVarEntry("debug.fields", 0, True, "fields: active blocks (red hot, blue wet, purple gas, green growing), burning cells orange, field areas, element states over what holds them, breath")
    if index == CVAR_FIELDS_DEMO:
        return CVarEntry("fields.demo", 0, True, "build a lit timber frame (posts, a plank roof, a stone block on it), magma under a seep, a gas pocket and a damp stone wall ahead of Maroon (journaled), once; not while recording")
    if index == CVAR_CAMERA_DITHER:
        return CVarEntry("camera.dither", 1, True, "thin out rock and props between the camera and Maroon around him (0: draw them solid, to compare)")
    if index == CVAR_SKIN_METHOD:
        return CVarEntry("skin.method", -1, False, "how skinned meshes deform: -1 as each asset's manifest says (linear blend when it does not), 0 linear blend, 1 dual quaternions, 2 centers of rotation")
    if index == CVAR_SKIN_GPU:
        return CVarEntry("skin.gpu", 1, True, "skin on the GPU when a GPU backend draws: the pose shown is one frame old (0: skin on the CPU, this frame's pose)")
    if index == CVAR_SKILLS_DEMO:
        return CVarEntry("skills.demo", 0, True, "own every bear skill (clap, hug, uproot, belly slide, burrow) and build snow and ice to slide on, a dirt bank to burrow and a thin stone wall three metres right to clap ahead of Maroon (journaled), once; not while recording")
    if index == CVAR_FORMS_DEMO:
        return CVarEntry("forms.demo", 0, True, "unlock every form with full energy and build a stone wall to wall run along (right), a metal block to smash (ahead), a thin-roofed box to break through (left) and an updraft to glide up (further ahead) by Maroon (journaled; the updraft is not saved), once; not while recording")
    if index == CVAR_FX:
        return CVarEntry("fx.on", 1, True, "the effects library: decals, beams, swipes, flashes, dissolves, pickups, screen effects, ambient emitters, fire (package 2.13; 0: none of them)")
    if index == CVAR_FX_VIGNETTE:
        return CVarEntry("fx.vignette", 1, True, "comfort scale of the red vignette when Maroon is hurt, 0 to 1")
    if index == CVAR_FX_LOW_HEALTH:
        return CVarEntry("fx.low_health", 1, True, "comfort scale of the low-health pulse, 0 to 1")
    if index == CVAR_FX_SPEED_LINES:
        return CVarEntry("fx.speed_lines", 1, True, "comfort scale of the Dasher's speed lines, 0 to 1")
    if index == CVAR_FX_DROPLETS:
        return CVarEntry("fx.droplets", 1, True, "comfort scale of the drops on the lens after leaving water, 0 to 1")
    if index == CVAR_FX_DEMO:
        return CVarEntry("fx.demo", 0, True, "show every effect of the library around Maroon for a while (decals, a light beam, lightning, ambient emitters, a pickup, the screen effects), once; presentation only")
    if index == CVAR_DEBUG_FX:
        return CVarEntry("debug.fx", 0, True, "the effects library: decal boxes (green live, red refused), ambient emitter areas and their counts, the budget on the debug HUD")
    if index == CVAR_MECH_DEMO:
        return CVarEntry("mechanics.demo", 0, True, "level the ground ahead of Maroon and build the mechanics proving ground there: a lift (stand on it), a conveyor, a piston that crushes, a door that stops, a tank that floods and drains, a magnet wall with an iron staircase and a three-mirror beam puzzle that opens a gate (journaled, recorded as a dev record), once; not in scripted runs")
    if index == CVAR_DEBUG_MECH:
        return CVarEntry("debug.mechanics", 0, True, "movers (box, path, blocked), conveyor areas and their direction, magnets (reach, what they hold), pools (level), beam paths and receivers, and the signals on the debug HUD")
    if index == CVAR_LOGIC_DEMO:
        return CVarEntry("logic.demo", 0, True, "level the ground ahead of Maroon and build the gameplay logic proving ground there: a checkpoint pad, a door that opens while he stands at it, a lever (E) that opens a gate, a roar-then-clap seal, and the Frost Warden's locked arena with three ice cores and two sleeping guards (journaled, recorded as a dev record), once; not in scripted runs")
    if index >= CVAR_MEM_BUDGET and index < CVAR_COUNT:
        var tag = index - CVAR_MEM_BUDGET
        return CVarEntry("mem.budget." + mem_tag_key(tag), default_budget_mib(tag), False, "MiB the " + mem_tag_key(tag) + " memory tag may hold; 0 = no budget")
    return CVarEntry("", 0, False, "")


def show_value(value: Float32) -> String:
    """A whole number without its decimals, otherwise as Float32 prints."""
    if value == Float32(Int(value)):
        return String(Int(value))
    return String(value)


def parse_value(text: String) -> Optional[Float32]:
    try:
        return Float32(Float64(text))
    except:
        return None


struct ConsoleVars(Copyable, Movable):
    var values: List[Float32]

    def __init__(out self):
        self.values = List[Float32](capacity=CVAR_COUNT)
        for index in range(CVAR_COUNT):
            self.values.append(cvar_entry(index).default)

    def get(self, index: Int) -> Float32:
        return self.values[index]

    def on(self, index: Int) -> Bool:
        return self.values[index] != 0

    def set(mut self, index: Int, value: Float32):
        self.values[index] = (Float32(1) if value != 0 else Float32(0)) if cvar_entry(index).flag else value

    def find(self, name: String) -> Int:
        for index in range(CVAR_COUNT):
            if cvar_entry(index).name == name:
                return index
        return -1

    def describe(self, index: Int) -> String:
        return cvar_entry(index).name + " = " + show_value(self.values[index])

    def text(self) -> String:
        """Every variable, one `name = value` line each, as a crash report lists them."""
        var out = String()
        for index in range(CVAR_COUNT):
            out += self.describe(index) + "\n"
        return out^

    def apply(mut self, text: String) -> Int:
        """Set variables from `name=value,name=value`. Returns how many it set;
        an unknown name or a bad value is reported and skipped."""
        var applied = 0
        for part in text.split(","):
            var item = String(part).strip()
            if item.byte_length() == 0:
                continue
            var pieces = String(item).split("=")
            var name = String(pieces[0]).strip()
            var index = self.find(String(name))
            if index < 0:
                print("CVAR_UNKNOWN " + String(name))
                continue
            if len(pieces) == 1:
                if not cvar_entry(index).flag:
                    print("CVAR_NEEDS_VALUE " + String(name))
                    continue
                self.set(index, 1)
                applied += 1
                continue
            var value = parse_value(String(String(pieces[1]).strip()))
            if value is None:
                print("CVAR_BAD_VALUE " + String(item))
                continue
            self.set(index, value.value())
            applied += 1
        return applied
