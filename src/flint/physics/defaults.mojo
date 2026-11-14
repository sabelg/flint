"""Default 3D physics solver and force-volume tuning parameters for Flint.

Decouples `flint.physics.solver` and `flint.physics.forces` from game-specific
tunables while keeping the exact verified Box2D v3 Soft-Step baseline values.
"""

comptime PHYSICS_SUBSTEPS: Int = 4
comptime CONTACT_HERTZ: Float32 = 30.0
comptime CONTACT_DAMPING_RATIO: Float32 = 10.0
comptime CONTACT_PUSH_MAX_SPEED: Float32 = 3.0
comptime RESTITUTION_THRESHOLD: Float32 = 1.0
comptime JOINT_HERTZ: Float32 = 60.0
comptime JOINT_DAMPING_RATIO: Float32 = 2.0
comptime ROLLING_RESISTANCE: Float32 = 0.04
comptime BODY_MAX_SPEED: Float32 = 400.0
comptime WATER_ANGULAR_DRAG: Float32 = 1.6
