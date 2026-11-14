"""Flat-level water: buoyancy and drag shared by bodies and characters.

Buoyancy compares a material's density with WATER_DENSITY: soil, clay,
grass and sand float; stone, crystal and metal sink, slowed by drag.
"""

comptime NO_WATER: Float32 = -1000
comptime WATER_DENSITY: Float32 = 1.6
comptime WATER_DRAG: Float32 = 2.4


def floats(material_density: Float32) -> Bool:
    return material_density > 0 and material_density < WATER_DENSITY


def submerged_fraction(bottom: Float32, height: Float32, water_level: Float32) -> Float32:
    if water_level <= NO_WATER + 1 or height <= 0:
        return 0
    return max(Float32(0), min(Float32(1), (water_level - bottom) / height))


def buoyant_acceleration(fraction: Float32, material_density: Float32, gravity: Float32) -> Float32:
    """Upward acceleration from displaced water (Archimedes)."""
    if fraction <= 0 or material_density <= 0:
        return 0
    return gravity * fraction * (WATER_DENSITY / material_density)
