"""Generated from assets/deepsong/materials.json by tools/tunables.mojo; edit that file, not this one.

Mass per voxel in thousandths by material id (package 2.11, from the 2.4 material
table's mass_density), and the table's solver factors. bash tools/build.sh
regenerates this module before it compiles anything.
"""

comptime DENSITY_MATERIALS: Int = 25


def material_density_milli(material: Int) -> Int:
    """A material's mass per voxel in thousandths; air and unknown ids weigh nothing."""
    if material == 1:  # soil
        return 1200
    if material == 2:  # clay
        return 1500
    if material == 3:  # grass
        return 1100
    if material == 4:  # stone
        return 2500
    if material == 5:  # crystal
        return 2800
    if material == 6:  # metal
        return 5200
    if material == 7:  # sand
        return 1400
    if material == 8:  # explosive
        return 2100
    if material == 9:  # foundation
        return 8000
    if material == 10:  # granite
        return 2700
    if material == 11:  # bark
        return 700
    if material == 12:  # ice
        return 917
    if material == 13:  # gravel
        return 1800
    if material == 14:  # snow
        return 300
    if material == 15:  # mud
        return 1700
    if material == 16:  # wood
        return 600
    if material == 17:  # basalt
        return 2900
    if material == 18:  # dirt
        return 1300
    if material == 19:  # ash
        return 600
    if material == 20:  # magma
        return 2700
    if material == 21:  # moss
        return 2400
    if material == 22:  # springcap
        return 600
    if material == 23:  # lodestone
        return 5000
    if material == 24:  # belt
        return 1300
    return 0


def material_rolling_factor(material: Int) -> Float32:
    """Times the solver's ROLLING_RESISTANCE for a body of this material."""
    return 1


def material_wind_drag(material: Int) -> Float32:
    """Times WIND_DRAG for a body of this material."""
    if material == 14:  # snow
        return 1.5
    if material == 19:  # ash
        return 1.6
    return 1
