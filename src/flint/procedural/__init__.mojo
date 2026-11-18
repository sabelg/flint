"""Flint Engine-Level Pure-Mojo Procedural Surface Modeling, Botanical SSS & Rigged Crowd Forge (`flint.procedural`).

Re-exports all public procedural modeling primitives across:
  - `subd`: Pixar/OpenSubdiv Catmull-Clark Subdivision Surfaces with semi-sharp creases & vehicle/coaster cages
  - `loft_bevel`: Beveled 2D extrusions, lathed 5-spoke wheels/brakes, and Wang Double-Reflection RMF spline lofts
  - `foliage`: `512x512` botanical alpha-cutout leaf/needle/frond atlases, fissured bark, gnarled trees & understory shrubs
  - `crowd_rig`: 3-submesh `16`-bone rigged SubD `Cyber-Enforcer` mech & `12`-pose multi-tone humanoid crowd bank
  - `arch_kit`: Clean back-wall `4x4` WindowBox atlas, `8` `512x512` cyber billboards & emissive-split urban kits.
"""
from flint.procedural.subd import (
    SubdCage,
    creased_subdivide,
    subdivide_to_mesh,
    build_subd_supercar_body_and_glass,
    build_subd_spinner_hovercar_body_and_canopy,
    build_subd_traffic_vehicle,
    build_subd_traffic_vehicle_parts,
    build_subd_coaster_car_shell,
    build_subd_coaster_car_parts,
)
from flint.procedural.loft_bevel import (
    extrude_profile_beveled,
    build_lathed_wheel_assembly,
    compute_rmf_frames,
    sweep_profile_rmf,
    sweep_profile_rmf_banked,
    make_chamfered_girder_profile,
    make_curb_profile,
    sweep_catenary_cable,
)
from flint.procedural.foliage import (
    bake_oak_leaf_sprig_512,
    bake_conifer_needle_sprig_512,
    bake_palm_frond_512,
    bake_fissured_bark_256,
    build_botanical_pine,
    build_botanical_oak,
    build_botanical_palm,
    build_urban_ginkgo_tree,
    build_grass_wildflower_tuft,
    build_manicured_hedge_and_planter,
    build_flowering_bush,
)
from flint.procedural.crowd_rig import (
    ENFORCER_BONE_COUNT,
    SkinnedMeshData,
    build_subd_cyber_enforcer_rigged,
    build_subd_cyber_enforcer_rigged_v3,
    build_articulated_crowd_pose_bank,
)
from flint.procedural.arch_kit import (
    bake_interior_windowbox_atlas_512,
    bake_cyber_billboard_512,
    build_colonnade_arcade_bay,
    build_facade_mullion_spandrel_grid,
    build_rooftop_hvac_chiller,
    build_rooftop_vtol_helipad,
    build_rooftop_vtol_helipad_parts,
    build_street_traffic_cantilever,
    build_street_traffic_cantilever_parts,
    build_street_bollard_and_grate,
    build_street_bollard_and_grate_parts,
    build_transit_shelter,
    build_transit_shelter_parts,
)
