"""`flint.ecs` — Flint v5 Hybrid Archetype + SparseSet ECS v2.

Re-exports all public ECS v2 primitives from `flint.ecs.world`.
"""
from flint.core.ecs import EntityHandle
from flint.ecs.world import (
    MAX_COMPONENTS,
    STORAGE_TABLE,
    STORAGE_SPARSE,
    CHUNK_ALIGN,
    SIMD_LANES,
    DEFAULT_CHUNK_ROWS,
    SPARSE_PAGE_SIZE,
    INVALID_ENTITY_INDEX,
    PROVISIONAL_GEN_BIT,
    MAX_COMMAND_LANES,
    WORLD_BLOB_TAG,
    WORLD_BLOB_VERSION,
    REMOVAL_EVENT_SOFT_CAP,
    DespawnHookFn,
    NULL_ENTITY,
    comp_hash64,
    comp_hash64_str,
    ecs_runtime_shutdown,
    Entity,
    Component,
    Changed,
    ComponentBitSet,
    BitSet256,
    ComponentInfo,
    ArchetypeChunk,
    Archetype,
    SparseSet,
    CommandLane,
    ParallelCommands,
    Events,
    EntityRecord,
    World,
)
