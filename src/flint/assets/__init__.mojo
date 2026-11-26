"""Flint v5 Assets Package (`flint.assets`)."""
from flint.assets.asset_plugin import AssetPlugin, SceneAssetResolver
from flint.assets.asset_ref import (
    AssetRef,
    is_safe_asset_rel_path,
    normalize_asset_rel_path,
)
from flint.assets.flmat import (
    FlmatAsset,
    MAT_SHADING_NPR_CEL,
    MAT_SHADING_PBR,
    MAT_SHADING_STYLIZED_VFX,
    load_flmat_bytes,
    load_flmat_file,
    save_flmat_file,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_FONT,
    ASSET_KIND_HDR,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    AssetMeta,
    AssetUuid,
    COOKED_RIG_SLAB_MAGIC,
    SUB_ID_RIG,
    decode_cooked_rig_slab,
    encode_cooked_rig_slab,
)
from flint.assets.server import AssetServer
