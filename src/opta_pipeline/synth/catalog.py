"""Compatibility alias: the procedural star catalog lives in
:mod:`opta_pipeline.procedural_catalog` (it needs no opta-model)."""

from opta_pipeline.procedural_catalog import (  # noqa: F401
    STARS_PER_SQ_DEG,
    CatalogStar,
    catalog_stars_in_fov,
    cone_search,
    star_field_at,
)

__all__ = [
    "CatalogStar",
    "star_field_at",
    "catalog_stars_in_fov",
    "cone_search",
    "STARS_PER_SQ_DEG",
]
