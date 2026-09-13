"""Pluggable reference-catalog backends for blind plate solving (I-03).

The blind solver needs candidate stars from an *approximate* pointing, with no
truth coupling — a cone search ``(ra, dec, radius, mag_limit) → stars``.  Two
interchangeable backends implement that one interface:

* :class:`ProceduralCatalog` — the deterministic, file-free synthetic catalog
  (:mod:`opta_pipeline.procedural_catalog`), the default for tests and development.
* :class:`GaiaDR3Catalog` — a **cached Gaia DR3** cone search (I-03): a local
  on-disk source extract indexed with a KD-tree for fast spherical queries,
  with cone results memoised to a disk cache so repeated pointings during a
  pass do not re-scan the extract.

The pipeline takes a :class:`CatalogBackend` by dependency injection
(``run_frame(..., catalog_backend=...)``), so swapping the procedural catalog
for Gaia DR3 in production touches no pipeline code.  Both backends return the
shared :class:`~opta_pipeline.procedural_catalog.CatalogStar` record, which satisfies
the matcher's ``CatalogEntry`` protocol.

Production note: staging the Gaia extract (auth, bulk download, indexing) is out
of scope here; :meth:`GaiaDR3Catalog.write_extract` defines the on-disk format a
fetcher would populate, and :meth:`GaiaDR3Catalog.from_extract` loads it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.spatial import KDTree

from opta_pipeline.procedural_catalog import CatalogStar
from opta_pipeline.procedural_catalog import cone_search as _procedural_cone_search

__all__ = [
    "CatalogStar",
    "CatalogBackend",
    "ProceduralCatalog",
    "GaiaDR3Catalog",
]

_MAG_LIMIT_DEFAULT = 14.0


@runtime_checkable
class CatalogBackend(Protocol):
    """A reference catalog that answers pointing-only cone searches."""

    def cone_search(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        mag_limit: float = _MAG_LIMIT_DEFAULT,
    ) -> list[CatalogStar]:
        """Return stars within ``radius_deg`` of ``(ra_deg, dec_deg)``,
        no brighter-than ``mag_limit``, brightest first."""
        ...


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Map RA/Dec (degrees) to unit vectors on the celestial sphere."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack(
        [cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)]
    )


# ---------------------------------------------------------------------------
# Procedural backend (default)
# ---------------------------------------------------------------------------


class ProceduralCatalog:
    """The deterministic synthetic catalog as a :class:`CatalogBackend`.

    Thin adapter over :func:`opta_pipeline.procedural_catalog.cone_search`; the
    default backend so the blind solve runs file-free in tests.
    """

    def cone_search(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        mag_limit: float = _MAG_LIMIT_DEFAULT,
    ) -> list[CatalogStar]:
        return _procedural_cone_search(
            ra_deg, dec_deg, radius_deg, mag_limit=mag_limit
        )


# ---------------------------------------------------------------------------
# Gaia DR3 backend (cached cone search)
# ---------------------------------------------------------------------------


class GaiaDR3Catalog:
    """Cached Gaia DR3 cone search over a local source extract (I-03).

    Builds a KD-tree over the extract's unit vectors for O(log N) spherical
    cone queries and (optionally) memoises each cone result to ``cache_dir`` so
    a pass that re-solves the same pointing reads the cache instead of
    re-querying.  Gaia ``phot_g_mean_mag`` maps to ``CatalogStar.magnitude``.

    Construct directly from arrays, or via :meth:`from_extract` to load a
    staged ``.npz`` produced by :meth:`write_extract`.
    """

    def __init__(
        self,
        ra_deg: np.ndarray | list[float],
        dec_deg: np.ndarray | list[float],
        magnitude: np.ndarray | list[float],
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._ra = np.asarray(ra_deg, dtype=np.float64)
        self._dec = np.asarray(dec_deg, dtype=np.float64)
        self._mag = np.asarray(magnitude, dtype=np.float64)
        if not (len(self._ra) == len(self._dec) == len(self._mag)):
            raise ValueError("ra_deg, dec_deg, magnitude must be equal length")
        self._tree = (
            KDTree(_unit_vectors(self._ra, self._dec))
            if len(self._ra)
            else None
        )
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._extract_key = ""
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            # Extract fingerprint: the cone key (pointing + radius + mag
            # limit) describes the *query*, not the catalog answering it, so
            # two catalogs sharing a cache_dir — or a re-staged extract at the
            # same path — would serve each other's cones.  Fold the extract's
            # own bytes into the key so a different source list simply misses
            # the cache.  Hashed from the buffers directly (no ``tobytes``
            # copy) and only when a cache is actually in use, so an uncached
            # multi-million-row extract pays nothing.
            digest = hashlib.md5(usedforsecurity=False)
            for arr in (self._ra, self._dec, self._mag):
                digest.update(np.ascontiguousarray(arr).tobytes())
            self._extract_key = digest.hexdigest()[:16]

    # -- on-disk extract format ------------------------------------------------

    @staticmethod
    def write_extract(
        path: str | Path,
        ra_deg: np.ndarray | list[float],
        dec_deg: np.ndarray | list[float],
        magnitude: np.ndarray | list[float],
    ) -> Path:
        """Write a Gaia source extract to ``path`` (``.npz``); returns the path.

        This is the format a production Gaia fetcher stages on disk; the
        backend reads it with :meth:`from_extract`.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            ra_deg=np.asarray(ra_deg, dtype=np.float64),
            dec_deg=np.asarray(dec_deg, dtype=np.float64),
            magnitude=np.asarray(magnitude, dtype=np.float64),
        )
        return out if out.suffix else out.with_suffix(".npz")

    @classmethod
    def from_extract(
        cls, path: str | Path, *, cache_dir: str | Path | None = None
    ) -> GaiaDR3Catalog:
        """Load a staged ``.npz`` extract (see :meth:`write_extract`)."""
        with np.load(Path(path)) as d:
            return cls(
                d["ra_deg"], d["dec_deg"], d["magnitude"], cache_dir=cache_dir
            )

    # -- query -----------------------------------------------------------------

    def cone_search(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        mag_limit: float = _MAG_LIMIT_DEFAULT,
    ) -> list[CatalogStar]:
        if radius_deg <= 0.0 or self._tree is None:
            return []

        cached = self._load_cache(ra_deg, dec_deg, radius_deg, mag_limit)
        if cached is not None:
            return cached

        # Great-circle radius → chord length between unit vectors.
        centre = _unit_vectors(
            np.array([ra_deg]), np.array([dec_deg])
        )[0]
        chord = 2.0 * math.sin(math.radians(radius_deg) / 2.0)
        idx = np.asarray(self._tree.query_ball_point(centre, chord), dtype=int)
        if idx.size:
            idx = idx[self._mag[idx] <= mag_limit]
            idx = idx[np.argsort(self._mag[idx], kind="stable")]

        result = [
            CatalogStar(
                ra_deg=float(self._ra[i]),
                dec_deg=float(self._dec[i]),
                magnitude=float(self._mag[i]),
            )
            for i in idx
        ]
        self._store_cache(ra_deg, dec_deg, radius_deg, mag_limit, result)
        return result

    # -- disk cache ------------------------------------------------------------

    def _cache_path(
        self, ra: float, dec: float, radius: float, mag_limit: float
    ) -> Path | None:
        if self._cache_dir is None:
            return None
        key = (
            f"{self._extract_key}_{ra:.4f}_{dec:.4f}_{radius:.4f}_{mag_limit:.2f}"
        ).encode()
        digest = hashlib.md5(key, usedforsecurity=False).hexdigest()[:16]
        return self._cache_dir / f"cone_{digest}.npz"

    def _load_cache(
        self, ra: float, dec: float, radius: float, mag_limit: float
    ) -> list[CatalogStar] | None:
        path = self._cache_path(ra, dec, radius, mag_limit)
        if path is None or not path.exists():
            return None
        with np.load(path) as d:
            return [
                CatalogStar(float(r), float(dc), float(m))
                for r, dc, m in zip(d["ra_deg"], d["dec_deg"], d["magnitude"])
            ]

    def _store_cache(
        self,
        ra: float,
        dec: float,
        radius: float,
        mag_limit: float,
        result: list[CatalogStar],
    ) -> None:
        path = self._cache_path(ra, dec, radius, mag_limit)
        if path is None:
            return
        np.savez(
            path,
            ra_deg=np.array([s.ra_deg for s in result], dtype=np.float64),
            dec_deg=np.array([s.dec_deg for s in result], dtype=np.float64),
            magnitude=np.array([s.magnitude for s in result], dtype=np.float64),
        )
