"""Tests for pluggable reference-catalog backends (I-03).

Covers the :class:`CatalogBackend` interface, the procedural adapter, and the
cached Gaia DR3 cone search: spherical-cone correctness against a brute-force
great-circle filter, magnitude limiting, brightest-first ordering, the on-disk
result cache, the staged-extract round trip, and an end-to-end blind solve
driven entirely by an injected Gaia backend (the production swap point).
"""

from __future__ import annotations

import numpy as np
import pytest

from opta_pipeline.catalog import (
    CatalogBackend,
    GaiaDR3Catalog,
    ProceduralCatalog,
)
from opta_pipeline.procedural_catalog import _angular_sep_deg
from opta_pipeline.procedural_catalog import cone_search as procedural_cone_search

_RA0, _DEC0 = 135.0, 45.0


@pytest.fixture(scope="module")
def gaia_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 5000
    return (
        rng.uniform(130.0, 140.0, n),
        rng.uniform(40.0, 50.0, n),
        rng.uniform(6.0, 15.0, n),
    )


# ---------------------------------------------------------------------------
# Interface + procedural adapter
# ---------------------------------------------------------------------------


class TestCatalogBackendInterface:
    def test_both_backends_satisfy_protocol(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        assert isinstance(ProceduralCatalog(), CatalogBackend)
        assert isinstance(GaiaDR3Catalog(*gaia_table), CatalogBackend)

    def test_procedural_matches_synth_cone_search(self) -> None:
        backend = ProceduralCatalog()
        a = backend.cone_search(_RA0, _DEC0, 1.5, mag_limit=12.0)
        b = procedural_cone_search(_RA0, _DEC0, 1.5, mag_limit=12.0)
        assert [(s.ra_deg, s.magnitude) for s in a] == [
            (s.ra_deg, s.magnitude) for s in b
        ]


# ---------------------------------------------------------------------------
# Gaia cone search
# ---------------------------------------------------------------------------


class TestGaiaConeSearch:
    def test_matches_brute_force(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        ra, dec, mag = gaia_table
        gaia = GaiaDR3Catalog(ra, dec, mag)
        radius, mag_limit = 1.0, 12.0
        got = gaia.cone_search(_RA0, _DEC0, radius, mag_limit=mag_limit)
        brute = {
            (float(ra[i]), float(dec[i]), float(mag[i]))
            for i in range(len(ra))
            if mag[i] <= mag_limit
            and _angular_sep_deg(_RA0, _DEC0, ra[i], dec[i]) <= radius
        }
        assert {(s.ra_deg, s.dec_deg, s.magnitude) for s in got} == brute
        assert len(got) > 0

    def test_brightest_first(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table)
        got = gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=14.0)
        assert got == sorted(got, key=lambda s: s.magnitude)

    def test_magnitude_limit_excludes_faint(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table)
        bright = gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=10.0)
        deep = gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=14.0)
        assert len(bright) < len(deep)
        assert all(s.magnitude <= 10.0 for s in bright)

    def test_radius_excludes_outside_cone(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table)
        got = gaia.cone_search(_RA0, _DEC0, 0.5, mag_limit=15.0)
        assert all(
            _angular_sep_deg(_RA0, _DEC0, s.ra_deg, s.dec_deg) <= 0.5 + 1e-9
            for s in got
        )

    def test_degenerate_queries_return_empty(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table)
        assert gaia.cone_search(_RA0, _DEC0, 0.0, mag_limit=14.0) == []
        empty = GaiaDR3Catalog([], [], [])
        assert empty.cone_search(_RA0, _DEC0, 1.0, mag_limit=14.0) == []


# ---------------------------------------------------------------------------
# Disk cache + extract round-trip
# ---------------------------------------------------------------------------


class TestGaiaDiskCache:
    def test_cache_hit_is_identical_and_writes_one_file(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table, cache_dir=tmp_path)
        first = gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=12.0)
        cached_files = list(tmp_path.glob("cone_*.npz"))
        second = gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=12.0)
        assert len(cached_files) == 1
        assert [(s.ra_deg, s.dec_deg, s.magnitude) for s in first] == [
            (s.ra_deg, s.dec_deg, s.magnitude) for s in second
        ]

    def test_distinct_queries_use_distinct_cache_entries(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path
    ) -> None:
        gaia = GaiaDR3Catalog(*gaia_table, cache_dir=tmp_path)
        gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=12.0)
        gaia.cone_search(_RA0, _DEC0, 1.0, mag_limit=11.0)
        assert len(list(tmp_path.glob("cone_*.npz"))) == 2

    def test_different_extracts_do_not_share_cached_cones(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path
    ) -> None:
        """The cache key must identify the *catalog*, not only the query.

        Keying on (ra, dec, radius, mag_limit) alone means two catalogs sharing
        a cache_dir — or a re-staged extract at the same path — serve each
        other's cones, silently plate-solving against the wrong star list.
        """
        ra, dec, mag = gaia_table
        a = GaiaDR3Catalog(ra, dec, mag, cache_dir=tmp_path)
        # A strict subset: same pointing, genuinely fewer stars in the cone.
        b = GaiaDR3Catalog(
            ra[::2], dec[::2], mag[::2], cache_dir=tmp_path
        )
        q = (_RA0, _DEC0, 1.0)
        first = a.cone_search(*q, mag_limit=12.0)
        second = b.cone_search(*q, mag_limit=12.0)
        assert len(first) > len(second) > 0
        # Two distinct cache entries, and each catalog re-reads its own.
        assert len(list(tmp_path.glob("cone_*.npz"))) == 2
        assert b.cone_search(*q, mag_limit=12.0) == second
        assert a.cone_search(*q, mag_limit=12.0) == first

    def test_extract_round_trip(
        self, gaia_table: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path
    ) -> None:
        ra, dec, mag = gaia_table
        path = GaiaDR3Catalog.write_extract(tmp_path / "gaia.npz", ra, dec, mag)
        loaded = GaiaDR3Catalog.from_extract(path)
        direct = GaiaDR3Catalog(ra, dec, mag)
        q = (_RA0, _DEC0, 1.0)
        assert len(loaded.cone_search(*q, mag_limit=12.0)) == len(
            direct.cone_search(*q, mag_limit=12.0)
        )


# ---------------------------------------------------------------------------
# End-to-end: blind solve driven by an injected Gaia backend
# ---------------------------------------------------------------------------


class TestGaiaBackendInPipeline:
    def test_blind_solve_through_gaia_backend(self) -> None:
        """A blind solve runs end-to-end on the injected Gaia backend and meets
        OpTA.NOD.ACC — the production catalog swap touches no pipeline code."""
        import test_pipeline_blind_solve as blind

        from opta_pipeline.astrometry import pixels_to_radec
        from opta_pipeline.pipeline import FrameContext, run_frame
        from opta_pipeline.synth import SatelliteSpec, generate_frame

        true = blind._true_wcs()
        stars = blind._star_specs(true)
        # Stage a Gaia extract from the field's stars, then query it blind.
        cs = procedural_cone_search(blind._RA0, blind._DEC0, 2.0, mag_limit=14.0)
        gaia = GaiaDR3Catalog(
            [s.ra_deg for s in cs], [s.dec_deg for s in cs], [s.magnitude for s in cs]
        )
        w, h, fps = blind._W, blind._H, blind._FPS
        errs: list[float] = []
        detected = 0
        for i in range(6):
            cx = w / 2.0 - 24.0 + i * 8.0
            cy = h / 2.0
            synth = generate_frame(
                sensor=blind._SENSOR, optics=blind._OPTICS, sky_mag_arcsec2=21.0,
                satellites=[SatelliteSpec(magnitude=7.0, angular_velocity_deg_s=1.2,
                                          x_center=cx, y_center=cy, angle_deg=0.0)],
                stars=stars, rng=np.random.default_rng(4000 + i),
            )
            ctx = FrameContext(utc_mjd=blind._MJD0 + i / fps / 86400.0, frame_id=i,
                               node_id=blind._NODE, pointing=blind._HINT)
            fd = run_frame(synth.data, ctx, catalog_backend=gaia)
            if fd is None:
                continue
            detected += 1
            ra_t, dec_t = pixels_to_radec(true, cx, cy)
            best = min(fd.detections, key=lambda a: abs(a.detection.x - cx))
            errs.append(blind._sky_err(best.ra_deg, best.dec_deg, ra_t, dec_t))

        assert detected >= 5
        assert max(errs) < 10.0  # OpTA.NOD.ACC
        assert float(np.mean(errs)) < 3.0
