# supertraj

Weather along aircraft trajectories.

| Package | Role |
|---|---|
| [`supertraj-met`](packages/supertraj-met) | Fast retrieval and interpolation of wind/temperature at aircraft positions: ERA5 (137 model levels or pressure levels), Météo-France ARPEGE (live and Open-Meteo archive), FastMeteo-compatible façade |

```bash
uv sync                      # whole workspace, with dev tools
uv run pytest                # unit + integration (offline)
uv run pytest -m live        # opt-in checks against the public buckets
uv run ruff check packages && uv run mypy
```
