"""
title: Spatial correlogram
summary: Moran's I at increasing neighbour orders, with a permutation envelope — how far spatial clustering reaches.
r: spdep::sp.correlogram(method = "I")
tags: [spatial autocorrelation, correlogram, Moran's I, higher-order neighbours, spatial scale, Great Lakes]
description: |
  Moran's I with first-order neighbours measures clustering at the shortest
  range. A spatial correlogram repeats the computation for neighbours of order
  2 (neighbours of neighbours, excluding closer ones), 3, and so on. The curve
  shows the **spatial scale** of a pattern: how many steps away two areas can
  be and still resemble each other, and whether distant areas become
  *opposites* — negative I — as they do when a region is split into two large
  blocks. It is the spatial counterpart of the autocorrelation function of a
  time series.

  The grey band is the 95% range of I under spatial randomness, from 999
  permutations at each order; points outside it are unlikely under no spatial
  structure. Two variables are compared: exposure to low water, which follows
  long stretches of the waterway, and the share of manufacturing jobs exposed
  to trade, which clusters tightly around a few industrial cities.

  Data are **simulated**, set in the context of the GVCdtLab, CIRANO's digital twin
  of the Great Lakes–St. Lawrence trade and transport corridor: 276 hexagonal
  zones along a schematic waterway from Duluth and Chicago to the Gulf of
  St. Lawrence, each with a 0–100 index of how exposed its shipping is to low
  water levels. Neither the zones nor the values are the lab's data or results.
  Weights are higher-order queen contiguity from
  `libpysal.weights.higher_order`, row-standardised; I comes from `esda.Moran`.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from esda.moran import Moran
from libpysal import weights
from shapely.geometry import MultiLineString, Point, Polygon

# Simulated data, set in the context of the CIRANO GVCdtLab [gvcdtlab]
# Schematic corridor: hexagonal zones within reach of the waterway, Duluth and Chicago to the Gulf
main = [(-92.1, 46.8), (-89.2, 48.4), (-84.4, 46.5), (-82.4, 44.0), (-83.0, 42.3), (-81.2, 42.2),
        (-79.0, 42.9), (-79.4, 43.6), (-76.5, 44.2), (-73.6, 45.5), (-71.2, 46.8), (-68.5, 48.5),
        (-64.5, 49.2)]
michigan = [(-84.0, 45.6), (-86.6, 45.4), (-87.0, 43.5), (-87.6, 41.9)]
k = np.cos(np.radians(45))                      # shrink longitudes so the hexagons are regular
water = MultiLineString([[(x * k, y) for x, y in line] for line in (main, michigan)])
r, hexes = 0.3, []
for col in range(90):
    for row in range(36):
        cx, cy = -66.5 + 1.5 * r * col, 40.5 + np.sqrt(3) * r * (row + (col % 2) / 2)
        if water.distance(Point(cx, cy)) < 0.95:
            hexes.append(Polygon([(round(cx + r * np.cos(a), 6), round(cy + r * np.sin(a), 6))
                                  for a in np.radians(range(0, 360, 60))]))
g = gpd.GeoDataFrame(geometry=hexes)
xy = np.column_stack([g.centroid.x / k, g.centroid.y])      # back to longitude, latitude

# Exposure to low water: high at shallow passages (Seaway, connecting channels, upper lakes),
# low in the deep tidal estuary, plus spatially smooth noise and one draft-limited harbour
def bump(lon, lat, s):
    return np.exp(-((xy[:, 0] - lon) ** 2 + (xy[:, 1] - lat) ** 2) / (2 * s ** 2))
w = weights.Queen.from_dataframe(g, use_index=False)
w.transform = "r"
rng = np.random.default_rng(12)
noise = rng.normal(0, 1, len(g))
for _ in range(2):
    noise = weights.lag_spatial(w, noise)       # smooth the noise over neighbours
exposure = (35 + 30 * bump(-74.8, 45.0, 1.6) + 22 * bump(-82.7, 42.7, 1.2) + 20 * bump(-90.3, 47.6, 1.8)
            - 25 * bump(-67.0, 48.8, 2.2) + 12 * noise + rng.normal(0, 4, len(g)))
exposure[np.argmin(np.hypot(xy[:, 0] + 66.0, xy[:, 1] - 49.0))] = 75        # outlier harbour
g["exposure"] = exposure.clip(0, 100).round(1)
# A second, short-range variable: trade-exposed manufacturing jobs around a few industrial cities
g["manuf"] = (18 + 12 * bump(-83.0, 42.3, 0.6) + 10 * bump(-79.8, 43.3, 0.6) + 9 * bump(-73.6, 45.5, 0.5)
              + 9 * bump(-87.6, 41.9, 0.5) + rng.normal(0, 2.5, len(g)))
variables = {"exposure": ("Exposure to low water", "#007AFF"),
             "manuf": ("Trade-exposed manufacturing jobs", "#FF9500")}

w1 = weights.Queen.from_dataframe(g, use_index=False)
orders = range(1, 11)
res = {v: [] for v in variables}
for o in orders:
    wk = weights.higher_order(w1, o, silence_warnings=True)    # exactly o steps away
    wk.transform = "r"
    for v in variables:
        m = Moran(g[v].to_numpy(), wk, permutations=999)
        res[v].append((m.I, *np.percentile(m.sim, [2.5, 97.5]), m.p_sim))

fig, ax = plt.subplots(figsize=(8, 5))
env = np.array(res["exposure"])[:, 1:3]                   # envelope (similar for both variables)
ax.fill_between(orders, env[:, 0], env[:, 1], color="#E5E5EA", lw=0, label="95% under randomness")
ax.axhline(0, color="#8E8E93", lw=1)
for v, (name, c) in variables.items():
    a = np.array(res[v])
    ax.plot(orders, a[:, 0], color=c, lw=2.2, marker="o", label=name)
    out = (a[:, 0] < a[:, 1]) | (a[:, 0] > a[:, 2])
    ax.scatter(np.array(orders)[out], a[out, 0], s=90, facecolor="none", edgecolor=c, lw=1.5, zorder=4)
ax.set_xticks(list(orders))
ax.set_xlabel("Neighbour order (1 = shares a border, 2 = neighbour of a neighbour, …)")
ax.set_ylabel("Moran's I")
ax.grid(axis="x")
ax.legend(loc="upper right")

reach = {v: max(o for o, (I, lo, hi, p) in zip(orders, res[v]) if I > hi) for v in variables}
ax.set_title(f"Low-water exposure clusters up to {reach['exposure']} steps away, "
             f"manufacturing only {reach['manuf']}", pad=26)
ax.text(0, 1.02, "Circled: outside the 95% permutation envelope · 999 permutations per order · "
        f"queen contiguity · n = {w1.n} zones (simulated)", transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
