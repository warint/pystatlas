# Choropleth with spatial weights

> A map of one variable with the neighbour network drawn on top — the structure every spatial statistic relies on.

![Choropleth with spatial weights](plot.png)

Every spatial statistic starts by deciding who is a neighbour of whom. This
chart shows that decision before any test is run: a choropleth of one
variable, with a line joining the centres of every pair of neighbouring
areas. Here neighbours share a border or a corner (**queen contiguity**), and
each row of the weights matrix is later standardised so that an area's
"spatial lag" is the average of its neighbours. Other choices — rook
contiguity, the k nearest neighbours, a distance band — give different
networks and can change the results, so the network is worth looking at.

Data are **simulated**, set in the context of the GVCdtLab, CIRANO's digital twin
of the Great Lakes–St. Lawrence trade and transport corridor: 276 hexagonal
zones laid along a schematic waterway from Duluth and Chicago to the Gulf of
St. Lawrence, each with a 0–100 index of how exposed its shipping is to low
water levels. Neither the zones nor the values are the lab's data or results.
Weights come from `libpysal.weights.Queen`; the zones are built with
**Shapely** and drawn with **GeoPandas**.

**R equivalent:** `spdep::poly2nb() + plot(nb, coords)`  
**Libraries:** GeoPandas, Matplotlib, NumPy, libpysal, Shapely

**Data:** Simulated data with a fixed random seed, created for this example. — No real respondents, documents or measurements: the numbers only illustrate how the chart and its statistics work.

**Data:** CIRANO. GVCdtLab: Digital twin of the Great Lakes–St. Lawrence trade and transport corridor (researcher in charge: T. Warin). Supported by Transport Canada's National Trade Corridors Fund; research on hydrological change, maritime traffic and trade funded by the Fonds de recherche du Québec. https://cirano.qc.ca/en/research/pole/Transport — The examples are set in the context of the lab's work; the hexagonal zones form a schematic map of the corridor and the values are simulated — they are not the lab's data, geography or results.


```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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

# Queen contiguity: neighbours share a border or a corner
cards = np.array(list(w.cardinalities.values()))
c = np.column_stack([g.centroid.x, g.centroid.y])

fig, ax = plt.subplots(figsize=(11, 5.2))
g.plot(column="exposure", cmap="Blues", edgecolor="white", linewidth=0.6, ax=ax, legend=True,
       legend_kwds=dict(label="Exposure to low water (index, 0–100)", shrink=0.55, pad=0.01))
for i, nbrs in w.neighbors.items():
    for j in nbrs:
        if i < j:                      # draw each link once
            ax.plot(*c[[i, j]].T, color="#FF9500", lw=0.7, alpha=0.8)
ax.scatter(*c.T, s=6, color="#1D1D1F", zorder=3)
for name, lon, lat, dx, dy in [("Duluth", -92.1, 46.8, -0.6, -0.9), ("Chicago", -87.6, 41.9, -1.4, -0.4),
                              ("Detroit", -83.0, 42.3, 0, -1.2), ("Toronto", -79.4, 43.6, -0.4, -1.4),
                              ("Montréal", -73.6, 45.5, 0.9, -1.1), ("Québec", -71.2, 46.8, 1.3, -0.9)]:
    ax.annotate(name, (lon * k, lat), (lon * k + dx, lat + dy), ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#C7C7CC", lw=0.6),
                arrowprops=dict(arrowstyle="-", color="#3A3A3C", lw=0.8))
ax.set_axis_off()
ax.set_title(f"Each corridor zone has {w.mean_neighbors:.1f} neighbours on average", pad=26)
ax.text(0, 1.02, f"Queen contiguity · n = {w.n} hexagonal zones (simulated, schematic map) · "
        f"{cards.sum() // 2} links · {cards.min()} to {cards.max()} neighbours",
        transform=ax.transAxes, fontsize=9.5, color="#6E6E73")
plt.show()
```

---

**PyStatlas** by [Thierry Warin](https://www.warin.ca) · [View on the website →](https://warint.github.io/pystatlas/spatial-data/spatial-weights/) · [← Spatial data](../README.md)

**Cite:** Warin, T. (2026). Choropleth with spatial weights. In PyStatlas: A Python Gallery of Statistical Graphics for the Social Sciences (Version 1.1.0) [Computer software]. https://warint.github.io/pystatlas/spatial-data/spatial-weights/
