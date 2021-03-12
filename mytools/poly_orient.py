from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
import numpy as np
coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
poly = Polygon(coords)
print(poly)
poly = orient(poly, sign=1)
print(poly)
print(list(poly.exterior.coords))
print(Polygon(np.array(coords)))