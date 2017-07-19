from matplotlib import pyplot
from shapely.geometry import MultiPoint, Point
from descartes.patch import PolygonPatch

from figures import SIZE, BLUE, GRAY

fig = pyplot.figure(1, figsize=SIZE, dpi=90) #1, figsize=SIZE, dpi=90)

p = Point(1, 1).buffer(1.5)

# 1
ax = fig.add_subplot(221)

q = p.simplify(0.1)

patch1a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch1a)

patch1b = PolygonPatch(q, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch1b)

ax.set_title('a) tolerance 0.1')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
#ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
#ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

#2
ax = fig.add_subplot(222)

r = p.simplify(0.2)

patch2a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch2a)

patch2b = PolygonPatch(r, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch2b)

ax.set_title('b) tolerance 0.2')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
#ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
#ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)


#3
ax = fig.add_subplot(223)

r = p.simplify(0.3)

patch2a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch2a)

patch2b = PolygonPatch(r, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch2b)

ax.set_title('b) tolerance 0.3')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
#ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
#ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)


#2
ax = fig.add_subplot(224)

r = p.simplify(0.4)

patch2a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch2a)

patch2b = PolygonPatch(r, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch2b)

ax.set_title('b) tolerance 0.4')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
#ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
#ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)


pyplot.show()


