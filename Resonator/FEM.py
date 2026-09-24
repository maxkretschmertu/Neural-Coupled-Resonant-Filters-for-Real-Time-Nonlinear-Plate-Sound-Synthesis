import numpy as np
from skfem import MeshTri
from skfem.visuals.matplotlib import draw
import matplotlib.pyplot as plt




def generate_circle_mesh(radius: float = 1.0, num_points: int = 100) -> MeshTri:
    """
    Generates a circular mesh using skfem.
    Returns:
        mesh: A skfem Mesh object representing the circular mesh.
    """
    # Generate points on the circle
    mesh = MeshTri.init_circle(nrefs=3, smoothed=False)
    mesh = mesh.scaled(radius)

    return mesh


def generate_ellipse_mesh(a: float = 1.0, b: float = 0.5, num_points: int = 100) -> MeshTri:
    """
    Generates an elliptical mesh using skfem.
    Returns:
        mesh: A skfem Mesh object representing the elliptical mesh.
    """
    # Generate points on the ellipse
    mesh = MeshTri.init_circle(nrefs=3, smoothed=False)
    p = mesh.p.copy()
    p[0,:] *= a
    p[1,:] *= b
    mesh = MeshTri(p, mesh.t)
    return mesh

def generate_rectangle_mesh(width: float = 1.0, height: float = 1.0, num_points: int = 100) -> MeshTri:
    """
    Generates a rectangular mesh using skfem.
    Returns:
        mesh: A skfem Mesh object representing the rectangular mesh.
    """
    # Generate points on the rectangle
    mesh = MeshTri.init_sqsymmetric()
    p = mesh.p.copy()
    p[0,:] *= width
    p[1,:] *= height
    mesh = MeshTri(p, mesh.t)
    return mesh


def replace_zeros_ones(mesh):
    return

def create_grid(a: int = 64, b: int = 64, mesh_array: np.ndarray = None) -> np.ndarray:
    grid = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            if mesh_array is not None and mesh_array[i, j] == 1:
                grid[i, j] = 1
    return grid

import numpy as np
from matplotlib.path import Path

def mesh_to_grid(mesh, resolution=64):
    """
    Convert a skfem MeshTri into a binary image.

    Parameters
    ----------
    mesh : MeshTri
        skfem mesh
    resolution : int
        image size (resolution x resolution)

    Returns
    -------
    grid : ndarray
        Binary image containing 0 and 1.
    """

    # mesh bounding box
    xmin, xmax = mesh.p[0].min(), mesh.p[0].max()
    ymin, ymax = mesh.p[1].min(), mesh.p[1].max()

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)

    X, Y = np.meshgrid(x, y)

    points = np.column_stack((X.ravel(), Y.ravel()))

    grid = np.zeros(points.shape[0], dtype=np.uint8)

    # mark points inside any triangle
    for tri in mesh.t.T:
        vertices = mesh.p[:, tri].T
        path = Path(vertices)

        mask = path.contains_points(points)
        grid[mask] = 1

    return grid.reshape((resolution, resolution))

#m_circle = generate_circle_mesh(2.0)
#draw(m_circle)

m_circle = generate_circle_mesh(2.0)
circle_img = mesh_to_grid(m_circle, 64)

print(circle_img)

m_ellipse = generate_ellipse_mesh(a=2.0, b=1.0)
draw(m_ellipse)

m_rectangle = generate_rectangle_mesh(width=2.0, height=1.0)
draw(m_rectangle)

#plt.plot(m_circle.p[0], m_circle.p[1], 'o')
#plt.plot(m_ellipse.p[0], m_ellipse.p[1], 'o')
#plt.plot(m_rectangle.p[0], m_rectangle.p[1], 'o')
plt.show()