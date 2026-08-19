import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from enum import Enum
from Polynomial2D import Polynomial2D
# ============================================================
# 1. Structured square mesh on (0,1)^2
# ============================================================
class StructuredSquareMesh:
    def __init__(self,nx=4,ny=4,xs=None,ys=None):
        self.nx = nx
        self.ny = ny
        self.custom_xs = xs
        self.custom_ys = ys
        self.build_mesh()
    def node_index(self, i, j):
        return j * (self.nx + 1) + i
    def cell_index(self, i, j):
        return j * self.nx + i
    def build_mesh(self):
        if self.custom_xs is None:
            xs = np.linspace(0.0,1.0, self.nx + 1)
        else :
            xs = np.array(self.custom_xs, dtype=float)
        if self.custom_ys is None:
            ys = np.linspace(0.0,1.0,self.ny+1)
        else:
            ys = np.array(self.custom_ys,dtype=float)
        points = []
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                points.append([xs[i], ys[j]])
        self.points = np.array(points, dtype=float)
        cells = []
        for j in range(self.ny):
            for i in range(self.nx):
                p00 = self.node_index(i, j)
                p10 = self.node_index(i + 1, j)
                p11 = self.node_index(i + 1, j + 1)
                p01 = self.node_index(i, j + 1)
                cells.append([p00, p10, p11, p01]) # Keep each square as a one rectangular cell.
        self.cells = np.array(cells, dtype=int)
        self.nsimplices = len(self.cells)
        self.centers = np.mean(self.points[self.cells], axis=1) # This computes center of every rectangle
        # edge order: bottom, right, top, left
        self.neighbors = -np.ones((self.nsimplices, 4), dtype=int)
        for j in range(self.ny):
            for i in range(self.nx):
                K = self.cell_index(i, j)
                if j > 0:
                    self.neighbors[K, 0] = self.cell_index(i, j - 1)
                if i < self.nx - 1:
                    self.neighbors[K, 1] = self.cell_index(i + 1, j)
                if j < self.ny - 1:
                    self.neighbors[K, 2] = self.cell_index(i, j + 1)
                if i > 0:
                    self.neighbors[K, 3] = self.cell_index(i - 1, j)
        self.build_vertex_weights()
    def build_vertex_weights(self):
        npoints = len(self.points)
        positions = [[] for _ in range(npoints)]
        for K in range(self.nsimplices):
            for p in self.cells[K]:
                positions[p].append(K)
        self.positions = []
        self.weights = []
        for p in range(npoints):
            x = self.points[p]
            cells = np.array(positions[p], dtype=int)
            self.positions.append(cells)
            if x[0] == 0.0 or x[0] == 1.0 or x[1] == 0.0 or x[1] == 1.0: 
                self.weights.append(np.array([]))
            else:
                self.weights.append(np.ones(len(cells)) / len(cells)) #An interior vertex are surrounded by 4 cells, this becomes exactly: wk(a)=1/4
    def refine_vertical_columns(self, x_targets):
        xs = sorted(np.unique(self.points[:,0]))
        new_xs = list(xs)
        for i in range(len(xs)-1):
            x_left = xs[i]
            x_right = xs[i+1]
            x_mid = 0.5*(x_left+x_right)
            for x_target in x_targets:
                if abs(x_mid-x_target) < 1e-12:
                    new_xs.append(x_mid)
                    break
        new_xs = sorted(np.unique(new_xs))
        self.custom_xs = new_xs
        self.nx = len(new_xs)-1
        self.build_mesh()
    def refine_horizontal_rows(self, y_targets):
        ys = sorted(np.unique(self.points[:,1]))
        new_ys = list(ys)
        for i in range(len(ys)-1):
            y_bottom = ys[i]
            y_top = ys[i+1]
            y_mid = 0.5*(y_bottom + y_top)
            for y_target in y_targets:
                if abs(y_mid-y_target) < 1e-12:
                    new_ys.append(y_mid)
                    break
        new_ys = sorted(np.unique(new_ys))
        self.custom_ys = new_ys
        self.ny = len(new_ys)-1
        self.build_mesh()
def structured_meshes(nx=4, ny=4, xs=None, ys=None):
    return StructuredSquareMesh(nx=nx, ny=ny, xs=xs, ys=ys)
def refine_horizontal_rows(self, y_targets):
        ys = sorted(np.unique(self.points[:,1]))
        new_ys = list(ys)
        for i in range(len(ys)-1):
            y_bottom = ys[i]
            y_top = ys[i+1]
            y_mid = 0.5*(y_bottom + y_top)
            for y_target in y_targets:
                if abs(y_mid-y_target) < 1e-12:
                    new_ys.append(y_mid)
                    break
        new_ys = sorted(np.unique(new_ys))
        self.custom_ys = new_ys
        self.ny = len(new_ys)-1
        self.build_mesh()
def structured_meshes(nx=4, ny=4, xs=None, ys=None):
    return StructuredSquareMesh(nx=nx, ny=ny, xs=xs, ys=ys)
# ============================================================
# 2. Boundary
# ============================================================
class Boundary(Enum):
    NEUMANN = 0
    DIRICHLET = 1
    def __eq__(self, other):
        return self.value == other.value
def boundary(point_1, point_2):
    return Boundary.DIRICHLET
# ============================================================
# 3. Exact solution and sourse term
# ============================================================
def exact_solution(x, t):
    return np.exp(-t) * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
def exact_gradient(x, t):
    x0, x1 = x
    e = np.exp(-t)
    ux = e * (1.0 - 2.0 * x0) * x1 * (1.0 - x1)
    uy = e * x0 * (1.0 - x0) * (1.0 - 2.0 * x1)
    return np.array([ux, uy])
def dirichlet(x, t):
    return exact_solution(x, t)
# SOURCE TERM
def forcing(x, t):
    x0 = x[0]
    x1 = x[1]
    # Distance from the singular point (1/2,1/2)
    r = np.linalg.norm(
        np.array([x0, x1]) -
        np.array([0.5, 0.5]))
    # Avoid division by zero
    eps = 1.0e-12
    r = max(r, eps)
    return (x0*(1.0-x0)*x1*(1.0-x1)*(np.exp(-t) + np.sin(t)/(r**0.5)))
# ============================================================
# 4. Geometry
# ============================================================
def rectangle_area(verts):
    return abs(np.cross(verts[1] - verts[0], verts[3] - verts[0]))
def rectangle_diameter(verts):
    return np.linalg.norm(verts[2] - verts[0])
def mesh_size(mesh):
    return 1.0 / max(mesh.nx,mesh.ny)
def edge_points_of_cell(mesh, K, E): # Edge height
    verts = mesh.points[mesh.cells[K]]
    if E == 0:
        return verts[0], verts[1]
    if E == 1:
        return verts[1], verts[2]
    if E == 2:
        return verts[2], verts[3]
    if E == 3:
        return verts[3], verts[0]
    raise ValueError("Edge index must be 0,1,2,3")
def h_EK_rectangle(verts, edge0, edge1):
    return rectangle_area(verts) / np.linalg.norm(edge1 - edge0)
def unit_tangent(edge0, edge1):
    t = edge1 - edge0
    return t / np.linalg.norm(t)
def unit_normal(edge0, edge1):
    n = np.array([-(edge1 - edge0)[1], (edge1 - edge0)[0]])
    return n / np.linalg.norm(n)
# ============================================================
# 5. Rectangular Morley element
# ============================================================
class MorleyRectangle:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float)
    def basis_polynomial(self, coeffs8):
        c = np.zeros(15)
        c[0] = coeffs8[0]
        c[1] = coeffs8[1]
        c[2] = coeffs8[2]
        c[3] = coeffs8[3]
        c[4] = coeffs8[4]
        c[5] = coeffs8[5]
        # x^3 - 3xy^2
        c[6] += coeffs8[6]
        c[7] += -3.0 * coeffs8[6]
        # y^3 - 3yx^2
        c[9] += coeffs8[7]
        c[8] += -3.0 * coeffs8[7]
        return Polynomial2D(c)
    def function(self, dofs):
        V = self.vertices
        edges = [
            (V[0], V[1], V[2]),
            (V[1], V[2], V[0]),
            (V[2], V[3], V[0]),
            (V[3], V[0], V[1]),
        ]
        A = np.zeros((8, 8))
        b = np.array(dofs, dtype=float)
        for j in range(8):
            coeffs = np.zeros(8)
            coeffs[j] = 1.0
            poly = self.basis_polynomial(coeffs)
            for i in range(4):
                A[i, j] = poly.evaluate_vec(V[i])
            for E, (p0, p1, other) in enumerate(edges):
                A[4 + E, j] = poly.integral_normal_line(
                    np.array([p0, p1]),
                    other
                )
        coeffs8 = np.linalg.solve(A, b)
        return self.basis_polynomial(coeffs8)
# ============================================================
# 6. Matrix assembly
# ============================================================
def assemble_stiffness_matrix(mesh):
    data = []
    row = []
    col = []
    for K in range(mesh.nsimplices):
        for E in range(4):
            edge0, edge1 = edge_points_of_cell(mesh, K, E)
            len_edge = np.linalg.norm(edge1 - edge0)
            if mesh.neighbors[K, E] != -1:
                L = mesh.neighbors[K, E]
                d_KL = np.linalg.norm(mesh.centers[L] - mesh.centers[K])
                coeff = len_edge / d_KL
                data.append(coeff)
                row.append(K)
                col.append(K)
                data.append(-coeff)
                row.append(K)
                col.append(L)
            else:
                if boundary(edge0, edge1) == Boundary.DIRICHLET:
                    d_boundary = abs(np.cross(edge1 - edge0, edge0 - mesh.centers[K])) / len_edge
                    coeff = len_edge / d_boundary
                    data.append(coeff)
                    row.append(K)
                    col.append(K)
    return sp.csr_matrix((data, (row, col)), shape=(mesh.nsimplices, mesh.nsimplices))
def assemble_mass_matrix(mesh):
    masses = np.zeros(mesh.nsimplices)
    for K in range(mesh.nsimplices):
        verts = mesh.points[mesh.cells[K]]
        masses[K] = rectangle_area(verts) # Mass is rectagle area.
    return sp.diags(masses, offsets=0, format="csr"), masses
def integrate_rectangle_function(func, verts):
    return (
        Polynomial2D.gen_integral_tri(func, verts[0], verts[1], verts[2])
        +
        Polynomial2D.gen_integral_tri(func, verts[0], verts[2], verts[3])
    )
def assemble_rhs(mesh, t):
    rhs = np.zeros(mesh.nsimplices)
    for K in range(mesh.nsimplices):
        verts = mesh.points[mesh.cells[K]]
        rhs[K] = integrate_rectangle_function(lambda x: forcing(x, t),verts)
        for E in range(4):
            edge0, edge1 = edge_points_of_cell(mesh, K, E)
            if mesh.neighbors[K, E] == -1:
                if boundary(edge0, edge1) == Boundary.DIRICHLET:
                    len_edge = np.linalg.norm(edge1 - edge0)
                    d_boundary = abs(np.cross(edge1 - edge0, edge0 - mesh.centers[K])) / len_edge
                    tau_E = len_edge / d_boundary
                    boundary_average = (
                        Polynomial2D.line_integral(
                            lambda x: dirichlet(x, t),
                            [edge0, edge1]
                        ) / len_edge
                    )
                    rhs[K] += tau_E * boundary_average
    return rhs
# ============================================================
# 7. Initial values
# ============================================================
def initial_cell_values(mesh):
    u0 = np.zeros(mesh.nsimplices)
    for K in range(mesh.nsimplices):
        verts = mesh.points[mesh.cells[K]]
        area = rectangle_area(verts)
        integral_value = integrate_rectangle_function(
            lambda x: exact_solution(x, 0.0),
            verts
        )
        u0[K] = integral_value / area
    return u0
def reconstructed_subcell_values_2D(mesh, K, u):
    verts = mesh.points[mesh.cells[K]]
    hx = np.max(verts[:,0]) - np.min(verts[:,0])
    hy = np.max(verts[:,1]) - np.min(verts[:,1])
    # ---------- x-direction slopes ----------
    left_grad = 0.0
    right_grad = 0.0
    if mesh.neighbors[K,3] != -1:
        L = mesh.neighbors[K,3]
        d = np.linalg.norm(mesh.centers[K] - mesh.centers[L])
        left_grad = (u[K]-u[L])/d
    if mesh.neighbors[K,1] != -1:
        R = mesh.neighbors[K,1]
        d = np.linalg.norm(mesh.centers[R]-mesh.centers[K])
        right_grad = (u[R]-u[K])/d
    if abs(left_grad) < abs(right_grad):
        gx = left_grad
    else:
        gx = right_grad
    # ---------- y-direction slopes ----------
    bottom_grad = 0.0
    top_grad = 0.0
    if mesh.neighbors[K,0] != -1:
        B = mesh.neighbors[K,0]
        d = np.linalg.norm(mesh.centers[K]-mesh.centers[B])
        bottom_grad = (u[K]-u[B])/d
    if mesh.neighbors[K,2] != -1:
        T = mesh.neighbors[K,2]
        d = np.linalg.norm(mesh.centers[T]-mesh.centers[K])
        top_grad = (u[T]-u[K])/d
    if abs(bottom_grad) < abs(top_grad):
        gy = bottom_grad
    else:
        gy = top_grad
    # ---------- Professor equations ----------
    ex = hx/2.0
    ey = hy/2.0
    uSW = u[K] - 0.5*ex*gx - 0.5*ey*gy
    uSE = u[K] + 0.5*ex*gx - 0.5*ey*gy
    uNW = u[K] - 0.5*ex*gx + 0.5*ey*gy
    uNE = u[K] + 0.5*ex*gx + 0.5*ey*gy
    return uSW,uSE,uNW,uNE,gx,gy
# ============================================================
# 8. Reconstruction
# ============================================================
def get_nodal_values(mesh, phi, p, time_value):
    x = mesh.points[p]
    if x[0] == 0.0 or x[0] == 1.0 or x[1] == 0.0 or x[1] == 1.0: # Boundary vertices are directly assigned to Drichlet value: I_Mu_h = 0 on boundary.
        return dirichlet(x, time_value), [], []
    return (
        np.dot(mesh.weights[p], phi[mesh.positions[p]]),
        mesh.weights[p],
        mesh.positions[p]
    )
def reconstruct(mesh, u, time_value):
    I_Mu = []
    for K in range(mesh.nsimplices):
        verts = mesh.points[mesh.cells[K]]
        dofs = np.zeros(8) # Rectangle has 8 DOFs, 4 vertex values and 4 edges normal derivative
        for i in range(4):
            dofs[i], _, _ = get_nodal_values(
                mesh, u, mesh.cells[K, i], time_value
            )
        for E in range(4):
            edge0, edge1 = edge_points_of_cell(mesh, K, E)
            len_edge = np.linalg.norm(edge1 - edge0)
            if mesh.neighbors[K, E] != -1:
                L = mesh.neighbors[K, E]
                d_KL = np.linalg.norm(mesh.centers[L] - mesh.centers[K])
                dofs[4 + E] = len_edge / d_KL * (u[L] - u[K])
            else:
                d_boundary = abs(
                    np.cross(edge1 - edge0, edge0 - mesh.centers[K])
                ) / len_edge
                tau_E = len_edge / d_boundary
                edge_average = (
                    Polynomial2D.line_integral(
                        lambda x: dirichlet(x, time_value),
                        [edge0, edge1]
                    ) / len_edge
                )
                dofs[4 + E] = tau_E * (edge_average - u[K])
        element = MorleyRectangle(verts)
        I_Mu.append(element.function(dofs)) # The polynomial space follows the paper rectangular Morley element. The space enriched by cubic functiond
    return I_Mu
# ============================================================
# ONE BACKWARD EULER STEP
# ============================================================
def solve_one_step(mesh,u_current,current_time,dt):
    A = assemble_stiffness_matrix(mesh)
    M, masses = assemble_mass_matrix(mesh)
    t_new = current_time + dt
    rhs = assemble_rhs(mesh, t_new)
    system_matrix = ((1.0/dt)*M +A)
    rhs_total = ((1.0/dt)*(M @ u_current) + rhs)
    u_next = spsolve(system_matrix,rhs_total)
    return u_next
# ============================================================
# 10. Polynomial derivatives
# ============================================================
def gradient_polynomial(poly):
    px = poly.differentiate('x')
    py = poly.differentiate('y')
    return px, py
def laplacian_polynomial(poly):
    px = poly.differentiate('x')
    py = poly.differentiate('y')
    pxx = px.differentiate('x')
    pyy = py.differentiate('y')
    return Polynomial2D.add(pxx, pyy, 1.0, 1.0)
# ============================================================
# 12. Estimator residual and jumps
# ============================================================
def cell_residual_norm_sq(mesh, K, I_Mu, u_new, u_old, dt, time_value):
    verts = mesh.points[mesh.cells[K]]
    lap_poly = laplacian_polynomial(I_Mu[K])
    time_term = (u_new[K] - u_old[K]) / dt 
    def residual_function(x):
        return forcing(x, time_value) - time_term + lap_poly.evaluate_vec(x)
    total = 0.0
    for tri in [(verts[0], verts[1], verts[2]), (verts[0], verts[2], verts[3])]: # Each residul is integrated over two triangles which equal to one rectangle.
        points, weights = Polynomial2D.integral_tri_points(*tri)
        residual_values = np.array([residual_function(p) for p in points])
        total += np.dot(weights, residual_values ** 2)
    return total
def normal_jump_norm_sq_on_edge(mesh, K, E, I_Mu):
    L = mesh.neighbors[K, E]
    if L == -1:
        return 0.0
    edge0, edge1 = edge_points_of_cell(mesh, K, E)
    nE = unit_normal(edge0, edge1)
    pxK, pyK = gradient_polynomial(I_Mu[K])
    pxL, pyL = gradient_polynomial(I_Mu[L])
    def jump_function(x):
        gradK = np.array([pxK.evaluate_vec(x), pyK.evaluate_vec(x)])
        gradL = np.array([pxL.evaluate_vec(x), pyL.evaluate_vec(x)])
        return np.dot(gradK, nE) - np.dot(gradL, nE)
    points, weights = Polynomial2D.line_integral_points([edge0, edge1])
    jump_values = np.array([jump_function(p) for p in points])
    return np.dot(weights, jump_values ** 2)
def tangential_jump_norm_sq_on_edge(mesh, K, E, I_Mu):
    L = mesh.neighbors[K, E]
    edge0, edge1 = edge_points_of_cell(mesh, K, E)
    tE = unit_tangent(edge0, edge1)
    pxK, pyK = gradient_polynomial(I_Mu[K])
    if L != -1:
        pxL, pyL = gradient_polynomial(I_Mu[L])
        def jump_function(x):
            gradK = np.array([pxK.evaluate_vec(x), pyK.evaluate_vec(x)])
            gradL = np.array([pxL.evaluate_vec(x), pyL.evaluate_vec(x)])
            return np.dot(gradK, tE) - np.dot(gradL, tE)
    else:
        def jump_function(x):
            gradK = np.array([pxK.evaluate_vec(x), pyK.evaluate_vec(x)])
            return np.dot(gradK, tE)
    points, weights = Polynomial2D.line_integral_points([edge0, edge1])
    jump_values = np.array([jump_function(p) for p in points])
    return np.dot(weights, jump_values ** 2)
def local_estimator_eta_K_sq(mesh, K, I_Mu, u_new, u_old, dt, time_value):
    verts = mesh.points[mesh.cells[K]]
    hK = rectangle_diameter(verts)
    residual_sq = cell_residual_norm_sq(mesh, K, I_Mu, u_new, u_old, dt, time_value)
    normal_part = 0.0
    tangential_part = 0.0
    for E in range(4): # Loop over 4 edges
        edge0, edge1 = edge_points_of_cell(mesh, K, E)
        hEK = h_EK_rectangle(verts, edge0, edge1)
        if mesh.neighbors[K, E] != -1:
            Jn_sq = normal_jump_norm_sq_on_edge(mesh, K, E, I_Mu)
            normal_part += (1.0 / hEK) * Jn_sq
        Jt_sq = tangential_jump_norm_sq_on_edge(mesh, K, E, I_Mu)
        tangential_part += (1.0 / hEK) * Jt_sq
    return hK ** 2 * (residual_sq + normal_part + tangential_part)
def global_estimator_eta(mesh, I_Mu, u_new, u_old, dt, time_value):
    eta_sq = 0.0
    eta_local = np.zeros(mesh.nsimplices)
    for K in range(mesh.nsimplices):
        eta_local[K] = local_estimator_eta_K_sq(mesh, K, I_Mu, u_new, u_old, dt, time_value)
        eta_sq += eta_local[K]
    return np.sqrt(eta_sq), eta_local,
# ============================================================
# MAXIMUM STRATEGY FOR 2D REFINEMENT
# ============================================================
def mark_xy_lines(mesh, eta_local, theta=0.5):
    eta_values = np.sqrt(eta_local)
    eta_max = np.max(eta_values)
    x_targets = []
    y_targets = []
    for K in range(mesh.nsimplices):
        etaK = eta_values[K]
        if etaK >= theta * eta_max:
            x_targets.append(mesh.centers[K][0])
            y_targets.append(mesh.centers[K][1])
    x_targets = np.unique(np.array(x_targets))
    y_targets = np.unique(np.array(y_targets))
    return x_targets, y_targets
# ============================================================
# FIND PARENT COARSE CELL
# ============================================================
def find_parent_cell(coarse_mesh,refined_center):
        x = refined_center[0]
        y = refined_center[1]
        for T in range(coarse_mesh.nsimplices):
            verts = coarse_mesh.points[
            coarse_mesh.cells[T]]
            xmin = np.min(verts[:,0])
            xmax = np.max(verts[:,0])
            ymin = np.min(verts[:,1])
            ymax = np.max(verts[:,1])
            if (xmin <= x <= xmax and ymin <= y <= ymax):
                return T
        return None
# ============================================================
# TRANSFER COARSE SOLUTION TO REFINED MESH (2D)
# ============================================================
def transfer_solution(old_mesh, old_u, new_mesh):
    new_u = np.zeros(new_mesh.nsimplices)
    for S in range(new_mesh.nsimplices):
        # centre of refined cell
        center = new_mesh.centers[S]
        # parent coarse cell
        T = find_parent_cell(old_mesh, center)
        if T is None:
            continue
        # reconstruct four child values
        uSW, uSE, uNW, uNE, gx, gy = \
            reconstructed_subcell_values_2D(old_mesh,T,old_u)
        # parent centre
        xc = old_mesh.centers[T][0]
        yc = old_mesh.centers[T][1]
        # choose correct quadrant
        if center[0] < xc:
            # left children
            if center[1] < yc:
                new_u[S] = uSW
            else:
                new_u[S] = uNW
        else:
            # right children
            if center[1] < yc:
                new_u[S] = uSE
            else:
                new_u[S] = uNE
    return new_u
def task(n=4,Tfinal=3.0,dt=0.1,tolerance=0.001,theta=0.2):
    print("="*90)
    mesh = structured_meshes(nx=n,ny=n)
    u_current = initial_cell_values(mesh)
    current_time = 0.0
    step = 0
    while current_time < Tfinal:
        step += 1
        print()
        print("="*90)
        print(f"STEP {step}")
        print(f"time = {current_time:.4f}")
        print(f"cells = {mesh.nsimplices}")
        # SOLVE ONE TIME STEP
        u_next = solve_one_step(mesh,u_current,current_time,dt)
        t_new = current_time + dt
        # RECONSTRUCTION
        I_Mu = reconstruct(mesh,u_next,t_new)
        # ESTIMATOR
        eta_global, eta_local = (global_estimator_eta(mesh,I_Mu,u_next,u_current,dt,t_new))
        print(f"eta = {eta_global:.10e}")
        # ACCEPT
        if eta_global < tolerance:
            print("ACCEPT STEP")
            u_current = u_next
            current_time = t_new
        # REFINE AND REDO
        else:
            print("REFINE MESH")
            x_targets, y_targets = mark_xy_lines(mesh,eta_local,theta)
            print("marked x = ",x_targets)
            print("marked y = ",y_targets)
            old_mesh = mesh
            old_u = u_current.copy()
            refined_mesh = structured_meshes(
                nx=mesh.nx,
                ny=mesh.ny,
                xs=np.unique(mesh.points[:,0]),
                ys=np.unique(mesh.points[:,1])
            )
            refined_mesh.refine_vertical_columns(x_targets)
            refined_mesh.refine_horizontal_rows(y_targets)
            mesh = refined_mesh
            u_current = transfer_solution(old_mesh,old_u,mesh)
            print("nx =", mesh.nx)
            print("ny =", mesh.ny)
# ============================================================
# RUN TASK 
# ============================================================
if __name__ == "__main__":task(n=4,Tfinal=1.0,dt=0.2,tolerance=0.003,theta=0.2)