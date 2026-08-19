import numpy as np
from scipy import integrate

class Polynomial2D:
    def __init__(self, coefficients): # Creates a polynomial from list of coefficients.
        self.coefficients = np.zeros(15) # first create an array of 15 zeros.
        self.coefficients[0:len(coefficients)] = np.array(coefficients) # then fill the beginning with that given coefficients
        
    def evaluate_vec(self,x): # Evaluates the polynomial at a point written as a vecctor x = [x0,x1]
        return self.evaluate(x[0],x[1])
    
    # Evaluate polynomial at point (x,y)
    def evaluate(self, x, y):
        terms = np.dot(self.coefficients, np.array([1,x,y,x**2,x*y,y**2,x**3,x*(y**2),y*(x**2),y**3,y*y*x*x,x**4,y*(x**3),x*(y**3),y**4])) # this is how polynomial is evaluated numerically.
        return np.sum(terms)
    def __repr__(self):
        monomials = ["1","x","y","x^2","xy","y^2","x^3","xy^2","yx^2","y^3","y^2x^2","x^4","yx^3","xy^3","y^4"]
        representation = ""
        for i in range(len(self.coefficients)):
            if self.coefficients[i] > 0:
                representation += "+"+str(self.coefficients[i])+monomials[i]
            elif self.coefficients[i] < 0:
                representation += str(self.coefficients[i])+monomials[i]
        return representation

    # Integrate function f over triangle given by z_0,z_1,z_2
    def gen_integral_tri(f, z_0, z_1, z_2):
        points, weights = Polynomial2D.integral_tri_points(z_0, z_1, z_2) # first gets quadrature points and wights from integral-tri-points then computes 
        integral = np.sum([weights[i]*f(points[i]) for i in range(len(weights))])
        return integral # this is quadrature formula 

    # Gives collocation points and weights to numerically integrate over triangle given by points z_0, z_1, z_2
    def integral_tri_points(z_0, z_1, z_2): # These are 16 quadrature points on a refrence triangle. 
        points = [[0.0571041961, 0.06546699455602246],\
        [0.2768430136, 0.05021012321401679],\
        [0.5835904324, 0.02891208422223085],\
        [0.8602401357, 0.009703785123906346],\
        [0.0571041961, 0.3111645522491480],\
        [0.2768430136, 0.2386486597440242],\
        [0.5835904324, 0.1374191041243166],\
        [0.8602401357, 0.04612207989200404],\
        [0.0571041961, 0.6317312516508520],\
        [0.2768430136, 0.4845083266559759],\
        [0.5835904324, 0.2789904634756834],\
        [0.8602401357, 0.09363778440799593],\
        [0.0571041961, 0.8774288093439775],\
        [0.2768430136, 0.6729468631859832],\
        [0.5835904324, 0.3874974833777692],\
        [0.8602401357, 0.1300560791760936]]
        weights = np.array([0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496, 0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504, 0.08837017702418863, 0.1326884322074010, 0.08467944903812383, 0.02033451909634504, 0.04713673637581137, 0.07077613579259895, 0.04516809856187617, 0.01084645180365496]) # these are the 16 corresponding quadrature weights.
        transform_matrix = np.transpose(np.array([z_1-z_0,z_2-z_0])) # Builds the linear part of the map from the refrence triangle to the actual triangle. The matrix uses the edges vecctors: z1-z0, z2-z0
        return [np.matmul(transform_matrix,points[i])+z_0 for i in range(len(points))], 0.5*abs(np.linalg.det(transform_matrix))*weights # Returns: Mapped quadrature points in the real triangle, Scaled weights of the real triangle.

    # Integrates polynomial numerically on triangle given by points z_0, z_1, z_2
    def integral_tri(self, z_0, z_1, z_2): # Integrates the current polynomial over a triangle.
        return Polynomial2D.gen_integral_tri(lambda x: self.evaluate_vec(x),z_0,z_1,z_2) # Calls gen_tri with the function equal to this polynomial.

    # Numerically integrates normal derivative of polynomial on edge of triangle
    def integral_normal_line(self, edge, other):
        dx = self.differentiate(variable='x') # Compute partial derivatives px, py
        dy = self.differentiate(variable='y')
        normal = np.flip(edge[1]-edge[0]) # construct a perpendiccular vector to the edge which is normal derivative.
        normal[1] = -normal[1]
        if np.dot(normal, other-edge[0])>= 0:
            normal = -normal # Possibly flip the normal direction.
        normal = normal/(np.linalg.norm(normal)) # turn normal vector into unit normal vector.
        dn = Polynomial2D.add(dx,dy,normal[0],normal[1]) # Form polynomial: dn = nxpx + nypy
        return Polynomial2D.line_integral(lambda x: dn.evaluate_vec(x), edge) # integrate that normal derivatives along the edges.

    # Numerically integrates function f on line
    def line_integral(f, line):
        points, weights = Polynomial2D.line_integral_points(line)
        integral = sum([weights[i]*f(points[i]) for i in range(5)])
        return integral
    
    # Returns collocation points and weights for numerically integration on given line
    def line_integral_points(line): # Integral on edge
        weights = np.array([0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908]) # these are the standard weights of the 5 points Guass-Legendre quadrature on [-1,1]
        points = np.array([-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]) # These are standard Gauss - Legendre nodes on [-1,1]
        coeff =[(line[0]+line[1])/2.0, 0.5*(line[1]-line[0])] # builds the affine map from [-1,1] to the actual edge by midpoint of edge and half edge vector
        return [coeff[0]+points[i]*coeff[1] for i in range(5)], np.linalg.norm(line[1]-line[0])*0.5*weights  # The integral over the real edge need length scaling

    # Differentiates polynomial with respect to the specified variable
    def differentiate(self, variable='x'):
        monomials = ["1","x","y","x^2","xy","y^2","x^3","xy^2","yx^2","y^3","y^2x^2","x^4","yx^3","xy^3","y^4"]
        if variable not in ['x', 'y']:
            raise ValueError("Variable must be 'x' or 'y'.")
        if variable == 'x': # Applies a matrix representing differentiation with respect to x.
            derivative = np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # Each monomial transform as: x ---> 1, x2---> 2x etc
                [0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,3,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,2,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,4,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,3,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]) # This is compact way to differentiate the polynbomial automatically.
            return Polynomial2D(np.matmul(derivative,self.coefficients).flatten())
        else:  # variable == 'y'
            derivative = np.array([[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,2,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,3,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,3,0],
                [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
            return Polynomial2D(np.matmul(derivative,self.coefficients).flatten())

    # Adds two  polynomials
    def add(polynomial_1, polynomial_2,a,b): # this return linear combination ap1+bp2
        return Polynomial2D(a*polynomial_1.coefficients+b*polynomial_2.coefficients)

