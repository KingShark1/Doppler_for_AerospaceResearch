"Numerical propgator based on RK4, takes into account J2 and drag perturbations"
#from propagation.cowell import drag, sdot,rkf45
import numpy as np
mu = 398600.4418  # gravitational parameter mu
J2 = 1.08262668e-3 # J2 coefficient
Re = 6378.137  # equatorial radius of the Earth
we = 7.292115e-5  # rotation rate of the Earth in rad/s
ee = 0.08181819  # eccentricity of the Earth's shape
def drag(s):
    """Returns the drag acceleration for a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x3 numpy array: the drag acceleration [ax,ay,az]
    """

    r = np.linalg.norm(s[0:3])
    v_atm = we*np.array([-s[1],s[0],0])   # calculate velocity of atmosphere
    v_rel = s[3:6] - v_atm

    rs = Re*(1-(ee*s[2]/r)**2)   # calculate radius of surface
    h = r-rs
    p = 0.6*np.exp(-(h-175)*(29.4-0.012*h)/915) # in kg/km^3
    coeff = 3.36131e-9     # in km^2/kg
    acc = -p*coeff*np.linalg.norm(v_rel)*v_rel

    return acc

def j2_pert(s):
    """Returns the J2 acceleration for a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x3 numpy array: the J2 acceleration [ax,ay,az]
    """

    r = np.linalg.norm(s[0:3])
    K = -3*mu*J2*(Re**2)/2/r**5
    comp = np.array([1,1,3])
    comp = comp - 5*(s[2]/r)**2
    comp = np.multiply(comp,s[0:3])
    comp = np.multiply(K,comp)

    return comp

def sdot(s):
    """Returns the time derivative of a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x6 numpy array: the time derivative of s [vx,vy,vz,ax,ay,az]
    """

    mu = 398600.4405
    r = np.linalg.norm(s[0:3])
    a = -mu/(r**3)*s[0:3]

    p_j2 = j2_pert(s)
    p_drag = drag(s)

    a = a+p_j2+p_drag
    return np.array([*s[3:6],*a])

s = np.array([2.87327861e+03,5.22872234e+03,3.23884457e+03,-3.49536799e+00,4.87267295e+00,-4.76846910e+00])
print(sdot(s))