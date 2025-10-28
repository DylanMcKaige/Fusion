"""
Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 

The shape of the domain here is specifically for DBS simulations.

To use this, from Scotty results, decide on the distance_to_ERMES_port variable (distance from launcher to launch boundary in ERMES). The rest is automatic

Spits out .txt of all the necessary data. 

All paths are w.r.t CWD

TODO
1. Separate this into 2-3 different files?

References
    [1] Two dimensional full-wave simulations of Doppler back-scattering in tokamak plasmas with COMSOL by Quinn Pratt et al (in-progress paper)
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin, pg 43-44
    [4] V. H. Hall-Chen, F. I. Parra, and J. C. Hillesheim, “Beam model of Doppler backscattering,” Plasma Phys. Control. Fusion, vol. 64, no. 9, p. 095002, Sep. 2022, doi: 10.1088/1361-6587/ac57a1.
    [5] F. M. A. Smits, “ELLIPTICAL POLARISATION FOR OBLIQUE EC-WAVE LAUNCH”.

NOTE
SCOTTY_cartesian refers to X Y Z where X, Z align with R, Z, Y is toroidal into the page. We will call this RtZ
ERMES_cartesian refers to X Y Z where X, Y align to R, Z, Z is toroidal out of the page
CYL referes to R zeta Z where zeta is toroidal angle
BEAM referes to xhat yhat ghat

Written by Dylan James Mc Kaige
Created: 16/5/2025
Updated: 28/10/2025
"""
import os, json, datatree
import numpy as np
import numpy as numpy
import pandas as pd
from tqdm import tqdm
from math import sin, cos, tan, acos, atan, sqrt, fabs
from scipy.constants import c, pi, m_e, m_p, elementary_charge, epsilon_0
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, griddata, CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colormaps as cm
from scotty.analysis import beam_width
from scotty.plotting import plot_poloidal_crosssection, plot_toroidal_contour, maybe_make_axis
from scotty.fun_general import find_vec_lab_Cartesian, find_Psi_3D_lab_Cartesian


def handle_scotty_launch_angle_sign(dt: datatree):
    """
    Because I keep messing up the sign conventions. Also returns it as an acute angle. 
    + Above the horizontal, - Below the horizontal (pointing into the plasma from the right)

    Args:
        dt (datatree): Scotty output file
    """
    
    if abs(dt.inputs.poloidal_launch_angle_Torbeam.values) < 180:
        return abs(dt.inputs.poloidal_launch_angle_Torbeam.values)
    else:
        return (abs(dt.inputs.poloidal_launch_angle_Torbeam.values)-360)

def scotty_cyl_to_RtZ(array, dt):
    """
    Convert Scotty Cylindrical to Scotty Cartesian, R zeta Z to R t Z
    
    Args:
        array (array): The array to convert
        dt (datatree): Scotty output file
    """
    cos_q_zeta = np.cos(dt.analysis.q_zeta.values)
    sin_q_zeta = np.sin(dt.analysis.q_zeta.values)
    cart = np.empty([dt.inputs.len_tau.values, 3])
    cart[:, 0] = array[:, 0] * cos_q_zeta - array[:, 1] * sin_q_zeta
    cart[:, 1] = array[:, 0] * sin_q_zeta + array[:, 1] * cos_q_zeta
    cart[:, 2] = array[:, 2]
    return cart

def RtZ_to_XYZ(a: np.array) -> np.array:
    """
    Convert a vector array from SCOTTY cartesian (R,t,Z) to ERMES cartesian (X,Y,Z), keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (R,t,Z) SCOTTY cartesian basis
    
    Returns:
        b (array): The transformed vector in (X,Y,Z) ERMES Cartesian basis

    """
    assert len(a) == 3, "This function only supports single arrays of length 3, if you do this on an array of arrays, you gotta stack it"
    
    b = np.array([a[0], a[2], -a[1]])
    return b

def XYZ_to_RtZ(a: np.array) -> np.array:
    """
    Convert a vector array from ERMES cartesian (X,Y,Z) to SCOTTY cartesian (R,t,Z), keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (X,Y,Z) ERMES Cartesian basis
    
    Returns:
        b (array): The transformed vector in (R,t,Z) SCOTTY cartesian basis

    """
    assert len(a) == 3, "This function only supports single arrays of length 3, if you do this on an array of arrays, you gotta stack it"
    
    b = np.array([a[0], a[2], -a[1]])
    return b
    
def load_scotty_data(path: str) -> datatree.DataTree:
    """
    Load data from scotty and return the datatree.
    
    Args: 
        path (str): Path (relative to cwd) to Scotty output file inclusive of the file name
    
    Returns:
        dt (DataTree): Scotty output datatree
    """
    assert os.path.exists(os.getcwd() + path), f"Path '{os.getcwd() + path}' does not exist."
    dt = datatree.open_datatree(os.getcwd() + path, engine="h5netcdf")
    
    return dt

def find_lcfs_entry_point(
    pol_flux_spline: RectBivariateSpline, 
    R0: float, 
    Z0: float, 
    launch_angle_rad: float, 
    psi_closed: float, 
    step: float = 0.001, 
    max_dist: float = 0.5
    ):
    """    
    Finds the point of entry along the Last Closed Flux Surface for calculation of polarization vector (Linear).
    Only used if no Scotty output file is provided (so generally you wouldnt be using this)

    Args:
        pol_flux_spline (RectBivariateSpline): Poloidal flux spline
        R0 (float): R of launcher (or centre of port since they are collinear)
        Z0 (float): Z of launcher (or centre of port since they are collinear)
        launch_angle_rad (float): Launch angle in rad w.r.t -ve R axis
        psi_closed (float): Value of poloidal flux for the LCFS
        step (float): Step size for line interpolation. Defaults to 0.001
        max_dist (float): Maximum distance away from launcher (or port) to search. Defaults to 0.5

    Returns:
        R_entry (float): R of point of entry
        Z_entry (float): Z of point of entry
    """
    print("Why are you using this? Get a Scotty output file, it'll make your life easier.")
    s_vals = np.arange(0, max_dist, step)
    R_vals = R0 - s_vals * cos(-launch_angle_rad)
    Z_vals = Z0 - s_vals * sin(-launch_angle_rad)

    psi_vals = pol_flux_spline.ev(R_vals, Z_vals)
    for i in range(len(psi_vals) - 1):
        psi1 = psi_vals[i]
        psi2 = psi_vals[i + 1]

        if (psi1 < psi_closed and psi2 >= psi_closed) or (psi1 > psi_closed and psi2 <= psi_closed):
            
            # Linear interpolation between psi1 and psi2
            frac = (psi_closed - psi1)/(psi2 - psi1 + 1e-12) # Arbitrarliy prevent divide-by-zero errors
            s_entry = s_vals[i] + frac*step
            R_entry = R0 - s_entry*cos(-launch_angle_rad)
            Z_entry = Z0 - s_entry*sin(-launch_angle_rad)
            return R_entry, Z_entry

    raise "Last Closed Flux Surface not intersected along beam path"
    
def get_pol_from_scotty(dt: datatree.DataTree):
    """
    Get the polarization vector from a Scotty output file. This function assumes
    1) Tau starts from point of entry (may not be exactly equal to the one calculated by find_lcfs_entry_point)
    2) y_hat_Cartesian, b_hat_Cartesian, g_hat_Cartesian are in the (R,t,Z) basis, e_hat is in the (u1,u2,bhat) basis
    3) Equations 104-106 from [4] apply exactly to let us derive u1_hat and u2_hat
    
    Produces an elliptical polarization vector.

    Args:
        dt (DataTree): Scotty output file
        
    Returns:
        e_hat_XYZ (array): Polarization vector in (X,Y,Z) basis
    """
 
    y_hat_RtZ = dt.analysis.y_hat_Cartesian.values
    g_hat_RtZ = dt.analysis.g_hat_Cartesian.values
    b_hat_RtZ = find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values)/np.linalg.norm(find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values))
    e_hat = dt.analysis.e_hat.values # This is in u1, u2, bhat basis
    u2_hat_RtZ = y_hat_RtZ/np.linalg.norm(y_hat_RtZ)
    u1_hat_RtZ = (np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))/np.linalg.norm(np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))

    # Form the basis transition vector from u1,u2,b to R,t,Z
    uub_to_RtZ_basis = np.column_stack((u1_hat_RtZ[0],u2_hat_RtZ[0],b_hat_RtZ[0]))

    # Pol vector at entry in R,t,Z
    e_hat_entry = e_hat[0]
    e_hat_RtZ = np.dot(uub_to_RtZ_basis, e_hat_entry)
    e_hat_RtZ = e_hat_RtZ/np.linalg.norm(e_hat_RtZ)
    e_hat_XYZ = RtZ_to_XYZ(e_hat_RtZ) # Project to ERMES cartesian
    
    return e_hat_XYZ

def scotty_pol_to_ERMES(pol: np.array, k_vec: np.array, E0: float = 1.0):
    """
    Convert a complex polarisation vector from Scotty to a real poalrisation vector in ERMES.
    
    In ERMES, rho is the pol vector, delta is the wave vector, eta = delta x rho. For consistency with everywhere else, k is the wavevector
    We define rho = real(pol) projected onto the plane defned by k (force rho to be perp to k) and find eta using eta = k x rho
    We then project rho and eta onto pol to determine their ratio
    We use the ratio to get E_rho and E_eta
    phi_rho and phi_eta are determined from these projections
    
    NOTE
    This was a bit of divine inspiration that happens to work so far (Literally came to me in a dream). 
    VERY good agreement for lower angle cases WITHOUT forcing the projection. 
    Seems to lose agreement as poloidal launch angle increases (why?, increase seems to be monotonic)
    
    Force projection to maintain k perp to rho relation
    tried with k at launch, k at entry, ghat at entry. Similar results for the 3. 
    None fully eliminate O mode at higher angles

    Args:
        pol (np.array): Complex polarization vector from Scotty in ERMES cartesian vasis
        E0 (float, optional): Magnitude of incident E field. Defaults to 1.0.
        
    Return:
        E_perp (float): |E_perp|
        phi_perp (float): Phase of E_perp
        E_par (float): |E_par|
        phi_par (float): Phase of E_par
    """
    v = np.asarray(pol, complex) # Force the type casting just in case
    k = np.asarray(k_vec, float) # Force the type casting just in case
    k = k/np.linalg.norm(k) # Normalize just in case
    
    # Define rho as real part of v projected onto plane perp to k
    rho0 = np.real(v)
    rho = rho0 - np.dot(rho0, k)*k
    rho = rho/np.linalg.norm(rho)

    # Build eta from k x rho
    cross = np.cross(k, rho)
    eta = cross/np.linalg.norm(cross)

    # Projections of v onto basis, these are the complex scaling factors
    a_rho = np.vdot(rho, v)
    a_eta = np.vdot(eta, v)

    E_rho = E0*a_rho
    E_eta = E0*a_eta

    # Results
    E_par, phi_par = np.abs(E_rho), float(np.angle(E_rho))
    E_perp, phi_perp = np.abs(E_eta), float(np.angle(E_eta))

    # Check orthogonality
    k_dot_rho = float(np.dot(k, rho))
    print("k dot rho:", k_dot_rho)

    return rho, E_perp, phi_perp, E_par, phi_par

def rotate_rodrigues(a: np.array, b: np.array, theta) -> np.array:
    """
    Rotate vector a about b by an angle of theta using Rodrigues' formula

    Args:
        a (np.array): The vector to be rotated about b
        b (np.array): The vector of reference
        theta (float or array): Angle of rotation following right-hand rule, relative to a, in rad

    Returns:
        a_prime (np.array): The rotated vector
    """
    bcrossa = np.reshape(np.cross(b, a), (3,1))
    bdota = np.dot(b, a)
    a = np.reshape(a, (3,1))
    b = np.reshape(b, (3,1))
    a_prime = a*(np.reshape(np.cos(theta), (1, len(theta)))) + \
        bcrossa*(np.reshape(np.sin(theta), (1, len(theta)))) + \
            b*bdota*(np.ones(len(theta)) - (np.reshape(np.cos(theta), (1, len(theta)))))
    return a_prime.T

def get_psi_normal_entry(psi_spline: RectBivariateSpline, R_entry, Z_entry) -> np.array:
    """
    Get the outward normal vector to the LCFS at the point of entry

    Args:
        psi_spline (RectBivariateSpline): Psi interpolation spline
        R_entry (float): R of entry point
        Z_entry (float): Z of entry point

    Returns:
        psi_normal (array): Outward normal vector at entry of LCFS in ERMES cartesian basis
    """
    dpsi_dR = psi_spline.ev(R_entry, Z_entry, dx=1, dy = 0)
    dpsi_dZ = psi_spline.ev(R_entry, Z_entry, dx=0, dy = 1)
    
    psi_normal = np.array([dpsi_dR, dpsi_dZ, 0])
    return psi_normal
   
def get_limits_from_scotty(dt: datatree, padding_R: float = 0.03, padding_Z: float = 0.01, padding_t: float = 0.01, cartesian_scotty: bool = False):
    """
    Get the min and max R t Z from Scotty in ERMES cartesian basis by adding padding to tor and pol width

    Args:
        dt (datatree): Scotty output file in .h5 format
        padding (float): padding
        cartesian_scotty (bool): Are we using cartesian_scotty (Since variable names are different)

    Returns:
        lims (array): In the form [[x, x, y, y, z, z]]
    """
    
    if cartesian_scotty:
        min_z = dt.inputs.Y[0] # Cus in Scotty, Y is toroidal for cartesian
        max_z = dt.inputs.Y[-1]
        width_at_tau = dt.analysis.beam_width_2
        beam_width_vector = dt.analysis.y_hat_cartesian*width_at_tau
        
        data_Yaxis = {
            "q_X": np.array(dt.solver_output.q_X),
            "q_Y": np.array(dt.solver_output.q_Y),
            "q_Z": np.array(dt.solver_output.q_Z),
        }
        
        beam_vector = np.column_stack([data_Yaxis["q_X"], data_Yaxis["q_Y"], data_Yaxis["q_Z"]])
        beam_plus = beam_vector + beam_width_vector
        beam_minus = beam_vector - beam_width_vector
        combined_beam_R = np.concatenate([beam_plus.sel(col="X"), beam_minus.sel(col="X")])
        combined_beam_Z = np.concatenate([beam_plus.sel(col="Z"), beam_minus.sel(col="Z")])
        
        min_x, max_x = (np.min(combined_beam_R) - padding_R), (np.max(combined_beam_R) + padding_R)
        min_y, max_y = (np.min(combined_beam_Z) - padding_Z), (np.max(combined_beam_Z) + padding_Z)
    else:
        width_tor = beam_width(dt.analysis.g_hat_Cartesian,np.array([0.0, 0.0, 1.0]),dt.analysis.Psi_3D_Cartesian)
        width_pol = beam_width(dt.analysis.g_hat,np.array([0.0, 1.0, 0.0]),dt.analysis.Psi_3D)
        beam_plus_tor, beam_minus_tor = dt.analysis.beam_cartesian + width_tor, dt.analysis.beam_cartesian - width_tor
        beam_plus_pol, beam_minus_pol = dt.analysis.beam + width_pol, dt.analysis.beam - width_pol
        combined_beam_X = np.concatenate([beam_plus_tor.sel(col_cart="X"), beam_minus_tor.sel(col_cart="X")])
        combined_beam_Y = np.concatenate([beam_plus_tor.sel(col_cart="Y"), beam_minus_tor.sel(col_cart="Y")])
        combined_beam_R = np.concatenate([beam_plus_pol.sel(col="R"), beam_minus_pol.sel(col="R"), combined_beam_X])
        combined_beam_Z = np.concatenate([beam_plus_pol.sel(col="Z"), beam_minus_pol.sel(col="Z")])

        # In ERMES cartesian frame
        min_x, max_x = (np.min(combined_beam_R) - padding_R), (np.max(combined_beam_R) + padding_R)
        min_y, max_y = (np.min(combined_beam_Z) - padding_Z), (np.max(combined_beam_Z) + padding_Z)
        min_z, max_z = (np.min(combined_beam_Y) - padding_t), (np.max(combined_beam_Y) + padding_t)
    
    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])

def get_ERMES_parameters(
    dt: datatree.DataTree,
    prefix: str = "",
    launch_position: np.array = None,
    dist_to_ERMES_port: float = None,
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    cartesian_scotty: bool = False
    ):
    """
    The main function
    
    Generate ERMES parameters for given input
    
    Args:
        dt (DataTree): Scotty output file in .h5 format
        prefix (str): Prefix for naming (e.g MAST-U, DIII-D, etc), defaults to None
        launch_position (array): Launch position in [R,t,Z] coordinates in m
        dist_to_ERMES_port (float): Distance from launcher to port in ERMES in m, stare at Scotty to decide this
        plot (bool): Plot everything
        save (bool): Save everything
        path (str): Path to save file in, defaults to cwd
        cartesian_scotty (bool): Are we using cartesian_scotty (Since variable names are different)
        
    Returns:
        Plots the position of the required points in the Z-R axes and saves a .txt file of the necessary values for ERMES
        Optionally generates the necessary points and launch parameters needed for ERMES
    """   
    launch_freq_GHz = dt.inputs.launch_freq_GHz.values
    launch_beam_width = dt.inputs.launch_beam_width.values
    launch_beam_curvature = dt.inputs.launch_beam_curvature.values
    launch_angle_pol = handle_scotty_launch_angle_sign(dt)
    launch_angle_tor = dt.inputs.toroidal_launch_angle_Torbeam.values
    k_vec_launch_RtZ = dt.inputs.launch_K.values
    k_vec_launch_XYZ = RtZ_to_XYZ(k_vec_launch_RtZ)
    k_vec_launch_XYZ_norm = k_vec_launch_XYZ / numpy.linalg.norm(k_vec_launch_XYZ)
    launch_R = dt.inputs.launch_position.values[0] if cartesian_scotty is False else launch_position[0]
    launch_Z = dt.inputs.launch_position.values[2] if cartesian_scotty is False else launch_position[2]
    # launch_zeta, t, always equal to 0
    
    degtorad = pi/180
    launch_beam_wavelength = c/(launch_freq_GHz*1e9)
    radius_of_curv = fabs(1/launch_beam_curvature)
    launch_angle_pol_rad = launch_angle_pol*degtorad
    launch_angle_tor_rad = launch_angle_tor*degtorad
    
    filename = prefix + str(launch_angle_pol) + "pol_degree_" + str(launch_angle_tor) + "tor_degree_" + str(launch_freq_GHz) + "GHz_"
    
    # Create subdirectory for saving:
    if save:
        if os.path.isdir(path + filename + 'folder'):
            path = path + filename + 'folder' + '\\'
        else:
            os.makedirs(path + filename + 'folder')
            path = path + filename + 'folder' + '\\'
    
    # Initial calculations
    if launch_beam_curvature != 0:
        distance_to_launcher = fabs((radius_of_curv * pi**2 * launch_beam_width**4)/(launch_beam_wavelength**2 * radius_of_curv**2+pi**2 * launch_beam_width**4))
        z_R = fabs((launch_beam_wavelength*radius_of_curv*distance_to_launcher)/(pi*launch_beam_width**2))
        w0 = sqrt((launch_beam_wavelength*z_R)/(pi))
    else:
        w0 = launch_beam_width
        z_R = pi*w0**2 / launch_beam_wavelength
        distance_to_launcher = 0
    
    w_ERMES = 2*w0*sqrt(1+(dist_to_ERMES_port/z_R)**2) # Width of beam at port position in ERMES
    xw = launch_R - distance_to_launcher*cos(launch_angle_pol_rad) # Centre of waist
    yw = launch_Z + distance_to_launcher*sin(launch_angle_pol_rad) # Centre of waist
    zw = 0 + distance_to_launcher*sin(launch_angle_tor_rad) # Centre of waist
    
    # This formula is from Quinn's thesis
    z0 = 377 # Impedance of free space
    E0 = sqrt(z0*2*1/(w0*sqrt(pi/2))) # For P_in = 1 W/m in 2D
    
    # 2D Port calculations
    
    # Centre of front face
    xp, yp = launch_R - dist_to_ERMES_port*cos(launch_angle_pol_rad), launch_Z + dist_to_ERMES_port*sin(launch_angle_pol_rad) 
    xp0, yp0 = xp - w_ERMES/2*sin(launch_angle_pol_rad), yp - w_ERMES/2*cos(launch_angle_pol_rad) # Bottom
    xp1, yp1 = xp + w_ERMES/2*sin(launch_angle_pol_rad), yp + w_ERMES/2*cos(launch_angle_pol_rad) # Top
    
    # 3D port calculations (project from 2D centre based on toroidal launch angle)
    port_delta_z = w_ERMES/2
    port_padded_delta_z = w_ERMES*1.1/2
    
    # Slightly wider to minimize numerical errors at boundary of port
    xp0_ext, yp0_ext = xp - w_ERMES*1.1/2*sin(launch_angle_pol_rad), yp - w_ERMES*1.1/2*cos(launch_angle_pol_rad) # Bottom
    xp1_ext, yp1_ext = xp + w_ERMES*1.1/2*sin(launch_angle_pol_rad), yp + w_ERMES*1.1/2*cos(launch_angle_pol_rad) # Top

    def generate_launcher_port_3D(launch_R, launch_Z, dist_to_ERMES_port, w_ERMES, k_hat):
        """
        Generate 3D launcher port geometry in ERMES coordinates.

        The port is centered dist_to_ERMES_port along the launch direction (k_hat),
        with a square face perpendicular to k_hat and side length w_ERMES.

        Args:
            launch_R (float): R coordinate of launcher position (m)
            launch_Z (float): Z coordinate of launcher position (m)
            dist_to_ERMES_port (float): Distance from launcher to ERMES port along k_hat (m)
            w_ERMES (float): Full width of the launch port (m)
            k_hat (array): 3D unit wavevector direction (includes both poloidal & toroidal angles)

        Returns:
            port_corners (array): (4,3) Coordinates of the four corners of the main port plane.
            padded_corners (array): (4,3) Coordinates of the padded (1.1xlarger) port plane.
            center (array): Center coordinates of the port plane.
            in_plane_basis (array): (2,3) Orthonormal basis vectors spanning the plane (u,v).
        """

        k_hat = np.asarray(k_hat, float)
        k_hat /= np.linalg.norm(k_hat)  # Ensure normalized

        # Centre
        center = np.array([launch_R, launch_Z, 0.0]) + dist_to_ERMES_port * k_hat

        # Construct orthonormal in-plane basis (u, v)
        # Choose a reference vector not parallel to k_hat
        ref = np.array([0, 0, 1]) if abs(k_hat[2]) < 0.9 else np.array([0, 1, 0])
        u_hat = np.cross(k_hat, ref)
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(k_hat, u_hat)
        v_hat /= np.linalg.norm(v_hat)

        # Get corners
        half_w = w_ERMES / 2
        port_corners = np.array([
            center + half_w * ( u_hat + v_hat),
            center + half_w * (-u_hat + v_hat),
            center + half_w * (-u_hat - v_hat),
            center + half_w * ( u_hat - v_hat)
        ])

        # Padded corners
        pad_w = w_ERMES * 1.1 / 2
        padded_corners = np.array([
            center + pad_w * ( u_hat + v_hat),
            center + pad_w * (-u_hat + v_hat),
            center + pad_w * (-u_hat - v_hat),
            center + pad_w * ( u_hat - v_hat)
        ])

        return port_corners, padded_corners, center, (u_hat, v_hat)

    port_corners, padded_corners, center, (u_hat, v_hat) = generate_launcher_port_3D(
        launch_R, launch_Z, dist_to_ERMES_port, w_ERMES, k_vec_launch_XYZ_norm
    )
    
    # Domain calculations
    min_x, max_x, min_y, max_y, min_z, max_z = get_limits_from_scotty(dt, cartesian_scotty = cartesian_scotty)
    # Bottom right (closest to port) first
    x_br, y_br = xp0_ext, yp0_ext
    x_bl, y_bl = min_x, y_br
    x_tl, y_tl = min_x, max_y
    x_tr, y_tr = xp1_ext, max_y
    
    # Handle entry point
    if cartesian_scotty:
        entry_point = [0, 0, 0] # Not needed for 2D Linear Layer
    else:
        entry_point = RtZ_to_XYZ(dt.inputs.initial_position.values)
    
    # Handle pol vector and E values and phases
    if cartesian_scotty:
        # TODO update this for new format
        rho_hat_perp, mod_E_par, mod_E_perp, rho_hat_rotated_set, rho_hat_rotated = None, E0, 0, None, [[0,0,0],[0,0,0],[0,0,0]] 
    else:
        B_entry_vec_RtZ = np.array([dt.analysis.B_R.values[0], dt.analysis.B_T.values[0], dt.analysis.B_Z.values[0]])
        B_entry_vec_XYZ = RtZ_to_XYZ(B_entry_vec_RtZ)
        
        # linear pol vec at launch point = beam k X B field at launch point in (X,Y,Z)
        k_vec_entry_XYZ = np.array([dt.analysis.K_R.values[0], dt.analysis.K_Z.values[0], 0]) # Nope, not all that better than vacuum k, try perp to g
        g_vec_entry_XYZ = RtZ_to_XYZ(dt.analysis.g_hat_Cartesian.values[0])
        # Polarization from Scotty and convert to ERMES basis
        complex_pol_vec = get_pol_from_scotty(dt = dt)

        rho_hat, mod_E_perp, phi_E_perp, mod_E_par, phi_E_par = scotty_pol_to_ERMES(pol = complex_pol_vec, k_vec = g_vec_entry_XYZ, E0 = E0)

    # For saving    
    def save_ERMES_params(path, filename, vec_names, vec_vals, params_names, params_val):
        """
        Cleaned up saving function

        Args:
            path (str): Path to save
            filename (str): Filename to save
            vec_names (array): Names of vectors
            vecs (array): Values of vectors
            params_names (array): Names of parameters (scalar)
            params_val (array): Value of parameters

        Raises:
            ValueError: If vecs don't have 3
        """
        # Ensure inputs are arrays
        vec_vals = np.asarray(vec_vals, float)
        vec_names = np.asarray(vec_names, str)
        params_names = np.asarray(params_names, str)
        params_val = np.asarray(params_val, float)

        # Validate shape (ERMES Cartesian)
        if vec_vals.shape[1] != 3:
            raise ValueError("`points` must have shape (N, 3) for x, y, z coordinates")

        file_path = f"{path}{filename}_ERMES_params.txt"

        # Write 3D points
        header_points = (
            "=== Cartesian Points in ERMES (3D) ===\n"
            f'{"Point":40s} {"X":>15s} {"Y":>15s} {"Z":>15s}\n'
        )

        with open(file_path, 'w') as f:
            f.write(header_points)
            for name, (x, y, z) in zip(vec_names, vec_vals):
                f.write(f"{name:40s} {x:15.10f} {y:15.10f} {z:15.10f}\n")

            # Write scalar parameters
            f.write("\n\n=== Beam and Simulation Parameters ===\n")
            f.write(f'{"Parameter":40s} {"Value":>15s}\n')
            for name, val in zip(params_names, params_val):
                f.write(f"{name:40s} {val:15.10g}\n")

        print(f"Saved ERMES parameters to {file_path}")

    vec_vals = np.vstack([
        np.asarray(center),
        port_corners.reshape(-1, 3),
        padded_corners.reshape(-1, 3),
        np.array([launch_R, launch_Z, 0]),
        np.array(entry_point),
        np.array([xw, yw, zw]),
        np.asarray(rho_hat),
        np.asarray(k_vec_launch_XYZ_norm)
    ])
    vec_names = np.array([
        'Source Position (front face of port)    ', 
        'Port 0    ', 
        'Port 1    ',
        'Port 2    ',
        'Port 3    ', 
        'Port Padding 0    ',
        'Port Padding 1    ',
        'Port Padding 2    ',
        'Port Padding 3    ',
        'Launch Position    ', 
        'Point of Entry    ',
        'Waist Position    ',
        'pol vec    ',
        'k vec    '
    ])
    
    # Beam params
    params_val = np.array([
        launch_angle_pol, 
        launch_angle_tor, 
        launch_beam_width, 
        radius_of_curv, 
        distance_to_launcher, 
        dist_to_ERMES_port, 
        w0, 
        z_R, 
        launch_freq_GHz, 
        launch_beam_wavelength, 
        E0, 
        mod_E_par,
        phi_E_par,
        mod_E_perp,
        phi_E_perp,
    ])
    params_names = np.array([
        'Poloidal Launch Angle    ', 
        'Toroidal Launch Angle    ', 
        'Launch Beam Width    ', 
        'launch Beam Radius of Curvature    ', 
        'Distance to Launcher (from waist)    ', 
        'Distance to ERMES Port (from launcher)    ', 
        'Beam Waist (w0)    ', 
        'Rayleigh Length (m)    ', 
        'Beam Frequency (GHz)    ', 
        'Beam Wavelength (m)    ', 
        'E0    ', 
        'E par (ERMES)    ',
        'phi E par (ERMES)    ',
        'E perp (ERMES)    ',  
        'phi E perp (ERMES)    ',
    ])

    # For plotting
    points_x = np.array([
        xp, 
        xp0, 
        xp1, 
        xp0_ext,
        xp1_ext,
        x_br,
        x_bl,
        x_tr,
        x_tl,
        launch_R, 
        xw,
        entry_point[0],
    ])
    points_y = np.array([
        yp, 
        yp0, 
        yp1, 
        yp0_ext,
        yp1_ext,
        y_br,
        y_bl,
        y_tr,
        y_tl,
        launch_Z,
        yw,
        entry_point[1],
    ])
    
    # Save it!
    if save:
        save_ERMES_params(path = path, filename = filename + 'ERMES_params', 
                          vec_names=vec_names, vec_vals=vec_vals, 
                          params_names = params_names, params_val=params_val)

    if plot:
        # Note, this only gives a poloidal cross section plot.
        plt.scatter(points_x, points_y, s = 8) 
        ax = plt.gca()
        
        if cartesian_scotty:
            
            width_at_tau = dt.analysis.beam_width_2
            beam_width_vector = dt.analysis.y_hat_cartesian*width_at_tau
            
            data_Yaxis = {
                "q_X": np.array(dt.solver_output.q_X),
                "q_Y": np.array(dt.solver_output.q_Y),
                "q_Z": np.array(dt.solver_output.q_Z),
            }
            
            beam_vector = np.column_stack([data_Yaxis["q_X"], data_Yaxis["q_Y"], data_Yaxis["q_Z"]])
            beam_plus = beam_vector + beam_width_vector
            beam_minus = beam_vector - beam_width_vector

            plt.plot(beam_plus.sel(col="X"), beam_plus.sel(col="Z"), "--k")
            plt.plot(beam_minus.sel(col="X"), beam_minus.sel(col="Z"), "--k", label="Beam width")
            plt.plot(data_Yaxis["q_X"], data_Yaxis["q_Z"], "-", c='black', linewidth=2, zorder=1, label = "Central ray")
        else:
            # Plot the plasma
            plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)
        
            width_pol = beam_width(
                dt.analysis.g_hat,
                np.array([0.0, 1.0, 0.0]),
                dt.analysis.Psi_3D,
            )   
            # Plot the beam
            beam_plus_pol = dt.analysis.beam + width_pol
            beam_minus_pol = dt.analysis.beam - width_pol
            ax.plot(beam_plus_pol.sel(col="R"), beam_plus_pol.sel(col="Z"), "--k")
            ax.plot(beam_minus_pol.sel(col="R"), beam_minus_pol.sel(col="Z"), "--k", label="Beam width")
            ax.plot(
                np.concatenate([dt.analysis.q_R]),
                np.concatenate([dt.analysis.q_Z]),
                "k",
                label="Central ray",
            )
            # Plot the g vector on exit
            #plt.quiver(dt.analysis.q_R.values[-1], dt.analysis.q_Z.values[-1], dt.analysis.g_hat.values[-1][0], dt.analysis.g_hat.values[-1][2], angles = 'xy', scale = 1, color='red')
            
        # Set the lims
        plt.legend()
        plt.xlabel("R (m)")
        plt.ylabel("Z (m)")
        plt.xlim(left=x_bl-0.01)
        plt.ylim(y_bl - 0.01, y_tl + 0.01)
        plt.gca().set_aspect('auto')
        plt.show()

def calc_Eb_from_scotty(dt: datatree, E0: float = 1.0):
    """
    Calcualte the probe beam Electric Field amplitude along the central ray using Eqn 33 of [4]
    Returns Eb(tau) where tau is the beam parameter

    Args:
        dt (datatree): Scotty output file
        E0 (float): For scaling
        
    Returns:
        Eb_tau (array): |Eb| at all the points along the ray w.r.t tau
    """
    # Naming convention, end in tau means index in terms of tau, RtZ means in RtZ basis, xyg means in xyg basis, neither means it is an element (likely), 
    # w means in plane perp to g, g means projected onto g
    # i means im component, launch and ant are equiv
    tau_len = dt.inputs.len_tau.values
    tau_values = dt.analysis.tau.values
    
    x_hat_RtZ = dt.analysis.x_hat.values
    y_hat_RtZ = dt.analysis.y_hat.values
    g_hat_RtZ = dt.analysis.g_hat.values
    g_mag = dt.analysis.g_magnitude.values
    
    q_R = dt.analysis.q_R.values
    q_Z = dt.analysis.q_Z.values
    q_zeta = dt.analysis.q_zeta.values
    launch_R = dt.inputs.launch_position.values[0]
    launch_zeta = dt.inputs.launch_position.values[1]
    launch_Z = dt.inputs.launch_position.values[2]
    launch_K_R = dt.inputs.launch_K.values[0]
    launch_K_zeta = dt.inputs.launch_K.values[1]
    launch_K_Z = dt.inputs.launch_K.values[2]
    K_R = dt.analysis.K_R.values
    K_zeta = dt.analysis.K_zeta_initial.values
    beam_trajectiry_RtZ = dt.analysis.beam_cartesian.values
    
    Psi_xx = dt.analysis.Psi_xx.values
    Psi_xy = dt.analysis.Psi_xy.values
    Psi_yy = dt.analysis.Psi_yy.values
    iPsi_xx_tau=np.imag(Psi_xx)
    iPsi_xy_tau=np.imag(Psi_xy)
    iPsi_yy_tau=np.imag(Psi_yy)
    
    # Rotation/Projection matrices, requires the input to be in RtZ basis
    P_RtZ_to_xyg = np.stack([x_hat_RtZ, y_hat_RtZ, g_hat_RtZ], axis = 1)
    
    def mat_RtZ_to_xyg(mat, index = 0):
        """
        Project a matrix in RtZ basis to xyg basis

        Args:
            mat (array): The vector to transform
            
        Returns:
            mat (array): The transformed vector
        """
        if np.ndim(mat) == 2:
            return np.einsum('ij,jk,lk->il', P_RtZ_to_xyg[index], mat, P_RtZ_to_xyg[index])
        else:
            return np.einsum('nij,njk,nlk->nil', P_RtZ_to_xyg, mat, P_RtZ_to_xyg)
    
    def mat_to_plane_perp_to_g(mat, ghat):
        """
        Project a matrix to the plane perp to g either over tau or a single tau

        Args:
            mat (array): The vector to transform
            
        Returns:
            M_proj (array): The transformed vector
        """
        I = np.eye(3)
        if np.ndim(ghat) == 1: 
            P = I - np.einsum('i,j->ij', ghat, ghat)
            M_proj = np.einsum('ij,jk,lk->il', P, mat, P)
        else: 
            P = I - np.einsum('ni,nj->nij', ghat, ghat)
            M_proj = np.einsum('nij,njk,nlk->nil', P, mat, P)
        return M_proj
    
    # Create Psi_w from Psi_xx,xy,yy. Alternatively, project Psi_3D onto w. TODO Check if these are equal
    Psi_w_xyg = np.zeros((tau_len, 3, 3), dtype=np.complex64) # Such that Psi_w_xyg_tau(N) is the Nth Psi_w corresponding to the Nth tau in xyg basis
    Psi_w_xyg[:, 0, 0] = Psi_xx
    Psi_w_xyg[:, 0, 1] = Psi_xy
    Psi_w_xyg[:, 1, 0] = Psi_xy
    Psi_w_xyg[:, 1, 1] = Psi_yy
    
    Psi_3D_ant_RtZ = find_Psi_3D_lab_Cartesian(dt.analysis.Psi_3D_lab_launch, launch_R, launch_zeta, launch_K_R, launch_K_zeta) # Since Psi_3D is in CYL basis
    g_hat_ant_RtZ = g_hat_RtZ[0]
    Psi_w_ant_RtZ = mat_to_plane_perp_to_g(Psi_3D_ant_RtZ, g_hat_ant_RtZ)
    Psi_w_ant_xyg = mat_RtZ_to_xyg(Psi_w_ant_RtZ)
    
    # 4th root piece (det_piece)
    det_im_Psi_w = iPsi_xx_tau*iPsi_yy_tau-iPsi_xy_tau**2 # Eqn A.67 from [4]
    det_im_Psi_w_ant = np.imag(Psi_w_ant_xyg[0,0])*np.imag(Psi_w_ant_xyg[1,1])-np.imag(Psi_w_ant_xyg[0,1])*np.imag(Psi_w_ant_xyg[1,0]) # Eqn A.67 from [4]
    det_piece = (det_im_Psi_w/det_im_Psi_w_ant)**0.25
    
    # g_piece
    g_mag_ant = 2*c/(2*pi*dt.inputs.launch_freq_GHz.values*1e9) # Eqn 195
    g_piece = (g_mag_ant/g_mag)**0.5
    
    # Finally, calculate |E_b|
    Eb_tau = det_piece*g_piece#*w_dot_Psi_w_dot_w_piece
    
    # A_ant piece, defined based off of first E in ERMES (make them equal)
    A_ant = E0/(Eb_tau[0])
    
    # Normalize to equate the first point
    Eb_tau = A_ant*Eb_tau
    
    return Eb_tau

def compute_torsion(dt: datatree = None):
    """
    Compute torsion, tau, of the central ray from Scotty. This follows the Frenet-Serret frame
    
    Args:
        dt (datatree): Scotty output file
    
    Returns
        tau (array): Torsion along the central ray as a function of beam parameter
    """
    s = dt.analysis.distance_along_line.values # Arc length
    g_hat_XYZ = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=dt.analysis.g_hat.values)
    g_hat_XYZ_norm = g_hat_XYZ / np.linalg.norm(g_hat_XYZ)
    
    dg_ds = np.gradient(g_hat_XYZ_norm, s, axis = 0)
    kappa = np.linalg.norm(dg_ds, axis = 1) # Curvature
    
    # Avoid division by zero
    eps = 1e-12
    N = np.zeros_like(g_hat_XYZ_norm)
    mask = kappa > eps
    N[mask] = dg_ds[mask] / kappa[mask, None]
    
    # Binormal vector B = g_hat × N
    B = np.cross(g_hat_XYZ_norm, N)
    
    # Derivative of B wrt s
    dB_ds = np.gradient(B, s, axis=0)

    # Torsion τ = - (dB/ds · N)
    tau = -np.einsum('ij,ij->i', dB_ds, N)
    
    return tau

def ERMES_nodes_to_XYZ(msh_file: str):
    """
    Load in the ASCII .msh file to read each node as nodeID and return the cartesian cooridnates of that node in xyz. 
    Note that to allow indexing by node, an additional (0,0,0) node is created.
    
    Args:
        msh_file (str): ERMES msh file
        
    Returns:
        node_to_xyz (dict): Dictionary of node xyz coordinates with nodeID as the key
    """
    print("Reading ERMES msh file")
    # Node ID as XYZ coords
    with open(os.getcwd() + msh_file, 'r') as f:
        lines = f.readlines()

    coords = {}
    reading = False

    for line in lines:
        line = line.strip()
        if line.startswith("Coordinates"):
            reading = True
            continue
        if line.startswith("End Coordinates"):
            break
        if reading:
            parts = line.split()
            if len(parts) == 4:
                node_id = int(parts[0])
                x, y, z = map(float, parts[1:])
                coords[node_id] = [x, y, z]

    # Determine max node ID to allocate array correctly
    max_node_id = max(coords.keys())
    node_to_xyz = np.zeros((max_node_id + 1, 3)) # + 1 to allow indexing by node ID, note that this creates a (0,0,0) node 0

    for node_id, xyz in coords.items():
        node_to_xyz[node_id] = xyz
    
    return node_to_xyz
        
def ERMES_results_to_node(res_file: str, result_name: str):
    """
    Load in the .res file to read each result as nodeID and return the value of result_name at that node. Supports scalar and vector results
    
    Args:
        res_file (str): ERMES res file
        result_name (str): Name of the result, as saved by ERMES, that is wanted
        
    Returns:
        result (dict): Dictionary of result value (scalar or vector) with nodeID as the key
    """
    print(f"Reading ERMES res file for '{result_name}' results")
    
    with open(os.getcwd() + res_file, 'r') as f:
        lines = f.readlines()

    result = {}
    reading = False
    inside_block = False

    for line in lines:
        line = line.strip()
        
        # Start of the correct block
        if line.startswith(f'Result "{result_name}"'):
            inside_block = True
            continue

        # Inside correct block, look for Values
        if inside_block and line.startswith("Values"):
            reading = True
            continue

        # End reading values
        if reading and line.startswith("End Values"):
            break

        # Read values line
        if reading:
            parts = line.split()
            
            # Scalar?
            if len(parts) == 2:
                node_id = int(parts[0])
                value = float(parts[1])
                result[node_id] = value
                
            # Vector?
            elif len(parts) == 4:
                node_id = int(parts[0])
                vector = np.array(list(map(float, parts[1:])))
                result[node_id] = vector
                
            else: 
                print("Did not expect to reach this, please fix")
                
    if inside_block == False:
        print(f"Invalid result name '{result_name}', consult ERMES 20.0 documentation or your res file on output result names, setting results to 0 to allow for other plots")
    
    return result

def gaussian_fit(x, A, x0, w):
    """
    Gaussian fit function for 1/e width

    Args:
        x (float or array): position
        A (float or array): scale
        x0 (float or array): centre
        w (float or array): width

    Returns:
        gaussian: Generic gaussian curve for fitting
    """
    return A * np.exp( -((x - x0)**2) / (w**2) )

def fit_gaussian_width(offsets_per_tau, modE_profiles):
    """
    Perform a gaussian fit to obtain width and chi**2 of fit

    Args:
        offsets_per_tau (list): offset in the direction of beam width from the current point along the central ray
        modE_profiles (list): Transverse modE profiles from ERMES

    Returns:
        fitted_widths: Fitted widths in m
        fit_params: Fitting parameters
        chi2_list: chi**2 of fit
    """
    fitted_widths = []
    fit_params = []
    chi2_list = []

    for x, E in zip(offsets_per_tau, modE_profiles):
        try:
            p0 = [np.max(E), 0.0, (np.max(x) - np.min(x)) / 2]
            popt, _ = curve_fit(gaussian_fit, x, E, p0=p0)
            A_fit, x0_fit, w_fit = popt

            # Evaluate fitted curve
            E_fit = gaussian_fit(x, *popt)
            chi2 = np.sum((E_fit - E) ** 2 / E)

            fit_params.append([A_fit, x0_fit, w_fit])
            fitted_widths.append(np.abs(w_fit))
            chi2_list.append(chi2)
        except RuntimeError:
            fit_params.append([np.nan, np.nan, np.nan])
            fitted_widths.append(np.nan)
            chi2_list.append(np.nan)

    return fitted_widths, fit_params, chi2_list
    
def get_relative_error(observed_data, actual_data):
    """
    Returns the relative error between observed and actual

    Args:
        observed_data (array like): Observed/ Experimental Data
        actual_data (array like): Actual/ Theoretical Data
    """
    observed_data = np.asarray(observed_data)
    actual_data = np.asarray(actual_data)
    
    assert observed_data.shape == actual_data.shape, "Input arrays must have the same shape"
    
    err = np.abs(observed_data - actual_data)/np.abs(actual_data)
    
    return err

def get_moving_RMS(observed_data, window_size: int):
    """
    Get the moving RMS of a dataset, used for smaller angles as the data becomes oscillatory.

    Args:
        observed_data (array like): The observed data
        window_size (float): The size of the RMS window
    """
    
    observed_data = np.asarray(observed_data)
    n = len(observed_data)
    smoothed_data = np.empty(n)
    observed_data2 = np.concatenate(([0.0], np.cumsum(observed_data**2)))
    for i in range(n):
        j = min(n, i+window_size)
        cnt = j-i
        smoothed_data[i] = np.sqrt((observed_data2[j]-observed_data2[i])/cnt)
    #smoothed_data = np.sqrt(np.convolve(observed_data2, kernel, mode='same'))
    
    return smoothed_data

# TO BE DEPRECATED
def ERMES_results_to_plots(res: str = None, msh: str = None, dt: datatree = None, plot: bool = False, save: bool = True, grid_resolution: float = 8e-4, prefix: str = "prefix"):
    """
    Plot ERMES modE in R,Z with Scotty overlaid, Plot ERMES modE, rE vec, transverse modE along central ray with Scotty calculations where applicable.
    
    Args:
        res (str): path to the .res file
        msh (str): path to the .msh file
        dt (datatree): Scotty output file
        plot (bool): Plot it!
        save (bool): Save the error results
        grid_resolution (float): Resolution of grid used for meshing
        prefix (str): Prefix for error file names
    """
    # Node ID as XYZ coords
    node_to_xyz = ERMES_nodes_to_XYZ(msh_file=msh)
        
    # modE as nodeID
    modE = ERMES_results_to_node(res_file=res, result_name="mod(E)")   

    # rE as nodeID
    vecE = ERMES_results_to_node(res_file=res, result_name="rE")
    
    # Poynting vec as nodeID
    vecS = ERMES_results_to_node(res_file=res, result_name="Poynting_vector")
    
    # Convert to array
    max_node = max(modE.keys()) # all results will have the same number of nodes
    modE_array = np.zeros(max_node + 1)
    vecE_array = np.zeros((max_node + 1, 3))
    vecS_array = np.zeros((max_node + 1, 3))
    for i, val in modE.items():
        modE_array[i] = val
    for i, vec in vecE.items():
        vecE_array[i] = vec
    for i, vec in vecS.items():
        vecS_array[i] = vec
    
    
    # modE convert to xyz
    common_nodes = min(node_to_xyz.shape[0], modE_array.shape[0])
    modE_xyz = np.hstack((node_to_xyz[:common_nodes], modE_array[:common_nodes].reshape(-1, 1)))

    # vecE convert to xyz
    common_nodes = min(node_to_xyz.shape[0], vecE_array.shape[0])
    vecE_xyz = np.hstack((node_to_xyz[:common_nodes], vecE_array[:common_nodes]))
    
    # vecS convert to xyz
    common_nodes = min(node_to_xyz.shape[0], vecS_array.shape[0])
    vecS_xyz = np.hstack((node_to_xyz[:common_nodes], vecS_array[:common_nodes]))
    
    # Central beam
    tau_len = dt.inputs.len_tau.values
    tau_cutoff = dt.analysis.cutoff_index.values
    beam_RZ = np.column_stack([dt.analysis.q_R.values,dt.analysis.q_Z.values])
    beam_xyz = np.column_stack([dt.analysis.q_R.values,dt.analysis.q_Z.values, -dt.analysis.beam_cartesian.values[:, 1]]) # If going 3D

    # Extract vecE and modE along central ray within tolerance
    tol = grid_resolution/2 # Half of mesh size
    
    modE_list = []
    data_xy = modE_xyz[1:, :2]
    modE_vals = modE_xyz[1:, 3]
    
    # Extract vecE along central ray within tolerance
    vecE_list = []
    data_xy = vecE_xyz[1:, :2]
    data_xyz = vecE_xyz[1:, :3]
    vecE_vecs = vecE_xyz[1:, 3:6]
    
    # to get only modE along the central ray
    for xq, yq, zq in beam_xyz:
        dx = np.abs(data_xyz[:, 0] - xq)
        dy = np.abs(data_xyz[:, 1] - yq)
        #dz = np.abs(data_xyz[:, 2] - zq)
        mask = (dx <= tol) & (dy <= tol)# & (dz <= tol)

        if np.any(mask):
            modE_list.append(modE_vals[mask][0]) # First match
            vecE_list.append(vecE_vecs[mask][0]) # First match
        else:
            modE_list.append(0)
            vecE_list.append(np.array([0.0, 0.0, 0.0]))

    vecE_array_beam = np.array(vecE_list) # convert to array
    
    # Get beam width for gaussian x-section, This needs to be updated to work for 3D and 2D. And I think this can be tidied up a lot
    tau_vals = np.arange(len(beam_RZ)) # Take this from Scotty instead for consistency
    width_factor = 2 # n times of the width to visualize
    distance_along_beam = dt.analysis.distance_along_line.values
    width_at_tau = np.linalg.norm(
                                np.array([
                                    beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D).sel(col="R"), 
                                    beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D).sel(col="Z")]).T, 
                                axis=1, 
                                keepdims=False
                                )*width_factor
    no_of_points_width_at_tau = (np.rint(width_at_tau/tol))
    beam_width_range = [np.linspace(-width_at_tau[i], width_at_tau[i], int(no_of_points_width_at_tau[i])) for i in range(tau_len)]
    ghat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=dt.analysis.g_hat.values)

    beamfront_vector = np.cross(np.array([0.0, 0.0, 1.0]), ghat_xyz)
    beamfront_vector /= np.linalg.norm(beamfront_vector, axis=1, keepdims=True)

    sample_points_per_tau = []
    theoretical_transverse_modE_tau = []
    offsets_per_tau = []

    for i in range(tau_len):
        Psi = dt.analysis.Psi_3D_Cartesian.values[i]

        offset = np.linspace(-width_at_tau[i], width_at_tau[i], int(no_of_points_width_at_tau[i]))
        offsets_per_tau.append(offset)
        offset_vecs = offset[:, None] * beamfront_vector[i] # shape (num_samples, 3), this is w vec in cartesian x,y plane
        sample_points = beam_xyz[i] + offset_vecs # shape (num_samples, 3)

        quad_vals = np.einsum('ni,ij,nj->n', offset_vecs, Psi, offset_vecs)
        envelope = np.exp(-0.5*np.imag(quad_vals))
        
        theoretical_transverse_modE_tau.append(modE_list[i]*envelope)
        sample_points_per_tau.append(sample_points)
        
    xyz_grid = modE_xyz[:, :3]
    modE_vals = modE_xyz[:, 3]
    xyz_tree = cKDTree(xyz_grid)

    gaussian_modE_profiles = []
    for points in sample_points_per_tau:
        _, indices = xyz_tree.query(points)
        gaussian_modE_profiles.append(modE_vals[indices])
    
    # Fitted widths
    fitted_widths, fit_params, chi2_list = fit_gaussian_width(offsets_per_tau, gaussian_modE_profiles)
    
    # Get the poynting flux
    vecS_tree = cKDTree(vecS_xyz[:, :3])
    vecS_vals = vecS_xyz[:, 3:] # Sx, Sy, Sz
        
    poynting_flux_per_tau = []

    for i in range(tau_len):
        points = sample_points_per_tau[i] # shape (num_samples, 3)
        ghat = ghat_xyz[i] # shape (3,)

        # Find nearest vecS values
        _, indices = vecS_tree.query(points)
        S_vecs = vecS_vals[indices] # shape (num_samples, 3)

        # Compute S · g
        S_dot_g = np.dot(S_vecs, ghat) # shape (num_samples,)

        # Integrate using trapezoidal rule over transverse line
        dx = grid_resolution # approx element width
        total_flux = np.trapz(S_dot_g, dx=dx)
        
        poynting_flux_per_tau.append(total_flux)
    
    # Plot it!
    if plot:
        print("Plotting modE in R,Z from ERMES 20.0 and Scotty")
        # Plot modE over R Z
        # Filter points near desired z-slice, arrays start from 1 cus first node in 0,0,0
        mask = np.where(np.abs(modE_xyz[1:, 2]) < tol)
        filtered = modE_xyz[mask]

        # Get the filtered values (only the level set of z = 0.0)
        x = filtered[1:, 0]
        y = filtered[1:, 1]
        modE = filtered[1:, 3]

        # Create grid
        grid_res = int((np.max(data_xy[1:, 0]) - np.min(data_xy[1:, 0]))/(grid_resolution))
        xi = np.linspace(np.min(x), np.max(x), grid_res)
        yi = np.linspace(np.min(y), np.max(y), grid_res)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate modE onto grid
        Z = griddata((x, y), modE, (X, Y), method='linear', fill_value=np.nan) # nan so it is white

        # Plot
        plt.figure(figsize=(6, 5))
        c = plt.pcolormesh(X, Y, Z, shading='auto', cmap='YlOrRd')
        plt.xlabel("R (m)")
        plt.ylabel("Z (m)")
        plt.title(r"Results for ERMES 20.0 & Scotty, $\theta_{pol}$="
                  + f"{handle_scotty_launch_angle_sign(dt=dt):.1f}" 
                  + r"$^\circ$" 
                  + f", f={dt.inputs.launch_freq_GHz.values}GHz")
        plt.colorbar(c, label='|E| (A.U.)')
        
        # Plot flux surfaces to 'visualize' the plasma
        plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)
        
        # Plot Scotty results
        width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
        beam = dt.analysis.beam
        beam_plus = beam + width
        beam_minus = beam - width
        
        plt.plot(beam_plus.sel(col="R"), beam_plus.sel(col="Z"), "--k")
        plt.plot(beam_minus.sel(col="R"), beam_minus.sel(col="Z"), "--k", label="Beam width")
        plt.plot(beam.sel(col="R"), beam.sel(col="Z"), "-", c='black', label = "Central ray")
        
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(y), np.max(y))
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.show()
        
        print("Plotting modE vs tau from ERMES and Scotty")
        # Plot modE vs tau
        plt.scatter(distance_along_beam, modE_list, marker='.', color = 'red', label='ERMES 20.0')
        theoretical_modE_tau = calc_Eb_from_scotty(dt=dt, wx=dt.inputs.launch_beam_width.values, wy=dt.inputs.launch_beam_width.values, E0 = modE_list[0])
        plt.scatter(distance_along_beam, theoretical_modE_tau, marker='.', color = 'orange', label='Scotty')
        smoothed_modE_list = get_moving_RMS(modE_list, 40)
        #plt.plot(distance_along_beam, smoothed_modE_list, 'g-', label="Smoothed ERMES")
        plt.vlines(distance_along_beam[tau_cutoff], ymin=plt.gca().get_ylim()[0], ymax = plt.gca().get_ylim()[1], linestyles='--', color='blue')
        
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("|E| (A.U.)")
        plt.title(r"|E| vs Distance along central ray, $\theta_{pol}$="
                  + f"{handle_scotty_launch_angle_sign(dt=dt):.1f}"
                  + r"$^\circ$" 
                  + f", f={dt.inputs.launch_freq_GHz.values}GHz")
        plt.xlim(0, distance_along_beam[-1])
        plt.ylim(bottom = 0)
        plt.tight_layout()
        plt.legend()
        plt.show()
        
        # Relative err of modE
        err_modE = get_relative_error(smoothed_modE_list, theoretical_modE_tau)
        plt.scatter(distance_along_beam, err_modE, color = 'red', s = 15)
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("Relative error")
        plt.title("Relative error between ERMES 20.0 and Scotty |E| along central ray")
        plt.show()
        
        print("Plotting rE vector vs tau from ERMES 20.0 and Scotty")
        # Define the 3D transverse basis
        e_y = np.array([0.0, 1.0, 0.0])
        t2 = np.cross(ghat_xyz, e_y)
        t2 = t2/np.linalg.norm(t2, axis=1, keepdims=True)
        t1 = np.tile(e_y, (ghat_xyz.shape[0], 1))
        
        # Project vecE onto this transverse plane
        E_t1 = np.einsum('ij,ij->i', vecE_array_beam, t1) # projection onto e_y (Ez)
        E_t2 = np.einsum('ij,ij->i', vecE_array_beam, t2) # projection onto transverse axis

        # Reconstruct transverse vectors (for quiver plotting)
        E_transverse = np.stack([E_t2, E_t1], axis=1)

        # Set up figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        q = ax.quiver(0, 0, E_transverse[0, 0], E_transverse[0, 1], angles='xy', scale_units='xy', scale=1, color='blue')
        ax.set_xlim(-1.1*np.max(np.abs(E_transverse)), 1.1*np.max(np.abs(E_transverse)))
        ax.set_ylim(-1.1*np.max(np.abs(E_transverse)), 1.1*np.max(np.abs(E_transverse)))
        ax.set_xlabel(r"$t_2$ direction (in-plane)")
        ax.set_ylabel(r"$t_1$ direction (z-axis)")
        ax.set_title(f"Transverse Electric Field at {distance_along_beam[0]:.3f}m along central ray")
        ax.grid(True)
        ax.set_aspect('equal')

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, r'$\tau$ Index', 0, len(tau_vals)-1, valinit=0, valstep=1)

        def update(val):
            i = int(slider.val)
            q.set_UVC(E_transverse[i, 0], E_transverse[i, 1])
            ax.set_title(f"Transverse Electric Field vector at {distance_along_beam[i]:.3f}m along central ray")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        
        # Transverse gaussian beam front
        print("Plotting transverse gaussian beamfront")

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        scatter = ax.scatter(beam_width_range[0], gaussian_modE_profiles[0], color = 'red', s=15)  # first slice
        line, = ax.plot(beam_width_range[0], theoretical_transverse_modE_tau[0], '-', linewidth=5, color = 'orange')
        
        ax.set_xlabel("Distance from beam center (m)")
        ax.set_ylabel("mod(E)")
        ax.set_title(f"Transverse gaussian beamfront at {distance_along_beam[0]:.3f}m along central ray")
        ax.set_xlim(-np.max(width_at_tau)/width_factor, np.max(width_at_tau)/width_factor)
        ax.set_ylim(0, 1.1*np.max(modE_list))

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        tau_slider = Slider(ax_slider, r'$\tau$ index', 0, tau_len - 1, valinit=0, valstep=1)

        def update(val):
            i = int(tau_slider.val)
            scatter.set_offsets(np.column_stack([beam_width_range[i], gaussian_modE_profiles[i]]))
            scatter.set_array(gaussian_modE_profiles[i])
            line.set_data(beam_width_range[i], theoretical_transverse_modE_tau[i])
            ax.autoscale_view()
            ax.set_title(f"Transverse gaussian beamfront at {distance_along_beam[i]:.3f}m along central ray")
            fig.canvas.draw_idle()

        tau_slider.on_changed(update)
        plt.show()
        
        # Start figure with 2 subplots, sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1])

        # Top subplot: width
        ax1.plot(distance_along_beam, np.linalg.norm(width.values, axis = 1), label="Scotty width", color='orange')
        ax1.plot(distance_along_beam, np.array(fitted_widths), label="Fitted ERMES 20.0 width", color='red')
        smoothed_width_list = get_moving_RMS(fitted_widths, 40)
        #ax1.plot(distance_along_beam, smoothed_width_list, 'g-', label="Smoothed Fitted ERMES 20.0 width")
        ax1.vlines(distance_along_beam[tau_cutoff], ymin=ax1.get_ylim()[0], ymax = ax1.get_ylim()[1], linestyles='dashed', color='blue')
        ax1.set_ylabel("Width (m)")
        ax1.set_title("Beam widths")
        ax1.legend()

        # Bottom subplot: chi**2
        ax2.plot(distance_along_beam, chi2_list, label=r"$\chi^2$ of ERMES 20.0 fit", color='red')
        ax2.vlines(distance_along_beam[tau_cutoff], ymin=ax2.get_ylim()[0], ymax = ax2.get_ylim()[1], linestyles='dashed', color='blue')
        ax2.set_xlabel("Distance along central ray (m)")
        ax2.set_ylabel(r"$\chi^2$")
        ax2.legend()
        plt.show()
        
        # Relative err of beam_width
        err_width = get_relative_error(smoothed_width_list, np.linalg.norm(width.values, axis = 1))
        plt.scatter(distance_along_beam, err_width, color = 'red', s = 15)
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("Relative error")
        plt.title("Relative error between ERMES 20.0 and Scotty beam widths")
        plt.show()
        
        # modE and Width together
        # Start figure with 2 subplots, sharing x-axis
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4, 10), sharex=True, height_ratios=[1, 1])
        #plt.suptitle(r"Results for ERMES 20.0 & Scotty, $\theta_{pol}$="
        #          + fr"{handle_scotty_launch_angle_sign(dt=dt):.1f}$^\circ$" 
        #          + f", f={dt.inputs.launch_freq_GHz.values}GHz")
        # Plot modE vs tau
        ax0.scatter(distance_along_beam, modE_list, marker='.', color = 'red', label='ERMES 20.0')
        theoretical_modE_tau = calc_Eb_from_scotty(dt=dt, wx=dt.inputs.launch_beam_width.values, wy=dt.inputs.launch_beam_width.values, E0 = modE_list[0])
        ax0.scatter(distance_along_beam, theoretical_modE_tau, marker='.', color = 'orange', label='Scotty')
        smoothed_modE_list = get_moving_RMS(modE_list, 40)
        #plt.plot(distance_along_beam, smoothed_modE_list, 'g-', label="Smoothed ERMES")
        ax0.vlines(distance_along_beam[tau_cutoff], ymin=ax0.get_ylim()[0], ymax = ax0.get_ylim()[1], linestyles='dashed', color='blue')
        ax0.set_ylabel("|E| (A.U.)")
        ax0.set_title("|E| vs Distance along central ray")
        ax0.set_ylim(bottom=0)
        ax0.legend()

        # Bottom subplot: width
        ax1.plot(distance_along_beam, np.linalg.norm(width.values, axis = 1), label="Scotty width", color='orange')
        ax1.plot(distance_along_beam, np.array(fitted_widths), label="Fitted ERMES 20.0 width", color='red')
        smoothed_width_list = get_moving_RMS(fitted_widths, 40)
        #ax1.plot(distance_along_beam, smoothed_width_list, 'g-', label="Smoothed Fitted ERMES 20.0 width")
        ax1.set_ylim(bottom = 0)
        ax1.vlines(distance_along_beam[tau_cutoff], ymin=ax1.get_ylim()[0], ymax = ax1.get_ylim()[1], linestyles='dashed', color='blue')
        ax1.set_xlabel("Distance along central ray (m)")
        ax1.set_ylabel("Width (m)")
        ax1.set_title("Beam widths")
        #ax1.legend()
        plt.show()
        
        # Poynting flux
        plt.scatter(distance_along_beam, poynting_flux_per_tau, color = 'red', s = 15)
        print("P_X at exit: ", poynting_flux_per_tau[-1])
        plt.ylim(0, 1.1) # Since it should be 1
        plt.vlines(distance_along_beam[tau_cutoff], ymin=plt.gca().get_ylim()[0], ymax = plt.gca().get_ylim()[1], linestyles='--', color='blue')
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("Power flux (W/m)")
        plt.title("Poynting flux across beamfront along central ray")
        plt.show()
        
        # Save the errors
        if save:
            np.savez(os.getcwd() + f"\\{prefix}_{dt.inputs.launch_freq_GHz.values}_{handle_scotty_launch_angle_sign(dt)}_errors", err_modE, err_width)
   
def ERMES_results_to_plots_3D(
    res: str,
    msh: str,
    dt,
    grid_resolution: float = 4e-4,
    normal_vector: np.array = None,
    plot_blocks = None,
    save: bool = False,
    prefix: str = "ERMES_results"
):
    """
    Modular ERMES analysis & plotting (3D-native, 2D just means Z = 0)

    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        grid_resolution (float): Sampling resolution from meshing (m)
        normal_vector (array): Normal vector to 2D planes for plotting
        plot_blocks (list[str]): Choose which logical plot blocks to render. If None, all are used. Available: "field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors"
        save (bool): If True, each plot block saves a PNG with the given prefix
        prefix (str): Prefix for saved figures
            
    Returns:
        Plots
    """

    if plot_blocks is None:
        plot_blocks = ["field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors"]

    # Load & prepare all data
    (
        modE_xyz, vecE_xyz, vecS_xyz,
        beam_xyz,
        distance_along_beam, tau_len, tau_cutoff
    ) = prepare_core_fields(res, msh, dt)

    # Sampling tolerance (Cheap way of choosing the nearest point)
    tol = grid_resolution / 2.0

    # Extract |E| and E-vector along beam path
    print("Sampling E field")
    modE_list, vecE_array_beam, xyz_tree, modE_vals_all = sample_fields_along_beam(modE_xyz, vecE_xyz, beam_xyz)

    # Build transverse sampling slices & profiles using gaussian fit
    print("Building transverse E and widths")
    fitted_widths, fit_params, chi2_list, offsets_per_tau, modE_profiles, mod_E_theoretical_profiles, poynting_flux_per_tau = build_transverse_profiles_and_fits(
        dt,
        beam_xyz, 
        modE_xyz, 
        vecS_xyz,
        modE_list,
        [0,0,1], 
        grid_resolution=grid_resolution
    )

    # Plot
    if "field_map" in plot_blocks:
        plot_field_map(
            modE_xyz=modE_xyz,
            dt=dt,
            tol=tol,
            grid_resolution=grid_resolution,
            norm_vec = normal_vector,
            prefix=prefix,
            save=save
        )

    if "3D field_map" in plot_blocks:
        plot_field_with_beam(
            dt=dt,
            modE_xyz= modE_xyz,
            norm_vec = normal_vector,
            prefix=prefix,
            save=save,
            #sample_rate=0.10
        )

    if "modE_vs_tau" in plot_blocks:
        plot_modE_vs_tau(
            dt=dt,
            modE_list=modE_list,
            tau_cutoff=tau_cutoff,
            distance_along_beam=distance_along_beam,
            prefix=prefix,
            save=save
        )

    if "transverse_profile" in plot_blocks:
        plot_transverse_profiles(
            offsets_per_tau, 
            modE_profiles, 
            fit_params, 
            modE_theoretical_profiles=mod_E_theoretical_profiles,
            beam_widths=fitted_widths,
            prefix=prefix,
            save=save
        )

    if "widths" in plot_blocks:
        plot_widths(
            dt=dt,
            distance_along_beam=distance_along_beam,
            tau_cutoff=tau_cutoff,
            fitted_widths=fitted_widths,
            chi2_list=chi2_list,
            prefix=prefix,
            save=save
        )

    if "flux" in plot_blocks:
        plot_flux(
            distance_along_beam=distance_along_beam,
            poynting_flux_per_tau=poynting_flux_per_tau,
            tau_cutoff=tau_cutoff,
            prefix=prefix,
            save=save
        )

    if "errors" in plot_blocks:
        plot_errors(
            dt=dt,
            distance_along_beam=distance_along_beam,
            tau_cutoff=tau_cutoff,
            modE_list=modE_list,
            fitted_widths=fitted_widths,
            prefix=prefix,
            save=save
        )

# Data preparation for analysis 
def prepare_core_fields(res, msh, dt):
    """
    Load ERMES data in XYZ, map node fields, and build beam arrays
    
    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        
    Returns:
        modE_xyz
        vecE_xyz
        vecS_xyz
        beam_xyz
        ghat_xyz
        distance_along_beam
        tau_len
        tau_cutoff
    """
    # Mesh nodes to XYZ
    node_to_xyz = ERMES_nodes_to_XYZ(msh_file=msh)  # (Nnodes, 3)

    # Results to node dicts
    modE = ERMES_results_to_node(res_file=res, result_name="mod(E)")          # {node_id: scalar}
    vecE = ERMES_results_to_node(res_file=res, result_name="rE")              # {node_id: (Ex,Ey,Ez)}
    vecS = ERMES_results_to_node(res_file=res, result_name="Poynting_vector") # {node_id: (Sx,Sy,Sz)}

    # Dicts to arrays (max node id used as length)
    max_node = max(modE.keys())
    modE_array = np.zeros(max_node + 1)
    vecE_array = np.zeros((max_node + 1, 3))
    vecS_array = np.zeros((max_node + 1, 3))

    for i, val in modE.items():
        modE_array[i] = val
    for i, v in vecE.items():
        vecE_array[i] = v
    for i, v in vecS.items():
        vecS_array[i] = v

    # Truncate to common
    common_nodes = min(node_to_xyz.shape[0], modE_array.shape[0])
    modE_xyz = np.hstack((node_to_xyz[:common_nodes], modE_array[:common_nodes].reshape(-1, 1)))

    common_nodes = min(node_to_xyz.shape[0], vecE_array.shape[0])
    vecE_xyz = np.hstack((node_to_xyz[:common_nodes], vecE_array[:common_nodes]))

    common_nodes = min(node_to_xyz.shape[0], vecS_array.shape[0])
    vecS_xyz = np.hstack((node_to_xyz[:common_nodes], vecS_array[:common_nodes]))

    # Beam and beam param from Scotty
    tau_len = int(dt.inputs.len_tau.values)
    tau_cutoff = int(dt.analysis.cutoff_index.values)
    distance_along_beam = dt.analysis.distance_along_line.values  # (N,)

    # Central beam in ERMES CARTESIAN. Same as RtZ_to_XYZ(beam_cartesian)
    beam_xyz = np.column_stack([
        dt.analysis.q_X.values,
        dt.analysis.q_Z.values,
        -dt.analysis.q_Y.values
    ])
    
    return (modE_xyz, vecE_xyz, vecS_xyz,
            beam_xyz,
            distance_along_beam, tau_len, tau_cutoff)

def sample_fields_along_beam(modE_xyz, vecE_xyz, beam_xyz):
    """
    Nearest sampling of |E| and E-vector at beam points.
    
    Args:
        modE_xyz (array): modE as a function of X Y Z coordinates in ERMES
        vecE_xyz (array): vecE as a function of X Y Z coordinates in ERMES
        beam_xyz (array): beam coordinates in ERMES
        tol (float): Tolerance of sampling
        
    Returns:
        modE_list (array): modE along beam
        vecE_array_beam (array): vecE along beam
        tree (cKDTree): Datatree of all nodes as coordinates
        modE_vals_all (array): modE everywhere
    """
    # Build KDTree over all nodes
    xyz_all = vecE_xyz[:, :3]
    tree = cKDTree(xyz_all)

    # Field arrays
    modE_vals_all = modE_xyz[:, 3]
    vecE_vals_all = vecE_xyz[:, 3:6]

    # Query all beam points at once
    _, indices = tree.query(beam_xyz)

    # Sample values at nearest nodes
    modE_list = modE_vals_all[indices]
    vecE_array_beam = vecE_vals_all[indices]

    return np.array(modE_list), vecE_array_beam, tree, modE_vals_all

def build_transverse_profiles_and_fits(dt, beam_xyz, modE_xyz, vecS_xyz, modE_list, normal_vec, grid_resolution):
    """
    Build and fit transverse |E| profiles along a beam in a 2D plane,
    and compute the theoretical transverse field envelope from Scotty.
    Also calculates the poynting flux within +-w

    Args:
        dt (datatree): Scotty output file.
        beam_xyz (array): (N,3) beam center coordinates.
        modE_xyz (ndaarrayrray): (M,4) ERMES data [x,y,z,|E|].
        vecS_xyz (array): (M,6) ERMES [x,y,z,Sx,Sy,Sz] Poynting vectors.
        modE_list (array): List of modE values for normalization
        normal_vec (array-like): Plane normal vector in ERMES X Y Z.
        grid_resolution (float): ERMES grid spacing (m).

    Returns:
        fitted_widths (array): (N,) fitted 1/e field widths from ERMES.
        fit_params (array): (N,3) [A_fit, x0_fit, w_fit].
        chi2_list (array): (N,) chi-squared values of fits.
        offsets_per_tau (list): offsets along b-hat for each tau.
        modE_profiles (list): sampled |E| profiles for each tau.
        modE_theoretical_profiles (list): theoretical Scotty envelopes along beamfront.
        poynting_flux_per_tau (array): Integrated poynting flux
    """
    n_hat = np.array(normal_vec, float)
    n_hat /= np.linalg.norm(n_hat)
    modE_coords = modE_xyz[:, :3]
    modE_vals = modE_xyz[:, 3]
    tree = cKDTree(modE_coords)
    
    S_coords = vecS_xyz[:, :3]
    S_vals = vecS_xyz[:, 3:]
    tree_S = cKDTree(S_coords)

    poynting_flux_per_tau = []

    N = len(beam_xyz)
    fitted_widths, fit_params, chi2_list = [], [], []
    offsets_per_tau, modE_profiles, modE_theoretical_profiles = [], [], []
    poynting_flux_per_tau = []
    
    # From Scotty
    ghat_cartesian = dt.analysis.g_hat_Cartesian  # in RtZ
    Psi_3D_Cartesian = dt.analysis.Psi_3D_Cartesian  # in RtZ
    # Since Scotty cartesian is RtZ
    n_hat_RtZ = XYZ_to_RtZ(n_hat)

    beam_width_at_tau = np.linalg.norm(
        beam_width(ghat_cartesian, n_hat_RtZ, Psi_3D_Cartesian).values,
        axis=1
    )

    ghat_xyz = np.apply_along_axis(
        RtZ_to_XYZ, axis=1,
        arr=ghat_cartesian.values /
            np.expand_dims(dt.analysis.g_magnitude.values, axis=1)
    )

    modE_scotty = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0]
    )

    # Loop over tau
    for i in range(N):
        g = ghat_xyz[i]
        g /= np.linalg.norm(g)

        # Projection onto plane
        g_proj = g - np.dot(g, n_hat) * n_hat
        g_proj /= np.linalg.norm(g_proj)
        b_hat = np.cross(n_hat, g_proj)
        b_hat /= np.linalg.norm(b_hat)

        # Offset sampling along ±width
        width = beam_width_at_tau[i]
        n_points = int(max(np.rint(2 * width / grid_resolution), 5))
        offsets = np.linspace(-width, width, n_points)
        offsets_per_tau.append(offsets)

        sample_points = beam_xyz[i] + offsets[:, None] * b_hat[None, :]
        _, idx = tree.query(sample_points)
        profile = modE_vals[idx]
        modE_profiles.append(profile)

        # Theoretical profile from Scotty
        Psi = Psi_3D_Cartesian.values[i]
        P_perp = np.eye(3) - np.outer(g, g)
        Psi_w = P_perp @ Psi @ P_perp # Project to plane perp to g
        w_vecs = np.apply_along_axis(XYZ_to_RtZ, axis = 1, arr = np.outer(offsets, b_hat)) # Since Psi_w is in RtZ
        quad = np.einsum('ni,ij,nj->n', w_vecs, np.imag(Psi_w), w_vecs)
        envelope = np.exp(-0.5 * quad)
        E_theory = modE_scotty[i] * envelope
        modE_theoretical_profiles.append(E_theory)

        # Gaussian Fit (ERMES data)
        try:
            p0 = [np.max(profile), 0.0, width]
            popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
            A_fit, x0_fit, w_fit = popt
            E_fit = gaussian_fit(offsets, *popt)
            chi2 = np.sum((profile - E_fit)**2 / (E_fit + 1e-12))
            fit_params.append([A_fit, x0_fit, w_fit])
            fitted_widths.append(abs(w_fit))
            chi2_list.append(chi2)
        except RuntimeError:
            fit_params.append([np.nan, np.nan, np.nan])
            fitted_widths.append(np.nan)
            chi2_list.append(np.nan)
            
        _, S_idx = tree_S.query(sample_points)
        S_vecs = S_vals[S_idx]
        S_dot_g = np.dot(S_vecs, g)
        total_flux = np.trapz(S_dot_g, x=offsets)
        poynting_flux_per_tau.append(total_flux)

    return (np.array(fitted_widths),
            np.array(fit_params),
            np.array(chi2_list),
            offsets_per_tau,
            modE_profiles,
            modE_theoretical_profiles,
            np.array(poynting_flux_per_tau))

# Plot functions
def plot_field_map(modE_xyz, dt, tol, grid_resolution, norm_vec, prefix, save):
    """
    2D |E| map on z=0 (poloidal cross-section) with transparent-under colormap.
    Uses nearest sampling onto a regular R-Z grid.
    """
    # Select near z=0 slice
    mask = np.abs(modE_xyz[:, 2]) < tol
    slice_pts = modE_xyz[mask]
    if slice_pts.shape[0] < 10:
        print("[field_map] Not enough points near z=0 to plot.")
        return

    R = slice_pts[1:, 0]
    Z = slice_pts[1:, 1]
    E = slice_pts[1:, 3]

    # Build uniform grid in R-Z
    Rmin, Rmax = np.min(R), np.max(R)
    Zmin, Zmax = np.min(Z), np.max(Z)
    nR = max(16, int((Rmax - Rmin) / grid_resolution))
    nZ = max(16, int((Zmax - Zmin) / grid_resolution))
    Ri = np.linspace(Rmin, Rmax, nR)
    Zi = np.linspace(Zmin, Zmax, nZ)
    RR, ZZ = np.meshgrid(Ri, Zi)

    # KDTree nearest sampling
    tree = cKDTree(np.column_stack([R, Z]))
    dist, idx = tree.query(np.column_stack([RR.ravel(), ZZ.ravel()]))
    
    Ei = np.full(RR.size, np.nan)
    valid = dist < grid_resolution * 1.5     # keep points only near mesh nodes (for edge, so padding of 1.5*res)
    Ei[valid] = E[idx[valid]]
    Ei = Ei.reshape(RR.shape)

    # Colormap with white as 0
    cmap = cm.get_cmap('bwr').copy() # Blue-White-Red
    cmap.set_bad(color=(1, 1, 1, 0)) # transparent for masked NaN regions

    # Dynamic normalization centered at zero
    finite_vals = Ei[np.isfinite(Ei)]
    vmax = np.nanpercentile(np.abs(finite_vals), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    plt.figure(figsize=(6, 5))
    pc = plt.pcolormesh(RR, ZZ, Ei, shading='auto', cmap=cmap, norm=norm)
    plt.colorbar(pc, label='|E| (A.U.)')

    # Poloidal flux surfaces
    plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)

    # Scotty beam center and width in RZ (projection)
    width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
    beam = dt.analysis.beam_cartesian 
    beam_plus = beam + width
    beam_minus = beam - width
    plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), "--k", lw=1)
    plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), "--k", lw=1, label="Beam width")
    plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), "-", c='black', lw=1.5, label="Central ray")

    plt.xlabel("R (m)")
    plt.ylabel("Z (m)")
    plt.title(r"|E|, Poloidal cross-section, $\theta_{pol}$="
              + f"{handle_scotty_launch_angle_sign(dt=dt):.1f}°, "
              + r"$\theta_{tor}$="
              + f"{dt.inputs.toroidal_launch_angle_Torbeam.values}°"
              + f",  f={dt.inputs.launch_freq_GHz.values} GHz")
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)    
    plt.gca().set_aspect('equal')
    plt.legend()
    

    if save:
        plt.tight_layout()
        plt.savefig(f"{prefix}_field_map.png", dpi=200)
    plt.show()

def plot_field_with_beam(dt, modE_xyz, norm_vec = None, save=False, prefix="", sample_rate = 0.25):
    """
    Plot the 3D ERMES |E| field (voxel-style scatter) with beam geometry overlaid.

    Args:
        dt (datatree): Scotty output file
        modE_xyz (ndarray): (N,4) array [x, y, z, |E|]
        norm_vec (array): Normal vector for 2D plot ERMES
        save (bool): Whether to save to file
        prefix (str): Output prefix (if save=True)
        sample_rate (float): Percentage of points to plot

    Returns:
        fig, ax: Matplotlib figure and 3D axis objects.
    """
    # Get the field
    x, y, z, e = modE_xyz.T
    e_norm = e / np.nanmax(e)
    mask = e_norm > 0.01
    
    idx = np.where(mask)[0]
    n_total = len(idx)
    
    n_keep = int(sample_rate*n_total)
    if n_keep < n_total:
        idx = np.random.choice(idx, n_keep, replace=False)
        
    finite_vals = e[np.isfinite(e)]
    vmax = np.nanpercentile(np.abs(finite_vals), 99)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap = cm.get_cmap('bwr').copy()
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent for masked / NaN region

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Field visualization (voxel-style scatter) for 3D
    sc = ax.scatter(x[idx], y[idx], z[idx],
                    c=e[idx], cmap=cmap, norm = norm, alpha=0.2, s=0.5,
                    label='ERMES |E| Field')
    
    # Overlay beam geometry
    width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
    beam = dt.analysis.beam_cartesian 
    beam_plus = beam + width
    beam_minus = beam - width
    plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), beam_plus.sel(col_cart="Y"),"--k", lw=2)
    plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), beam_minus.sel(col_cart="Y"), "--k", lw=2, label="Beam width")
    plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), beam.sel(col_cart="Y"),  "-", c='black', lw=4, label="Central ray")


    # Labels and formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    zmin, zmax = ax.get_zlim()
    ax.set_zticks([zmin, zmax])
    ax.set_title("ERMES |E| Field with Beam Geometry")
    ax.legend()
    ax.set_aspect('equal')
    ax.view_init(elev=33, azim=45, roll=123)
    fig.colorbar(sc, ax=ax, shrink=0.6, label='|E| (A.U.)')

    # Optional save/show
    if save:
        plt.tight_layout()
        plt.savefig(f"{prefix}_3D_field_map.png", dpi=200)
    plt.show()

def plot_modE_vs_tau(dt, modE_list, tau_cutoff, distance_along_beam, prefix, save):
    """
    |E| (ERMES vs Scotty) along beam distance.
    """
    theoretical_modE_tau = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0]
    )
    #smoothed_modE_list = get_moving_RMS(modE_list, 40)

    plt.figure(figsize=(7, 4))
    plt.scatter(distance_along_beam, modE_list, s=12, color='red', label='ERMES')
    plt.plot(distance_along_beam, theoretical_modE_tau, '--', color='orange', label='Scotty')
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("|E| (A.U.)")
    plt.title("|E| along central ray")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_modE_vs_tau.png", dpi=200)
    plt.show()

def plot_transverse_profiles(
    offsets_per_tau,
    modE_profiles,
    fit_params,
    modE_theoretical_profiles=None,
    beam_widths=None,
    save: bool = False,
    prefix: str = ""
):
    """
    Interactive plot of ERMES |E| transverse profiles, Gaussian fits,
    and Scotty theoretical |E| envelopes.

    Args:
        offsets_per_tau (list): list of 1D arrays of offsets along transverse axis.
        modE_profiles (list): list of 1D arrays of sampled |E| values from ERMES.
        fit_params (ndarray): (N,3) fitted Gaussian parameters [A, x0, w].
        modE_theoretical_profiles (list or None): list of 1D arrays of theoretical |E| profiles from Scotty.
        beam_widths (ndarray or None): optional fitted widths for marking ±w.
        save (bool): Whether to save figure.
        prefix (str): Prefix for save name.
    """
    N = len(offsets_per_tau)
    if N == 0:
        print("No profiles to plot.")
        return

    # Scaling for axes
    global_maxE = np.nanmax([np.nanmax(p) for p in modE_profiles])
    if modE_theoretical_profiles is not None:
        global_maxE = max(global_maxE, np.nanmax([np.nanmax(p) for p in modE_theoretical_profiles]))
    global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets_per_tau])

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)

    i0 = 0
    x0 = offsets_per_tau[i0]
    y0 = modE_profiles[i0]

    scatter = ax.scatter(x0, y0, color='red', s=15, label='ERMES samples')
    E_fit0 = gaussian_fit(x0, *fit_params[i0])
    line_fit, = ax.plot(x0, E_fit0, color='green', lw=3, label='Gaussian fit')

    # Add Scotty theoretical curve (if available)
    if modE_theoretical_profiles is not None:
        y_theory0 = modE_theoretical_profiles[i0]
        line_theory, = ax.plot(x0, y_theory0, '--', color='orange', label='Scotty')
    else:
        line_theory = None

    # Labels and limits
    ax.set_xlabel('Offset from beam center (m)')
    ax.set_ylabel('|E| (A.U.)')
    ax.set_title(r'Transverse profile at $\tau$ = ' + f'{i0}')
    ax.legend()
    ax.set_xlim(-global_max_offset, global_max_offset)
    ax.set_ylim(0, 1.1 * global_maxE)

    # Slider setup
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'τ index', 0, N - 1, valinit=i0, valstep=1)

    def update(val):
        j = int(slider.val)
        x = offsets_per_tau[j]
        y = modE_profiles[j]

        # Update ERMES data
        scatter.set_offsets(np.column_stack([x, y]))

        # Update Gaussian fit
        y_fit = gaussian_fit(x, *fit_params[j])
        line_fit.set_data(x, y_fit)

        # Update Scotty theoretical curve
        if line_theory is not None:
            y_theory = modE_theoretical_profiles[j]
            line_theory.set_data(x, y_theory)

        ax.set_title(r'Transverse profile at $\tau$ = ' + f'{j}')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save:
        plt.savefig(f"{prefix}_transverse_profile.png", dpi=200)

    plt.show()

def plot_widths(dt, distance_along_beam, tau_cutoff, fitted_widths, chi2_list, prefix, save):
    """
    Fitted vs Scotty widths + chi^2 vs distance (stacked).
    """
    width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    width_norm = np.linalg.norm(width.values, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1])

    ax1.plot(distance_along_beam, width_norm, '--', label="Scotty width", color='orange')
    ax1.plot(distance_along_beam, np.array(fitted_widths), label="Fitted ERMES width", color='red')
    ax1.vlines(distance_along_beam[tau_cutoff], *ax1.get_ylim(), linestyles='--', color='blue')
    ax1.set_ylabel("Width (m)")
    ax1.set_title("Beam widths")
    ax1.legend()

    ax2.plot(distance_along_beam, chi2_list, label=r"$\chi^2$ of ERMES fit", color='red')
    ax2.vlines(distance_along_beam[tau_cutoff], *ax2.get_ylim(), linestyles='--', color='blue')
    ax2.set_xlabel("Distance along central ray (m)")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_widths_and_chi2.png", dpi=200)
    plt.show()

def plot_flux(distance_along_beam, poynting_flux_per_tau, tau_cutoff, prefix, save):
    """
    Poynting flux S dot g integrated across beamfront vs distance.
    """
    plt.figure(figsize=(7, 4))
    plt.scatter(distance_along_beam, poynting_flux_per_tau, color='red', s=15)
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.ylim(0, max(1.1, 1.1*np.max(poynting_flux_per_tau)))
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("Power flux (arb. units)")
    plt.title("Poynting flux across beamfront")
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_poynting_flux.png", dpi=200)
    plt.show()

def plot_errors(dt, distance_along_beam, tau_cutoff, modE_list, fitted_widths, prefix, save):
    """
    Relative errors for |E| and width vs Scotty
    """
    # |E| error
    theoretical_modE_tau = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0]
    )
    smoothed_modE_list = get_moving_RMS(modE_list, 40)
    err_modE = get_relative_error(smoothed_modE_list, theoretical_modE_tau)

    plt.figure(figsize=(7, 3.6))
    plt.scatter(distance_along_beam, err_modE, color='red', s=14)
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("Rel. error (|E|)")
    plt.title("Relative error: ERMES vs Scotty |E|")
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_error_modE.png", dpi=200)
    plt.show()

    # Width error
    width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    width_norm = np.linalg.norm(width.values, axis=1)
    smoothed_width_list = get_moving_RMS(fitted_widths, 40)
    err_width = get_relative_error(smoothed_width_list, width_norm)

    plt.figure(figsize=(7, 3.6))
    plt.scatter(distance_along_beam, err_width, color='red', s=14)
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("Rel. error (width)")
    plt.title("Relative error: ERMES vs Scotty beam width")
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_error_width.png", dpi=200)
    plt.show()

    
if __name__ == '__main__':
    # MAST-U
    """
    get_ERMES_parameters(
        dt=load_scotty_data('\\MAST-U\\scotty_output_freq40.0_pol-13.0_rev.h5'),
        prefix="MAST-U_new_",
        launch_angle=13.0, 
        launch_freq_GHz=40, 
        port_width=0.01, 
        #launch_positon=[2.278,0,-0.01], 0
        #launch_beam_curvature=-0.7497156475519201, 
        #launch_beam_width=0.07596928872724663, 
        dist_to_ERMES_port=0.9, 
        plot=True,
        save=True,
        )
    #"""
    
    #DIII-D
    """
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\scotty_output_tor_ideal__freq72.5_pol-7.0_tor1.5726_rev.h5'),
        prefix="DIII-D_",
        #launch_positon=[3.01346,0,-0.09017],
        dist_to_ERMES_port=0.66,
        plot=True,
        save=True,
        cartesian_scotty=False
    )
    
    #"""
    
    #2D Linear Layer
    """
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\scotty_output_2D_linear_freq30_45deg.h5'),
        prefix="2D_linear_new_",
        launch_angle=45.0, 
        launch_freq_GHz=30.0, 
        #port_width=0.01, 
        launch_position=[0,0,0],
        #launch_beam_curvature=-0.95, 
        #launch_beam_width=0.125, 
        dist_to_ERMES_port=0, 
        plot=True,
        save=True,
        cartesian_scotty=True
    )
    
    #"""
    
    #get_pol_from_smits([cos(-30*pi/180), sin(-30*pi/180), 0], [0,0,1], [0,-1,0], 15, 109.69092568169533, mode = 1)
    
    # Text ERMES output to plots
    """
    ERMES_results_to_plots(
        res="\\Final_Ermes_output\\MAST-U_7_degree.post.res", 
        msh="\\Final_Ermes_output\\MAST-U_7_degree.post.msh",
        dt=load_scotty_data('\\MAST-U\\scotty_output_freq40.0_pol-7.0_rev.h5'),
    
        #res="\\Final_Ermes_output\\DIII-D_7_degree_oblique.post.res", 
        #msh="\\Final_Ermes_output\\DIII-D_7_degree_oblique.post.msh",
        #dt=load_scotty_data('\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5'),
        plot=True,
        grid_resolution=4e-4,
        save=False,
        prefix="MAST-U"
    )
    
    #"""
    
    # Try out the 3D one with 2D known results
    ERMES_results_to_plots_3D(
        res="\\Final_Ermes_output\\DIII-D_7_degree_oblique.post.res", 
        msh="\\Final_Ermes_output\\DIII-D_7_degree_oblique.post.msh",
        dt=load_scotty_data('\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5'),
        grid_resolution=4e-4,
        normal_vector=[-0.0191506, 0.13483198, 0.9906834],
        plot_blocks=["field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors"],
        save=False
    )
    
    """
    # Plotting stuffs to get summary of errors
    launch_angles = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    #toroidal_diff = np.array([np.max(load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values) for launch_angle in launch_angles]) - np.array([np.min(load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values) for launch_angle in launch_angles])
    q_Y_arrays = np.row_stack([load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values for launch_angle in launch_angles])
    
    #print(q_Y_arrays)
    #print(toroidal_diff)
    err_15 = np.load(os.getcwd()+"\\final_72.5_15_errors.npz")
    err_13 = np.load(os.getcwd()+"\\final_72.5_13_errors.npz")
    err_11 = np.load(os.getcwd()+"\\final_72.5_11_errors.npz")
    err_9 = np.load(os.getcwd()+"\\final_72.5_8.995_errors.npz")
    err_7 = np.load(os.getcwd()+"\\final_72.5_7_errors.npz")
    err_5 = np.load(os.getcwd()+"\\final_72.5_5_errors.npz")
    err_3 = np.load(os.getcwd()+"\\final_72.5_3_errors.npz")
    
    err_3_beam, err_3_width = err_3['arr_0'], err_3['arr_1']
    err_5_beam, err_5_width = err_5['arr_0'], err_5['arr_1']
    err_7_beam, err_7_width = err_7['arr_0'], err_7['arr_1']
    err_9_beam, err_9_width = err_9['arr_0'], err_9['arr_1']
    err_11_beam, err_11_width = err_11['arr_0'], err_11['arr_1']
    err_13_beam, err_13_width = err_13['arr_0'], err_13['arr_1']
    err_15_beam, err_15_width = err_15['arr_0'], err_15['arr_1']
    
    cutoff_index_3 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-3.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_5 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-5.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_7 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_9 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-9.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_11 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-11.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_13 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-13.0_rev.h5").analysis.cutoff_index.values
    cutoff_index_15 = load_scotty_data("\\Output\\scotty_output_freq72.5_pol-15.0_rev.h5").analysis.cutoff_index.values
    tau_vals = np.arange(0, 1002, 1, dtype=int)
    
    tau_3 = [tau_vals[1], tau_vals[cutoff_index_3], tau_vals[-1]]
    tau_5 = [tau_vals[1], tau_vals[cutoff_index_5], tau_vals[-1]]
    tau_7 = [tau_vals[1], tau_vals[cutoff_index_7], tau_vals[-1]]
    tau_9 = [tau_vals[1], tau_vals[cutoff_index_9], tau_vals[-1]]
    tau_11 = [tau_vals[1], tau_vals[cutoff_index_11], tau_vals[-1]]
    tau_13= [tau_vals[1], tau_vals[cutoff_index_13], tau_vals[-1]]
    tau_15 = [tau_vals[1], tau_vals[cutoff_index_15], tau_vals[-1]]
    
    err_beam_3 = err_3_beam[tau_3]
    err_beam_5 = err_5_beam[tau_5]
    err_beam_7 = err_7_beam[tau_7]
    err_beam_9 = err_9_beam[tau_9]
    err_beam_11 = err_11_beam[tau_11]
    err_beam_13 = err_13_beam[tau_13]
    err_beam_15 = err_15_beam[tau_15]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4, 8), sharex=True, height_ratios=[1,1])
    ax0.plot([0, 1, 2], err_beam_3, color='orange', label=r"$3^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_5, color='purple', label=r"$5^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_7, color='red', label=r"$7^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_9, color='gray', label=r"$9^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_11, color='brown', label=r"$11^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_13, color='pink', label=r"$13^\circ$", marker = 'x')
    ax0.plot([0, 1, 2], err_beam_15, color='blue', label=r"$15^\circ$", marker = 'x')
    #plt.xticks([0, 1, 2], ['Entry', 'Cutoff', 'Exit'])
    ax0.set_ylabel("Relative error")
    #plt.xlabel(r"Beam parameter $\tau$")
    ax0.set_title(r"Relative error of modE along central ray vs $\tau$")
    ax0.set_ylim(bottom = 0)
    ax0.legend()
    
    
    err_width_3 = err_3_width[tau_3]
    err_width_5 = err_5_width[tau_5]
    err_width_7 = err_7_width[tau_7]
    err_width_9 = err_9_width[tau_9]
    err_width_11 = err_11_width[tau_11]
    err_width_13 = err_13_width[tau_13]
    err_width_15 = err_15_width[tau_15]
    ax1.plot([0, 1, 2], err_width_3, color='orange', label=r"$3^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_5, color='purple', label=r"$5^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_7, color='red', label=r"$7^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_9, color='gray', label=r"$9^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_11, color='brown', label=r"$11^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_13, color='pink', label=r"$13^\circ$", marker = 'x')
    ax1.plot([0, 1, 2], err_width_15, color='blue', label=r"$15^\circ$", marker = 'x')
    ax1.set_xticks([0, 1, 2], ['Entry', 'Cutoff', 'Exit'])
    ax1.set_xlabel(r"Beam parameter $\tau$")
    ax1.set_title(r"Relative error of beam width vs $\tau$")
    ax1.set_ylabel("Relative error")
    ax1.set_ylim(bottom = 0, top = 1)
    ax1.legend()
    #plt.suptitle("Summarized Errors")
    plt.show()
    
    #"""
    
    """ Plot toroidal component of each beam (zoomed in)
    dt = load_scotty_data('\\Output\\scotty_output_2D_linear_freq15.0_pol-30.0_rev.h5')
    
    # Plot flux surfaces to 'visualize' the plasma
    ax = maybe_make_axis(plt.gca())

    lcfs = 1
    #flux_spline = CubicSpline(
    #    dt.analysis.R_midplane, dt.analysis.poloidal_flux_on_midplane - lcfs
    #)
    #all_R_lcfs = flux_spline.roots()

    zeta = np.linspace(-np.pi, np.pi, 1001)
    #for R_lcfs in all_R_lcfs:
    #    plot_toroidal_contour(ax, R_lcfs, zeta)

    #flux_min_index = dt.analysis.poloidal_flux_on_midplane.argmin()
    #R_axis = dt.analysis.R_midplane[flux_min_index].data
    #plot_toroidal_contour(ax, R_axis, zeta, "#00003f")
        
    # Plot Scotty results
    width_tor = beam_width(
        dt.analysis.g_hat_cartesian,
        np.array([0.0, 0.0, 1.0]),
        dt.analysis.Psi_3D_labframe_cartesian,
    )
    
    beam_plus_tor = dt.analysis.beam_cartesian + width_tor
    beam_minus_tor = dt.analysis.beam_cartesian - width_tor
    
    ax.plot(beam_plus_tor.sel(col_cart="X"), beam_plus_tor.sel(col_cart="Y"), "--k")
    ax.plot(beam_minus_tor.sel(col_cart="X"), beam_minus_tor.sel(col_cart="Y"), "--k", label="Beam width")
    ax.plot(
        np.concatenate([dt.analysis.q_X]),
        np.concatenate([dt.analysis.q_Y]),
        "k",
        label="Central (reference) ray",
    )
    
    # Get lims from beam width
    combined_beam_X = np.concatenate([beam_plus_tor.sel(col_cart="X"), beam_minus_tor.sel(col_cart="X")])
    combined_beam_Y = np.concatenate([beam_plus_tor.sel(col_cart="Y"), beam_minus_tor.sel(col_cart="Y")])
    
    plt.xlim(np.min(combined_beam_X)-0.01, np.max(combined_beam_X)+0.01)
    plt.ylim(np.min(combined_beam_Y)-0.01, np.max(combined_beam_Y)+0.01)
    plt.gca().set_aspect('equal')
    plt.show()
    
    # Poloidal
    ax.clear()
    plt.close()
    ax = maybe_make_axis(plt.gca())
    
    width_pol = beam_width(
        dt.analysis.g_hat,
        np.array([0.0, 1.0, 0.0]),
        dt.analysis.Psi_3D,
    )
    
    beam_plus_pol = dt.analysis.beam + width_pol
    beam_minus_pol = dt.analysis.beam - width_pol
    ax.plot(beam_plus_pol.sel(col="R"), beam_plus_pol.sel(col="Z"), "--k")
    ax.plot(beam_minus_pol.sel(col="R"), beam_minus_pol.sel(col="Z"), "--k", label="Beam width")
    ax.plot(
        np.concatenate([dt.analysis.q_R]),
        np.concatenate([dt.analysis.q_Z]),
        "k",
        label="Central (reference) ray",
    )
    
    # Get lims from beam width
    combined_beam_R = np.concatenate([beam_plus_pol.sel(col="R"), beam_minus_pol.sel(col="R")])
    combined_beam_Z = np.concatenate([beam_plus_pol.sel(col="Z"), beam_minus_pol.sel(col="Z")])
    
    plt.xlim(np.min(combined_beam_R)-0.01, np.max(combined_beam_R)+0.01)
    plt.ylim(np.min(combined_beam_Z)-0.01, np.max(combined_beam_Z)+0.01)
    plt.gca().set_aspect('equal')
    plt.show()
    #"""