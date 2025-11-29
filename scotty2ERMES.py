"""
Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 

The shape of the domain here is specifically for DBS simulations.

To use this, from Scotty results, decide on the distance_to_ERMES_port variable (distance from launcher to launch boundary in ERMES). The rest is automatic

Spits out .txt of all the necessary data. 

All paths are w.r.t CWD

TODO
1. SEPARATE THIS INTO MULTIPLE logical SCRIPTS
2. Make a figure that describes everything
3. Documentation
4. Transition saving to also include a .h5 file for ease of use
5. Tidy up plotting functions (a lot of repeated variables)

References
    [1] Two dimensional full-wave simulations of Doppler back-scattering in tokamak plasmas with COMSOL by Quinn Pratt et al (in-progress paper)
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin, pg 43-44
    [4] V. H. Hall-Chen, F. I. Parra, and J. C. Hillesheim, “Beam model of Doppler backscattering,” Plasma Phys. Control. Fusion, vol. 64, no. 9, p. 095002, Sep. 2022, doi: 10.1088/1361-6587/ac57a1.
    [5] F. M. A. Smits, “ELLIPTICAL POLARISATION FOR OBLIQUE EC-WAVE LAUNCH”.

NOTE on Coordinate systems
SCOTTY_cartesian refers to X Y Z where X, Z align with R, Z, Y is toroidal into the page. We will call this RtZ
ERMES_cartesian refers to X Y Z where X, Y align to R, Z. Z is (instantaneously) toroidal out of the page 
CYL referes to R zeta Z where zeta is toroidal angle
BEAM referes to xhat yhat ghat

NOTE
It is perfectly possible to NOT need any of these conversions between the two cartesians. 
I simply unfortunately started with this and stuck with it

Written by Dylan James Mc Kaige
Created: 16/5/2025
Updated: 29/11/2025
"""
import os, datatree
import numpy as np
from tqdm import tqdm
from math import sin, cos, tan, sqrt, fabs
from scipy.constants import c, pi
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import colormaps as cm
from scotty.analysis import beam_width
from scotty.plotting import plot_poloidal_crosssection
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
 
# TO DEPRECATE   
def get_pol_from_scotty(dt: datatree.DataTree, index: int):
    """
    Get the polarization vector from a Scotty output file. This function assumes
    1) Tau starts from point of entry (may not be exactly equal to the one calculated by find_lcfs_entry_point)
    2) y_hat_Cartesian, b_hat_Cartesian, g_hat_Cartesian are in the (R,t,Z) basis, e_hat is in the (u1,u2,bhat) basis
    3) Equations 104-106 from [4] apply exactly to let us derive u1_hat and u2_hat
    
    Produces an elliptical polarization vector.

    Args:
        dt (DataTree): Scotty output file
        index (int): Tau index (0 = entry)
        
    Returns:
        e_hat_XYZ (array): Polarization vector in ERMES (X,Y,Z) basis
    """
 
    y_hat_RtZ = dt.analysis.y_hat_Cartesian.values
    g_hat_RtZ = dt.analysis.g_hat_Cartesian.values
    b_hat_RtZ = find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values)/np.linalg.norm(find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values))
    e_hat = dt.analysis.e_hat.values # This is in u1, u2, bhat basis
    u2_hat_RtZ = y_hat_RtZ/np.linalg.norm(y_hat_RtZ)
    u1_hat_RtZ = (np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))/np.linalg.norm(np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))

    # Form the basis transition vector from u1,u2,b to R,t,Z
    uub_to_RtZ_basis = np.column_stack((u1_hat_RtZ[index],u2_hat_RtZ[index],b_hat_RtZ[index]))

    # Pol vector at tau in R,t,Z
    e_hat_RtZ = np.dot(uub_to_RtZ_basis, e_hat[index])
    e_hat_RtZ = e_hat_RtZ/np.linalg.norm(e_hat_RtZ)
    e_hat_XYZ = RtZ_to_XYZ(e_hat_RtZ) # Project to ERMES cartesian
    
    return e_hat_XYZ

# TO DEPRECATE
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
    Now there's VERY good agreement at the 0 mistmatch angle case as well 
    Seems to lose agreement as poloidal launch angle increases (why?, increase seems to be monotonic)
    
    Force projection to maintain k perp to rho relation
    tried with k at launch, k at entry, ghat at entry. Similar results for the 3. 
    None fully eliminate O mode at higher angles

    Args:
        pol (np.array): Complex polarization vector from Scotty in ERMES cartesian vasis
        E0 (float, optional): Magnitude of incident E field. Defaults to 1.0.
        
    Return:
        rho (array): Linear polarization vector in ERMES Cartesian
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
    #print("k dot rho:", k_dot_rho)

    return rho, E_perp, phi_perp, E_par, phi_par

def scotty_pol_to_ERMES_final(dt: datatree, E0: float, cartesian_scotty = False):
    """
    Convert a complex polarisation vector from Scotty to a real poalrisation vector in ERMES.
    
    Project e_hat to beam basis (xyg), the purely real component is rho, the purely imaginary component is eta (xhat is rho, yhat is eta)
    In this way, phi_par = 0, phi_perp = -pi/2
    """
    x_hat_RtZ = dt.analysis.x_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.x_hat_cartesian.values
    y_hat_RtZ = dt.analysis.y_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.y_hat_cartesian.values
    g_hat_RtZ = dt.analysis.g_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.g_hat_cartesian.values

    e_hat_uub = dt.analysis.e_hat.values # This is in u1, u2, bhat basis

    if cartesian_scotty == False:
        b_hat_RtZ = find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values)/np.linalg.norm(find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values))
    else:
        b_hat_RtZ = dt.analysis.b_hat.values
    u2_hat_RtZ = y_hat_RtZ/np.linalg.norm(y_hat_RtZ)
    u1_hat_RtZ = (np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))/np.linalg.norm(np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))

    # Form the basis transition vector from u1,u2,b to R,t,Z
    uub_to_RtZ_basis = np.column_stack((u1_hat_RtZ[0], u2_hat_RtZ[0], b_hat_RtZ[0])) # at entry
    RtZ_to_xyg_basis = np.column_stack((x_hat_RtZ[0], y_hat_RtZ[0], g_hat_RtZ[0])) # at entry

    # Pol vector at tau in R,t,Z
    e_hat_xyg = RtZ_to_xyg_basis.T @ uub_to_RtZ_basis @ e_hat_uub[0]

    # x_hat is RHO (PAR), y_hat is ETA (PERP)
    E_par_to_perp_ratio = np.linalg.norm(e_hat_xyg[0])/np.linalg.norm(e_hat_xyg[1])
    E_perp = E0/np.sqrt(E_par_to_perp_ratio**2+1)
    E_par = E_par_to_perp_ratio*E_perp
    
    return E_par, E_perp, RtZ_to_XYZ(x_hat_RtZ[0])
    
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
        min_z, max_z = -(np.min(combined_beam_Y) - padding_t), -(np.max(combined_beam_Y) + padding_t)
    
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
    degtorad = pi/180
    
    launch_freq_GHz = dt.inputs.launch_freq_GHz.values
    launch_beam_width = dt.inputs.launch_beam_width.values
    launch_beam_curvature = dt.inputs.launch_beam_curvature.values
    launch_angle_pol = handle_scotty_launch_angle_sign(dt) 
    launch_angle_tor = dt.inputs.toroidal_launch_angle_Torbeam.values
    
    kx_temp = 1/np.sqrt(tan(launch_angle_pol*degtorad)**2+tan(launch_angle_tor*degtorad)**2)
    ky_temp = tan(launch_angle_pol*degtorad)*kx_temp
    kz_temp = tan(launch_angle_tor*degtorad)*kx_temp
    k_vec_launch_XYZ = np.array([-kx_temp, ky_temp, kz_temp])
    k_vec_launch_XYZ_norm = k_vec_launch_XYZ / np.linalg.norm(k_vec_launch_XYZ) # Just in case
    launch_R = dt.inputs.launch_position.values[0] if cartesian_scotty is False else launch_position[0]
    launch_Z = dt.inputs.launch_position.values[2] if cartesian_scotty is False else launch_position[2]
    # launch_zeta, t, always equal to 0
    
    
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
        
        # 2D line down the centre
        port_2D = np.array([
            center + half_w * u_hat,
            center - half_w * u_hat,
            center + pad_w * u_hat,
            center - pad_w * u_hat,
        ])

        return port_corners, padded_corners, center, port_2D, (u_hat, v_hat)

    port_corners, padded_corners, center, port_2D, (u_hat, v_hat) = generate_launcher_port_3D(
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
    mod_E_par, mod_E_perp, rho_hat = scotty_pol_to_ERMES_final(dt, E0, cartesian_scotty=cartesian_scotty)
    phi_E_par = 0
    phi_E_perp = -pi/2

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

        file_path = f"{path}{filename}.txt"

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
        port_2D.reshape(-1, 3),
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
        '2D Port 0    ',
        '2D Port 1    ',
        '2D Port Padding 0    ',
        '2D Port Padding 1    ',
        'Launch Position    ', 
        'Point of Entry    ',
        'Waist Position    ',
        'pol vec    ',
        'k vec    '
    ])
    
    # Beam params
    params_val = np.array([
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
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
        'Min x(R)    ',
        'Max x(R)    ',
        'Min y(Z)    ',
        'Max y(Z)    ',
        'Min z(t)    ',
        'Max z(t)    ',
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

def calc_Eb_from_scotty(dt: datatree, E0: float = 1.0, cartesian_scotty: bool = False):
    """
    Calcualte the probe beam Electric Field amplitude along the central ray using Eqn 33 of [4]
    Returns Eb(tau) where tau is the beam parameter

    Args:
        dt (datatree): Scotty output file
        E0 (float): For scaling
        cartesian_scotty (bool): Cartesian Scotty
        
    Returns:
        Eb_tau (array): |Eb| at all the points along the ray w.r.t tau
    """
    # Naming convention, end in tau means index in terms of tau, RtZ means in RtZ basis, xyg means in xyg basis, neither means it is an element (likely), 
    # w means in plane perp to g, g means projected onto g
    # i means im component, launch and ant are equiv
    tau_len = len(dt.analysis.Psi_xx.values)
    
    x_hat_RtZ = dt.analysis.x_hat.values
    y_hat_RtZ = dt.analysis.y_hat.values
    g_hat_RtZ = dt.analysis.g_hat.values
    g_mag = dt.analysis.g_magnitude.values
    
    launch_R = dt.inputs.launch_position.values[0] if not cartesian_scotty else 0
    launch_zeta = dt.inputs.launch_position.values[1] if not cartesian_scotty else 0
    launch_K_R = dt.inputs.launch_K.values[0] if not cartesian_scotty else 0
    launch_K_zeta = dt.inputs.launch_K.values[1] if not cartesian_scotty else 0
    
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
    
    if cartesian_scotty:
        Psi_3D_ant_RtZ = dt.inputs.initial_Psi_3D_lab_cartesian
    else:
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

def ERMES_nodes_to_XYZ(msh_file: str, show_progress = True):
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
    path = os.getcwd() + msh_file
    print("Reading ERMES .msh file (streaming mode)")

    reading = False

    # First pass: determine maximum node ID for allocation (lightweight scan)
    max_id = 0
    with open(path, 'r') as f:
        iterator = f
        if show_progress:
            iterator = tqdm(f, desc="Reading mesh nodes", ncols=100)
        for line in iterator:
            if line.startswith("Coordinates"):
                reading = True
                continue
            if line.startswith("End Coordinates"):
                break
            if reading:
                try:
                    node_id = int(line.split()[0])
                    if node_id > max_id:
                        max_id = node_id
                except Exception:
                    continue

    node_to_xyz = np.zeros((max_id + 1, 3), dtype=np.float64) # +1 to allow indexing by nodeID

    # Second pass: fill directly
    with open(path, 'r') as f:
        reading = False
        iterator = f
        if show_progress:
            iterator = tqdm(f, desc="Loading mesh nodes", ncols=100)
        for line in iterator:
            if line.startswith("Coordinates"):
                reading = True
                continue
            if line.startswith("End Coordinates"):
                break
            if reading:
                parts = line.split()
                if len(parts) == 4:
                    node_id = int(parts[0])
                    node_to_xyz[node_id] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return node_to_xyz
        
def ERMES_results_to_node(res_file: str, result_name: str, show_progress = True):
    """
    Load in the .res file to read each result as nodeID and return the value of result_name at that node. Supports scalar and vector results
    
    Args:
        res_file (str): ERMES res file
        result_name (str): Name of the result, as saved by ERMES, that is wanted
        
    Returns:
        result (dict): Dictionary of result value (scalar or vector) with nodeID as the key
    """
    path = os.getcwd() + res_file
    print(f"Reading ERMES res file for '{result_name}' results")
    
    result = {}
    reading = False
    inside_block = False
    num_entries = 0
    available_results = []

    # Stream read
    with open(path, 'r') as f:
        iterator = tqdm(f, desc=f"Parsing {result_name}", ncols=100) if show_progress else f
        for line in iterator:
            line = line.strip()

            # Skip blank lines or comments
            if not line or line.startswith("#"):
                continue

            # Detect result block
            if line.startswith("Result"):
                # Extract the FIRST quoted substring
                if '"' in line:
                    q1 = line.find('"')
                    q2 = line.find('"', q1 + 1)
                    if q2 > q1:
                        found_name = line[q1+1:q2]
                        available_results.append(found_name)
                        # Activate block if it matches
                        inside_block = (found_name == result_name)
                        reading = False
                        continue

            # Start of Values
            if inside_block and line.startswith("Values"):
                reading = True
                continue

            # --- FIXED end condition ---
            if reading and line.startswith("End Values"):
                reading = False
                inside_block = False
                break

            # Actual data
            if reading:
                if reading:
                    parts = line.split()

                    # Scalar: node_id value
                    if len(parts) == 2:
                        try:
                            node_id = int(parts[0])
                            result[node_id] = float(parts[1])
                            num_entries += 1
                        except ValueError:
                            # e.g. "End Values" slipping through: ignore
                            pass

                    # Vector-like: node_id v1 v2 v3 [extra stuff...]
                    elif len(parts) >= 4:
                        try:
                            node_id = int(parts[0])
                            # Take ONLY the first 3 numeric components (e.g. Sx, Sy, Sz)
                            vec = np.array(list(map(float, parts[1:4])), dtype=float)
                            result[node_id] = vec
                            num_entries += 1
                        except ValueError:
                            # Skip malformed lines
                            pass

                    else:
                        # Unexpected line format inside Values; ignore
                        pass

    if num_entries == 0:
        print(f"[Warning] '{result_name}' not found in file; returning empty dict.")
        if available_results:
            print("  Available result names found:")
            for n in sorted(set(available_results)):
                print("   -", repr(n))
    else:
        print(f"Loaded {num_entries:,} entries for '{result_name}'.")
    return result

def debug_read_block(res_file: str, result_name: str, max_lines: int = 10):
    """
    Print the first few 'Values' lines from a given Result block.
    """
    path = os.getcwd() + res_file
    print(f"\nInspecting result block: {result_name}")
    inside_block = False
    reading = False
    printed = 0

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("Result"):
                inside_block = (f'"{result_name}"' in line)
                print(line)
                reading = False
                continue

            if not inside_block:
                continue

            if line.startswith("Values"):
                reading = True
                print("Found 'Values' section")
                continue

            if reading:
                if line.startswith("End Values"):
                    print("Reached 'End Values'")
                    break

                if not line:
                    continue

                parts = line.split()
                print(f"LINE: {line}")
                print(f"  parts ({len(parts)}): {parts}")
                printed += 1
                if printed >= max_lines:
                    print("... stopping after", max_lines, "lines")
                    break

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
   
def ERMES_results_to_plots(
    res: str,
    msh: str,
    dt,
    grid_resolution: float = 4e-4,
    normal_vector: np.array = None,
    plot_blocks = None,
    save: bool = False,
    prefix: str = "ERMES_results",
    cartesian_scotty: bool = False
):
    """
    Modular ERMES analysis & plotting (3D-native, 2D just means Z = 0)
    If plotting 3D results, set normal_vector to None

    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        grid_resolution (float): Sampling resolution from meshing (m)
        normal_vector (array or ndarray): Normal vector in ERMES XYZ that is perp to g for plotting and beam width calculations. (The width in the OTHER normal will be calculated). If it is None, automatically assume this is in 3D.
        plot_blocks (list[str]): Choose which logical plot blocks to render. If None, all are used. Available: "field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors", "cross-section"
        save (bool): If True, each plot block saves a PNG with the given prefix
        prefix (str): Prefix for saved figures
        cartesian_Scotty (bool): Cartesian Scotty?
            
    Returns:
        Plots
    """

    if normal_vector is not None:
        is2D = True
    else:
        is2D = False # Means it's 3D, so we don't need the normal vector. Use xhat and yhat for widths

    if plot_blocks is None:
        plot_blocks = ["field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors", "cross-section"]
    
    # Load & prepare all data
    (
        modE_xyz, vecE_xyz, vecS_xyz,
        beam_xyz,
        distance_along_beam, tau_len, tau_cutoff
    ) = prepare_core_fields(res, msh, dt, cartesian_scotty)

    # Sampling tolerance (Cheap way of choosing the nearest point)
    tol = grid_resolution / 2.0

    # Extract |E| and E-vector along beam path
    print("Sampling E field")
    modE_list, vecE_array_beam, xyz_tree, modE_vals_all = sample_fields_along_beam(modE_xyz, vecE_xyz, beam_xyz)

    # Build transverse sampling slices & profiles using gaussian fit
 
    print("Building transverse E and widths")
    (
    fitted_widths, fitted_principle_width_1, fitted_principle_width_2, 
    fit_params, chi2_list, offsets_per_tau, 
    offsets_per_tau_x, offsets_per_tau_y, 
    modE_profiles,  mod_E_theoretical_profiles, poynting_flux_per_tau,
    fit_params_x, fit_params_y,
    modE_profiles_x, modE_profiles_y, # x means in the x hat direction, y means in the y hat direction
    modE_theoretical_profiles_x, modE_theoretical_profiles_y,
    ) = build_transverse_profiles_and_fits(
        dt,
        beam_xyz, 
        modE_xyz, 
        vecS_xyz,
        modE_list,
        normal_vector, # fittied principle widths only generated if normal vector is None
        cartesian_scotty=cartesian_scotty,
        grid_resolution=grid_resolution
    )

    # Plot
    
    # Only for 2D
    if "field_map" in plot_blocks:
        if is2D: # Check if 2D. Essentially only plot this if we're doing a 2D plot
            plot_field_map(
                modE_xyz=modE_xyz,
                dt=dt,
                tol=tol,
                grid_resolution=grid_resolution,
                norm_vec = normal_vector,
                prefix=prefix,
                save=save,
                cartesian_scotty=cartesian_scotty
            )
        else:
            pass

    if "3D field_map" in plot_blocks:
        plot_field_with_beam(
            dt=dt,
            modE_xyz= modE_xyz,
            norm_vec = normal_vector if is2D else None,
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
            save=save,
            cartesian_scotty=cartesian_scotty
        )
        save_modE_vs_tau_results(
            save_path = os.getcwd() + "\\analysis_data\\",
            prefix = prefix,
            tau_positions = distance_along_beam,
            modE_ermes_central=modE_list
        )

    if "transverse_profile" in plot_blocks:
        if is2D:
            plot_transverse_profiles_2D(
                offsets_per_tau, 
                modE_profiles, 
                fit_params, 
                modE_theoretical_profiles=mod_E_theoretical_profiles,
                prefix=prefix,
                save=save
            )
        else:
            plot_transverse_profiles_3D(
                distance_along_beam,
                offsets_per_tau_x, 
                offsets_per_tau_y,
                modE_profiles_x, modE_profiles_y, 
                fit_params_x, 
                fit_params_y,
                modE_theoretical_profiles_x, modE_theoretical_profiles_y,
                prefix=prefix,
                save=save
            )

    if "widths" in plot_blocks:
        if is2D:
            plot_2D_widths(
                dt=dt,
                distance_along_beam=distance_along_beam,
                tau_cutoff=tau_cutoff,
                fitted_widths=fitted_widths,
                chi2_list=chi2_list,
                norm_vec=normal_vector,
                prefix=prefix,
                save=save,
                cartesian_scotty=cartesian_scotty
            )
        else: # Plot 3D principle widths and error of fitting
            plot_3D_widths(
                dt=dt,
                distance_along_beam=distance_along_beam,
                tau_cutoff=tau_cutoff,
                fitted_widths_x=fitted_principle_width_1, 
                fitted_widths_y=fitted_principle_width_2,
                prefix=prefix,
                save=save
            )
            plot_3D_width_var_covar(
                fit_params_x,
                fit_params_y,
                prefix,
                save
            )

    if "cross-section" in plot_blocks:
        plot_cross_section(
            dt=dt,
            modE_xyz=modE_xyz,
            vecS_xyz=vecS_xyz,
            save=save,
            prefix=prefix
        )
    # Only for 2D
    if "flux" in plot_blocks:
        plot_flux(
            distance_along_beam=distance_along_beam,
            poynting_flux_per_tau=poynting_flux_per_tau,
            tau_cutoff=tau_cutoff,
            prefix=prefix,
            save=save
        )

    if "errors" in plot_blocks: # NOT IMPLEMENTED PROPERLY YET, Likely won't bother
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
def prepare_core_fields(res, msh, dt, cartesian_scotty):
    """
    Load ERMES data in XYZ, map node fields, and build beam arrays
    
    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        cartesian_scotty (bool): Cartesian Scotty
        
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

    #ebug_read_block(res, "Poynting_vector")
    #exit
    
    # Results to node dicts
    modE = ERMES_results_to_node(res_file=res, result_name="mod(E)") # {node_id: scalar}
    #vecE = ERMES_results_to_node(res_file=res, result_name="rE") # {node_id: (Ex,Ey,Ez)}
    vecS = ERMES_results_to_node(res_file=res, result_name="Poynting_vector") # {node_id: (Sx,Sy,Sz)}
    
    # Dicts to arrays (max node id used as length)
    max_node = max(modE.keys())
    modE_array = np.zeros(max_node + 1)
    vecE_array = np.zeros((max_node + 1, 3))
    vecS_array = np.zeros((max_node + 1, 3))

    for i, val in modE.items():
        modE_array[i] = val
    #for i, v in vecE.items():
    #    vecE_array[i] = v
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
    distance_along_beam = dt.analysis.distance_along_line.values if not cartesian_scotty else dt.analysis.arc_length.values # (N,)

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

def build_transverse_profiles_and_fits(dt, beam_xyz, modE_xyz, vecS_xyz, modE_list, normal_vec, grid_resolution, cartesian_scotty, show_progress = True):
    """
    Build and fit transverse |E| profiles along a beam in a 2D plane,
    and compute the theoretical transverse field envelope from Scotty.
    Also calculates the poynting flux within +-w

    Args:
        dt (datatree): Scotty output file
        beam_xyz (array): (N,3) beam center coordinates
        modE_xyz (ndaarrayrray): (M,4) ERMES data [x,y,z,|E|]
        vecS_xyz (array): (M,6) ERMES [x,y,z,Sx,Sy,Sz] Poynting vectors
        modE_list (array): List of modE values for normalization
        normal_vec (array-like): Plane normal vector in ERMES X Y Z
        grid_resolution (float): ERMES grid spacing (m)
        cartesian_scotty (bool): Cartesian Scotty

    Returns:
        fitted_widths (array): (N,) fitted 1/e field widths from ERMES.
        fitted_x: x_hat dir
        fitted_y: y_hat dir
        fit_params (array): (N,3) [A_fit, x0_fit, w_fit].
        chi2_list (array): (N,) chi-squared values of fits.
        offsets_per_tau (list): offsets along b-hat for each tau.
        offsets_per_tau_x: For 3D
        offsets_per_tau_y: For 3D
        modE_profiles (list): sampled |E| profiles for each tau.
        modE_theoretical_profiles (list): theoretical Scotty envelopes along beamfront.
        poynting_flux_per_tau (array): Integrated poynting flux
        fit_params_x: For 3D
        fit_params_y: For 3D
        modE_profiles_x: For 3D 
        modE_profiles_y: For 3D
        modE_theoretical_profiles_x: For 3D
        modE_theoretical_profiles_y: For 3D
    """

    # Prepare ERMES data lookup
    coords = modE_xyz[:, :3]
    values = modE_xyz[:, 3]
    tree = cKDTree(coords)

    N = len(beam_xyz)
    iterator = tqdm(range(N), desc="Building transverse profiles", ncols=100) if show_progress else range(N)

    # Containers for 2D
    fitted_widths, fit_params, chi2_list = [], [], []
    offsets_per_tau, modE_profiles, modE_theoretical_profiles = [], [], []
    
    # Containers for 3D
    fitted_x, fitted_y = [], []
    fit_params_x, fit_params_y = [], []
    chi2_x, chi2_y = [], []
    modE_profiles_x, modE_profiles_y = [], []
    offsets_per_tau_x, offsets_per_tau_y = [], []
    modE_theoretical_profiles_x, modE_theoretical_profiles_y = [], []
    
    S_coords = vecS_xyz[:, :3]
    S_vals = vecS_xyz[:, 3:]
    tree_S = cKDTree(S_coords)

    poynting_flux_per_tau = []

    # From Scotty
    ghat_cartesian = dt.analysis.g_hat_Cartesian if not cartesian_scotty else dt.analysis.g_hat_cartesian # in RtZ
    Psi_3D_Cartesian = dt.analysis.Psi_3D_Cartesian if not cartesian_scotty else dt.analysis.Psi_3D_labframe_cartesian # in RtZ
    # 2D
    if normal_vec is not None:
        n_hat = np.array(normal_vec, float)
        n_hat /= np.linalg.norm(n_hat)
        n_hat_RtZ = XYZ_to_RtZ(n_hat)
        beam_width_at_tau = np.linalg.norm(beam_width(ghat_cartesian, n_hat_RtZ, Psi_3D_Cartesian).values, axis=1)
    # 3D
    else:
        n_hat = None
        # For 3D, we will project onto x_hat and y_hat separately below
        
        x_hat_RtZ = dt.analysis.x_hat_Cartesian.values if not cartesian_scotty else dt.analysis.x_hat_cartesian.values
        y_hat_RtZ = dt.analysis.y_hat_Cartesian.values if not cartesian_scotty else dt.analysis.y_hat_cartesian.values
        
        x_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis = 1, arr = x_hat_RtZ)
        y_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis = 1, arr = y_hat_RtZ)
        beam_width_x = np.linalg.norm(beam_width(ghat_cartesian, dt.analysis.y_hat_Cartesian, Psi_3D_Cartesian).values, axis=1)
        beam_width_y = np.linalg.norm(beam_width(ghat_cartesian, dt.analysis.x_hat_Cartesian, Psi_3D_Cartesian).values, axis=1)
    

    ghat_xyz = np.apply_along_axis(
        RtZ_to_XYZ, axis=1,
        arr=ghat_cartesian.values /
            np.expand_dims(dt.analysis.g_magnitude.values, axis=1)
    )

    modE_scotty = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0],
        cartesian_scotty=cartesian_scotty
    )
    if normal_vec is not None:
        for i in iterator:
            g_xyz = ghat_xyz[i]
            g_rtz = ghat_cartesian[i]
            Psi = Psi_3D_Cartesian.values[i]

            # Project beam onto plane perp to n_hat
            g_proj = g_xyz - np.dot(g_xyz, n_hat) * n_hat
            g_proj /= np.linalg.norm(g_proj)
            b_hat = np.cross(n_hat, g_proj)
            b_hat /= np.linalg.norm(b_hat)
            width = beam_width_at_tau[i]
            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)

            # Sample ERMES data
            sample_points = beam_xyz[i] + offsets[:, None] * b_hat[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles.append(profile)
            offsets_per_tau.append(offsets)

            # Theoretical envelope (Scotty) in RtZ coords WLOG
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, b_hat)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles.append(E_theory)

            # Fit Gaussian
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
    else:
        # in x_hat direction
        for i in iterator:
            hat_vec = x_hat_xyz[i]
            width = float(beam_width_x[i])

            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)
            offsets_per_tau_x.append(offsets)

            # Sample points along x_hat
            sample_points = beam_xyz[i] + offsets[:, None] * hat_vec[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles_x.append(profile)

            # Theoretical profile
            Psi = Psi_3D_Cartesian.values[i]
            g_rtz = ghat_cartesian.values[i]
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, hat_vec)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles_x.append(E_theory)

            # Gaussian fit
            try:
                p0 = [np.max(profile), 0.0, width]
                popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
                A_fit, x0_fit, w_fit = popt
                chi2 = np.sum((profile - gaussian_fit(offsets, *popt))**2 / (np.abs(profile) + 1e-12))
                fit_params_x.append([A_fit, x0_fit, w_fit])
                fitted_x.append(abs(w_fit))
                chi2_x.append(chi2)
            except RuntimeError:
                fit_params_x.append([np.nan, np.nan, np.nan])
                fitted_x.append(np.nan)
                chi2_x.append(np.nan)

        # In y_hat direction
        for i in iterator:
            hat_vec = y_hat_xyz[i]
            width = float(beam_width_y[i])

            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)
            offsets_per_tau_y.append(offsets)

            # Sample points along y_hat
            sample_points = beam_xyz[i] + offsets[:, None] * hat_vec[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles_y.append(profile)

            # Theoretical profile
            Psi = Psi_3D_Cartesian.values[i]
            g_rtz = ghat_cartesian.values[i]
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, hat_vec)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles_y.append(E_theory)

            # Gaussian fit
            try:
                p0 = [np.max(profile), 0.0, width]
                popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
                A_fit, x0_fit, w_fit = popt
                chi2 = np.sum((profile - gaussian_fit(offsets, *popt))**2 / (np.abs(profile) + 1e-12))
                fit_params_y.append([A_fit, x0_fit, w_fit])
                fitted_y.append(abs(w_fit))
                chi2_y.append(chi2)
            except RuntimeError:
                fit_params_y.append([np.nan, np.nan, np.nan])
                fitted_y.append(np.nan)
                chi2_y.append(np.nan)
                
    # Flux regardless (maybe deprecate)
    for i in iterator:
        g_xyz = ghat_xyz[i]
        _, S_idx = tree_S.query(sample_points)
        S_vecs = S_vals[S_idx]
        S_dot_g = np.dot(S_vecs, g_xyz)
        total_flux = np.trapz(S_dot_g, x=offsets)
        poynting_flux_per_tau.append(total_flux)

    return (
        np.array(fitted_widths),
        np.array(fitted_x),
        np.array(fitted_y),
        np.array(fit_params),
        np.array(chi2_list),
        offsets_per_tau,
        offsets_per_tau_x,
        offsets_per_tau_y,
        modE_profiles,
        modE_theoretical_profiles,
        np.array(poynting_flux_per_tau),
        np.array(fit_params_x),
        np.array(fit_params_y),
        np.array(modE_profiles_x, dtype=object),
        np.array(modE_profiles_y, dtype=object),
        np.array(modE_theoretical_profiles_x, dtype=object),
        np.array(modE_theoretical_profiles_y, dtype=object)
    )

# Plot functions
def plot_field_map(modE_xyz, dt, tol, grid_resolution, norm_vec, prefix, save, cartesian_scotty):
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

    # Colormap with white as 0, red as max
    base = cm.get_cmap('Reds')
    colors = [(0, 0, 0, 0)] + [base(i) for i in np.linspace(0, 1, 256)]
    white_to_red = LinearSegmentedColormap.from_list('white_to_red', colors)
    white_to_red.set_bad(color=(1, 1, 1, 0)) # Handle NaNs (set to transparent)
    
    finite_vals = Ei[np.isfinite(Ei)]
    vmax = np.nanmax(finite_vals)
    norm = Normalize(vmin=0.0, vmax=vmax)
    plt.figure(figsize=(6, 5))
    pc = plt.pcolormesh(RR, ZZ, Ei, shading='auto', cmap=white_to_red, norm=norm)
    plt.colorbar(pc, label='|E| (A.U.)')

    # Poloidal flux surfaces
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

        plt.plot(beam_plus.sel(col="X"), beam_plus.sel(col="Z"), "--k", lw=1)
        plt.plot(beam_minus.sel(col="X"), beam_minus.sel(col="Z"), "--k", lw=1, label="Beam width")
        plt.plot(data_Yaxis["q_X"], data_Yaxis["q_Z"], "-", c='black', lw=1.5, zorder=1, label = "Central ray")
    else:
        plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)

        ghat_cartesian = dt.analysis.g_hat_Cartesian # in RtZ
        Psi_3D_Cartesian = dt.analysis.Psi_3D_Cartesian # in RtZ
        
        width = beam_width(ghat_cartesian, XYZ_to_RtZ(norm_vec), Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian if not cartesian_scotty else {
            "X": np.array(dt.solver_output.q_X),
            "Y": np.array(dt.solver_output.q_Y),
            "Z": np.array(dt.solver_output.q_Z),
        }
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), "--k", lw=1)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), "--k", lw=1, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), "-", c='black', lw=1.5, zorder=1, label="Central ray")

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

def plot_field_with_beam(dt, modE_xyz, norm_vec = None, save=False, prefix="", sample_rate = 0.05):
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
    mask = e_norm > 0.05
    
    idx = np.where(mask)[0]
    n_total = len(idx)
    
    n_keep = int(sample_rate*n_total)
    if n_keep < n_total:
        idx = np.random.choice(idx, n_keep, replace=False)
        
    base = cm.get_cmap('Reds')
    colors = [(1, 1, 1, 1)] + [base(i) for i in np.linspace(0.3, 1, 256)]
    white_to_red = LinearSegmentedColormap.from_list('white_to_red', colors)
    white_to_red.set_bad(color=(1, 1, 1, 0)) # Handle NaNs (set to transparent)
    
    finite_vals = e[np.isfinite(e)]
    vmax = np.nanmax(finite_vals)
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Field visualization (voxel-style scatter) for 3D
    sc = ax.scatter(x[idx], y[idx], z[idx],
                    c=e[idx], cmap=white_to_red, norm = norm, alpha=0.2, s=0.5,
                    label='ERMES |E| Field')
    
    # Overlay beam geometry
    if norm_vec is not None: # Means this is 2D ERMES
        width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian 
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), -beam_plus.sel(col_cart="Y"),"--k", lw=2)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), -beam_minus.sel(col_cart="Y"), "--k", lw=2, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), -beam.sel(col_cart="Y"),  "-", c='black', lw=4, label="Central ray")
    else: # Means this is 3D ERMES, use principal widths in y_hat direction for plotting
        width = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.x_hat_Cartesian.values, dt.analysis.Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian 
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), -beam_plus.sel(col_cart="Y"),"--k", lw=2)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), -beam_minus.sel(col_cart="Y"), "--k", lw=2, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), -beam.sel(col_cart="Y"),  "-", c='black', lw=4, label="Central ray")


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

def plot_modE_vs_tau(dt, modE_list, tau_cutoff, distance_along_beam, prefix, save, cartesian_scotty):
    """
    |E| (ERMES vs Scotty) along beam distance.
    """
    theoretical_modE_tau = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0],
        cartesian_scotty=cartesian_scotty
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

def plot_transverse_profiles_2D(
    offsets_per_tau,
    modE_profiles,
    fit_params,
    modE_theoretical_profiles=None,
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

def plot_transverse_profiles_3D(
    distance_along_line,
    offsets_per_tau_x,
    offsets_per_tau_y,
    modE_profiles_x,
    modE_profiles_y,
    fit_params_x,
    fit_params_y,
    modE_theoretical_profiles_x,
    modE_theoretical_profiles_y,
    save: bool = False,
    prefix: str = ""
):
    """
    Interactive 3D transverse profile plot showing ERMES |E| samples,
    Gaussian fits, and Scotty theoretical envelopes along x_hat and y_hat.

    Args:
        distance_along_line (array): From scotty output
        offsets_per_tau (list): list of offset arrays for each τ.
        modE_profiles_x, modE_profiles_y: sampled |E| profiles from ERMES.
        fit_params (tuple): (fit_params_x, fit_params_y), Gaussian parameters [A, x0, w].
        modE_theoretical_profiles (tuple, optional): (theory_x, theory_y) from Scotty.
        beam_widths (tuple, optional): (fitted_principle_width_1, fitted_principle_width_2) for ±w markers.
        prefix (str): File save prefix.
        save (bool): Whether to save figure.
    """

    N = len(offsets_per_tau_x)

    # Determine scaling
    global_maxE = np.nanmax([
        np.nanmax(np.abs(p)) for p in (modE_profiles_x)
    ])
    global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets_per_tau_x])

    # Figure setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    plt.subplots_adjust(bottom=0.15, hspace=0.25)

    x0 = offsets_per_tau_x[0]
    yx = modE_profiles_x[0]
    yy = modE_profiles_y[0]

    # ERMES scatter points
    scatter_x = ax1.scatter(x0, yx, color='red', s=15, label=r'ERMES samples $(\hat{x})$')
    scatter_y = ax2.scatter(x0, yy, color='red', s=15, label=r'ERMES samples $(\hat{y})$')

    # Gaussian fits
    fit_yx = gaussian_fit(x0, *fit_params_x[0])
    fit_yy = gaussian_fit(x0, *fit_params_y[0])
    line_fit_x, = ax1.plot(x0, fit_yx, color='green', lw=2, label='Gaussian fit')
    line_fit_y, = ax2.plot(x0, fit_yy, color='green', lw=2, label='Gaussian fit')

    # Scotty theoretical envelopes
    thx = modE_theoretical_profiles_x[0]
    thy = modE_theoretical_profiles_y[0]
    line_theory_x, = ax1.plot(x0, thx, '--', color='orange', lw=2, label='Scotty theory')
    line_theory_y, = ax2.plot(x0, thy, '--', color='orange', lw=2, label='Scotty theory')

    # Set limits, labels
    for ax, label in zip([ax1, ax2], [r"$\hat{x}$ direction", r"$\hat{y}$ direction"]):
        ax.set_xlim(-global_max_offset, global_max_offset)
        ax.set_ylim(0, 1.1 * global_maxE)
        ax.set_ylabel('|E| (A.U.)')
        ax.legend()
        ax.grid(True)
    ax2.set_xlabel('Offset from beam center (m)')
    ax1.set_title(f'Transverse |E| profiles at {distance_along_line[0]:.3f}m along the central ray')
    
    # Slider over distance along beam
    s_min, s_max = float(distance_along_line[0]), float(distance_along_line[-1])
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        '',
        s_min,
        s_max,
        valinit=s_min
    )

    # Helper functiom to map slider value (s) to nearest index
    def s_to_index(s_val):
        return int(np.argmin(np.abs(distance_along_line - s_val)))
    
    # Update function
    def update(val):
        s_val = slider.val
        j = s_to_index(s_val)

        # x-hat plot
        x = offsets_per_tau_x[j]
        yx = modE_profiles_x[j]
        if len(x) != len(yx):
            min_len = min(len(x), len(yx))
            x, yx = x[:min_len], yx[:min_len]
        scatter_x.set_offsets(np.column_stack([x, yx]))
        line_fit_x.set_data(x, gaussian_fit(x, *fit_params_x[j]))
        if line_theory_x is not None:
            y_theory_x = modE_theoretical_profiles_x[j]
            if len(y_theory_x) != len(x):
                y_theory_x = y_theory_x[:len(x)]
            line_theory_x.set_data(x, y_theory_x)

        # y-hat plot 
        y = offsets_per_tau_y[j]
        yy = modE_profiles_y[j]
        if len(y) != len(yy):
            min_len = min(len(y), len(yy))
            y, yy = y[:min_len], yy[:min_len]
        scatter_y.set_offsets(np.column_stack([y, yy]))
        line_fit_y.set_data(y, gaussian_fit(y, *fit_params_y[j]))
        if line_theory_y is not None:
            y_theory_y = modE_theoretical_profiles_y[j]
            if len(y_theory_y) != len(y):
                y_theory_y = y_theory_y[:len(y)]
            line_theory_y.set_data(y, y_theory_y)

        ax1.set_title(f"Transverse |E| profiles at {distance_along_line[j]:.3f}m along the central ray")
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Save option
    if save:
        fig.savefig(f"{prefix}_transverse_profiles_3D.png", dpi=250, bbox_inches='tight')
        print(f"Saved {prefix}_transverse_profiles_3D.png")

    plt.show()

def plot_cross_section(dt, modE_xyz, vecS_xyz, save=False, prefix="", show_progress=True):
    """
    Plot a heatmap cross section of the beam in the plane perpendicular to g
    
    Args:
    
    
    """
    coords = modE_xyz[:, :3]
    values = modE_xyz[:, 3]
    tree = cKDTree(coords)
    
    S_coords = vecS_xyz[:, :3]
    S_vals = vecS_xyz[:, 3:]
    tree_S = cKDTree(S_coords)
    
    beam_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=dt.analysis.beam_cartesian.values)
    
    N = len(beam_xyz)

    iterator = tqdm(range(N), desc="Computing transverse slices", ncols=100) \
               if show_progress else range(N)

    distance_along_line = dt.analysis.distance_along_line.values
    
    g_hat_rtz = dt.analysis.g_hat_Cartesian
    g_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=g_hat_rtz.values)
    
    Psi = dt.analysis.Psi_3D_Cartesian
    x_hat_rtz = dt.analysis.x_hat_Cartesian
    y_hat_rtz = dt.analysis.y_hat_Cartesian

    x_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=x_hat_rtz.values)
    y_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=y_hat_rtz.values)

    width_x = np.linalg.norm(beam_width(g_hat_rtz, y_hat_rtz, Psi).values, axis=1) # in x_hat direction
    width_y = np.linalg.norm(beam_width(g_hat_rtz, x_hat_rtz, Psi).values, axis=1) # in y_hat direction
    
    E_slices = []
    extents = [] # store extents per tau since the beam evolves
    power_slices = []
    
    for i in iterator:

        # Center of slice
        r0 = beam_xyz[i]

        # Basis vectors of plane:
        g = g_hat_xyz[i] / np.linalg.norm(g_hat_xyz[i])

        # Use Scotty's principal axes for the transverse plane
        u = x_hat_xyz[i] / np.linalg.norm(x_hat_xyz[i])
        v = y_hat_xyz[i] / np.linalg.norm(y_hat_xyz[i])

        # Beam extents (4 times width )
        extent_u = 2.5*width_x[i]
        extent_v = 2.5*width_y[i]

        extents.append((extent_u, extent_v))

        # 2D grid in uv space
        u_lin = np.linspace(-extent_u, extent_u, 100)
        v_lin = np.linspace(-extent_v, extent_v, 100)
        U, V = np.meshgrid(u_lin, v_lin)

        # Convert (u,v) to XYZ
        pts = r0 + U[..., None]*u + V[..., None]*v
        pts_flat = pts.reshape(-1, 3)

        # Query ERMES field to get the vals
        _, idx = tree.query(pts_flat)
        E_slice = values[idx].reshape(U.shape)
        E_slices.append(E_slice)
        
        _, idxS = tree_S.query(pts_flat)
        S_slice = S_vals[idxS].reshape(U.shape + (3,))
        
        S_dot_g = np.einsum('ijk,k->ij', S_slice, g)
        total_flux = np.sum(S_dot_g)*(2*extent_u/99)*(2*extent_v/99) # int s.g.dA
        power_slices.append(total_flux)
    power_slices /= power_slices[0] # Normalize to input
    print("Power in: ", power_slices[0])
    print("Power out: ", power_slices[-1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(bottom=0.25)

    # Initial data
    extent_u0, extent_v0 = extents[0]
    im = ax.imshow(
        E_slices[0],
        extent=[-extent_u0, extent_u0, -extent_v0, extent_v0],
        origin='lower',
        cmap='inferno',
        interpolation='nearest'
    )

    ax.set_title(f"Transverse |E| slice at {distance_along_line[0]:.3f}m along the central ray")
    ax.set_xlabel(r"$\hat{x}$ direction (m)")
    ax.set_ylabel(r"$\hat{y}$ direction (m)")

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("|E| (A.U.)")

    # Slider over distance along beam
    s_min, s_max = float(distance_along_line[0]), float(distance_along_line[-1])
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        '',
        s_min,
        s_max,
        valinit=s_min
    )

    # Helper function to map slider value (s) to nearest index
    def s_to_index(s_val):
        return int(np.argmin(np.abs(distance_along_line - s_val)))

    # Update function
    def update(val):
        s_val = slider.val
        j = s_to_index(s_val)

        extent_u, extent_v = extents[j]
        E_j = E_slices[j]

        im.set_data(E_j)
        im.set_extent([-extent_u, extent_u, -extent_v, extent_v])
        E_val = E_j[np.isfinite(E_j)]
        if E_val.size > 0: # Quick bit of safety
            vmax = np.nanpercentile(E_val, 99)  # or np.nanmax(finite)
            im.set_clim(0.0, vmax)
            cb.update_normal(im)
        
        ax.set_title(f"Transverse |E| slice at {distance_along_line[j]:.3f}m along the central ray")
        print("Power: ", power_slices[j])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save:
        fig.savefig(f"{prefix}_transverse_field_slices_3D.png", dpi=200)

    plt.show()
  
def plot_2D_widths(dt, distance_along_beam, tau_cutoff, fitted_widths, norm_vec, chi2_list, prefix, save, cartesian_scotty):
    """
    Fitted vs Scotty widths + chi^2 vs distance (stacked).
    """
    if cartesian_scotty:
        width_norm = dt.analysis.beam_width_2
    else:
        width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
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
    
def plot_3D_widths(dt, distance_along_beam, tau_cutoff, fitted_widths_x, fitted_widths_y, prefix, save):
    """
    Fitted vs Scotty widths for x_hat and y_hat directions (principle widths)
    """
    width_x = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.y_hat_Cartesian, dt.analysis.Psi_3D_Cartesian) # in x_hat direction
    width_y = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.x_hat_Cartesian, dt.analysis.Psi_3D_Cartesian) # in y_hat direction
    width_x_norm = np.linalg.norm(width_x.values, axis=1)
    width_y_norm = np.linalg.norm(width_y.values, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[1, 1])

    ax1.plot(distance_along_beam, width_x_norm, '--', label=r"Scotty width $\hat{x}$ direction", color='orange')
    ax1.plot(distance_along_beam, fitted_widths_x, label=r"Fitted ERMES width $\hat{x}$ direction", color='red') # width in x_hat direction
    ax1.vlines(distance_along_beam[tau_cutoff], *ax1.get_ylim(), linestyles='--', color='blue')
    ax1.set_ylabel("Width (m)")
    ax1.set_title("Beam widths")
    ax1.legend()
    
    ax2.plot(distance_along_beam, width_y_norm, '--', label=r"Scotty width $\hat{y}$ direction", color='orange')
    ax2.plot(distance_along_beam, fitted_widths_y, label=r"Fitted ERMES width $\hat{y}$ direction", color='red') # width in y_hat direction
    ax2.vlines(distance_along_beam[tau_cutoff], *ax2.get_ylim(), linestyles='--', color='blue')
    ax2.set_ylabel("Width (m)")
    ax2.set_title("Beam widths")
    ax2.legend()


    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_principle_widths.png", dpi=200)
    plt.show()

def plot_3D_width_var_covar(
    fit_params_x, 
    fit_params_y, 
    prefix: str = "", 
    save: bool = False
):
    """
    Plot the variance and covariance of 3D Gaussian beam widths
    fitted along the x_hat andy_hat principal axes.

    Args:
        fit_params_x (ndarray): (N,3) fitted Gaussian parameters [A, x0, w] for x_hat direction.
        fit_params_y (ndarray): (N,3) fitted Gaussian parameters [A, x0, w] for y_hat direction.
        prefix (str): Optional filename prefix for saving figures.
        save (bool): Whether to save the generated figure.
    """

    # Extract valid widths (w corresponds to 1/e field width)
    sigma_x = np.array([p[2] for p in fit_params_x if np.all(np.isfinite(p))])
    sigma_y = np.array([p[2] for p in fit_params_y if np.all(np.isfinite(p))])

    # tau index range (ensure equal length for plotting)
    N = min(len(sigma_x), len(sigma_y))
    tau_vals = np.arange(N)
    sigma_x, sigma_y = sigma_x[:N], sigma_y[:N]

    # Variance and covariance
    variance_x = sigma_x**2
    variance_y = sigma_y**2
    covariance_xy = sigma_x * sigma_y

    # Plot it
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    plt.subplots_adjust(wspace=0.3)

    # Left: Variances
    ax1.plot(tau_vals, variance_x, label=r'$\sigma_x^2$', color='red')
    ax1.plot(tau_vals, variance_y, label=r'$\sigma_y^2$', color='blue')
    ax1.set_title("Variance of Fitted Gaussian Widths")
    ax1.set_xlabel(r"$\tau$ index")
    ax1.set_ylabel(r"Variance (m$^2$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Covariance
    ax2.plot(tau_vals, covariance_xy, color='purple', label=r'$\sigma_x\sigma_y$')
    ax2.set_title("Covariance of Fitted Gaussian Widths")
    ax2.set_xlabel(r"$\tau$ index")
    ax2.set_ylabel(r"Covariance (m$^2$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("3D Gaussian Beam Width Variance and Covariance", fontsize=12)

    # Save optional
    if save:
        filename = f"{prefix}_3D_width_var_covar.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved 3D width variance/covariance plot to {filename}")

    plt.show()

def plot_flux(distance_along_beam, poynting_flux_per_tau, tau_cutoff, prefix, save):
    """
    Poynting flux S dot g integrated across beamfront vs distance. Only valid for 2D since in 3D, we can just calculate it from ERMES
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

# NOT PROPERLY IMPLEMENTED YET AND LIKELY WON'T BOTHER
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

# For saving
def save_modE_vs_tau_results(
    save_path: str,
    prefix: str,
    tau_positions,
    modE_ermes_central,
):
    """
    Save |E| vs tau along central ray results to .npz for later comparison.

    Args:
        save_path (str): Folder path to save results
        prefix (str): Filename prefix
        tau_positions (array): Positions along beam
        modE_ermes_central (array): |E| from ERMES along central ray
    Returns:
        file_path (str): Path to saved .npz file.
    """

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{prefix}_modE_vs_tau.npz")

    np.savez_compressed(
        file_path,
        tau_positions=np.asarray(tau_positions),
        modE_ermes_central=np.asarray(modE_ermes_central),
    )

    print(rf"Saved |E|($\tau$) data to {file_path}")

# Getting an oblique plane for 2D ERMES defined by a known normal
def define_plane_from_normal_and_point(n_vec, o1_vec):
    """
    Define a 2D plane in 3D space given its normal vector in ERMES Cartesian basis

    Args:
        n_vec (array): Normal vector in ERMES Cartesian coordinates.
        o1_vec (array): reference point of n_vec (centre of launcher)

    Returns:
        n (array): Normal vector (need not be normalized)
        u_hat (array): first in-plane basis vector
        v_hat (array): second in-plane basis vector
    """

    n_hat = np.array(n_vec, dtype=float)
    n_hat /= np.linalg.norm(n_hat)

    # Choose a stable reference vector not parallel to n_hat
    ref = np.array([0, 1.0, 0.0])

    # Use Gram–Schmidt to make u_hat perpendicular to n_hat
    u_hat = ref - np.dot(ref, n_hat) * n_hat
    u_hat /= np.linalg.norm(u_hat)

    # u_hat completes the right-handed basis
    v_hat = np.cross(n_hat, u_hat)
    v_hat /= np.linalg.norm(v_hat)

    return n_hat, u_hat, v_hat

def best_fit_plane(beam_xyz, o, v):
    """
    Compute centroid and orthonormal basis vectors of best-fit plane through the central way with an anchor on the centre of the launch port
    
    Args:
        beam_xyz: Beam coords in ERMES Cartesian
        o: Anchor point
        v: launch vector
        
    Returns:
        n_hat: normal vec
        u_hat, v_hat: In plane basis
    """
    v = v/np.linalg.norm(v)
    shifted_beam = beam_xyz - o # Make o the origin so the plane passes through it

    proj = shifted_beam - np.outer(shifted_beam @ v, v)
    _, _, vh = np.linalg.svd(proj)

    # The best-fit normal must be perp to v
    # and approximately perp the main spread in the projected points
    n_hat = vh[-1]
    # Re-orthogonalize to ensure exact orthogonality to v
    n_hat = n_hat - np.dot(n_hat, v) * v
    n_hat = n_hat/np.linalg.norm(n_hat)
    u_hat = v
    v_hat = np.cross(n_hat, u_hat)
    v_hat = v_hat/np.linalg.norm(v_hat)
    return n_hat, u_hat, v_hat

def project_point_onto_plane(r: np.array, n: np.array, o: np.array):
    """
    Project a point to a plane defined by some normal and origin point. Note this does 1 point at a time.

    Args:
        r (array): Point to project
        n (array): Normal vector of plane to project to
        o (array): Origin point of plane to project to (Known point on the plane)

    Returns:
        r_proj (array): Projected point
    """
    return np.array(r - np.dot(r - o, n) * n)

def offset_point_along_plane_normal(point_on_plane, plane_normal, dz):
    """
    Offset a point perpendicular to a plane by a distance dz since we need 1 offset point to extrude the volume in ERMES

    Args:
    point_on_plane (array): Any domain point
    plane_normal (array): Normal vector of the plane
    dz (float): Offset distance (m)

    Returns:
        new_point (array):Offset point
    """
    n_hat = plane_normal / np.linalg.norm(plane_normal)
    new_point = point_on_plane + dz * n_hat
    return np.array(new_point)

def pure_best_fit_plane(beam_xyz):
    """
    This version doesnt care about an anchor k or anchor point. Purely best fit
    Args:
        beam_xyz: Beam coords in ERMES Cartesian
    Returns:
        centroid (array): Of the plane
        n_hat (array): Of the plane
        u_hat, v_hat (array): Of the plane
    """
    centroid = np.mean(beam_xyz, axis=0)
    X = beam_xyz - centroid
    _, _, vh = np.linalg.svd(X)
    n_hat = vh[-1] / np.linalg.norm(vh[-1])
    u_hat = vh[0] / np.linalg.norm(vh[0])
    v_hat = vh[1] / np.linalg.norm(vh[1])
    return centroid, n_hat, u_hat, v_hat

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
        dt=load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-3.0_tor0.0000_rev.h5'),
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
        dt=load_scotty_data('\\Output\\Linear\\scotty_output_2D_linear_freq.h5'),
        prefix="2D_linear_new_",
        #launch_angle=45.0, 
        #launch_freq_GHz=30.0, 
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
    
    # The actual function for analysis
    #"""
    
    #scotty_output_tor_ideal__freq72.5_pol-7.0_tor1.5726_rev.h5
    #scotty_output_freq72.5_pol-7.0_rev.h5
    #n_LS_oblique = [-0.0191506, 0.13483198, 0.9906834]
    dt_1_5726_tor=load_scotty_data('\\Output\\scotty_output_tor_ideal__freq72.5_pol-7.0_tor1.5726_rev.h5')
    dt_0_tor=load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-7.0_tor0.0000_rev.h5')
    dt_0_tor_OMODE=load_scotty_data('\\Output\\scotty_output_OMODE_72.5_pol-7.0_rev.h5')
    dt_1_5726_tor_OMODE=load_scotty_data('\\Output\\scotty_output_IDEALTOROMODE_72.5_pol-7.0_rev.h5')
    
    dt_5_pol_0_tor = load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-5.0_tor0.0000_rev.h5')
    dt_3_pol_0_tor = load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-3.0_tor0.0000_rev.h5')
    
    res_1_5726_tor = "\\Final_Ermes_output\\7_pol_1_5726_tor_3D_new.post.res"
    msh_1_5726_tor = "\\Final_Ermes_output\\7_pol_1_5726_tor_3D_new.post.msh"
    res_0_tor = "\\Final_Ermes_output\\7_pol_0_tor_b.post.res"
    msh_0_tor = "\\Final_Ermes_output\\7_pol_0_tor_b.post.msh"
    
    res_5_pol_0_tor = "\\Final_Ermes_output\\5_pol_0_tor_2D.post.res"
    msh_5_pol_0_tor = "\\Final_Ermes_output\\5_pol_0_tor_2D.post.msh"
    res_3_pol_0_tor = "\\Final_Ermes_output\\3_pol_0_tor_2D.post.res"
    msh_3_pol_0_tor = "\\Final_Ermes_output\\3_pol_0_tor_2D.post.msh"
    
    n_0_tor_2D = np.array([0,0,1])
    n_7_pol_0_tor_LS = np.array([-0.01914288,  0.13482606,  0.99068435])
    n_7_pol_0_tor_b = np.array([0.03274802, -0.17511896,  0.9840025])
    n_7_pol_1_5726_tor_2D_NEW = np.array([-0.02723919, 0, -0.99217786])
    n_7_pol_1_5726_tor_LS_NEW = np.array([0.00499752, -0.17842067,  0.98394161])
    n_7_pol_1_5726_tor_b_NEW = RtZ_to_XYZ(dt_1_5726_tor.analysis.b_hat.values[dt_1_5726_tor.analysis.cutoff_index])
    
    # Cartesian Scotty, likely need to do a lot of bug fixing
    dt_2D_linear=load_scotty_data('\\Output\\Linear\\scotty_output_2D_linear_freq.h5')
    
    res_2D_lin = "\\Final_Ermes_output\\2D_linear_layer_0_001_RME1.post.res"
    msh_2D_lin = "\\Final_Ermes_output\\2D_linear_layer_0_001_RME1.post.msh"
    
    ERMES_results_to_plots(
        res=res_2D_lin,
        msh=msh_2D_lin,
        dt=dt_2D_linear,
        grid_resolution=1e-3,
        normal_vector = n_0_tor_2D,
        plot_blocks=["field_map", "modE_vs_tau", "widths", "transverse_profile"],
        prefix="2D_Linear_Layer_30_pol_15GHz",
        save=False,
        cartesian_scotty=True
    )
    #"""

    # Plotting stuffs to get summary of modE vs tau
    """
    dt=load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-7.0_tor0.0000_rev.h5')
    
    data_3d = np.load(os.getcwd() + "\\analysis_data\\DIII-D_7_pol_0_tor_3D_modE_vs_tau.npz")
    data_2d_pol = os.getcwd() + "\\analysis_data\\DIII-D_7_pol_0_tor_2D_modE_vs_tau.npz"
    data_2d_ls = os.getcwd() + "\\analysis_data\\DIII-D_7_pol_0_tor_LS_modE_vs_tau.npz"
    data_2d_bvec = os.getcwd() + "\\analysis_data\\DIII-D_7_pol_0_tor_b_modE_vs_tau.npz"
    
    COMSOL_data = np.loadtxt("COMSOL_modE.txt", delimiter=",")
    
    COMSOL_distance_along_line = COMSOL_data[:,0]/100 * (0.3/0.32)
    COMSOL_modE = COMSOL_data[:,1]
    
    modE_3d = data_3d["modE_ermes_central"]
    tau_data = data_3d["tau_positions"]
    
    E0 = modE_3d[0] # Use 3D as reference as this shld be most accurate
    modE_theory = calc_Eb_from_scotty(dt=dt, E0=E0)
    colors = ["tab:blue", "black", "tab:green", "tab:red", "tab:orange"]
    labels = ["2D Poloidal", "2D LS Oblique", r"2D $\hat{b}_{cutoff}$ Oblique", "3D", "SCOTTY"]
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    files_2d = [data_2d_pol, data_2d_ls, data_2d_bvec]
    
    # Plot the 2D curves
    for i, file in enumerate(files_2d):
        data = np.load(file)
        modE = data["modE_ermes_central"]
        tau = data["tau_positions"]
        plt.plot(tau, modE, ls = linestyles[i], color=colors[i], lw=2, label=labels[i])

    # Temp plotting of COMSOL data from QUinn (Thank you!)
    plt.plot(COMSOL_distance_along_line, COMSOL_modE, '-', color='tab:purple', lw = 2, label = "2D COMSOL")
    
    # Plot the 3D curve
    plt.plot(tau_data, modE_3d, color=colors[3], lw=2.5, label=labels[3])

    # Plot the SCOTTY paper
    plt.plot(dt.analysis.distance_along_line.values, modE_theory, '-', color=colors[4], lw=2, label=labels[4])

    
    plt.xlim(0, dt.analysis.distance_along_line.values[-1])
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("|E| (A.U.)")
    plt.title(r"$|E|$ vs Distance along central ray. 2D vs 3D Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    #"""
    
    # Getting the oblique planes
    
    """
    dt = load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-7.0_tor0.0000_rev.h5')
    launch_angle_pol = handle_scotty_launch_angle_sign(dt) 
    launch_angle_tor = dt.inputs.toroidal_launch_angle_Torbeam.values
    
    degtorad = pi/180
    
    kx_temp = 1/np.sqrt(tan(launch_angle_pol*degtorad)**2+tan(launch_angle_tor*degtorad)**2)
    ky_temp = tan(launch_angle_pol*degtorad)*kx_temp
    kz_temp = tan(launch_angle_tor*degtorad)*kx_temp
    k_vec_launch_XYZ = np.array([-kx_temp, ky_temp, kz_temp])
    k_vec_launch_XYZ_norm = k_vec_launch_XYZ / np.linalg.norm(k_vec_launch_XYZ) # Just in case
    
    beam_xyz = np.apply_along_axis(RtZ_to_XYZ, arr = dt.analysis.beam_cartesian.values, axis = 1)
    b = RtZ_to_XYZ(dt.analysis.b_hat.values[dt.analysis.cutoff_index])
    b_pos = RtZ_to_XYZ(dt.analysis.beam_cartesian.values[dt.analysis.cutoff_index.values])
    
    # LS
    o_LS = np.array([2.3579195399,   -0.0097362334,    0.0000000000])
    o_b = b_pos
    o_to_use = o_LS
    
    points = [
        [2.3680624694,    0.0691248984,    0.0000000000], # Port
        [2.3486966105,   -0.0885973651,    0.0000000000], # Port
        [2.3690307623,    0.0770110116,    0.0000000000], # Port padding
        [2.3477283175,   -0.0964834783,    0.0000000000], # Port padding
        [2.13, -0.01, 0], # Domain
        [2.13, 0.11, 0], # Domain
        [2.287, 0.206, 0], # Domain
    ]
    
    centroid, LS_plane_normal, u_hat, v_hat = pure_best_fit_plane(beam_xyz)
    
    easy2Dplane_normal = np.cross(k_vec_launch_XYZ_norm, np.array([0,1,0]))
    plane_normal = b
    
    print(plane_normal)
    projected_points = np.array([
        project_point_onto_plane(p, plane_normal, o = o_to_use) for p in points
    ])
    
    print(offset_point_along_plane_normal(projected_points[0], b, 0.0004))
    print(projected_points)
    
    # Create mesh grid for visualizing the plane patch
    u_vals = np.linspace(-1, 1, 10)
    v_vals = np.linspace(-1, 1, 10)
    U, V = np.meshgrid(u_vals, v_vals)

    u_hat = k_vec_launch_XYZ_norm
    v_hat = np.cross(k_vec_launch_XYZ_norm, plane_normal)
    
    # Plane points in 3D
    plane_patch = o_to_use + U[..., None]*u_hat + V[..., None]*v_hat
    X, Y, Z = plane_patch[..., 0], plane_patch[..., 1], plane_patch[..., 2]

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the plane surface
    ax.plot_surface(X, Y, Z, alpha=0.3, color='skyblue', rstride=1, cstride=1, edgecolor='none')

    # Plot projected points
    ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2],
               color='red', s=30, label='Projected Points')

    # Plot launcher origin
    ax.scatter(*o_to_use, color='black', s=80, marker='*', label='Launcher Centre')
    ax.view_init(elev=33, azim=45, roll=123)

    # Plot normal vector
    ax.quiver(*o_to_use, *plane_normal, length=0.1, color='green', linewidth=2, label='Plane normal')

    # Plot in-plane basis vectors
    ax.quiver(*o_to_use, *u_hat, length=0.1, color='blue', linewidth=2, label=r'$\hat{u}$ (in-plane)')
    ax.quiver(*o_to_use, *v_hat, length=0.1, color='red', linewidth=2, label=r'$\hat{v}$ (in-plane)')

    #plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), -beam_plus.sel(col_cart="Y"), "--k", lw=1)
    #plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), -beam_minus.sel(col_cart="Y"), "--k", lw=1, label="Beam width")
    plt.plot(dt.analysis.beam_cartesian.sel(col_cart="X"), dt.analysis.beam_cartesian.sel(col_cart="Z"),  -dt.analysis.beam_cartesian.sel(col_cart="Y"),"-", c='black', lw=1.5, label="Central ray")

    # Formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Oblique Plane Projection and Launcher Anchor")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.show()
    #"""