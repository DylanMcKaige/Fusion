"""
Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 

The shape of the domain here is specifically for DBS simulations.

To use this, from Scotty results, decide on the distance_to_ERMES_port variable (distance from launcher to launch boundary in ERMES). The rest is automatic

Spits out .txt of all the necessary data. 

All paths are w.r.t CWD

TODO
1. Tidy up saving
2. Tidy up ERMES plotting, could probably make it more generalized

References
    [1] Two dimensional full-wave simulations of Doppler back-scattering in tokamak plasmas with COMSOL by Quinn Pratt et al (in-progress paper)
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin, pg 43-44
    [4] V. H. Hall-Chen, F. I. Parra, and J. C. Hillesheim, “Beam model of Doppler backscattering,” Plasma Phys. Control. Fusion, vol. 64, no. 9, p. 095002, Sep. 2022, doi: 10.1088/1361-6587/ac57a1.
    [5] F. M. A. Smits, “ELLIPTICAL POLARISATION FOR OBLIQUE EC-WAVE LAUNCH”.

Written by Dylan James Mc Kaige
Created: 16/5/2025
Updated: 31/8/2025
"""
import os, json, datatree
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sin, cos, tan, acos, atan, sqrt, fabs
from scipy.constants import c, pi, m_e, m_p, elementary_charge, epsilon_0
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, griddata
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scotty.analysis import beam_width
from scotty.plotting import plot_poloidal_crosssection

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

def RtZ_to_XYZ(a: np.array) -> np.array:
    """
    Convert an array in (R,t,Z) to TRUE CARTESIAN (X,Y,Z) for ERMES, keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (R,t,Z) basis
    
    Returns:
        b (array): The transformed vector in (X,Y,Z) basis (ERMES Cartesian)

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
    2) y_hat, b_hat, g_hat are in the (R,t,Z) basis, e_hat is in the (u1,u2,bhat) basis
    3) Equations 104-106 from [4] apply exactly to let us derive u1_hat and u2_hat
    
    Produces an elliptical polarization vector.

    Args:
        dt (DataTree): Scotty output file
        
    Returns:
        e_hat_XYZ (array): Polarization vector in (X,Y,Z) basis
    """
 
    y_hat = dt.analysis.y_hat.values
    g_hat = dt.analysis.g_hat.values
    b_hat = dt.analysis.b_hat.values/np.linalg.norm(dt.analysis.b_hat.values)
    e_hat = dt.analysis.e_hat.values
    u2_hat = y_hat/np.linalg.norm(y_hat)
    u1_hat = (np.cross(np.cross(b_hat, g_hat), b_hat))/np.linalg.norm(np.cross(np.cross(b_hat, g_hat), b_hat))

    # Form the basis transition vector from u1,u2,b to R,t,Z
    uub_to_RtZ_basis = np.column_stack((u1_hat[0],u2_hat[0],b_hat[0]))

    # Pol vector in R,t,Z
    e_hat_uub = e_hat[0]
    e_hat_RtZ = np.dot(uub_to_RtZ_basis, e_hat_uub)
    e_hat_RtZ = e_hat_RtZ/np.linalg.norm(e_hat_RtZ)
    e_hat_XYZ = RtZ_to_XYZ(e_hat_RtZ)
    
    return e_hat_XYZ

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
        psi_normal (array): Outward normal vector at entry of LCFS
    """
    dpsi_dR = psi_spline.ev(R_entry, Z_entry, dx=1, dy = 0)
    dpsi_dZ = psi_spline.ev(R_entry, Z_entry, dx=0, dy = 1)
    
    psi_normal = np.array([dpsi_dR, dpsi_dZ, 0])
    return psi_normal

def get_pol_from_smits(k_vec: np.array, B_entry_vec_XYZ: np.array, B_entry_vec_RtZ, launch_freq_GHz: float, E0: float):
    """
    Get the polarization vector and values of E_perp and E_par using Smits [5]
    
    Args:
        k_vec (np.array): k vector 
        B_entry_vec_XYZ (np.array): B vector at point of entry
        B_entry_vec_RtZ (np.array): B vector at point of entry
        launch_freq_GHz (float): Launch frequency in GHz
        E0 (float): E0 value

    Returns:
        rho_hat (array): Polarization vector in XYZ basis
        mod_E_par (float): Mod E_par in ERMES
        mod_E_perp (float): Mod E_per in ERMES
        rho_hat_rotated_set (array): Array of [pol theta, pol vector]
    """
    # Normalize to make life simpler
    k_vec_hat = k_vec/np.linalg.norm(k_vec)
    B_entry_vec_XYZ_hat = B_entry_vec_XYZ/np.linalg.norm(B_entry_vec_XYZ)
    
    # Angle between k and b: Pitch angle
    theta = acos(np.dot(k_vec_hat, B_entry_vec_XYZ_hat))
    
    # c from [5]
    C = (1.6e-19*np.linalg.norm(B_entry_vec_XYZ) / 9.11e-31) / (2*pi*launch_freq_GHz*1e9)
    p_prime = sqrt(sin(theta)**4 + 4*cos(theta)**2 / C**2)
    
    # Angle of rho above kb plane and perp to k for QX-mode
    gamma = atan(-2*cos(theta) / (C*(sin(theta)**2 + p_prime)) ) # Gamma from theta from [5] (best coupling)
    
    # Get the rho unit vector that is perp to k and b. This is our initial & supposedly ideal pol ( It is not, seems to be ~20deg off )
    rho_hat_perp = np.cross(k_vec_hat, B_entry_vec_XYZ_hat) 

    # Get Bp, Bt at entry
    B_t_entry = B_entry_vec_RtZ[1]
    B_p_entry = sqrt(B_entry_vec_RtZ[0]**2 + B_entry_vec_RtZ[2]**2)
    pitch_angle_entry = abs(atan(B_p_entry/B_t_entry))
    
    print("pitch angle at entry:", pitch_angle_entry*180/pi)
    print("kb angle at entry: ", theta*180/pi)
    
    # Smits polarization tilt
    phi = acos(cos(pitch_angle_entry)/sin(theta))
    print("Smits pol tilt angle: ", phi*180/pi)
    # Hypothesized ideal pol angle
    print("Hypthesis pol angle: ", (phi+pitch_angle_entry)*180/pi) # abs cus it's cos
    
    # To test for different pol vectors
    theta_rho = np.array([phi+pitch_angle_entry])
    #theta_rho = np.arange(start = -pi/2, stop = pi/2, step = pi/18) # 10 deg increments

    rho_hat_rotated = rotate_rodrigues(rho_hat_perp, k_vec_hat, theta_rho)
    print(rho_hat_rotated)
    mod_E_rho = np.sqrt(E0**2 / (1 + np.tan(gamma)**2))# rho_hat_kb
    mod_E_eta = np.sqrt(E0**2 - mod_E_rho**2) # eta_hat_kb
    
    # Check if rho perp k and perp b
    print("rho dot k: ", np.dot(rho_hat_rotated,k_vec_hat))
    print("rho dot b: ", np.dot(rho_hat_rotated,B_entry_vec_XYZ_hat))

    return rho_hat_perp, mod_E_rho, mod_E_eta, np.column_stack((theta_rho*180/pi, rho_hat_rotated)), rho_hat_rotated

def get_limits_from_scotty(dt: datatree, padding: float = 0.01):
    """
    Get the min and max R, Z, t from Scotty by adding padding to tor and pol width

    Args:
        dt (datatree): Scotty output file in .h5 format

    Returns:
       lims (array): In the form [[x, x, y, y, z, z]]
    """
    width_tor = beam_width(dt.analysis.g_hat_Cartesian,np.array([0.0, 0.0, 1.0]),dt.analysis.Psi_3D_Cartesian)
    width_pol = beam_width(dt.analysis.g_hat,np.array([0.0, 1.0, 0.0]),dt.analysis.Psi_3D)
    beam_plus_tor, beam_minus_tor = dt.analysis.beam_cartesian + width_tor, dt.analysis.beam_cartesian - width_tor
    beam_plus_pol, beam_minus_pol = dt.analysis.beam + width_pol, dt.analysis.beam - width_pol
    combined_beam_X = np.concatenate([beam_plus_tor.sel(col_cart="X"), beam_minus_tor.sel(col_cart="X")])
    combined_beam_Y = np.concatenate([beam_plus_tor.sel(col_cart="Y"), beam_minus_tor.sel(col_cart="Y")])
    combined_beam_R = np.concatenate([beam_plus_pol.sel(col="R"), beam_minus_pol.sel(col="R"), combined_beam_X])
    combined_beam_Z = np.concatenate([beam_plus_pol.sel(col="Z"), beam_minus_pol.sel(col="Z")])

    # In GiD frame, accounting for sign 
    min_x, max_x = np.sign(np.min(combined_beam_R))*(abs(np.min(combined_beam_R)) + padding), np.sign(np.max(combined_beam_R))*(abs(np.max(combined_beam_R)) + padding)
    min_y, max_y = np.sign(np.min(combined_beam_Z))*(abs(np.min(combined_beam_Z)) + padding), np.sign(np.max(combined_beam_Z))*(abs(np.max(combined_beam_Z)) + padding)
    min_z, max_z = np.sign(np.min(combined_beam_Y))*(abs(np.min(combined_beam_Y)) + padding), np.sign(np.max(combined_beam_Y))*(abs(np.max(combined_beam_Y)) + padding) # min_z would be negative so it points into the page
    
    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])

# TO BE DEPRECATED
def process_scotty_input_data(
    ne_path: str,
    topfile_path: str,
    filename: str,
    dt: datatree.DataTree = None,
    plot: bool = True,
    save: bool = True,
    path: str = os.getcwd() + '\\',
    save_path: str = os.getcwd() + '\\',
    ERMES_R = None,
    ERMES_Z = None,
    ERMES_port = None,
    ERMES_launch_centre = None,
    launch_angle_rad = None,
    num_RZ = 100
    ):
    """
    Loads ne and topfile data from the given .dat and .json files and saves ne, Br, Bt, Bz as a function of R & Z in a .csv file. Also plots for sanity check against Scotty

    Args:
        ne_path (str): Relative (to cwd) paht of ne.dat file
        topfile_path (str): Relative (to cwd) path of topfile.json file
        filename (str): Filename for saving
        dt (DataTree): Scotty output file in .h5 format
        plot (bool): Plot the data
        save (bool): Save the data
        path (str): Path to save file in, defaults to cwd
        ERMES_R (tuple): Range of R to zoom in and save ne of defaults to full range from topfile
        ERMES_Z (tuple): Range of Z to zoom in and save ne of defaults to full range from topfile
        ERMES_port (tuple): Coordinates of port to check if it is within the plasma 
        ERMES_launch_centre (tuple): Coordinates of centre of launch point in ERMES
        launch_angle_rad (float): Launch angle in rad w.r.t -ve R axis
        num_RZ (int): Number of RZ points for ERMES
        
    Returns:
        entry_point (array): Coordinates of point of entry defined by intersection of the last closed flux surface and the initial launch beam
        B_entry_vec_RtZ (array): B field at entry point of beam into plasma
        Plots ne in R,Z space w.r.t pol flux and saves a .csv files of ne, R, Z, Br, Bt, Bz for ERMES (Need to be converted using fullwavedensfile.py & fullwavemagfile.py)
    """
    
    # Load in ne data and spline it
    ne_path = path + ne_path
    df = pd.read_csv(ne_path, sep=' ', header=None, skiprows=1)
    ne_data = df.to_numpy(dtype=float).T
    ne_spline = UnivariateSpline(ne_data[0]**2, ne_data[1], s =  0, ext=1) # Squared cus input data is in sqrt(psi_p) convention following TORBEAM

    # Load in topfile data and spline the flux
    topfile_path = path + topfile_path
    with open(topfile_path, 'r') as file:
        topfile_data = json.load(file)
    topfile_arrays = {key: np.array(value) for key, value in topfile_data.items()}
    
    R = np.asarray(topfile_arrays['R'])
    Z = np.asarray(topfile_arrays['Z'])
    Br = np.asarray(topfile_arrays['Br'])
    Bt = np.asarray(topfile_arrays['Bt'])
    Bz = np.asarray(topfile_arrays['Bz'])
    pol_flux_RZ = np.asarray(topfile_arrays['pol_flux'])
    
    R_grid, Z_grid = np.meshgrid(R, Z)
    pol_flux_spline = RectBivariateSpline(R, Z, pol_flux_RZ.T)
    pol_flux_vals = pol_flux_spline.ev(R_grid, Z_grid)

    ne_vals_total = ne_spline(pol_flux_vals)

    # Save only the necessary range for ERMES
    R_range = np.linspace(ERMES_R[0], ERMES_R[1], num_RZ)
    Z_range = np.linspace(ERMES_Z[0], ERMES_Z[1], num_RZ)
    R_grid_to_ERMES, Z_grid_to_ERMES = np.meshgrid(R_range, Z_range)

    pol_flux_vals_to_ERMES = pol_flux_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    ne_vals_to_ERMES = ne_spline(pol_flux_vals_to_ERMES)

    Br_spline = RectBivariateSpline(R, Z, Br.T)
    Bt_spline = RectBivariateSpline(R, Z, Bt.T)
    Bz_spline = RectBivariateSpline(R, Z, Bz.T)
    
    Br_vals_ERMES = Br_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    Bt_vals_ERMES = Bt_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    Bz_vals_ERMES = Bz_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    
    # Get cut-offs and resonances
    launch_freq_GHz = 72.5
    w = 2*pi*launch_freq_GHz*1e9
    w_ce = elementary_charge*np.sqrt(Br_vals_ERMES**2 + Bt_vals_ERMES**2 + Bz_vals_ERMES**2)/m_e # (Z,R)
    w_pe = np.sqrt(ne_vals_to_ERMES*elementary_charge**2 / (epsilon_0*m_e)) # (Z,R)
    
    # assuming it's a H1 plasma
    w_ci = elementary_charge*np.sqrt(Br_vals_ERMES**2 + Bt_vals_ERMES**2 + Bz_vals_ERMES**2)/m_p # (Z,R)
    w_pi = np.sqrt(ne_vals_to_ERMES*elementary_charge**2 / (epsilon_0*m_p)) # (Z,R)
    
    w_UH = np.sqrt(w_pe**2 + w_ce**2) - w
    w_LH = np.sqrt(w_ce**2 * w_pi**2 / (w_ce**2 + w_pe**2)) - w
    
    w_R = np.sqrt(w_pe**2 + w_pi**2 + (w_ci + w_ce)**2 / 4) - (w_ci - w_ce)/2 - w
    
    # Get the line from launcher to plasma then find the point where ne goes from 0 to +ve, this is the point of entry of the beam into the plasma
    # Only used if no Scotty output file used
    if dt is None:
        entry_point = find_lcfs_entry_point(
            pol_flux_spline,
            R0=ERMES_launch_centre[0],
            Z0=ERMES_launch_centre[1],
            launch_angle_rad=launch_angle_rad,
            psi_closed=ne_data[0,-1],
            
        )
    else:
        entry_point = [dt.inputs.initial_position.values[0], dt.inputs.initial_position.values[2]]
    
    psi_normal_at_entry = get_psi_normal_entry(pol_flux_spline, entry_point[0], entry_point[1]) # Not needed

    B_entry_vec_RtZ = np.array([
        Br_spline.ev(entry_point[0], entry_point[1]),
        Bt_spline.ev(entry_point[0], entry_point[1]),
        Bz_spline.ev(entry_point[0], entry_point[1])
        ]
    )
    
    RZ_ERMES = np.column_stack((R_range, Z_range))

    if save:
        np.savetxt(save_path + filename + "RZ_ERMES.csv", RZ_ERMES, delimiter=",", fmt="%.6e")

        np.savetxt(save_path + filename + "Br_ERMES.csv", Br_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt(save_path + filename + "Bz_ERMES.csv", Bz_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt(save_path + filename + "Bt_ERMES.csv", Bt_vals_ERMES.T, delimiter=",", fmt="%.6e")

        np.savetxt(save_path + filename + "ne_ERMES.csv", ne_vals_to_ERMES.T, delimiter=",", fmt="%.6e")
    
    if plot:
        w_levels = [0]#np.linspace(-0.01*w, 0.01, 10, endpoint=True)
        
        plt.figure(figsize=(8, 6))

        cf = plt.contourf(R_grid, Z_grid, ne_vals_total*1e19, levels=100, cmap='plasma')
        pol_flux_levels = np.linspace(0.0, 1.0, 11)
        cs = plt.contour(R_grid, Z_grid, pol_flux_vals, levels=pol_flux_levels, colors='black', linewidths=0.8)
        
        plt.clabel(cs, fmt=r'%.1f', fontsize=8, colors='black')
        plt.colorbar(cf, label=r'$n_e\ (10^{19}\ \mathrm{m}^{-3})$')
        plt.xlabel("R [m]")
        plt.ylabel("Z [m]")
        plt.title(r'Heatmap of $n_e(R,Z)$')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        cf = plt.contourf(R_grid_to_ERMES, Z_grid_to_ERMES, ne_vals_to_ERMES*1e19, levels=100, cmap='plasma')
        pol_flux_levels = np.linspace(0.0, 1.0, 11)
        cs = plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, pol_flux_vals_to_ERMES, levels=pol_flux_levels, colors='black', linewidths=0.8)
        #plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_R, levels = w_levels, colors='white', linewidths=2)
        #plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_UH, levels = w_levels, colors='blue', linewidths=2)
        #plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_LH, levels = w_levels, colors='green', linewidths=2)
        plt.clabel(cs, fmt=r'%.1f', fontsize=8, colors='black')
        plt.scatter(*zip(*ERMES_port), s = 2, color='white')
        plt.scatter(*zip(entry_point), s = 2, color='white')

        plt.colorbar(cf, label=r'$n_e\ (10^{19}\ \mathrm{m}^{-3})$')
        plt.xlabel("R [m]")
        plt.ylabel("Z [m]")
        plt.xlim(ERMES_R[0], ERMES_R[1])
        plt.ylim(ERMES_Z[0], ERMES_Z[1])
        plt.title(r'Heatmap of $n_e(R,Z)$')
        
        ax = plt.gca()
        ax.use_sticky_edges = True
        ax.set_aspect('equal')
        plt.show()
        
    return entry_point, B_entry_vec_RtZ, psi_normal_at_entry

def get_ERMES_parameters(
    dt: datatree.DataTree = None,
    suffix: str = "",
    launch_angle: float = None,
    launch_freq_GHz: float = None,
    launch_position: np.array = None,
    launch_beam_curvature: float = None,
    launch_beam_width: float = None,
    port_width: float = None,
    dist_to_ERMES_port: float = None,
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    ):
    """
    Generate ERMES parameters for given input
    
    Args:
        dt (DataTree): Scotty output file in .h5 format
        suffix (str): Suffix for naming (e.g MAST-U, DII-D, etc), defaults to None
        launch_angle (float): Launch angle in degrees, w.r.t -ve R axis
        launch_freq_GHz (float): Launch frequency in GHz
        port_width (float): Width of port in ERMES in m, so far seems to be arbitrary
        launch_position (array): Launch position in [R,t,Z] coordinates in m
        launch_beam_curvature (float): Curvature of beam at launch position, in m^{-1}
        launch_beam_width (float): Width of beam at launch position, in m
        dist_to_ERMES_port (float): Distance from launcher to port in ERMES in m, stare at Scotty to decide this
        plot (bool): Plot everything
        save (bool): Save everything
        path (str): Path to save file in, defaults to cwd
        
    Returns:
        Plots the position of the required points in the Z-R axes and saves a .txt file of the necessary values for ERMES
        Optionally generates the necessary ne, R and Z files for ERMES
    """
    degtorad = pi/180
    
    if dt is None:
        # Take inputs
        launch_R = launch_position[0]
        launch_Z = launch_position[2]
    else:
        # Else, take from Scotty
        launch_freq_GHz = dt.inputs.launch_freq_GHz.values
        launch_beam_width = dt.inputs.launch_beam_width.values
        launch_beam_curvature = dt.inputs.launch_beam_curvature.values
        launch_R = dt.inputs.launch_position.values[0]
        launch_Z = dt.inputs.launch_position.values[2]
        
    launch_beam_wavelength = c/(launch_freq_GHz*1e9)
    radius_of_curv = fabs(1/launch_beam_curvature)
    launch_angle_rad = launch_angle*degtorad
    
    filename = suffix + str(launch_angle) + "_degree_" + str(launch_freq_GHz) + "GHz_"
    
    # Create subdirectory for saving:
    if save:
        if os.path.isdir(path + filename + 'folder'):
            path = path + filename + 'folder' + '\\'
        else:
            os.makedirs(path + filename + 'folder')
            path = path + filename + 'folder' + '\\'
    
    # Initial calculations
    distance_to_launcher = fabs((radius_of_curv * pi**2 * launch_beam_width**4)/(launch_beam_wavelength**2 * radius_of_curv**2+pi**2 * launch_beam_width**4))
    z_R = fabs((launch_beam_wavelength*radius_of_curv*distance_to_launcher)/(pi*launch_beam_width**2))
    w0 = sqrt((launch_beam_wavelength*z_R)/(pi))
    
    w_ERMES = 2*w0*sqrt(1+(dist_to_ERMES_port/z_R)**2) # Width of beam at port position in ERMES
    kx_norm = -cos(launch_angle_rad) # Normalized k vector, -ve because leftwards by definition of geometry
    ky_norm = sin(launch_angle_rad) # Normalized k vector 
    xw = launch_R - distance_to_launcher*cos(launch_angle_rad) # Centre of waist
    yw = launch_Z + distance_to_launcher*sin(launch_angle_rad) # Centre of waist
    
    # This formula is from Quinn's thesis
    z0 = 377 # Impedance of free space
    E0 = sqrt(z0*2*1/(w0*sqrt(pi/2))) # For P_in = 1 W/m in 2D
    
    # Port calculations
    xp, yp = launch_R - dist_to_ERMES_port*cos(launch_angle_rad), launch_Z + dist_to_ERMES_port*sin(launch_angle_rad) # Centre of front face
    xp0, yp0 = xp - w_ERMES/2*sin(launch_angle_rad), yp - w_ERMES/2*cos(launch_angle_rad) # Bottom
    xp1, yp1 = xp + w_ERMES/2*sin(launch_angle_rad), yp + w_ERMES/2*cos(launch_angle_rad) # Top
    
    # Domain calculations
    min_x, max_x, min_y, max_y, min_z, max_z = get_limits_from_scotty(dt, padding = 0.02)
    # Bottom right (closest to port) first
    x_br, y_br = xp0, min(yp0, min_y)
    x_bl, y_bl = min_x, y_br
    x_tl, y_tl = min_x, max_y
    x_tr, y_tr = xp1, max_y
    port_delta_z = w_ERMES/2
        
    # To be deprecated once I test a bit more to confirm things work
    """
    # For COMSOL
    xp01, yp01 = xp0 + port_width*cos(launch_angle_rad), yp0 - port_width*sin(launch_angle_rad)
    xp11, yp11 = xp1 + port_width*cos(launch_angle_rad), yp1 - port_width*sin(launch_angle_rad)
    
    # Domain calculations
    # Lazy fix for -ve Launch Angles
    if np.sign(launch_angle) == 1:
        # Arbitrary padding for plotting in Scotty
        xd1, yd0 = xp11 + 0.005, yp01 - 0.005 
        xd0, yd1 = xd1-domain_size, yd0+domain_size
        
        # More slimmed down with an initial guess, within bounds of original domain so no new ne or B need to be calculated. 4 sided figure with xpyp being bottom right, then go clockwise.
        trimmed_xd0 = xd0
        trimmed_yd0 = yp0 + (xp0-trimmed_xd0)*tan(launch_angle_rad)
        trimmed_xd2 = xp1
        trimmed_yd2 = yd1
        trimmed_yd1 = trimmed_yd2
        trimmed_xd1 = trimmed_xd0 + (trimmed_yd1 - trimmed_yd0)*tan(launch_angle_rad)
        trimmed_xd3 = trimmed_xd2
        trimmed_yd3 = trimmed_yd1 - (trimmed_xd2 - trimmed_xd1)*tan(launch_angle_rad)
        
        ERMES_Z=(yd0, yd1)
    else:
        # Arbitrary padding for plotting in Scotty
        xd1, yd0 = xp01 + 0.005, yp11 + 0.005 
        xd0, yd1 = xd1-domain_size, yd0-domain_size 
        
        # More slimmed down with an initial guess, within bounds of original domain so no new ne or B need to be calculated. 4 sided figure with xpyp being bottom right, then go clockwise.
        trimmed_xd0 = xd0
        trimmed_yd0 = yp0 - (xp0-trimmed_xd0)*tan(launch_angle_rad)
        trimmed_xd2 = xp0
        trimmed_yd2 = yd1
        trimmed_yd1 = trimmed_yd2
        trimmed_xd1 = trimmed_xd0 + (trimmed_yd1 - trimmed_yd0)*tan(launch_angle_rad)
        trimmed_xd3 = trimmed_xd2
        trimmed_yd3 = trimmed_yd1 - (trimmed_xd2 - trimmed_xd1)*tan(launch_angle_rad)
        
        ERMES_Z=(yd1, yd0)
    """
    # To be deprecated
    # Convert Scotty input files into RZ format (& Do some sanity plots)
    #entry_point, B_entry_vec_RtZ, psi_normal_at_entry = process_scotty_input_data(
    #    ne_path, 
    #    topfile_path, 
    #    filename = filename,
    #    plot=plot,
    #    save=save,
    #    dt=dt,
    #    ERMES_R=(xd0, xd1), 
    #    ERMES_Z=ERMES_Z, 
    #    ERMES_port=([xp0, yp0], [xp1, yp1], [xp01, yp01], [xp11, yp11]), 
    #    ERMES_launch_centre=(xp, yp), 
    #    launch_angle_rad=launch_angle_rad,
    #    num_RZ=num_RZ,
    #    save_path=path
    #    )
    
    entry_point = [dt.inputs.initial_position.values[0], dt.inputs.initial_position.values[2]]
    B_entry_vec_RtZ = np.array([dt.analysis.B_R.values[0], dt.analysis.B_T.values[0], dt.analysis.B_Z.values[0]])
    B_entry_vec_XYZ = RtZ_to_XYZ(B_entry_vec_RtZ)
    
    # linear pol vec at launch point = beam k X B field at launch point in (X,Y,Z)
    k_vec = np.array([kx_norm, ky_norm, 0])
    lin_pol_vec = np.cross(k_vec, B_entry_vec_XYZ)
    
    # Elliptical pol vector from Scotty
    if dt is not None: ellip_pol_vec = get_pol_from_scotty(dt)
    else: ellip_pol_vec = np.array([0, 0, 0])
    
    # Polarization from Smits
    rho_hat_perp, mod_E_par, mod_E_perp, rho_hat_rotated_set, rho_hat_rotated = get_pol_from_smits(k_vec, B_entry_vec_XYZ, B_entry_vec_RtZ, launch_freq_GHz, E0)

    # Arrays for saving
    # Surely there is a neater way of doing this, but I'm lazy and want to get this working first before making it pretty
    points_x = np.array([
        xp, 
        xp0, 
        xp1, 
        x_br,
        x_bl,
        x_tr,
        x_tl,
        port_delta_z,
        min_z,
        max_z,
        launch_R, 
        xw,
        entry_point[0],
        ]
    )
    points_y = np.array([
        yp, 
        yp0, 
        yp1, 
        y_br,
        y_bl,
        y_tr,
        y_tl,
        0,
        0,
        0,
        launch_Z,
        yw,
        entry_point[1],
        ]
    )
    points_names = np.array([
        'Source Position (front face of port)    ', 
        'Port BL    ', 
        'Port TL    ', 
        'Domain BR    ',
        'Domain BL    ',
        'Domain TR    ',
        'Domain TL    ',
        '3D: Port delta z    ',
        '3D: min z    ',
        '3D: max z    ',
        'Launch Position    ', 
        'Waist Position    ',
        'Point of Entry    ',
        ]
    )

    # Beam params
    params_val = np.array([
        launch_angle, 
        launch_beam_width, 
        radius_of_curv, 
        distance_to_launcher, 
        dist_to_ERMES_port, 
        w0, 
        z_R, 
        port_width, 
        launch_freq_GHz, 
        launch_beam_wavelength, 
        kx_norm, 
        ky_norm, 
        rho_hat_rotated[0][0],
        rho_hat_rotated[0][1],
        rho_hat_rotated[0][2],
        E0, 
        mod_E_par,
        mod_E_perp,
        ]
    )
    params_names = np.array([
        'Launch Angle    ', 
        'Launch Beam Width    ', 
        'launch Beam Radius of Curvature    ', 
        'Distance to Launcher (from waist)    ', 
        'Distance to ERMES Port (from launcher)    ', 
        'Beam Waist (w0)    ', 
        'Rayleigh Length (m)    ', 
        'Port Width    ', 
        'Beam Frequency (GHz)    ', 
        'Beam Wavelength (m)    ', 
        'kx (normalized)    ', 
        'ky (normalized)    ', 
        'polx (normalized)    ', 
        'poly (normalized)    ', 
        'polz (normalized)    ', 
        'E0    ', 
        'E par (ERMES)    ',
        'E perp (ERMES)    ',  
        ]
    )
    
    # Save it!
    if save:
        np.savetxt(path + filename + 'ERMES_params', np.array([points_names, points_x, points_y], dtype=object).T, delimiter=' ', header = 'Cartesian points in ERMES', fmt='%s')
        # There's probably a better way to do this 
        with open(path + filename + 'ERMES_params', 'a') as file:
            np.savetxt(file, np.array([params_names, params_val], dtype=object).T, delimiter=' ', header='Beam Params', fmt = '%s')
        #with open(path + filename + 'ERMES_params', 'a') as file:
        #    np.savetxt(file, np.round(rho_hat_rotated_set, 1), header='Rotated Polarization Vector', fmt = '%s')

    if plot:
        plt.scatter(points_x, points_y, s = 2)
        plt.xlim(left=x_bl-0.01)
        plt.gca().set_aspect('equal')
        plt.show()

def calc_Eb_from_scotty(dt: datatree, E0: float = 1.0, wx: float = 0.0, wy: float = 0.0):
    """
    Calcualte the probe beam Electric Field amplitude along the central ray using Eqn 33 of [4]
    Returns Eb(R, zeta, Z) where the coordinates (R, zeta, Z) are determined by tau

    Args:
        dt (datatree): Scotty output file
        E0 (float): For scaling
        wx (float): width in xhat direction
        wy (float): width in yhat direciton
        
    Returns:
        Eb_tau (array): |Eb| at all the points along the ray w.r.t tau
    """
    # Naming convention, end in tau means index in terms of tau, RtZ means in RtZ basis, xyg means in xyg basis, neither means it is an element (likely), 
    # w means in plane perp to g, g means projected onto g
    # i means im component, launch and ant are equiv
    tau_len = dt.inputs.len_tau.values
    tau_values = dt.analysis.tau.values
    
    x_hat_RtZ_tau = dt.analysis.x_hat.values
    y_hat_RtZ_tau = dt.analysis.y_hat.values
    g_hat_RtZ_tau = dt.analysis.g_hat.values
    g_mag_tau = dt.analysis.g_magnitude.values
    
    R_tau=dt.analysis.q_R.values
    Z_tau=dt.analysis.q_Z.values
    zeta_tau=dt.analysis.q_zeta.values
    beam_trajectiry_RtZ_tau = dt.analysis.beam.values # Vector made of the prev 3
    
    Psi_xx_tau = dt.analysis.Psi_xx.values
    Psi_xy_tau = dt.analysis.Psi_xy.values
    Psi_yy_tau = dt.analysis.Psi_yy.values
    iPsi_xx_tau=np.imag(Psi_xx_tau)
    iPsi_xy_tau=np.imag(Psi_xy_tau)
    iPsi_yy_tau=np.imag(Psi_yy_tau)
    
    # Rotation/Projection matrices, requires the input to be in RtZ basis
    P_RtZ_to_xyg = np.stack([x_hat_RtZ_tau, y_hat_RtZ_tau, g_hat_RtZ_tau], axis = 1)
    
    def vec_RtZ_to_xyg_tau(vec):
        """
        Transform a vector in RtZ basis to xyg basis

        Args:
            vec (array): The vector to transform
            
        Returns:
            vec (array): The transformed vector
        """
        return np.einsum('nij,nj->ni', P_RtZ_to_xyg, vec)
    
    def mat_RtZ_to_xyg_tau(mat, index = 0):
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
    
    def mat_to_plane_perp_to_g_tau(mat, ghat):
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
    Psi_w_xyg_tau = np.zeros((tau_len, 3, 3), dtype=np.complex64) # Such that Psi_w_xyg_tau(N) is the Nth Psi_w corresponding to the Nth tau in xyg basis
    Psi_w_xyg_tau[:, 0, 0] = Psi_xx_tau
    Psi_w_xyg_tau[:, 0, 1] = Psi_xy_tau
    Psi_w_xyg_tau[:, 1, 0] = Psi_xy_tau
    Psi_w_xyg_tau[:, 1, 1] = Psi_yy_tau
    
    Psi_3D_ant_RtZ = dt.analysis.Psi_3D_lab_launch
    g_hat_ant_RtZ = g_hat_RtZ_tau[0]
    Psi_w_ant_RtZ = mat_to_plane_perp_to_g_tau(Psi_3D_ant_RtZ, g_hat_ant_RtZ)
    Psi_w_ant_xyg = mat_RtZ_to_xyg_tau(Psi_w_ant_RtZ)
    
    # Get w. w = w_x xhat + w_y y_hat in (wx,wy,0) xhat, yhat, ghat
    """
    # To be deprecated since along central ray, w = 0
    kappa_dot_xhat = dt.analysis.kappa_dot_xhat.values
    kappa_dot_yhat = dt.analysis.kappa_dot_yhat.values
    d_xhat_d_tau_dot_yhat = dt.analysis.d_xhat_d_tau_dot_yhat.values
    
    wx_0 = wx
    wy_0 = wy
    wx_tau = np.zeros((tau_len,))
    wy_tau = np.zeros((tau_len,))
    wx_ghat_factor_tau = ( wy_tau/(g_mag_tau*(1 - (wx_tau*kappa_dot_xhat + wy_tau*kappa_dot_yhat))) ) * (d_xhat_d_tau_dot_yhat)
    wy_ghat_factor_tau = ( wx_tau/(g_mag_tau*(1 - (wx_tau*kappa_dot_xhat + wy_tau*kappa_dot_yhat))) ) * (d_xhat_d_tau_dot_yhat)
    grad_wx_tau = x_hat_RtZ_tau + g_hat_RtZ_tau * wx_ghat_factor_tau[:, np.newaxis] # Eqn D.3
    grad_wy_tau = y_hat_RtZ_tau - g_hat_RtZ_tau * wy_ghat_factor_tau[:, np.newaxis] # Eqn D.4
    d_r_d_tau = np.gradient(beam_trajectiry_RtZ_tau, tau_values, axis = 0)
    grad_wx_dot_d_r_d_tau = np.einsum('ij,ij->i', grad_wx_tau, d_r_d_tau)
    grad_wy_dot_d_r_d_tau = np.einsum('ij,ij->i', grad_wy_tau, d_r_d_tau)
    
    wx_tau = wx_0 + cumulative_trapezoid(grad_wx_dot_d_r_d_tau, tau_values, initial = 0.0)
    wy_tau = wy_0 + cumulative_trapezoid(grad_wy_dot_d_r_d_tau, tau_values, initial = 0.0)
    
    w_xyg_tau = np.array([wx_tau, wy_tau, np.zeros(tau_len)]).T
    #w_ant_xyg = w_xyg_tau[0]
    # wPsiw pieces
    def quad_form(vec, mat):
    
        Calcualte the quadratic form vec dot mat dot vec

        Args:
            vec (array): Nx1 vec
            mat (array): NxN mat
            
        Returns:
            Scalar or list of Scalars of len N

        if np.ndim(vec) == 1:
            return np.dot(vec, np.dot(mat, vec))
        else:
            return np.einsum('ni,nij,nj->n', vec, mat, vec)
    
    w_dot_Psi_w_ant_dot_w_piece = np.abs(np.exp( 1j/2 * quad_form(w_ant_xyg, Psi_w_ant_xyg) ))
    w_dot_Psi_w_dot_w_piece = np.abs(np.exp( 1j/2 * quad_form(w_xyg_tau, Psi_w_xyg_tau) ))
    """
    
    # 4th root piece (det_piece)
    det_im_Psi_w = iPsi_xx_tau*iPsi_yy_tau-iPsi_xy_tau**2 # Eqn A.67 from [4]
    det_im_Psi_w_ant = np.imag(Psi_w_ant_xyg[0,0])*np.imag(Psi_w_ant_xyg[1,1])-np.imag(Psi_w_ant_xyg[0,1])*np.imag(Psi_w_ant_xyg[1,0]) # Eqn A.67 from [4]
    det_piece = (det_im_Psi_w/det_im_Psi_w_ant)**0.25
    
    # g_piece
    g_mag_ant = 2*c/(2*pi*dt.inputs.launch_freq_GHz.values*1e9) # Eqn 195
    g_piece = (g_mag_ant/g_mag_tau)**0.5
    
    # Finally, calculate |E_b|
    Eb_tau = det_piece*g_piece#*w_dot_Psi_w_dot_w_piece
    
    # A_ant piece, defined based off of first E in ERMES (make them equal)
    A_ant = E0/(Eb_tau[0])
    
    # Normalize to equate the first point
    Eb_tau = A_ant*Eb_tau
    
    return Eb_tau

def ERMES_nodes_to_xyz(msh_file: str):
    """
    Load in the .msh file to read each node as nodeID and return the cartesian cooridnates of that node in xyz. 
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
    Gert the moving RMS of a dataset, used for smaller angles as the data becomes oscillatory.

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

def ERMES_results_to_plots(res: str = None, msh: str = None, dt: datatree = None, plot: bool = False, save: bool = True, grid_resolution: float = 8e-4):
    """
    Plot ERMES modE in R,Z with Scotty overlaid, Plot ERMES modE, rE vec, transverse modE along central ray with Scotty calculations where applicable.
    
    Args:
        res (str): path to the .res file
        msh (str): path to the .msh file
        dt (datatree): Scotty output file
        plot (bool): Plot it!
    """
    # Node ID as XYZ coords
    node_to_xyz = ERMES_nodes_to_xyz(msh_file=msh)
        
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
    beam_xyz = np.column_stack([dt.analysis.q_R.values,dt.analysis.q_Z.values, np.zeros(tau_len)])

    # Extract vecE and modE along central ray within tolerance
    tol = grid_resolution/2 # Half of mesh size
    
    modE_list = []
    data_xy = modE_xyz[1:, :2]
    modE_vals = modE_xyz[1:, 3]
    
    # Extract vecE along central ray within tolerance
    vecE_list = []
    data_xy = vecE_xyz[1:, :2]
    vecE_vecs = vecE_xyz[1:, 3:6]
    
    # to get only modE along the central ray
    for xq, yq in beam_RZ:
        dx = np.abs(data_xy[:, 0] - xq)
        dy = np.abs(data_xy[:, 1] - yq)
        mask = (dx <= tol) & (dy <= tol)

        if np.any(mask):
            modE_list.append(modE_vals[mask][0]) # First match
            vecE_list.append(vecE_vecs[mask][0]) # First match
        else:
            modE_list.append(0)
            vecE_list.append(np.array([0.0, 0.0, 0.0]))

    vecE_array_beam = np.array(vecE_list) # convert to array
    
    # Get beam width for gaussian x-section
    tau_vals = np.arange(len(beam_RZ))
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
        print("Plotting modE in R,Z from ERMES and Scotty")
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
        c = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.title(r"Results for ERMES & Scotty, $\theta_{launch}$="
                  + f"{handle_scotty_launch_angle_sign(dt=dt)}" 
                  + r"$^\circ$" 
                  + f", f={dt.inputs.launch_freq_GHz.values}GHz")
        plt.colorbar(c, label='modE')
        
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
        plt.gca().set_aspect('equal')
        plt.show()
        
        print("Plotting modE vs tau from ERMES and Scotty")
        # Plot modE vs tau
        plt.scatter(distance_along_beam, modE_list, marker='.', color = 'red', label='ERMES')
        theoretical_modE_tau = calc_Eb_from_scotty(dt=dt, wx=dt.inputs.launch_beam_width.values, wy=dt.inputs.launch_beam_width.values, E0 = modE_list[0])
        plt.scatter(distance_along_beam, theoretical_modE_tau, marker='.', color = 'orange', label='Scotty')
        smoothed_modE_list = get_moving_RMS(modE_list, 40)
        plt.plot(distance_along_beam, smoothed_modE_list, 'g-', label="Smoothed ERMES")
        plt.vlines(distance_along_beam[tau_cutoff], ymin=plt.gca().get_ylim()[0], ymax = plt.gca().get_ylim()[1], linestyles='--', color='blue')
        
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("mod(E) (A.U.)")
        plt.title(r"mod(E) vs Distance along central ray, $\theta_{launch}$="
                  + f"{handle_scotty_launch_angle_sign(dt=dt)}" 
                  + r"$^\circ$" 
                  + f", f={dt.inputs.launch_freq_GHz.values}GHz")
        plt.xlim(0, distance_along_beam[-1])
        plt.ylim(bottom = 0)
        plt.tight_layout()
        plt.show()
        
        # Relative err of modE
        err_modE = get_relative_error(smoothed_modE_list, theoretical_modE_tau)
        plt.scatter(distance_along_beam, err_modE, color = 'red', s = 15)
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("Relative error")
        plt.title("Relative error between ERMES and Scotty modE along central ray")
        plt.show()
        
        print("Plotting rE vector vs tau from ERMES and Scotty")
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
        ax1.plot(distance_along_beam, np.array(fitted_widths), label="Fitted ERMES width", color='red')
        smoothed_width_list = get_moving_RMS(fitted_widths, 40)
        ax1.plot(distance_along_beam, smoothed_width_list, 'g-', label="Smoothed Fitted ERMES width")
        ax1.vlines(distance_along_beam[tau_cutoff], ymin=ax1.get_ylim()[0], ymax = ax1.get_ylim()[1], linestyles='dashed', color='blue')
        ax1.set_ylabel("Width (m)")
        ax1.set_title("Beam widths")
        ax1.legend()

        # Bottom subplot: chi**2
        ax2.plot(distance_along_beam, chi2_list, label=r"$\chi^2$ of ERMES fit", color='red')
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
        plt.title("Relative error between ERMES and Scotty beam widths")
        plt.show()
        
        # Poynting flux
        plt.scatter(distance_along_beam, poynting_flux_per_tau, color = 'red', s = 15)
        plt.ylim(0, 1.1) # Since it should be 1
        plt.vlines(distance_along_beam[tau_cutoff], ymin=plt.gca().get_ylim()[0], ymax = plt.gca().get_ylim()[1], linestyles='--', color='blue')
        plt.xlabel("Distance along central ray (m)")
        plt.ylabel("Power flux (W/m)")
        plt.title("Poynting flux across beamfront along central ray")
        plt.show()
        
        # Save the errors
        if save:
            np.savez(os.getcwd() + f"\\{dt.inputs.launch_freq_GHz.values}_{handle_scotty_launch_angle_sign(dt)}_errors", err_modE, err_width)
         
if __name__ == '__main__':
    # MAST-U
    """
    get_ERMES_parameters(
        dt=load_scotty_data('\\MAST-U\\scotty_output_freq40.0_pol-7.0_rev.h5'),
        suffix="MAST-U_",
        launch_angle=7.0, 
        launch_freq_GHz=40, 
        port_width=0.01, 
        #launch_positon=[2.278,0,-0.01], 
        #launch_beam_curvature=-0.7497156475519201, 
        #launch_beam_width=0.07596928872724663, 
        dist_to_ERMES_port=0.85, 
        plot=False,
        save=False,
        )
    #"""
    
    #DIII-D
    #"""
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5'),
        suffix="DIII-D_NEW_",
        launch_angle=7.0, 
        launch_freq_GHz=72.5, 
        port_width=0.01, 
        #launch_positon=[3.01346,0,-0.09017],
        #launch_beam_curvature=-0.95, 
        #launch_beam_width=0.125, 
        dist_to_ERMES_port=0.68, 
        plot=True,
        save=True,
        )
    
    #"""
    
    """
    Some notes on the work I am doing right now so that I don't forget
    5,7,9,11,13,15,17 deg pol sweep on DIII-D & MAST-U
    Compare w/ Scotty & analytical
    
    Interestingly, Scotty crashes for launch_angle = 9 deg on the DIII-D data , so I ran at 8.995 deg since 8.996 deg will crash. 
   
    """
    # Text ERMES output to plots
    """
    ERMES_results_to_plots(
        #res="\\ERMES_output_files\\MAST-U_7_degree_40_EDG2.post.res", 
        #msh="\\ERMES_output_files\\MAST-U_7_degree_40_EDG2.post.msh",
        #dt=load_scotty_data('\\MAST-U\\scotty_output_freq40.0_pol-7.0_rev.h5'),
        res="\\ERMES_output_files\\DIII-D_7_degree_72_5_EDG2_no_damping_morene.post.res", 
        msh="\\ERMES_output_files\\DIII-D_7_degree_72_5_EDG2_no_damping_morene.post.msh",
        dt=load_scotty_data('\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5'),
        plot=True,
        grid_resolution=8e-4,
        save=False,
    )
    #"""
    
    """
    # Plotting stuffs
    launch_angles = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    #toroidal_diff = np.array([np.max(load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values) for launch_angle in launch_angles]) - np.array([np.min(load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values) for launch_angle in launch_angles])
    q_Y_arrays = np.row_stack([load_scotty_data(f"\\Output\\scotty_output_freq72.5_pol-{launch_angle}_rev.h5").analysis.q_Y.values for launch_angle in launch_angles])
    
    #print(q_Y_arrays)
    #print(toroidal_diff)
    err_15 = np.load(os.getcwd()+"\\72.5_15_errors.npz")
    err_13 = np.load(os.getcwd()+"\\72.5_13_errors.npz")
    err_11 = np.load(os.getcwd()+"\\72.5_11_errors.npz")
    err_9 = np.load(os.getcwd()+"\\72.5_8.995_errors.npz")
    err_7 = np.load(os.getcwd()+"\\72.5_7_errors.npz")
    err_5 = np.load(os.getcwd()+"\\72.5_5_errors.npz")
    err_3 = np.load(os.getcwd()+"\\72.5_3_errors.npz")
    
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
    plt.plot([0, 1, 2], err_beam_3, color='orange', label=r"$3^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_5, color='purple', label=r"$5^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_7, color='red', label=r"$7^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_9, color='gray', label=r"$9^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_11, color='brown', label=r"$11^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_13, color='pink', label=r"$13^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_beam_15, color='blue', label=r"$15^\circ$", marker = 'x')
    plt.xticks([0, 1, 2], ['Entry', 'Cutoff', 'Exit'])
    plt.ylabel("Relative error")
    plt.xlabel(r"Beam parameter $\tau$")
    plt.title(r"Relative error of modE along central ray vs $\tau$")
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()
    
    err_width_3 = err_3_width[tau_3]
    err_width_5 = err_5_width[tau_5]
    err_width_7 = err_7_width[tau_7]
    err_width_9 = err_9_width[tau_9]
    err_width_11 = err_11_width[tau_11]
    err_width_13 = err_13_width[tau_13]
    err_width_15 = err_15_width[tau_15]
    plt.plot([0, 1, 2], err_width_3, color='orange', label=r"$3^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_5, color='purple', label=r"$5^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_7, color='red', label=r"$7^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_9, color='gray', label=r"$9^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_11, color='brown', label=r"$11^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_13, color='pink', label=r"$13^\circ$", marker = 'x')
    plt.plot([0, 1, 2], err_width_15, color='blue', label=r"$15^\circ$", marker = 'x')
    plt.xticks([0, 1, 2], ['Entry', 'Cutoff', 'Exit'])
    plt.xlabel(r"Beam parameter $\tau$")
    plt.title(r"Relative error of beam width vs $\tau$")
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()
    
    #"""
    
    """ Plot toroidal component of each beam (zoomed in)
    dt = load_scotty_data('\\Output\\scotty_output_freq72.5_pol-7.0_rev.h5')
    
    # Plot flux surfaces to 'visualize' the plasma
    ax = maybe_make_axis(plt.gca())

    lcfs = 1
    flux_spline = CubicSpline(
        dt.analysis.R_midplane, dt.analysis.poloidal_flux_on_midplane - lcfs
    )
    all_R_lcfs = flux_spline.roots()

    zeta = np.linspace(-np.pi, np.pi, 1001)
    for R_lcfs in all_R_lcfs:
        plot_toroidal_contour(ax, R_lcfs, zeta)

    flux_min_index = dt.analysis.poloidal_flux_on_midplane.argmin()
    R_axis = dt.analysis.R_midplane[flux_min_index].data
    plot_toroidal_contour(ax, R_axis, zeta, "#00003f")
        
    # Plot Scotty results
    width_tor = beam_width(
        dt.analysis.g_hat_Cartesian,
        np.array([0.0, 0.0, 1.0]),
        dt.analysis.Psi_3D_Cartesian,
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