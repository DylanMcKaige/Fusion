"""
Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 
Transposes ne data to R,Z coords as well

Spits out .txt and .csv files of all the necessary data. 

TODO
1. Run in ERMES and compare to scotty (7 deg done, 15 deg done)
2. Project pol onto rho,eta plane
3. Figure out E0 -> Matthew & valerian & Ruben's script

References
    [1] Two dimensional full-wave simulations of Doppler back-scattering in tokamak plasmas with COMSOL by Quinn Pratt et al (in-progress paper)
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin, pg 43-44
    [4] V. H. Hall-Chen, F. I. Parra, and J. C. Hillesheim, “Beam model of Doppler backscattering,” Plasma Phys. Control. Fusion, vol. 64, no. 9, p. 095002, Sep. 2022, doi: 10.1088/1361-6587/ac57a1.
    [5] F. M. A. Smits, “ELLIPTICAL POLARISATION FOR OBLIQUE EC-WAVE LAUNCH”.

Written by Dylan James Mc Kaige
Created: 16/5/2025
Updated: 11/6/2025
"""
import os, json, datatree
import numpy as np
import pandas as pd
from math import sin, cos, tan, acos, atan, sqrt, fabs
from scipy.constants import c, pi, m_e, m_p, elementary_charge, epsilon_0
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from matplotlib import pyplot as plt

def RtZ_to_XYZ(a: np.array) -> np.array:
    """
    Convert an array in (R,t,Z) to (X,Y,Z) for ERMES, keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (R,t,Z) basis
    
    Returns:
        b (array): The transformed vector in (X,Y,Z) basis (ERMES Cartesian)

    """
    assert len(a) == 3, "This function only supports arrays of length 3"
    
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
    Only used if no Scotty output file is provided

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
        theta (float or array): Angle of rotation following right-hand rule, relative to a

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

def get_pol_from_smits(k_vec: np.array, B_entry_vec_XYZ: np.array, launch_freq_GHz: float, E0: float):
    """
    Get the polarization vector and values of E_perp and E_par using Smits [5]
    
    Args:
        k_vec (np.array): k vector 
        B_entry_vec_XYZ (np.array): B vector at point of entry
        launch_freq_GHz (float): Launch frequency in GHz
        E0 (float): E0 value

    Returns:
        rho_hat (array): Polarization vector in XYZ basis
        mod_E_par (float): Mod E_par in ERMES
        mod_E_perp (float): Mod E_per in ERMES
    """
    # Normalize to make life simpler
    k_vec_hat = k_vec/np.linalg.norm(k_vec)
    B_entry_vec_XYZ_hat = B_entry_vec_XYZ/np.linalg.norm(B_entry_vec_XYZ)
    
    # Angle between k and b
    theta = acos(np.dot(k_vec_hat, B_entry_vec_XYZ_hat))
    
    # c from [5]
    C = (1.6e-19*np.linalg.norm(B_entry_vec_XYZ) / 9.11e-31) / (2*pi*launch_freq_GHz*1e9)
    p_prime = sqrt(sin(theta)**4 + 4*cos(theta)**2 / C**2)
    # Angle of rho above kb plane and perp to k for QX-mode
    gamma = atan(-2*cos(theta) / (C*(sin(theta)**2 + p_prime)) ) # Gamma from theta -> should have BEST coupling!
    #gamma = np.linspace(0, 2*pi, 36, endpoint = 0)
    #print(gamma*180/pi)
    #gamma = [9]
    #print(gamma*180/pi)
    
    # Get the rho unit vector that is perp to k and b. This is our initial pol
    rho_hat_perp = np.cross(k_vec_hat, B_entry_vec_XYZ_hat) 
    theta_rho = np.linspace(0, 2*pi, 10, endpoint=False)
    print(theta_rho*180/pi)
    rho_hat_rotated = rotate_rodrigues(rho_hat_perp, k_vec_hat, theta_rho)
           
    # Get the ratio
    mod_E_rho = np.sqrt(E0**2 / (1 + np.tan(gamma)**2))# rho_hat_kb
    mod_E_eta = np.sqrt(E0**2 - mod_E_rho**2) # eta_hat_kb
    
    print(f'Gamma: {gamma*180/pi:.5}')
    print(f'E_par: {mod_E_rho:.5}')
    print(f'E_perp: {mod_E_eta:.5}')
    print(rho_hat_rotated)

    return rho_hat_perp, mod_E_rho, mod_E_eta

def process_scotty_input_data(
    ne_path: str,
    topfile_path: str,
    filename: str,
    dt: datatree.DataTree = None,
    plot: bool = True,
    save: bool = True,
    path: str = os.getcwd() + '\\',
    ERMES_R = None,
    ERMES_Z = None,
    ERMES_port = None,
    ERMES_launch_centre = None,
    launch_angle_rad = None,
    num_RZ = 25
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
    path = os.getcwd() + '\\'
    
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
    ne_vals_to_ERMES = ne_spline(pol_flux_vals_to_ERMES)*1e19 # Since the data was normalised by 1e19

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
        
    B_entry_vec_RtZ = np.array([
        Br_spline.ev(entry_point[0], entry_point[1]),
        Bt_spline.ev(entry_point[0], entry_point[1]),
        Bz_spline.ev(entry_point[0], entry_point[1])
        ]
    )
 
    RZ_ERMES = np.column_stack((R_range, Z_range))

    if save:
        np.savetxt(filename + "RZ_ERMES.csv", RZ_ERMES, delimiter=",", fmt="%.6e")

        np.savetxt(filename + "Br_ERMES.csv", Br_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt(filename + "Bz_ERMES.csv", Bz_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt(filename + "Bt_ERMES.csv", Bt_vals_ERMES.T, delimiter=",", fmt="%.6e")

        np.savetxt(filename + "ne_ERMES.csv", ne_vals_to_ERMES.T, delimiter=",", fmt="%.6e")
    
    if plot:
        w_levels = [0]#np.linspace(-0.01*w, 0.01, 10, endpoint=True)
        
        plt.figure(figsize=(8, 6))

        cf = plt.contourf(R_grid, Z_grid, ne_vals_total, levels=100, cmap='plasma')
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
        
        cf = plt.contourf(R_grid_to_ERMES, Z_grid_to_ERMES, ne_vals_to_ERMES, levels=100, cmap='plasma')
        pol_flux_levels = np.linspace(0.0, 1.0, 11)
        cs = plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, pol_flux_vals_to_ERMES, levels=pol_flux_levels, colors='black', linewidths=0.8)
        plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_R, levels = w_levels, colors='white', linewidths=2)
        plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_UH, levels = w_levels, colors='blue', linewidths=2)
        plt.contour(R_grid_to_ERMES, Z_grid_to_ERMES, w_LH, levels = w_levels, colors='green', linewidths=2)
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
        
    return entry_point, B_entry_vec_RtZ

def get_ERMES_parameters(
    dt: datatree.DataTree = None,
    launch_angle: float = None,
    launch_freq_GHz: float = None,
    launch_positon: np.array = None,
    launch_beam_curvature: float = None,
    launch_beam_width: float = None,
    port_width: float = None,
    dist_to_ERMES_port: float = None,
    domain_size: float = None,
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    ne_path = None,
    topfile_path = None,
    num_RZ = 20,
    ):
    """
    Generate ERMES parameters for given input
    
    Args:
        dt (DataTree): Scotty output file in .h5 format
        launch_angle (float): Launch angle in degrees, w.r.t -ve R axis
        launch_freq_GHz (float): Launch frequency in GHz
        port_width (float): Width of port in ERMES in m, so far seems to be arbitrary
        launch_position (array): Launch position in [R,t,Z] coordinates in m
        launch_beam_curvature (float): Curvature of beam at launch position, in m^{-1}
        launch_beam_width (float): Width of beam at launch position, in m
        dist_to_ERMES_port (float): Distance from launcher to port in ERMES in m, stare at Scotty to decide this
        domain_size (float): Size of domain for ERMES in m, stare at Scotty to decide this
        plot (bool): Plot everything
        save (bool): Save everything
        path (str): Path to save file in, defaults to cwd
        ne_path (str): Relative (to cwd) paht of ne.dat file
        topfile_path (str): Relative (to cwd) path of topfile.json file
        num_RZ (int): Number of RZ points for ERMES
        
    Returns:
        Plots the position of the required points in the Z-R axes and saves a .txt file of the necessary values for ERMES
        Optionally generates the necessary ne, R and Z files for ERMES
    """
    degtorad = pi/180
    
    if dt is None:
        # Take inputs
        launch_R = launch_positon[0]
        launch_Z = launch_positon[2]
    else:
        # Else, take from Scotty
        launch_freq_GHz = dt.inputs.launch_freq_GHz.values
        launch_beam_width = dt.inputs.launch_beam_width.values
        launch_beam_curvature = dt.inputs.launch_beam_curvature.values
        launch_angle = fabs(dt.inputs.poloidal_launch_angle_Torbeam.values)
        launch_R = dt.inputs.launch_position.values[0]
        launch_Z = dt.inputs.launch_position.values[2]
        
    launch_beam_wavelength = c/(launch_freq_GHz*1e9)
    radius_of_curv = fabs(1/launch_beam_curvature)
    launch_angle_rad = launch_angle*degtorad
    filename = str(launch_angle) + "_degree_" + str(launch_freq_GHz) + "GHz_"
    
    # Initial calculations
    distance_to_launcher = fabs((radius_of_curv * pi**2 * launch_beam_width**4)/(launch_beam_wavelength**2 * radius_of_curv**2+pi**2 * launch_beam_width**4))
    z_R = fabs((launch_beam_wavelength*radius_of_curv*distance_to_launcher)/(pi*launch_beam_width**2))
    w0 = sqrt((launch_beam_wavelength*z_R)/(pi))
    
    w_ERMES = 2*w0*sqrt(1+(dist_to_ERMES_port/z_R)**2) # Width of beam at port position in ERMES
    kx_norm = -cos(launch_angle_rad) # Normalized k vector, -ve because leftwards by definition of geometry
    ky_norm = sin(launch_angle_rad) # Normalized k vector 
    xw = launch_R - distance_to_launcher*cos(launch_angle_rad) # Centre of waist
    yw = launch_Z + distance_to_launcher*sin(launch_angle_rad) # Centre of waist
    
    # This formula is from a .txt file from UCLA collaborators. I didn't look for a proper source/derivation of this yet
    z0 = 377 # Impedance of free space
    E0 = sqrt(z0*2*1/(w0*sqrt(pi/2))) # For P_in = 1 W/m
    
    # Port calculations
    xp, yp = launch_R - dist_to_ERMES_port*cos(launch_angle_rad), launch_Z + dist_to_ERMES_port*sin(launch_angle_rad)
    xp0, yp0 = xp - w_ERMES/2*sin(launch_angle_rad), yp - w_ERMES/2*cos(launch_angle_rad)
    xp1, yp1 = xp + w_ERMES/2*sin(launch_angle_rad), yp + w_ERMES/2*cos(launch_angle_rad)
    xp01, yp01 = xp0 + port_width*cos(launch_angle_rad), yp0 - port_width*sin(launch_angle_rad)
    xp11, yp11 = xp1 + port_width*cos(launch_angle_rad), yp1 - port_width*sin(launch_angle_rad)
    
    # Domain calculations
    # Arbitrary padding (~10 x order of mesh spacing?) so that the volume can be generated. 
    # If padding is too small, will cause errors in ERMES OR mesh generation will take stupidly long... I don't make the rules!
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
    
    # Convert Scotty input files into RZ format (& Do some sanity plots)
    entry_point, B_entry_vec_RtZ = process_scotty_input_data(
        ne_path, 
        topfile_path, 
        filename = filename,
        plot=plot,
        dt=dt,
        ERMES_R=(xd0, xd1), 
        ERMES_Z=(yd0, yd1), 
        ERMES_port=([xp0, yp0], [xp1, yp1], [xp01, yp01], [xp11, yp11]), 
        ERMES_launch_centre=(xp, yp), 
        launch_angle_rad=launch_angle_rad,
        num_RZ=num_RZ,
        )
    
    B_entry_vec_XYZ = RtZ_to_XYZ(B_entry_vec_RtZ)
    
    # linear pol vec at launch point = beam k X B field at launch point in (X,Y,Z)
    k_vec = np.array([kx_norm, ky_norm, 0])
    lin_pol_vec = np.cross(k_vec, B_entry_vec_XYZ)

    # Elliptical pol vector from Scotty
    if dt is not None: ellip_pol_vec = get_pol_from_scotty(dt)
    else: ellip_pol_vec = np.array([0, 0, 0])
    
    # Polarization from Smits
    rho_hat, mod_E_par, mod_E_perp = get_pol_from_smits(k_vec, B_entry_vec_XYZ, launch_freq_GHz, E0)

    # Arrays for saving
    # Surely there is a neater way of doing this, but I'm lazy and want to get this working first before making it pretty
    points_x = np.array([
        xp, 
        xp0, 
        xp1, 
        xp01, 
        xp11, 
        launch_R, 
        xd0, 
        xd1, 
        xd0, 
        xd1, 
        xp0,
        trimmed_xd0,
        trimmed_xd1,
        trimmed_xd3,
        xw,
        entry_point[0],
        ]
    )
    points_y = np.array([
        yp, 
        yp0, 
        yp1, 
        yp01, 
        yp11, 
        launch_Z, 
        yd0, 
        yd0, 
        yd1, 
        yd1, 
        yp0,
        trimmed_yd0,
        trimmed_yd1,
        trimmed_yd3,
        yw,
        entry_point[1],
        ]
    )
    points_names = np.array([
        'Source Position (front face of port)    ', 
        'Port BL    ', 
        'Port TL    ', 
        'Port BR    ', 'Port TR    ', 
        'Launch Position    ', 
        'Domain BL    ', 
        'Domain BR    ', 
        'Domain TL    ', 
        'Domain TR    ', 
        'Trimmed Domain p0    ', 
        'Trimmed Domain d0    ',
        'Trimmed Domain d1    ',
        'Trimmed Domain d3    ',
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
        lin_pol_vec[0],
        lin_pol_vec[1],
        lin_pol_vec[2],
        ellip_pol_vec[0],
        ellip_pol_vec[1],
        ellip_pol_vec[2],
        rho_hat[0],
        rho_hat[1],
        rho_hat[2],
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
        'Linear polx (normalized)    ', 
        'Linear poly (normalized)    ', 
        'Linear polz (normalized)    ', 
        'Ellipitcal polx (normalized)    ', 
        'Ellipitcal poly (normalized)    ', 
        'Ellipitcal polz (normalized)    ', 
        'Smits polx (normalized)    ',
        'Smits poly (normalized)    ',
        'Smits polz (normalized)    ',
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

    if plot:
        plt.scatter(points_x, points_y, s = 2)
        plt.gca().set_aspect('equal')
        plt.show()

if __name__ == '__main__':
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\scotty_output_freq72.5_pol-15.0_rev.h5'),
        launch_angle=15.0, 
        launch_freq_GHz=72.5, 
        port_width=0.01, 
        launch_positon=[3.01346,0,-0.09017], 
        launch_beam_curvature=-0.95, 
        launch_beam_width=0.125, 
        dist_to_ERMES_port=0.65, 
        domain_size=0.45, 
        ne_path = '\\source_data\\ne_189998_3000ms_quinn.dat', 
        topfile_path = '\\source_data\\topfile_189998_3000ms_quinn.json',
        num_RZ = 25,
        plot=False,
        save=False,
        )