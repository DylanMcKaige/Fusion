"""
Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 
Transposes ne data to R,Z coords as well

Spits out .txt and .csv files of all the necessary data. 

TODO
1. Run in ERMES and compare to scotty

References
    [1] Two dimensional full-wave simulations of Doppler back-scattering in tokamak plasmas with COMSOL by Quinn Pratt et al (in-progress paper)
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin, pg 43-44

Written by Dylan James Mc Kaige
Created: 16/5/2025
Updated: 20/5/2025
"""
import os, json
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, fabs
from scipy.constants import c, pi
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from matplotlib import pyplot as plt

def ne_pol_to_RZ(
    ne_path: str,
    topfile_path: str,
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    ERMES_R = None,
    ERMES_Z = None,
    ERMES_port = None,
    num_RZ = 20
    ):
    """
    Loads ne and topfile data from the given .dat and .json files and saves ne as a function of R & Z in a .txt file. Also plots for sanity check against Scotty

    Args:
        ne_path (str): Relative (to cwd) paht of ne.dat file
        topfile_path (str): Relative (to cwd) path of topfile.json file
        plot (bool): Plot the data
        save (bool): Save the data
        path (str): Path to save file in, defaults to cwd
        ERMES_R (tuple): Range of R to zoom in and save ne of defaults to full range from topfile
        ERMES_Z (tuple): Range of Z to zoom in and save ne of defaults to full range from topfile
        ERMES_port (tuple): Coordinates of port to check if it is within the plasma 
        
    Returns:
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
    ne_vals_to_ERMES = ne_spline(pol_flux_vals_to_ERMES)*1e19

    Br_spline = RectBivariateSpline(R, Z, Br.T)
    Bz_spline = RectBivariateSpline(R, Z, Bz.T)
    Bt_spline = RectBivariateSpline(R, Z, Bt.T)
    
    Br_vals_ERMES = Br_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    Bz_vals_ERMES = Bz_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
    Bt_vals_ERMES = Bt_spline.ev(R_grid_to_ERMES, Z_grid_to_ERMES)
 
    RZ_ERMES = np.column_stack((R_range, Z_range))

    if save:
        np.savetxt("RZ_ERMES.csv", RZ_ERMES, delimiter=",", fmt="%.6e")

        np.savetxt("Br_ERMES.csv", Br_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt("Bz_ERMES.csv", Bz_vals_ERMES.T, delimiter=",", fmt="%.6e")
        np.savetxt("Bt_ERMES.csv", Bt_vals_ERMES.T, delimiter=",", fmt="%.6e")

        np.savetxt("ne_ERMES.csv", ne_vals_to_ERMES.T, delimiter=",", fmt="%.6e")
    
    if plot:
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
        plt.scatter(*zip(*ERMES_port), s = 2, color='white')
        plt.clabel(cs, fmt=r'%.1f', fontsize=8, colors='black')

        plt.colorbar(cf, label=r'$n_e\ (10^{19}\ \mathrm{m}^{-3})$')
        plt.xlabel("R [m]")
        plt.ylabel("Z [m]")
        plt.title(r'Heatmap of $n_e(R,Z)$')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def get_ERMES_parameters(
    launch_angle: float,
    launch_freq_GHz: float,
    port_width: float,
    launch_positon: np.array,
    launch_beam_curvature: float,
    launch_beam_width: float,
    dist_to_ERMES_port: float,
    domain_size: float,
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    gen_ne = True,
    ne_path = None,
    topfile_path = None,
    num_RZ = 20,
    ):
    """
    Generate ERMES parameters for given input
    
    Args:
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
        gen_ne (bool): Generte ne, R and Z files for ERMES
        ne_path (str): Relative (to cwd) paht of ne.dat file
        topfile_path (str): Relative (to cwd) path of topfile.json file
        num_RZ (int): Number of RZ points for ERMES
        
    Returns:
        Plots the position of the required points in the Z-R axes and saves a .txt file of the necessary values for ERMES
        Optionally generates the necessary ne, R and Z files for ERMES
    """
    degtorad = pi/180
    
    launch_beam_wavelength = c/(launch_freq_GHz*1e9)
    radius_of_curv = fabs(1/launch_beam_curvature)
    launch_angle_rad = launch_angle*degtorad
    launch_R = launch_positon[0]
    launch_Z = launch_positon[2]
    
    # Initial calculations
    distance_to_launcher = fabs((radius_of_curv* pi**2 * launch_beam_width**4)/(launch_beam_wavelength**2 * radius_of_curv**2+pi**2 * launch_beam_width**4))
    z_R = fabs((launch_beam_wavelength*radius_of_curv*distance_to_launcher)/(pi*launch_beam_width**2))
    w0 = sqrt((launch_beam_wavelength*z_R)/(pi))
    
    w_ERMES = 2*w0*sqrt(1+(dist_to_ERMES_port/z_R)**2) # Width of beam at port position in ERMES
    kx_norm = -cos(launch_angle_rad) # Normalized k vector, -ve because leftwards by definition of geometry
    ky_norm = sin(launch_angle_rad) # Normalized k vector 
    xw = launch_R - distance_to_launcher*cos(launch_angle_rad) # Centre of waist
    yw = launch_Z + distance_to_launcher*sin(launch_angle_rad) # Centre of waist
    
    # Scuffed normalized polarization vector generation (Forcing it to be polarized out of the plane of the screen)
    # Since we define k to be in the plane of the screen, the vector pointing out is definitely perp to it, satisfying ERMES
    # TODO update this to get the actual one from scotty
    # THIS JUST ENSURES THAT ERMES WORKS, it is very wrong please update this 
    polx = 0
    poly = 0
    polz = 1
    
    # This formula is from a .txt file from UCLA collaborators. I didn't look for a proper source/derivation of this yet
    z0 = 377 # Impedance of free space
    E0 = sqrt(z0*2*1/(w0*sqrt(pi/2))) # For P_in = 1 W/m
    
    # Port calculations
    xp, yp = launch_R - dist_to_ERMES_port*cos(launch_angle_rad), launch_Z + dist_to_ERMES_port*sin(launch_angle_rad)
    xp0, yp0 = xp - w_ERMES/2 * sin(launch_angle_rad), yp - w_ERMES/2 * cos(launch_angle_rad)
    xp1, yp1 = xp + w_ERMES/2 * sin(launch_angle_rad), yp + w_ERMES/2 * cos(launch_angle_rad)
    xp01, yp01 = xp0 + port_width*cos(launch_angle_rad), yp0 - port_width*sin(launch_angle_rad)
    xp11, yp11 = xp1 + port_width*cos(launch_angle_rad), yp1 - port_width*sin(launch_angle_rad)
    
    # Domain calculations
    # Arbitrary padding (~10 x order of mesh spacing?) so that the volume can be generated. 
    # If padding is too small, will cause errors in ERMES
    xd1, yd0 = xp11 + 0.005, yp01 - 0.005 
    xd0, yd1 = xd1-domain_size, yd0+domain_size

    # Arrays for saving
    # Surely there is a neater way of doing this, but I'm lazy and want to get this working first before making it pretty
    points_x = np.array(
        [xp, xp0, xp1, xp01, xp11, launch_R, xd0, xd1, xd0, xd1, xw]
    )
    points_y = np.array(
        [yp, yp0, yp1, yp01, yp11, launch_Z, yd0, yd0, yd1, yd1, yw]
    )
    points_names = np.array(
        ['Source Position (front face of port)    ', 
         'Port BL    ', 
         'Port TL    ', 
         'Port BR    ', 'Port TR    ', 
         'Launch Position    ', 
         'Domain BL    ', 
         'Domain BR    ', 
         'Domain TL    ', 
         'Domain TR    ', 
         'Waist Position    '
         ]
    )

    # Beam params
    params_val = np.array(
        [launch_angle, 
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
         polx, 
         poly, 
         polz, 
         E0, 
         ]
    )
    params_names = np.array(
        ['Launch Angle    ', 
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
         ]
    )
    
    # Save it!
    if save:
        np.savetxt(path + 'ERMES_params', np.array([points_names, points_x, points_y], dtype=object).T, delimiter=' ', header = 'Cartesian points in ERMES', fmt='%s')
        # There's probably a better way to do this 
        with open(path + 'ERMES_params', 'a') as file:
            np.savetxt(file, np.array([params_names, params_val], dtype=object).T, delimiter=' ', header='Beam Params', fmt = '%s')
    
    if gen_ne:
        ne_pol_to_RZ(ne_path, topfile_path, ERMES_R = (xd0, xd1), ERMES_Z = (yd0, yd1), ERMES_port = ([xp0, yp0], [xp1, yp1], [xp01, yp01], [xp11, yp11]), num_RZ = num_RZ, plot=plot)
    
    if plot:
        plt.scatter(points_x, points_y, s = 2)
        plt.gca().set_aspect('equal')
        plt.show()

if __name__ == '__main__':
    get_ERMES_parameters(
        launch_angle=7.0, 
        launch_freq_GHz=72.5, 
        port_width=0.01, 
        launch_positon=[3.01346,0,-0.09017], 
        launch_beam_curvature=-0.95, 
        launch_beam_width=0.125, 
        dist_to_ERMES_port=0.65, 
        domain_size=0.3, 
        ne_path = '\\source_data\\ne_189998_3000ms_quinn.dat', 
        topfile_path = '\\source_data\\topfile_189998_3000ms_quinn.json',
        num_RZ = 25,
        plot=True,
        save = False,
        )