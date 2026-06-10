# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.io.wavfile import write
from scipy.linalg import qr
from scipy.linalg import null_space
from scipy.linalg import eig
from scipy.signal import resample
from scipy.linalg import svd
import matplotlib.cm as cm
import time

#%% Functions 

def sample(sig, t, Fe_target):
    Te                      = t[1] - t[0]
    Fe                      = 1 / Te
    N                       = sig.size 
    n                       = int(np.ceil(N * Fe_target / Fe))
    sig_sampled, t_sampled  = resample(sig, n, t)
    Fe_new                  = int(np.round(1 / (t_sampled[1] - t_sampled[0])))
    return sig_sampled, t_sampled, Fe_new

def low_pass(x, y, beta):
    return beta * y + (1 - beta) * x

def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Aucun fichier trouvé parmi : " + str(paths))

#%% Load the body's modal basis

name_struct    = "Plaque_Chevalet_Modal_Primal_Dx_0.103_Dy_3.500_cm_z_Nmc_99_Nmp_4784"

file_c_path = first_existing([os.path.join("Data", name_struct + ".npz"),name_struct + ".npz"])
file_c  = np.load(file_c_path, allow_pickle=True)
print("File " + file_c_path + " opened.")
        
#%%

w_max_c = 2 * np.pi * 20e3

x_c             = file_c['x']
y_c             = file_c['y']
z_c             = file_c['z']
wn_c            = file_c['wn']
mn_c            = file_c['mn']
kn_c            = file_c['kn']
cn_c            = file_c['cn']
phinx_c         = file_c['phinx']
phiny_c         = file_c['phiny']
phinz_c         = file_c['phinz']

N_c             = x_c.size

f_lim           = np.arange(20000,0,-5000)
idx_c_init      = np.isfinite(wn_c) & (wn_c < 2 * np.pi * np.max(f_lim))

wn_c            = wn_c[idx_c_init]
mn_c            = mn_c[idx_c_init]
kn_c            = kn_c[idx_c_init]
cn_c            = cn_c[idx_c_init] 
phinx_c         = phinx_c[idx_c_init]
phiny_c         = phiny_c[idx_c_init]
phinz_c         = phinz_c[idx_c_init]

#%% Import string data

w_max_s = 2 * np.pi * 20e3

Dy = 2.031e-2
 
string_names = np.array(["E2", "A2", "D3", "G3", "B3", "E4"]) 

string_plucked = string_names[0] #corde excitée

string_mu       = {"E4" : 0.38e-3,
                   "B3" : 0.52e-3,
                   "G3" : 0.90e-3,
                   "D3" : 1.95e-3,
                   "A2" : 3.61e-3,
                   "E2" : 6.24e-3}

string_T        = {"E4" : 70.3,
                   "B3" : 53.4,
                   "G3" : 58.3,
                   "D3" : 71.2,
                   "A2" : 73.9,
                   "E2" : 71.6} 

x_s         = np.zeros((0))
y_s         = np.zeros((0))
z_s         = np.zeros((0))
phinx_s     = np.zeros((0,0))
phiny_s     = np.zeros((0,0))
phinz_s     = np.zeros((0,0))
f_nl_s      = np.zeros((0,0))
wn_s        = np.zeros((0))
mn_s        = np.zeros((0))
kn_s        = np.zeros((0))
cn_s        = np.zeros((0))
kappan_s    = np.zeros((0))
init_disp_s = np.zeros((0))
params_s    = np.zeros((0))

for (i,string_chosen) in enumerate(string_names):
     
    T       = string_T[string_chosen]
    mu      = string_mu[string_chosen]
    
    name_string = "String_modal_basis_" + string_chosen + "_Dy_" + f"{(np.round(Dy*100,3)):.3f}" +"_cm_T_"+\
            f"{(np.round(T,0)):.0f}"+"_N_mu_"+f"{(np.round(mu*1e6,0)):.0f}"+"_mg.m-1_Woodhouse_2012"
    
    file_s_path = first_existing([os.path.join("Data", name_string + ".npz"), name_string + ".npz", os.path.join("Data", name_string + " (1).npz"), name_string + " (1).npz"])
    file_s  = np.load(file_s_path, allow_pickle=True)
    print("File " + file_s_path + " opened.")
    
    # Rotate string and attach to bridge
    x_s_temp = file_s['x'].astype(float) + x_c[z_c==z_c.max()][i] - file_s['x'].astype(float).min()
    y_s_temp = -file_s['y'].astype(float)
    y_s_temp += y_c[z_c==z_c.max()][i] - y_s_temp.min()
    z_s_temp = file_s['z'].astype(float) + z_c[z_c==z_c.max()][i] - file_s['z'].astype(float).min()
    
    x_s             = np.concatenate((x_s, x_s_temp))
    y_s             = np.concatenate((y_s, y_s_temp))
    z_s             = np.concatenate((z_s, z_s_temp))
    wn_s            = np.concatenate((wn_s, file_s['wn']))
    mn_s            = np.concatenate((mn_s, file_s['mn']))
    kn_s            = np.concatenate((kn_s, file_s['kn']))
    cn_s            = np.concatenate((cn_s, file_s['cn']))
    kappan_s        = np.concatenate((kappan_s, file_s['kappan']))
    phinx_s_temp    = np.zeros((phinx_s.shape[0] + file_s['phinx'].shape[0],
                                phinx_s.shape[1] + file_s['phinx'].shape[1]))
    phinx_s_temp[:phinx_s.shape[0], 
                 :phinx_s.shape[1]] = phinx_s
    phinx_s_temp[phinx_s.shape[0]:, 
                 phinx_s.shape[1]:] = file_s['phinx']
    phinx_s         = phinx_s_temp
    phiny_s_temp    = np.zeros((phiny_s.shape[0] + file_s['phiny'].shape[0],
                                phiny_s.shape[1] + file_s['phiny'].shape[1]))
    phiny_s_temp[:phiny_s.shape[0], 
                 :phiny_s.shape[1]] = phiny_s
    phiny_s_temp[phiny_s.shape[0]:, 
                 phiny_s.shape[1]:] = file_s['phiny']
    phiny_s         = phiny_s_temp
    phinz_s_temp    = np.zeros((phinz_s.shape[0] + file_s['phinz'].shape[0],
                                phinz_s.shape[1] + file_s['phinz'].shape[1]))
    phinz_s_temp[:phinz_s.shape[0], 
                 :phinz_s.shape[1]] = phinz_s
    phinz_s_temp[phinz_s.shape[0]:, 
                 phinz_s.shape[1]:] = file_s['phinz']
    phinz_s         = phinz_s_temp
    
    f_nl_s_temp    = np.zeros((f_nl_s.shape[0] + file_s['f_nl'].shape[0],
                                f_nl_s.shape[1] + file_s['f_nl'].shape[1]))
    f_nl_s_temp[:f_nl_s.shape[0], 
                 :f_nl_s.shape[1]] = f_nl_s
    f_nl_s_temp[f_nl_s.shape[0]:, 
                 f_nl_s.shape[1]:] = file_s['f_nl']
    f_nl_s         = f_nl_s_temp
    params_s        = np.concatenate((params_s, file_s["params"]))
    
    L_s  = params_s[0]
    mu_s = params_s[3]
    
    y_exc           = 2 * L_s / 3 
    if string_plucked == string_chosen:
        init_disp_temp  = np.sqrt((L_s * mu_s) / 2) * 2 / \
                        (L_s * (L_s - y_exc) * y_exc * file_s['kappan']**2) * \
                        (L_s * np.sin(file_s['kappan'] * y_exc) - y_exc * np.sin(file_s['kappan']*L_s))
    else:
        init_disp_temp = np.zeros(file_s['kappan'].size)
                    
    init_disp_s     = np.concatenate((init_disp_s, init_disp_temp))  
    
# Select modes under 20kHz

idx = wn_s < w_max_s

wn_s        = wn_s[idx]
mn_s        = mn_s[idx]
kn_s        = kn_s[idx]
cn_s        = cn_s[idx]
phinx_s     = phinx_s[idx]
phiny_s     = phiny_s[idx]
phinz_s     = phinz_s[idx]
f_nl_s      = f_nl_s[idx][:,idx] 
init_disp_s = init_disp_s[idx] 

phinx_s     = phinx_s.real
phiny_s     = phiny_s.real
phinz_s     = phinz_s.real

# Select vertical modes only
idx = np.sum(np.abs(phinz_s),axis=-1) != 0 
wn_s        = wn_s[idx]
mn_s        = mn_s[idx]
kn_s        = kn_s[idx]
cn_s        = cn_s[idx]
phinx_s     = phinx_s[idx]
phiny_s     = phiny_s[idx]
phinz_s     = phinz_s[idx]
f_nl_s      = f_nl_s[idx][:,idx] 
init_disp_s = init_disp_s[idx]

phin_s      = np.concatenate((phinx_s, phiny_s, phinz_s), axis=1)

Nm_s        = wn_s.size
N_s         = x_s.size

#%% Concatenate coordinates

N       = N_c + N_s
N_3     = 3 * N

x                   = np.concatenate((x_c, x_s))
y                   = np.concatenate((y_c, y_s))
z                   = np.concatenate((z_c, z_s))

#%% Pre-compute the modal radiation using the Rayleigh integral

rho_a  = 1.293 #masse volumqiue de l'air
c_a = 343 #célérité du son dans l'air
Fe_ac = 40e3 #sert à convertir les retards acoustiques en nb d'échantillons, doit etre égale à Fe donnée plus bas

sound_dir = os.path.join("Results", "Sound")
os.makedirs(sound_dir, exist_ok=True)

idx_surface = z_c == np.min(z_c)

xg = x_c[idx_surface] #points de grille
yg = y_c[idx_surface]
zg = z_c[idx_surface]

phiz_points_all = np.real(phinz_c[:, idx_surface]).T

x_ligne = np.unique(xg) #tri les valeurs des points de grille
y_ligne = np.unique(yg)

dx = np.mean(np.diff(x_ligne)) #ecart entre deux points selon l'axe x
dy = np.mean(np.diff(y_ligne)) #ecart entre deux points selon l'axe y
dS = dx * dy #element de surface utilisé lors des calculs d'intégrales

Ntheta_sigma = 24 #pas angulaires pour le calcul d'intégral du calcul d'efficacité
Nphi_sigma = 48

def sigma_rayleigh_mode(Phi, omega):
    Phi = np.real(Phi)
    omega = np.real(omega)
    int_phi2 = np.sum(Phi**2)*dS
    k = omega/ c_a
    dtheta = (np.pi / 2) / Ntheta_sigma
    dphi = (2 * np.pi) / Nphi_sigma
    P_rad = 0

    for it in range(Ntheta_sigma):
        theta = (it + 0.5) * dtheta
        sint = np.sin(theta)

        for ip in range(Nphi_sigma):
            phi_ang = (ip + 0.5) * dphi
            kx = k * sint * np.cos(phi_ang)
            ky = k * sint * np.sin(phi_ang)
            phase = kx * xg + ky * yg
            W_chapeau = np.sum(Phi * dS * np.exp(1j * phase)) #transformée de Fourier
            p_amp = rho_a * omega**2 / (2 * np.pi) * W_chapeau
            P_rad += (np.abs(p_amp)**2 / (2 * rho_a * c_a)) * sint * dtheta * dphi

    P_ref = 0.5 * rho_a * c_a * omega**2 * int_phi2
    sigma = P_rad / (P_ref + 1e-30)
    return float(np.real(sigma))

sigma_air_c = np.zeros(wn_c.size)

for im in range(wn_c.size):
    sigma_air_c[im] = sigma_rayleigh_mode(phiz_points_all[:, im], wn_c[im])

int_phi2_c = np.sum(phiz_points_all**2, axis=0) * dS
rhoh_eff_c = mn_c / np.maximum(int_phi2_c, 1e-30)
alpha_air_c = rho_a * c_a * sigma_air_c / (2 * rhoh_eff_c) #facteur d'amortissement de l'air
cn_air_c = 2 * mn_c * alpha_air_c

cn_c = cn_c + cn_air_c #ajout de l'amortissement de l'air

xc = 0.5 * (xg.min() + xg.max()) #abscisse du centre de la grille acoustique
yc = 0.5 * (yg.min() + yg.max()) #ordonnée du centre de la grille acoustique
zc = np.mean(zg)

r_spectator         = 2.0 #distance absolue entre le centre de la plaque et le spectateur
theta_spectator_deg = 5.0 #le spectateur est situé en face de la guitare
phi_spectator_deg   = 0.0

r_musician          = 0.50 #distance absolue entre le centre de la plaque et le musicien
theta_musician_deg  = 75.0 #incidence rasante pour le guitariste
phi_musician_deg    = 180.0

def point_micro(r, theta_deg, phi_deg): #renvoie les coordonnées cartésiennes des points d'écoute
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    xm = xc + r * np.sin(theta) * np.cos(phi)
    ym = yc + r * np.sin(theta) * np.sin(phi)
    zm = zc + r * np.cos(theta)
    return xm, ym, zm

x_mic_spectator, y_mic_spectator, z_mic_spectator = point_micro(r_spectator, theta_spectator_deg, phi_spectator_deg)
x_mic_musician, y_mic_musician, z_mic_musician = point_micro(r_musician, theta_musician_deg, phi_musician_deg)

def ajoute_retard(p, s, retard, coef):
    #ajoute dans p le signal s retardé de "retard" échantillons multiplié par le coefficient acoustique coef.
    #p: signal de pression à construire
    #s: contribution d'un signal dans p, dans notre cas c'est l'accélération modale
    #retard : retard entier en nombre d'échantillons
    #coef: coefficient acoustique dans la formule de l'intégrale de Rayleigh

    if retard < p.size:
       p[retard:] += coef * s[:p.size-retard]  #comme le signal est retardé, on tronque la fin de s pour ne pas dépasser p.size.


def regroupe_retards(rets, coefs): #regroupe les contributions qui ont le même retard temporel
    if len(rets) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    rets = np.array(rets, dtype=int)
    coefs = np.array(coefs, dtype=float)
    coeff = np.bincount(rets, weights=coefs, minlength=rets.max()+1) #np.bincount additionne les coefficients par valeur de retard
    ind = np.nonzero(np.abs(coeff) > 1e-18)[0]
    return ind.astype(int), coeff[ind].astype(float)


def termes_rayleigh_mode(Phi, x_mic, y_mic, z_mic):  #pré-calcule les termes Rayleigh pour un mode
    r = np.sqrt((x_mic - xg)**2 + (y_mic - yg)**2 + (z_mic - zg)**2) #distance entre chaque point de surface et le microphone
    ret = np.rint((r / c_a) * Fe_ac).astype(int) #retard converti en nb d'échantillons
    coef = rho_a / (2*np.pi) * Phi * dS / r #coeff acoustique pour la fonction retard
    return regroupe_retards(ret, coef)


def pression_depuis_termes(qdd_modes, termes):
    p = np.zeros(qdd_modes.shape[1])
    for im, (rets, coefs) in enumerate(termes):
        sig = qdd_modes[im] #accélération modale
        for ret, coef in zip(rets, coefs):
            ajoute_retard(p, sig, ret, coef)
    return p

def filtre_termes(termes, idx): #idx = wn_c < 2*pi*f_lim
    ind = np.where(idx)[0]
    return [termes[k] for k in ind]

freq_c_all = np.real(wn_c) / (2*np.pi)

rad_rayleigh_spectator  = []
rad_rayleigh_musician   = []


print("=== pré-calcul rayonnement modal ===")
print("points acoustiques =", xg.size)
print("modes structure initiaux =", phiz_points_all.shape[1]) #combien de modes gardés avec telle f_lim
print("micro spectateur =", x_mic_spectator, y_mic_spectator, z_mic_spectator)
print("r spectateur =", r_spectator, "theta =", theta_spectator_deg, "phi =", phi_spectator_deg)
print("micro musicien =", x_mic_musician, y_mic_musician, z_mic_musician)
print("r musicien =", r_musician, "theta =", theta_musician_deg, "phi =", phi_musician_deg)
print("dx =", dx, "dy =", dy, "dS =", dS)

for im in range(phiz_points_all.shape[1]):
    Phi = phiz_points_all[:, im]
    rad_rayleigh_spectator.append(termes_rayleigh_mode(Phi, x_mic_spectator, y_mic_spectator, z_mic_spectator))
    rad_rayleigh_musician.append(termes_rayleigh_mode(Phi, x_mic_musician, y_mic_musician, z_mic_musician))

print("Pré-traitement Rayleigh terminé")

#%% Time-domain parameters

T       = 5
Fe_simu = 70e3
Te_simu = 1/Fe_simu
t_simu  = np.arange(0,T,Te_simu)
N_t_simu= t_simu.size
Fe      = 40e3
Te      = 1 / Fe 
t       = np.arange(0,T,Te)
N_t     = t.size

uddz            = np.zeros((f_lim.size, N_t)) # Bridge acceleration
P_spectator     = np.zeros((f_lim.size, N_t)) # Pressure at spectator position
P_musician      = np.zeros((f_lim.size, N_t)) # Pressure at musician position


for j in range(f_lim.size):
    # Select modes whose modal frequencies are under f_lim
    
    idx = np.isfinite(wn_c) & (wn_c < (2 * np.pi * f_lim[j]))

    wn_c        = wn_c[idx]
    mn_c        = mn_c[idx]
    kn_c        = kn_c[idx]
    cn_c        = cn_c[idx]
    phinx_c     = phinx_c[idx]
    phiny_c     = phiny_c[idx]
    phinz_c     = phinz_c[idx]

    rad_rayleigh_spectator = filtre_termes(rad_rayleigh_spectator, idx)
    rad_rayleigh_musician = filtre_termes(rad_rayleigh_musician, idx)


    # Select only the radiation terms for modes under f_lim

    phin_c = np.concatenate((phinx_c, phiny_c, phinz_c), axis=-1)

    Nm_c            = wn_c.size
    
    print("Structure with "+str(Nm_c)+" modes and flim = "+str(f_lim[j]) + " Hz")
    
    Nm      = Nm_c + Nm_s
    N_c_3   = 3 * N_c
    N_s_3   = 3 * N_s
    
    phix                = np.zeros((N, Nm))
    phix[:N_c,:Nm_c]    = phinx_c.T 
    phix[N_c:,Nm_c:]    = phinx_s.T
    phiy                = np.zeros((N, Nm))
    phiy[:N_c,:Nm_c]    = phiny_c.T
    phiy[N_c:,Nm_c:]    = phiny_s.T
    phiz                = np.zeros((N, Nm))
    phiz[:N_c,:Nm_c]    = phinz_c.T
    phiz[N_c:,Nm_c:]    = phinz_s.T
    phi                 = np.zeros((N_3, Nm))
    phi[:N_c_3, :Nm_c]  = phin_c.T
    phi[N_c_3:, Nm_c:]  = phin_s.T 
    f_nl                = np.zeros((Nm,Nm))
    f_nl[Nm_c:,Nm_c:]   = f_nl_s
    M                   = np.diag(np.concatenate((mn_c, mn_s)))
    C                   = np.diag(np.concatenate((cn_c, cn_s)))
    K                   = np.diag(np.concatenate((kn_c, kn_s)))
    
    #%% Compute coupling surface
    
    idx_temp    = np.zeros((0,2), dtype=int)
    
    for i in range(N_c):
        d = (x[N_c:] - x[i])**2 + (y[N_c:] - y[i])**2 + (z[N_c:] - z[i])**2
        if np.sqrt(d.min()) < 1e-5:
            idx_temp = np.concatenate((idx_temp, 
                                        np.array((i, N_c_3+d.argmin()))[np.newaxis]), 
                                       axis=0)
    
    idx_coupl = np.zeros((0,2), dtype=int)
    idx_coupl = np.concatenate((idx_coupl,idx_temp + 2 * np.array([N_c,N_s])))
    
    #%% Constraint matrix formualtion A qdd = 0
    
    A   = phi[idx_coupl[:,0]] - phi[idx_coupl[:,1]]
    
    # Normalize each constraint line
    
    A = A / np.max(np.abs(A),axis=1)[:,np.newaxis]
    
    # Select only the linearly independant lines
    AT              = np.transpose(A)
    Q,R,P           = qr(AT, pivoting=True)
    idx_independant = (np.abs(np.diag(R)) / np.abs(np.diag(R)).max()) > 1e-2
    idx_independant = P[idx_independant]
    AT              = AT[:,idx_independant]
    A               = np.transpose(AT)
    
    idx_coupl       = idx_coupl[idx_independant]
    
    #%% UK initial matrices formulation
    
    A_plus      = np.linalg.pinv(A)
    M_inv       = np.diag(1/M.diagonal())
    M_rt_inv    = np.diag(M.diagonal()**(-1/2))
    B           = A @ M_rt_inv
    B_plus      = np.linalg.pinv(B)
    V           = np.eye(Nm) - M_rt_inv @ B_plus @ A @ M_inv
    W           = - M_rt_inv @ B_plus @ A @ M_inv
    
    CORR        =  np.eye(Nm) - A_plus @ A
    
    #%% Matrix optimization 
    
    k_diag = K.diagonal()
    c_diag = C.diagonal()
    
    #%% UK excitation
    
    def give_F_gauss_ponctual(t, phix_exc, phiz_exc, angle):
        T_exc   = 0.0001
        F_exc   = 10 * np.exp(-(t-5*T_exc)**2/(2*T_exc**2))
        F_mod   = F_exc * phix_exc * np.cos(angle)
        F_mod   += F_exc * phiz_exc * np.sin(angle)
        return F_mod
    
    def give_F_lin_ponctual(t, phix_exc, phiz_exc, angle):
        T_exc   = 0.01
        F_exc   = 0
        if t<T_exc:
            F_exc = 10 * t/T_exc
        F_mod   = F_exc * phix_exc * np.cos(angle)
        F_mod   += F_exc * phiz_exc * np.sin(angle)
        return F_mod
    
    def give_F_null_ponctual(t, phix_exc, phiz_exc, angle):
        F_mod = np.zeros(phix_exc.size)
        return F_mod
    
    def give_F_cos_ponctual(t, phix_exc, phiz_exc, angle):
        T_exc   = 0.0001
        F_exc   = 0
        if t<T_exc:
            F_exc   = 10/2 * (1 - np.cos(2*np.pi*t/T_exc))
        F_mod   = F_exc * phix_exc * np.cos(angle)
        F_mod   += F_exc * phiz_exc * np.sin(angle)
        return F_mod
    
    exc_func    = give_F_null_ponctual
    idx_exc     = np.argmin(np.abs(y - y_exc))
    angle_exc   = np.pi/2
    
    #%% UK initial conditions / vector initialization
    
    angle_pinch     = np.pi/2
    idx_hor         = np.zeros(Nm, dtype=bool)
    idx_hor[Nm_c:]  = ((phix[:,Nm_c:]!=0).sum(axis=0) != 0)
    idx_vert        = np.zeros(Nm, dtype=bool)
    idx_vert[Nm_c:] = ((phiz[:,Nm_c:]!=0).sum(axis=0) != 0)
    
    q               = np.zeros((Nm,N_t), dtype=float)
    q[idx_hor,0]    = init_disp_s[idx_hor[Nm_c:]]  * 3e-3 * np.cos(angle_pinch)
    q[idx_vert,0]   = init_disp_s[idx_vert[Nm_c:]] * 3e-3 * np.sin(angle_pinch)
    qd              = np.zeros((Nm,N_t), dtype=float)
    qdd             = np.zeros((Nm,N_t), dtype=float)
    F_c             = np.zeros((Nm,N_t), dtype=float)
    F_ext           = np.zeros((Nm,N_t), dtype=float)
    
    plt.figure()
    plt.plot(y[N_c:],phi[N_c_3+2*N_s:] @ q[:,0])
    plt.axis("equal")
    
    q_next          = q[:,0]
    qd_next         = qd[:,0]
    qdd_next        = qdd[:,0]
    F_ext_next      = F_ext[:,0]
    F_c_next        = F_c[:,0]
    q_low_next      = q[:,0]
    qd_low_next     = qd[:,0]
    qdd_low_next    = qdd[:,0]
    F_ext_low_next  = F_ext[:,0]
    F_c_low_next    = F_c[:,0]
     
    delay       = 0
    n           = 1 
    beta        = np.exp(-2*np.pi*Fe/2*Te_simu) 
    
    #%% UK loop
    
    start = time.time()
    print("----------- Begin UK simulation -----------")
    for i in range(N_t_simu-1):
        if i%(N_t_simu//(100/5))==0 and i!=0: 
            time_diff = time.time() - start
            remaining_time = (100 - i//(N_t_simu//(100/1))) * time_diff/(i//(N_t_simu//(100/1)))
            print(str(i//(N_t_simu//(100/1)))+"% (time = "+str(np.round(time_diff))+
                  " s)\tEstimated time before end : "+str(int(remaining_time//60))+" min "
                  + str(int(remaining_time%60))+" s")
        
        q_prev          = q_next
        qd_prev         = qd_next 
        qdd_prev        = qdd_next
        F_ext_prev      = F_ext_next
        F_c_prev        = F_c_next
        q_low_prev      = q_low_next
        qd_low_prev     = qd_low_next
        qdd_low_prev    = qdd_low_next
        F_ext_low_prev  = F_ext_low_next
        F_c_low_prev    = F_c_low_next
        
        q_next    = q_prev + Te_simu * qd_prev + 0.5 * Te_simu**2 * qdd_prev
        q_next    = CORR @ q_next
        qd_half   = qd_prev + 0.5 * Te_simu * qdd_prev
        
        F_ext_next  = exc_func(t_simu[i+1], phix[idx_exc], phiz[idx_exc], angle_exc)
        F_ext_next  = F_ext_next + (q_next.real) * (f_nl @ (q_next.real)**2) # Nonlinearity
        F           = - k_diag * q_next - c_diag * qd_half + F_ext_next
        
        qdd_next    = V @ F
        F_c_next    = W @ F
        
        qd_next   = qd_prev + 0.5 * Te_simu * (qdd_prev + qdd_next)
        qd_next   = CORR @ qd_next
        
        delay += Te_simu
        
        q_low_next      = low_pass(q_next,      q_low_prev, beta) 
        qd_low_next     = low_pass(qd_next,     qd_low_prev, beta)
        qdd_low_next    = low_pass(qdd_next,    qdd_low_prev, beta)
        F_ext_low_next  = low_pass(F_ext_next,  F_ext_low_prev, beta) 
        F_c_low_next    = low_pass(F_c_next,    F_c_low_prev, beta)
        
        if delay > Te:
            q[:,n]      = q_low_prev        + ((n*(Te%Te_simu))%Te_simu)/ Te_simu * (q_low_next - q_low_prev)
            qd[:,n]     = qd_low_prev       + ((n*(Te%Te_simu))%Te_simu)/ Te_simu* (qd_low_next - qd_low_prev)
            qdd[:,n]    = qdd_low_prev      + ((n*(Te%Te_simu))%Te_simu)/ Te_simu* (qdd_low_next - qdd_low_prev)
            F_ext[:,n]  = F_ext_low_prev    + ((n*(Te%Te_simu))%Te_simu)/ Te_simu* (F_ext_low_next - F_ext_low_prev)
            F_c[:,n]    = F_c_low_prev      + ((n*(Te%Te_simu))%Te_simu)/ Te_simu* (F_c_low_next - F_c_low_prev)
            n           += 1
            delay       = delay - Te
       
    print("\nTemps total : " + str(int(time_diff//60)) + " min " + (str(int(time_diff%60)) + ' s'))
    print("------------ End UK simulation ------------")
    
    
    uddz[j]         = phi[idx_temp[0][0]] @ qdd
    
    qdd_struct = qdd[:Nm_c]

    print("calcul pression Rayleigh direct...")
    t0_ray          = time.perf_counter()
    P_spectator[j]  = pression_depuis_termes(qdd_struct, rad_rayleigh_spectator)
    P_musician[j]   = pression_depuis_termes(qdd_struct, rad_rayleigh_musician)
    temps_rayleigh  = time.perf_counter() - t0_ray

    print("=== rayonnement f_lim =", f_lim[j], "Hz ===")
    print("rayleigh direct :", temps_rayleigh, "s")
    print("===========================================")


#%%

np.savez(
    os.path.join(sound_dir, "pressions_acoustiques_flim.npz"),
    t=t,
    Fe=Fe,
    f_lim=f_lim,
    uddz=uddz,
    P_spectator_rayleigh=P_spectator,
    P_musician_rayleigh=P_musician,
    x_mic_spectator=x_mic_spectator,
    y_mic_spectator=y_mic_spectator,
    z_mic_spectator=z_mic_spectator,
    r_spectator=r_spectator,
    theta_spectator_deg=theta_spectator_deg,
    phi_spectator_deg=phi_spectator_deg,
    x_mic_musician=x_mic_musician,
    y_mic_musician=y_mic_musician,
    z_mic_musician=z_mic_musician,
    r_musician=r_musician,
    theta_musician_deg=theta_musician_deg,
    phi_musician_deg=phi_musician_deg,
    name_struct=name_struct,
    file_c_path=file_c_path,
    string_plucked=string_plucked,
)

scaled = (uddz / np.max(np.abs(uddz)) * np.iinfo(np.int32).max).astype(np.int32)

for i in range(uddz.shape[0]):
    write(os.path.join(sound_dir, "audio_f-lim_"+str(int(f_lim[i]))+"_Hz_bridge_acc.wav"), int(Fe), np.pad(scaled[i], (int(1*Fe),0)))

    write(os.path.join(sound_dir, "audio_f-lim_"+str(int(f_lim[i]))+"_Hz_rayleigh_spectateur.wav"),
          int(Fe), np.pad(scaled[i], (int(1*Fe),0)))

    write(os.path.join(sound_dir, "audio_f-lim_"+str(int(f_lim[i]))+"_Hz_rayleigh_musician.wav"),
          int(Fe), np.pad(scaled[i], (int(1*Fe),0)))

