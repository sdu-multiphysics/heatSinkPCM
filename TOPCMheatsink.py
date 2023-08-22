""" 
This code is made available under a BSD 3-clause license. See LICENSE for further information.

This script is a minimal working example of how to perform gradient-based topology optimisation 
based on a homogenisation interpolation and GCMMA with FEniCS. This exaple minimises the temporal
average variance of the spatial average temperature at the heat source boundary by placing phase
change material (PCM) and highly thermally conductive material (HCM).


To run this you need
- legacy FEniCS (https://fenicsproject.org/documentation/), to install in VSC in Linux use following commands:
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics
- dolfin adjoint (https://www.dolfin-adjoint.org/en/latest/), to install in VSC in Linux use following commands:
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
git clone https://bitbucket.org/dolfin-adjoint/pyadjoint.git
- MMA/GCMMA code (the MMA.py file) from Arjee Deetman (https://github.com/arjendeetman/GCMMA-MMA-Python), download from github

If you do not have a linux computer available a virtual machine with Linux can be used.
I have had good experiences with WMware and Oracle VM Virtualbox.

By Mark Christensen 
28-07-23

Disclaimer:                                                            
The author does not guarantee that the code is free from errors.     
Furthermore, the author shall not be liable in any event caused by the use of the program.
"""
# ##################################################### #
#                       LIBRARIES                       #
# ##################################################### #

from fenics import *
from fenics_adjoint import *    # Note: It is important that fenics_adjoint is called after fenics
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

from MMA import gcmmasub,subsolv,kktcheck,asymp,concheck,raaupdate # Import MMA functions

set_log_level(LogLevel.ERROR) # Only print message if error message

# ##################################################### #
#                     PREPROCESSING                     #
# ##################################################### #

# ----- PHYSICAL MODEL DATA -----
# Material constants
khcm = 10           # Thermal conductivity og HCM [W/(m*K)]
cphcm = 1           # Specific heat capacity of HCM [J/(K*kg)]
rhohcm = 1          # Mass density density of HCM [kg/m³]

kpcm = 1e-3*khcm    # Thermal conductivity og PCM [W/(m*K)]
cppcm = 1           # Specific heat capacity of PCM [J/(K*kg)]
rhopcm = 1          # Mass density density of PCM [kg/m³] 
Tmelt = 0.5         # Melting temperature of PCM [K]
dTmelt = 0.5        # Melting temperture range of PCM [K] 
Lheat = 10          # Latent heat of fusion [J/kg]

# Thermal data
qheat = 1           # Heat production rate in electronics [W]
w = 1               # Ocsilation frequency of heat source [Hz]
hconv = 5           # Het atransfer coefficient [W/(m²*K)]
Tinf = 0            # Temperature of surroundings [K]
Tinitial = 0        # Initial temperature [K]

# Temporal data
tfin = 20           # Final time [s]
num_steps = 500     # Number of time steps
dt = tfin/num_steps # Time step size [s]

# Finite element data
nx = ny = 100       # Number of element in x and y direction
lx = ly = 1         # Dimension of the 2D heatsink [m]
lz = 1              # Out of plane thickness of the 2D heatsink [m]
degree_phys = 1     # Element order for physical problem


# ----- OPTIMIZATION ELEMENT DATA -----
# Loop
niter = 300         # Max number of optimisation iterations
tol_opt = 1E-3      # Convergence tolerance, based on the absolute error

# Problem parameters
volfrac = 0.3       # Maximum allowable volume fraction used in the optimisation

# Filter data
r = 0.01            # Length scale for the Helmholzt PDE filterer            

# ----- DEFINING DOMAIN AND FUNCTION SPACES -----
# Create mesh
mesh = RectangleMesh(Point(-lx/2, -ly/2), Point(lx/2, ly/2), nx, ny, "crossed")

# Function space
V0 = FunctionSpace(mesh, "DG", 0)                   # Discontinous function space
Vphys = FunctionSpace(mesh, "CG", degree_phys)      # Continous function space 

# ----- BOUNDARY CONDITIONS -----
# Define the Heat source and convection boundaries were the Neumann BC are a aplied.
class HeatSourceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol_bc = 1E-14
        return on_boundary and near(x[1], -ly/2, tol_bc) and x[0]+lx/4>=-tol_bc and x[0]-lx/4<=tol_bc
HS_boundary = HeatSourceBoundary()

# Applying Neuman boundary condition to simulate the heat loss due to convection
class ConvectionBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol_bc = 1E-14
        return on_boundary and near(x[1], ly/2, tol_bc)
C_boundary = ConvectionBoundary()

# Marking the facets (boundaries) that coresspond to HeatSourceBoundary and ConvectionBoundary
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)  # store all facets in array
HS_boundary.mark(boundaries, 1)     # Mark all facest that are on the HS_boundary with 1
C_boundary.mark(boundaries, 2)      # Mark all facest that are on the c_boundary with 2
# Denfine the line intrgrals
dsHS = ds(subdomain_id=1, subdomain_data=boundaries)    
dsC = ds(subdomain_id=2, subdomain_data=boundaries)

# Define volume and area for later use
voltot = assemble(Constant(lz)*dx(domain=mesh))     # Compute total volume Of design domain
AHS = assemble(Constant(lz)*dsHS(domain=mesh))      # Compute crossectionl area of heat source

# ##################################################### #
#                       FUNCTIONS                       #
# ##################################################### #

# Apparent heat capacity method
def cppcmmod(T_,cppcm_, Lheat_):
    """
    Incoorporates the latent heat of fusion into the heat capacity with
    the aparent heat capacity method with heaviside step functions. 
    This function assumes no change in     spefic heat capacity due to phase change.
    
    Inputs: T_ is the temperture field, cppcm_ is the heat capacity without 
    phase change, Lheat_ is the latent heat of fusion.
    Outputs: Field of the modified heat capacity
    """
    k_H = 25        # Steepness of Heviside step function 
    return cppcm_ + Lheat_ / (dTmelt) * \
        (
            1 / (1 + exp(-2 * k_H * (T_ - (Tmelt - dTmelt/2))))
            - (1 / (1 + exp(-2 * k_H * (T_ - (Tmelt + dTmelt/2)))))
        )

def PDEfilter(u_, r_, mesh_):
    """
    Applies ering to a field by solving af PDEfilter type PDE 
    based on the work of B.S. Lazarov and O. Sigmund DOI: 10.1002/nme.3072
    
    Inputs: u_ is the unfiltered field, r_ is the characteristic length
    Outputs: filtered field u_tilde
    """
    Vold = u_.function_space()            # Saving function space from u_
    VCG = FunctionSpace(mesh_, "CG", 1)    # Defining continous function space

    # Defining variational problem
    u_tilde = TrialFunction(VCG)
    vf = TestFunction(VCG)
    # HELMHOLTZ-TYPE PDE (u_ is projected to a continouos function space)
    F = u_tilde*vf*dx + r_**2*dot(grad(u_tilde), grad(vf))*dx - project(u_, VCG)*vf*dx(mesh_)
    # a: unknowns, L: knowns
    a, L = lhs(F), rhs(F)

    u_temp = Function(VCG)
    u_tilde = Function(Vold)

    # Compute solution and project onto inital function space
    solve(a == L, u_temp)
    u_tilde.assign(project(u_temp, Vold))

    return u_tilde

def forward(rho_tilde_):
    """
    Solves the forward problem (solving PDE) using homogenisation

    Inputs: rho_tilde is the filtered material density variable field
    Outputs: Variables used for computing the objective function
    """
    # ----- DEFINING PROBLEM -----
    # Material interpolations
    def rhocppcmphys(rho_,T_):
        return (rhohcm*cphcm)*(rho_)+(1-rho_)*(rhopcm*cppcmmod(T_,cppcm, Lheat))

    def kphys(rho_):
        a = 1-(1-rho_)**0.5
        return 1/(a/khcm + (1-a)/(kpcm*(1-a)+khcm*a))

    # Defining the flucturating heat source
    qelec = Expression("qheat/(AHS)*(1+sin(2*DOLFIN_PI*w*t))",
                        qheat=qheat, AHS=AHS, lz=lz, w=w, t=0, degree=0)

    # Define initial values
    T_ini = Constant(Tinitial)
    T_n = interpolate(T_ini, Vphys)

    # Defining the trial and test functions
    T = TrialFunction(Vphys)
    v = TestFunction(Vphys)

    # Weak form rho_k=0 -> PCM, rho_k = 1 -> HCM
    F = rhocppcmphys(rho_tilde_,T_n)*(T-T_n)/dt*v*dx \
        + kphys(rho_tilde_)*dot(grad(T),grad(v))*dx \
        - qelec*v*dsHS \
        + hconv*(T-Tinf)*v*dsC
    # a: unknowns, L: knowns
    a, L = lhs(F), rhs(F)

    # ------ SOLVING THE PROBLEM -----

    # Initialise the field for the Temperature
    T = Function(Vphys) 

    # Set initial condistions
    T_n.assign(interpolate(T_ini, Vphys))
    T.assign(interpolate(T_ini, Vphys))
    t = 0
    qelec.t = t

    # Save the average temperture at the heat source
    T_elec_array = [assemble(T*dsHS)/AHS]   

    # ------ Solving Physics  -----
    for timestep in range(num_steps):
        t += float(dt)      # Update time
        qelec.t = t         # Update heat source
        solve(a == L, T)    # Solve problem for current time step with lieaner solver

        T_n.assign(T)       # Update previous temperature
        T_elec_array.append(assemble(T*dsHS)/AHS)

    # Compute the variance of the average temperture at the heat source over time
    T_time_mean = stat.mean(T_elec_array)        # Average over time
    T_var = []

    for timesteps in range(len(T_elec_array)):
        T_var.append((T_elec_array[timesteps] - T_time_mean)**2)

    return T_var
   
def Optimization(rho_,rho_tilde_, rf_f0_, rf_h_,f0_history_,Mnd_history_):
    """
    Performs gradient based topology optimazation using the Methode of moving assymptotes
    based on a MMA and GCMMA script by Krister Svanberg (https://people.kth.se/~krille/mmagcmma.pdf)
    which is translated to Python Arjeen Deetman (https://github.com/arjendeetman/GCMMA-MMA-Python)

    Inputs: rho is the density variable field, rho_tilde is the filtered density variable field, 
    rf_f0 is the reduced functional for f0, rf_h is the reduced functional for h, f0_history is 
    a empty list for storing the history of f0, and Mnd_history is a empty list for storing the
    history of h.
    Outputs: rho and rho_tilde at the final design and f0_history, Mnd_history
    """
    with pyadjoint.stop_annotating() as _: # Stop saving opterations with dolfin-adjoint
        
        # ------ INITIALASING LOOP -----
        # Preallocate rho and rho_tilde for each iteration
        rho_k = rho_
        rho_tilde_k = rho_tilde_

        # Initialize GCMMA parameters
        m = 1                                       # Number of constraints
        n = rho.vector()[:].size                    # Number of design variables
        xval = rho.vector()[:].reshape(-1, 1)       # Initial design variables
        epsimin = 0.0000001
        eeen = np.ones((n,1))
        eeem = np.ones((m,1))
        zeron = np.zeros((n,1))
        zerom = np.zeros((m,1))
        xold1 = xval.copy()
        xold2 = xval.copy()
        xmin = 0*eeen.copy()
        xmax = 1*eeen.copy()
        low = xmin.copy()
        upp = xmax.copy()
        c = 1000*eeem
        d = eeem.copy()
        a0 = 1
        a = zerom.copy()
        raa0 = 0.01
        raa = 0.01*eeem
        raa0eps = 0.000001
        raaeps = 0.000001*eeem
        outeriter = 0	

        # Initilize preliminary design 
        f0_k = rf_f0_(rho_k)
        h_k = rf_h_(rho_k)
        Mnd = assemble(4*rho_tilde_k*(1-rho_tilde_k)*dx)/(lx*ly)*100
        # Calculate Sensitivities
        df0drho = Function(V0, name="Object function Sensitivity")
        dhdrho = Function(V0, name="Constraint function Sensitivity")
        df0drho.assign(rf_f0_.derivative())                     
        dhdrho.assign(rf_h_.derivative())

        # Scaling functions and storing them in matrices for the MMA solver
        xval = rho_k.vector()[:].reshape(-1, 1)                       # Saving design variable in xval
        scale = np.abs(f0_k/10)
        f0val = 1 + f0_k/scale                                        # Scaling f0 at xval
        df0dx = df0drho.vector()[:].reshape(-1, 1)/scale           # nx1 matrix of sensitivities of f0 at xval
        fval = h_k                                                    # Array of constraint functions at xval
        dfdx = dhdrho.vector()[:].reshape(m, n)  
    
        # Initialize the outer iteration and convergence counter
        outit = 0
        cc = 0 

        # ----- OPTIMIZATION LOOP -----
        while (outit < niter):  
            outit += 1
            outeriter += 1

            if outit > 1:   # update prev values for convergence check
                f0_kprev = f0_k
                Mndprev = Mnd
            
            # The parameters low, upp, raa0 and raa are calculated:
            low,upp,raa0,raa= \
                asymp(outeriter,n,xval,xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx,dfdx)
            # The MMA subproblem is solved at the point xval:
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp= \
                gcmmasub(m,n,iter,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d)

            # Evaluate the functionals with the new deisgn
            rho_k.vector()[:] = xmma.flatten()          # Overwrite rho with the new rho from the MMA solver
            f0_knew = rf_f0_(rho_k)
            h_knew = rf_h_(rho_k)
            f0valnew = np.array([1 + f0_knew/scale])    # Scaling f0 at xval
            fvalnew = np.array([h_knew])

            # It is checked if the approximations are conservative:
            conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)

            # While the approximations are non-conservative (conserv=0), repeated inner iterations are made:
            innerit = 0
            if conserv == 0:
                while conserv == 0 and innerit < 2:
                    innerit += 1
                    # New values on the parameters raa0 and raa are calculated:
                    raa0,raa = raaupdate(xmma,xval,xmin,xmax,low,upp,f0valnew,fvalnew,f0app,fapp,raa0, \
                        raa,raa0eps,raaeps,epsimin)
                    # The GCMMA subproblem is solved with these new raa0 and raa:
                    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,n,iter,epsimin,xval,xmin, \
                        xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d)
                    
                    # Evaluate the functionals with the new deisgn
                    rho_k.vector()[:] = xmma.flatten()    # Overwrite rho with the new rho from the MMA solver
                    f0_knew = rf_f0_(rho_k)
                    h_knew = rf_h_(rho_k)
                    f0valnew = np.array([1 + f0_knew/scale])                                        # Scaling f0 at xval
                    fvalnew = np.array([h_knew])
                    # It is checked if the approximations have become conservative:
                    conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)

            # Updat vectors:
            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()

            # Re-calculate function values and gradients of the objective and constraints functions
            rho_k.vector()[:] = xmma.flatten()    # Overwrite rho with the new rho from the MMA solver
            rho_tilde_k.assign(PDEfilter(rho_k, r, mesh))
            f0_k = rf_f0_(rho_k)
            h_k = rf_h_(rho_k)
            df0drho.assign(rf_f0_.derivative())                     
            dhdrho.assign(rf_h_.derivative())

            # Scaling functions and storing them in matrices for the MMA solver
            f0val = 1 + f0_k/scale                                        # Scaling f0 at xval
            df0dx = df0drho.vector()[:].reshape(-1, 1)/scale           # nx1 matrix of sensitivities of f0 at xval
            fval = h_k                                                    # Array of constraint functions at xval
            dfdx = dhdrho.vector()[:].reshape(m, n)  

            # Store f0
            f0_history_.append(f0_k)   # Save f0 in f0 history

            Mnd = assemble(4*rho_tilde_k*(1-rho_tilde_k)*dx)/(lx*ly)*100
            Mnd_history_.append(Mnd)

            # Chect for convergence
            if outit > 1:
                 # Check if converged
                f0ichange = abs((f0_k-f0_kprev)/(f0_k))
                Mndchange = abs((Mnd-Mndprev)/(Mnd) )

                if f0ichange <= tol_opt and Mndchange <= tol_opt:
                    cc += 1
                    if cc > 3:
                        print("Tolerance Reached!!")
                        break 
                else:
                    cc = 0
                print("Iteration " + str(outeriter) + "." + str(innerit) + ": f0 = " + str(f'{f0_k:.3f}') + ", Mnd = " + str(f'{Mnd:.2f}')+", f0ichange = " + str("{:.2e}".format(f0ichange))+", Mndchange = " + str("{:.2e}".format(Mndchange)))               

            
            # ----- PLOTS While Solving -----               
            plt.figure(1)       # plot of current design
            plt.clf()
            plt.colorbar(plot(rho_tilde_k, cmap="Greys", vmin=0, vmax=1))
            plt.xlim([-lx/2, lx/2])
            plt.ylim([-ly/2, ly/2])
            plt.title("rho_tilde at iter " + str(outit)+": f0 = "+ str(f'{f0_history_[-1]:.2f}')+ ", Mnd = " + str(f'{Mnd_history[-1]:.2f}'))
            plt.pause(0.05)

            plt.figure(2)       # plot of f0 and Mnd history
            plt.clf()
            plt.plot(np.arange(1, len(f0_history_)+1)-1, f0_history_, color="red")
            ax = plt.gca()
            plt.xlabel("Number of iterations")
            plt.ylabel("Objective", color="red")
            plt.tick_params(axis="y", which="both", labelcolor="red")
            plt.title("Convergence history")
            ax2 = ax.twinx()
            ax2.plot(np.arange(1, len(Mnd_history_)+1)-1, Mnd_history_, color="blue")
            ax2.set_ylabel("Measure of non-discreteness", color="blue")
            ax2.tick_params(axis="y", which="both", labelcolor="blue")
            plt.pause(0.05)

    return rho_k,rho_tilde_k, f0_history_, Mnd_history_

if __name__ == '__main__':
    # ----- INITIALIZE DENSITIES -----
    rho = interpolate(Constant(volfrac),V0) # Define initial design
    rho_tilde = PDEfilter(rho, r, mesh)           # Apply filter

    T_var = forward(rho_tilde) # Solve forward problem (physical model)
    
    # Objective and Constraint function
    h = assemble(rho_tilde*lz*dx)/(volfrac*voltot) - 1
    f0 = sum(T_var)/len(T_var)

    # Reduced functionals
    control = Control(rho)  
    rf_f0 = ReducedFunctional(f0, control)  
    rf_h = ReducedFunctional(h, control)  

    # Initiate f0 and Mnd historry lists
    f0_history = []
    Mnd_history = []
    
    rho,rho_tilde, f0_history, Mnd_history = Optimization(rho, rho_tilde, rf_f0, rf_h, (f0_history), (Mnd_history))

    # ----- PLOTS -----  
    plt.figure(1) #The filal design
    plt.clf()
    plt.colorbar(plot(rho_tilde, cmap="Greys", vmin=0, vmax=1))
    plt.xlim([-lx/2, lx/2])
    plt.ylim([-ly/2, ly/2])
    plt.title("rho_tilde at iter " + str(iter)+": f0 = "+ str(f'{f0_history[-1]:.2f}')+ ", Mnd = "+ str(f'{Mnd_history[-1]:.2f}'))

    plt.figure(2) #The f0 and Mnd history
    plt.clf()
    plt.plot(np.arange(1, len(f0_history)+1)-1, f0_history, color="red")
    ax = plt.gca()
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective", color="red")
    plt.tick_params(axis="y", which="both", labelcolor="red")
    plt.title("Convergence history")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(Mnd_history)+1)-1, Mnd_history, color="blue")
    ax2.set_ylabel("Measure of non-discreteness", color="blue")
    ax2.tick_params(axis="y", which="both", labelcolor="blue")
    plt.show()