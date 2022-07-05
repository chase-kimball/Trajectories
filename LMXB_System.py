import Observed #dictionary file
import TrajTools
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.stats as stats
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy import constants as C
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.integrate import ode


Msun=C.M_sun.value
m_kpc=u.kpc.to(u.m)


## Class for systems

class LMXB_Sys:
    def __init__(self, sysname):
        self.system = Observed.getSysDict(sysname)
        
        ## Assigns the parameters of each system to a variable
        self.d = self.system.get('d')
        self.pm_ra = self.system.get('pm_ra')
        self.pm_dec = self.system.get('pm_dec')
        self.v_rad = self.system.get('v_rad')
        
        self.ra = self.system.get('ra')
        self.dec = self.system.get('dec')
        if not self.ra:
            
            Sys_gal = SkyCoord(l = self.system.get('l')*u.deg, b = self.system.get('b')*u.deg, frame = 'galactic')
            self.ra = Sys_gal.icrs.ra.value
            self.dec = Sys_gal.icrs.dec.value
             
        def getCartesian(self):
            self.X0, self.Y0, self.Z0, self.Vx0, self.Vy0, self.Vz0, self.randparams = TrajTools.getXYZUVW(self.ra, self.dec, self.d[0], self.pm_ra[0], self.pm_dec[0], self.v_rad[0])
        getCartesian(self)
        
    def setRandUVWXYZ(self,skew=False):
        if skew: d = self.system.get('d_skew')
        else: d = self.d
        self.X0, self.Y0, self.Z0, self.Vx0, self.Vy0, self.Vz0, self.randparams = TrajTools.getRandXYZUVW(self.ra, self.dec, d, self.pm_ra, self.pm_dec, self.v_rad, skew=skew)
        
        
 
        
        
    def plotTrajectory(self, T, dt, integrator):
        
        R0 = np.array([self.U0*1000, self.V0*1000, self.W0*1000, self.X0*u.kpc.to(u.m), self.Y0*u.kpc.to(u.m), self.Z0*u.kpc.to(u.m)]) 
        #print(R0,dt*u.Gyr.to(u.s),T*u.Gyr.to(u.s), TrajTools.vdot(T, R0))
        R_t, t = integrator(R0,dt*u.Gyr.to(u.s),T*u.Gyr.to(u.s), TrajTools.vdot)
        self.R_t = R_t
        self.R_t[0] = R_t[0]/1000
        self.R_t[1] = R_t[1]/1000
        self.R_t[2] = R_t[2]/1000
        self.R_t[3] = R_t[3]*u.m.to(u.kpc)
        self.R_t[4] = R_t[4]*u.m.to(u.kpc)
        self.R_t[5] = R_t[5]*u.m.to(u.kpc)
        self.t = t
        print(self.U0,self.V0,self.W0,self.X0,self.Y0,self.Z0)
        print(R0)
        plt.plot(self.R_t[3], self.R_t[4])
    
       
def doMotion(lmxb, T, Vdot = TrajTools.vdot, M=[1.45e11*Msun, 9.3e9*Msun, 1.0e10*Msun, 3.205e6 * 2.325e5 * Msun ], backend='dopri5', NSTEPS=1e13, MAX_STEP=u.year.to(u.s)*1e6, RTOL=1e-11):
    """ 
    Second order equation ma=-grad(U) converted to 2 sets of first order equations, with
    e.g. x1 = x
            x2 = vx

            x1dot = vx = x2
            x2dot = ax = -grad_x(U)/m
    """
    X0,Y0,Z0 = lmxb.X0*u.kpc.to(u.m), lmxb.Y0*u.kpc.to(u.m), lmxb.Z0*u.kpc.to(u.m)
    Vx0,Vy0,Vz0 = lmxb.Vx0*u.km.to(u.m), lmxb.Vy0*u.km.to(u.m), lmxb.Vz0*u.km.to(u.m)
    Tstop = T*u.Gyr.to(u.s)#-T*u.Gyr.to(u.s)
    RR = np.array([Vx0,Vy0,Vz0,X0,Y0,Z0])
   
    lmxb.RR=RR
    sol = []
    args = dict(M=M)
    solver=ode(Vdot,).set_integrator(backend,nsteps=NSTEPS,max_step=MAX_STEP,rtol=RTOL)
    solver.set_f_params(M)

    def solout(t,y):
        """ 
        function for saving integration results to sol[]
        """
        temp=list(y)
        temp.append(t)
        sol.append(temp)


    solver.set_solout(solout)
    solver.set_initial_value(RR,0)
    solver.integrate(Tstop)
    sol=np.array(sol)

    lmxb.Vx = sol[:,0]*u.m.to(u.km)
    lmxb.Vy = sol[:,1]*u.m.to(u.km)
    lmxb.Vz = sol[:,2]*u.m.to(u.km)

    lmxb.X = sol[:,3]*u.m.to(u.kpc)
    lmxb.Y = sol[:,4]*u.m.to(u.kpc)
    lmxb.Z = sol[:,5]*u.m.to(u.kpc)
    lmxb.Vpec = np.array([TrajTools.getPec(sol[j,3],sol[j,4],sol[j,5],sol[j,0],sol[j,1],sol[j,2]) for j in range(len(sol))])*u.m.to(u.km)

    lmxb.t = sol[:,6]*u.s.to(u.Gyr)

def getCrossings(lmxb):
    Zcross = []
    sol = np.array([lmxb.t,lmxb.Vx,lmxb.Vy,lmxb.Vz,lmxb.X,lmxb.Y,lmxb.Z,lmxb.Vpec]).T
    for j in range(len(sol)-1):
        if sol[j+1][-2]>=0 and sol[j][-2]<0:
            intersol=[]
            for k in range(len(sol[0])):
                zs=[sol[j][-2],sol[j+1][-2]]
                ks=[sol[j][k],sol[j+1][k]]
                intervar=interp1d(zs,ks)
                intersol.append(intervar(0))
            
            Zcross.append(intersol)
        elif sol[j+1][-2]<=0 and sol[j][-2]>0:
            intersol=[]
            for k in range(len(sol[0])):
                zs=[sol[j][-2],sol[j+1][-2]]
                ks=[sol[j][k],sol[j+1][k]]
                intervar=interp1d(zs,ks)
                intersol.append(intervar(0))
            
            Zcross.append(intersol)

    return np.array(Zcross).T
        
                
