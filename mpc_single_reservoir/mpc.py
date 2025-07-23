# -*- coding: utf-8 -*-

# pip install cyipopt numpy matplotlib scipy

from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp
import math
from scipy.stats import lognorm
from scipy import interpolate
import cyipopt as ipopt
import os
from scipy.sparse import csr_matrix


class model_parameters:
    def __init__(self, name, demand, xmin, xmax, qmin, qmax, av_efficiency, tail_coef,
                 for_coef, cost=0.0002):
        '''Class constructor with hydropower plant parameters
        -  tail_coef is an array [a0,a1,a2,a3,a4] with coefficients of tailrace elevation
           function: h_tail(q) = a0 + a1x + a2x^2 + a3x^3 + a4x^4.
        -  for_coef is an array [a0,a1,a2,a3,a4] with coefficients of forebay elevation 
           function: h_for(x) = a0 + a1x + a2x^2 + a3x^3 + a4x^4.
        '''
        self.name = name
        self.k = av_efficiency  #k^{ef}
        self.demand = demand    #k^{dem} load demand
        self.xmin = xmin  # min. storage
        self.xmax = xmax  # max. storage
        self.qmin = qmin  # min. discharge
        self.qmax = qmax  # max. discharge
        self.tail_coef = tail_coef  # tailrace elevation coefficients
        self.for_coef = for_coef  # forebay elevation coefficients
        self.c = cost  # cost constant
        self.kconv = (86400/(10**6))*np.array([31,28,31,30,31,30,31,31,30,31,30,31]) # k^{con} 
        self.VAZ = self.get_inflows()
        self.W_avg = self.get_inflows_avg()

    def efficiency(self, x0, q, v):
        ''' Efficiency function'''
        return self.k*(self.forebay_elevation(x0) - self.tailrace_elevation(q,v))
  
    def tailrace_elevation(self, discharge, spill):
        '''Tailrace elevation function'''
        return self.tail_coef[0] + self.tail_coef[1]*(discharge + spill) +\
            self.tail_coef[2]*(discharge + spill)** 2 +\
            self.tail_coef[3]*(discharge + spill)** 3 +\
            self.tail_coef[4]*(discharge + spill)**4

    def forebay_elevation(self, storage):
        '''Forebay elevation function'''
        return self.for_coef[0] + self.for_coef[1]*storage + self.for_coef[2]*(storage**2) + self.for_coef[3]*(storage**3) + self.for_coef[4]*(storage**4)

    def generation(self, storage, discharge, spill):
        '''Hydropower generation function'''
        return self.k*(self.forebay_elevation(storage) - self.tailrace_elevation(discharge, spill))*discharge

    def present_cost(self, storage, discharge, spill, month):
        '''Quadratic operation cost'''
        return self.c*((self.kconv[month]*10**6)/3600)*(self.demand - self.generation(storage, discharge, spill))**2

    def f_objective(self, X, Q, V, initial_month):
        '''Objective function'''
        n = len(Q)
        m = initial_month
        c = 0
        for i in range(n):
            c = self.present_cost(X[i], Q[i], V[i], m) + c
            m = m + 1
            if m == 12:
                m = 0
        return c/n
    
    def forebay_deriv(self, x):
        ''' Derivative of forebay elevation function'''
        return self.for_coef[1] + 2*self.for_coef[2]*x + 3*self.for_coef[3]*(x**2) + 4*self.for_coef[4]*(x**3)

    def tailrace_deriv(self, q, v):
        ''' Derivative of tailrace elevation function'''
        return self.tail_coef[1] + 2*self.tail_coef[2]*(q + v) + 3*self.tail_coef[3]*(q+v)**2 + 4*self.tail_coef[4]*(q + v)**3

    def get_inflows(self):
        ''' Method to load historical inflows'''
        #ref_arq = open('path/to/data/vazoes_'+self.name+'.txt',"r")
        #ref_arq = open(os.path.join(data, 'vazoes_'+self.name+'.txt',"r"))
        dir = os.getcwd() + '/data'
        try:           
            ref_arq = open('../data/vazoes_'+self.name+'.txt',"r")
        except:    
            ref_arq = open(dir + '/vazoes_'+self.name+'.txt',"r")            
        VAZ=[]
        i=-1
        for linha in ref_arq:
            i+=1           
            values = linha.split()
            values = [float(i) for i in values]
            VAZ.append(values)            
        ref_arq.close()
        return np.array(VAZ)

    def get_years(self):
        ''' Method to calculate the number of years'''
        return len(self.VAZ)

    def get_inflows_avg(self):
        ''' Method to calculate monthly inflows mean'''
        return np.mean(self.VAZ, axis=0)


class prob:
    '''Classe dos problemas a serem resolvidos pelo otimizador do MPC'''
    def __init__(self, uhe, H, m_initial, R0, Wpred):
        self.H = H  # optimization horizon (number of months)
        self.m_initial = m_initial  # initial month
        self.R0 = R0  # initial storage (hm^3)
        self.Wpred = Wpred  # inflow prediction (m^3/s)
        self.uhe = uhe  # hydro plant 

    def objective(self, x):
        # objective function
        X = x[0: self.H]  # storage vector (hm^3)
        Q = x[self.H+1: 2*self.H+1]  # discharge (m^3/s)
        V = x[2*self.H+1:]  # spillage vector (m^3/s)
        return self.uhe.f_objective(X, Q, V, self.m_initial)

    def gradient(self, x):
        X = x[0: self.H]
        Q = x[self.H + 1: 2*self.H + 1]
        V = x[2*self.H + 1: ]
        grad1 = [None]*self.H
        grad2a = [None]*self.H
        grad2b = [None]*self.H
        grad2c = [None]*self.H
        m = self.m_initial
        for i in range(self.H):
            grad1[i] = 2 * self.uhe.k * self.uhe.c * ((10 ** 6) / 3600) * self.uhe.kconv[m] * (
                self.uhe.demand - self.uhe.k*(self.uhe.forebay_elevation(X[i]) -
                self.uhe.tailrace_elevation(Q[i],V[i]))*Q[i])
            grad2a[i] = - Q[i]*self.uhe.forebay_deriv(X[i])
            grad2b[i] = - (self.uhe.forebay_elevation(X[i]) - self.uhe.tailrace_elevation(Q[i],V[i]) - Q[i]*self.uhe.tailrace_deriv(Q[i],V[i]))
            grad2c[i] = Q[i]*self.uhe.tailrace_deriv(Q[i],V[i])
            m = m + 1
            if m == 12: m = 0        
        grad2  = [*grad2a, 0, *grad2b, *grad2c]
        g = np.transpose(np.array(grad1+[0]+grad1+grad1))
        grad = g*grad2
        return grad

    def Aeq_beq(self):
        kcon = []
        month = self.m_initial
        for _ in range(self.H):
            month = month + 1
            if month==12: month = 0
            kcon = kcon + [self.uhe.kconv[month]]      
        Aeq1 = np.eye(self.H + 1) - np.array([*np.zeros((1,self.H+1)),*np.eye(self.H,self.H+1)])
        Aeq2 = np.array([*np.zeros((1,self.H)),*np.diag(kcon)])
        Aeq  = np.concatenate([Aeq1, Aeq2, Aeq2],axis=1)
        beq  = np.array([self.R0, *(kcon*self.Wpred.transpose())])
        return Aeq, beq

    def constraints(self, x):
        Aeq, _ = self.Aeq_beq()
        return Aeq.dot(x)

    def jacobian(self, x):
        Aeq, _ = self.Aeq_beq()
        return Aeq

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        # Example for the use of the intermediate callback.
        print ("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        return


class simulation:
    def __init__(self, uhe, m_initial, m_final, y_initial, y_final, x0, H = 36, prediction=True):
        self.m_initial = m_initial
        self.m_final = m_final
        self.y_initial = y_initial
        self.y_final = y_final
        self.x0 = x0
        self.H = H
        self.prob = prob
        self.prediction = prediction
        
    def __get_next_month(self, m0):
        m1 = m0 + 1
        if m1 == 12:
            m1 = 0
        return m1 

    def simulation_algo(self, uhe, sint_inflows = False, VAZ = None, verbose=True):
        if not sint_inflows:
            VAZ = uhe.VAZ
        print("Simulation using Model Predictive Control")
        H = self.H
        n = int(H / 12)

        inflows = []
        cost = []
        generation = []
        spillage = []
        efficiency = []
        discharges = []
        x0 = self.x0
        storage = [x0]
        turb_spill = []
        deficit = []
        horizon = self.y_final - self.y_initial + 1
        #perc = 1.09*np.array([1,1,1,1,1,1,1,1,1,1,1,1]) serra da mesa
        
        #perc = np.array([1.13951876, 1.12938004, 1.0815181,  1.03680347, 0.87261596, 0.83109418, 0.90476193, 0.90720135, 1.0879476,  1.10231925, 1.12446062, 1.07868424]) # nova ponte
        
        #perc = 1.07*np.array([1,1,1,1,1,1,1,1,1,1,1,1])
        #perc = np.array([1.10926314, 1.17045385, 1.01134995, 1.0519899 , 0.87397629, 1.01084973, 0.95465032, 1.11021935, 1.14720905, 1.06677353, 1.09250396, 1.16490802]) #serra da mesa

        # perc = np.array([1.10133143, 1.05012081, 0.80136086, 0.80110177, 0.82475168, 0.84533339, 0.81113776, 0.84683009, 1.15667524, 1.08725905, 1.12661458, 1.0454848 ]) #sobradinho

        #perc= np.array([1.11883473, 1.10374346, 1.11634858, 0.85221613, 0.86742181, 0.86745105, 1.06553272, 1.08536061, 1.03501226, 1.10376298, 1.11836273, 1.11800213]) #furnas

        #perc = np.array([1.11769293, 1.11337027, 1.06670563, 0.86042628, 0.89905925, 0.86722735, 1.0384126 , 1.07344075, 0.9059726 , 1.08932487, 1.11153316, 1.11538088]) # emborcacao

        for t in range(horizon):
            print('>>>>>>>>Ano: '+str(t) + '>>>>>>>>>>')
            for m in range(12):
                if not (t==0 and m < self.m_initial) and not (t==horizon-1 and m > self.m_final):
                    m0 = m - 1
                    vt = 0
                    if m0 == -1: m0 = 11
                    v = 0
                    d = 0
                    W_now = VAZ[self.y_initial + t, m]

                    Wpred = []
                    for i in range(n):
                        if i==0:
                            Wpred0 = np.concatenate((uhe.W_avg[m:12],uhe.W_avg[0:m])) #*np.concatenate((perc[m:12],perc[0:m]))
                        else:
                            Wpred0 = np.concatenate((uhe.W_avg[m:12], uhe.W_avg[0:m])) 
                        Wpred = np.concatenate((Wpred,Wpred0))
                    prob1 = prob(uhe, H, m, x0, Wpred)
                    q,_ = self.MPC_optimizer(uhe,x0, H, Wpred, prob1)

                    x1 = x0 + uhe.kconv[m]*(W_now - q)
                    if x1 > uhe.xmax:
                        v = (x1 - uhe.xmax)/uhe.kconv[m]
                        x1 = uhe.xmax                        
                        # if q < uhe.qmax and self.elim_turb_sp:
                        #     vt = min(uhe.qmax, q + v) - q
                        #     v = v - vt
                        #     q = q + vt                        
                    elif x1 < uhe.xmin:
                        d = (uhe.xmin - x1)/uhe.kconv[m]
                        q = max(0,q - d)
                        x1 = uhe.xmin                            
                    c = uhe.present_cost(x0, q, v, m)
                    g = uhe.generation(x0, q, v)
                    p = uhe.efficiency(x0, q, v)
                    inflows.append(W_now)
                    cost.append(c)
                    generation.append(g)
                    spillage.append(v)
                    efficiency.append(p)
                    discharges.append(q)
                    storage.append(x1)
                    turb_spill.append(vt)
                    deficit.append(d)
                    x0 = x1
        return cost, generation, efficiency, spillage, turb_spill, inflows, discharges, storage

    def simulation_stats(self, uhe, cost, generation, efficiency, spillage, turb_spill):
        cost_avg = np.mean(cost)
        cost_std = np.std(cost)
        gen_avg  = np.mean(generation)
        gen_std  = np.std(generation)
        eff_avg = np.mean(efficiency)
        spill_avg = np.mean(spillage)
        turb_spill_avg = np.mean(turb_spill)
        n = len(generation)
        mix_t = np.mean((uhe.demand*np.ones((1,n)) - generation)/uhe.demand)
        mix_h = 1 - mix_t        
        return cost_avg, cost_std, gen_avg, gen_std, eff_avg, spill_avg, turb_spill_avg, mix_t, mix_h

    def monthly_cost(self, cost, m_init):
        cost_matrix=[[],[],[],[],[],[],[],[],[],[],[],[]]        
        cost_avg_month = [None]*12
        month = m_init
        for i in range(len(cost)):
            if month == 12:
                month = 0          
            cost_matrix[month].append(cost[i])
            month += 1
        for j in range(12):
            cost_avg_month[j] =  np.mean(cost_matrix[j])
        return cost_avg_month

    def plot_results_simulation(self,uhe, cost, generation, efficiency, spillage, turb_spill, inflows, discharges, storage):
        plt.subplot(2,1,1)
        n = len(storage)
        plt.title('Storage Trajectory')
        plt.xlabel('Stage')
        plt.ylabel('Storage [$hm^3$]') 
        plt.plot(storage, label='Storage')
        plt.plot(uhe.xmin*np.ones(n))
        plt.plot(uhe.xmax*np.ones(n),'--')
        plt.autoscale(tight = True)  
        plt.legend()
        plt.subplot(2,1,2)
        plt.title('Flows')
        plt.xlabel('Stage')
        plt.ylabel('Flows [$m^3/s$]')
        plt.plot(inflows,'r', label='inflow')
        plt.plot(discharges,'b',label='discharge')
        plt.plot(spillage,'g',label='spillage')
        plt.plot(turb_spill,'k', label='turbined spillage')        
        plt.plot(uhe.qmax*np.ones(n-1),'--', label='maximum discharge')
        plt.legend()
        plt.autoscale(tight=True)
        plt.savefig('../results/graficos') 
        return plt.show
    
    def MPC_optimizer(self, uhe, R0, H, Wpred, prob1):
        x0 = [R0]+[(uhe.xmin+(uhe.xmax-uhe.xmin)/2)]*H + [*Wpred] + [0]*H
        lb = [R0] + [uhe.xmin]*H + [uhe.qmin]*H + [0]*H
        ub = [R0] + [uhe.xmax]*H + [uhe.qmax]*H + [6*uhe.qmax]*H
        _, beq = prob1.Aeq_beq()
        cl = beq
        cu = beq
        nlp = ipopt.problem(n=len(x0), m=len(cl), problem_obj = prob1,lb=lb,
            ub=ub,cl=cl,cu=cu)
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-4)
        q_opt, info = nlp.solve(x0)
        nlp.close()
        return q_opt[H+1], info


def main():
    plant = int(input("Choose a hydropower plant: 1-Emborcacao, 2-Sobradinho, 3-Serra da Mesa, 4-Furnas:  "))
    if plant == 1:
        for_coef_sob = np.array([5.680898E+02, 1.450600E-02, -1.202799E-06, 5.830299E-11, -1.124500E-15])
        tail_coef_sob = np.array([5.193198E+02, 3.939997E-03, -3.599999E-07, 4.329999E-11, -2.600000E-15])
        uhe = model_parameters('emborcacao', 1192, 4669, 17725, 0, 984.5, 0.008731, tail_coef_sob, for_coef_sob)
        m_initial = 4
        m_final   = 1
        y_initial = 1
        y_final   = 76
                
    elif plant == 2:
        for_coef_sob = np.array([3.741790E+02, 1.396690E-03, -5.351590E-08, 1.155989E-12, -9.545989E-18])
        tail_coef_sob = np.array([3.596538E+02, 1.964010E-03, -2.968730E-07, 2.508280E-11, -7.702299E-16])
        # Parâmetros da usina:
        uhe = model_parameters('sobradinho',1050, 5447, 34116, 0, 3860, 0.009023, tail_coef_sob, for_coef_sob)
        m_initial = 4
        m_final   = 2
        y_initial = 1
        y_final   = 12  # 76

    elif plant == 3:
        for_coef_sob = np.array([3.914048E+02, 2.772160E-03, -4.357250E-08,  2.903040E-13, 0])
        tail_coef_sob = np.array([3.327979E+02, 1.342970E-03,  8.819558E-08, -1.627669E-11, 0])
        uhe = model_parameters('serra da mesa', 1275, 11150, 54400, 0, 1074.6, 0.009124, tail_coef_sob, for_coef_sob)
        # m_initial, m_final, y_initial, y_final
    
    elif plant == 4:
        for_coef_sob = np.array([7.352458E+02, 3.496580E-03, -1.974370E-07, 6.917049E-12,-9.773650E-17])
        tail_coef_sob = np.array([672.9, 1.017380E-03, -1.799719E-07, 2.513280E-11, 0])
        uhe = model_parameters('furnas', 1312, 5733, 22950, 0, 1231.6, 0.008633, tail_coef_sob, for_coef_sob)
        #m_initial, m_final, y_initial, y_final

    H = 36

    ini_rodada = time()

    sim = simulation(uhe, m_initial, m_final, y_initial, y_final, x0=uhe.xmax, H = H, prediction=False)
    cost, generation, productivity, spillage, turb_spill, inflows, discharges, storage = sim.simulation_algo(uhe)
    
    print('Execução completa concluída em: ' + 
        str((time() - ini_rodada) / 60) + 'm')

    cost_avg, cost_std, gen_avg, gen_std, prod_avg, spill_avg, turb_spill_avg, mix_t, mix_h = sim.simulation_stats(uhe,cost, generation, productivity, spillage, turb_spill)
    sim.plot_results_simulation(uhe, cost, generation, productivity, spillage, turb_spill, inflows, discharges, storage)

    print("Cost - mean: ", cost_avg)
    print("Cost - standard deviation: ", cost_std)
    print("Generation: ", gen_avg)
    print("Generation: ", gen_std)
    print("Average efficiency: ", prod_avg)
    print("Average spillage: ", spill_avg)
    print("Average dischargeable spillage: ", turb_spill_avg)
    print("Thermal mix:", mix_t)
    print("Hydro mix: ", mix_h)

main()