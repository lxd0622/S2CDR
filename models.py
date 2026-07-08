import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import *
import pdb
import numpy as np
import scipy.sparse as sp
import math
import sys
import time
from sparsesvd import sparsesvd
from torchdiffeq import odeint



class SSBasemodel(object):
    def __init__(self, adj_mat, mask_matrix, device):
        self.adj_mat = adj_mat
        self.mask_matrix = mask_matrix
        self.factor_dim = 256
        self.device = device
        
        idl_T = float(1)
        idl_K = int(1)
        blur_T = float(1)
        blur_K = int(1)
        self.idl_beta = 1
        self.blu_alpha = 0.01
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float()
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float()

        self.idl_solver = 'euler'
        self.blur_solver = 'euler'
        self.sharpen_solver = "rk4"
        
        sharpen_T = float(1.8)
        sharpen_K = int(1)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float()

        self.sharpening_off = False
        self.final_sharpening = True
        self.t_point_combination = False

    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))  
        d_inv = np.power(rowsum, -0.5).flatten()  
        d_inv[np.isinf(d_inv)] = 0.  
        d_mat = sp.diags(d_inv)  
        norm_adj = d_mat.dot(adj_mat)  

        colsum = np.array(adj_mat.sum(axis=0))  
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  
        self.d_mat_i = d_mat
        safe_d_inv = np.where(d_inv == 0, 1e-10, d_inv)
        self.d_mat_i_inv = sp.diags(1 / safe_d_inv)
        norm_adj = norm_adj.dot(d_mat)  
        self.norm_adj = norm_adj.tocsc()  
        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)  
        end = time.time()
        print('training time for BSCDR', end-start)

    def IDLFunction(self, t, r):
        out = r.numpy() @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        out = out - r.numpy()
        return torch.Tensor(out)
    
    def blurFunction(self, t, r):
        R = self.mask_matrix
        out = r.numpy() @ R.T @ R
        out = out - r.numpy()
        return torch.Tensor(out)
    
    def sharpenFunction(self, t, r):
        R = self.mask_matrix
        out = r.numpy() @ R.T @ R
        return torch.Tensor(-out)
    
    def getUsersRating(self, batch_users):
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())

        with torch.no_grad():
            idl_out = odeint(func=self.IDLFunction, y0=torch.Tensor(batch_test), t=self.idl_times, method=self.idl_solver)
            blurred_out = odeint(func=self.blurFunction, y0=torch.Tensor(batch_test), t=self.blurring_times, method=self.blur_solver)
            
            if self.sharpening_off == False:
                if self.final_sharpening == True:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out[-1]+ self.blu_alpha * blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)

        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out[1:,...],sharpened_out[1:,...]],axis=0),axis=0)  
            else:
                U_2 =  torch.mean(blurred_out[1:,...],axis=0)
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
            else:
                U_2 = self.blu_alpha * blurred_out[-1]
        
        if self.final_sharpening == True:
            if self.sharpening_off == False:
                ret = U_2.numpy()
            elif self.sharpening_off == True:
                ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()
        else:
            ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()

        return ret
