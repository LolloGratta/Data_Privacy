import os
import numpy as np
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.paa import paa
from math import sqrt
from scipy.spatial import distance
import time


class Report:
    def __init__(self, total_time, leaf_list: list(), initial_table, final_table, k_value, p_value, paa_value, max_level ):
        self.k = k_value
        self.p = p_value
        self.paa = paa_value
        self.max = max_level
        self.time = round(total_time,2)
        self.leaf_list = leaf_list
        self.initial_table = initial_table
        self.final_table = final_table
        self.table_PL = self.pattern_loss()
        self.table_VL = self.value_loss()
    
    def printReport(self,path_name):
        print("Report of {}".format(os.path.basename(os.path.normpath(path_name))))
        print("Total time passed {}s ".format(self.time))
        print("The table pattern loss is {}".format(round(self.table_PL,2)))
        print("The table value loss is {}".format(round(self.table_VL,2)))


    def writeReport(self, path_name):
        f = open(path_name, "w")
        f.write("*****************************************************************************************************\n")
        f.write("Report of {} with K={}, P={}, PAA={}, MAX_LEVEL={}\n".format(os.path.basename(os.path.normpath(path_name)),self.k,self.p,self.paa,self.max))
        f.write("Total time passed {}s \n".format(self.time))
        f.write("The table pattern loss is {} \n".format(round(self.table_PL,2)))
        f.write("The table value loss is {} \n".format(round(self.table_VL,2)))



    def get_level(self,letter):
        if "a" <= letter < "t":
            return ord(letter) - 97

    def reconstruct(self, pr, level, paa_value):
        paa_recon = list()
        if level ==1:
            for letter in pr:
                paa_recon.append(0)
            return paa_recon
        B_levels = cuts_for_asize(level)
        normal_distribution = np.random.normal(size=1000)
        for letter in pr:
            relative_level = self.get_level(letter)
            b_level_low = B_levels[relative_level]
            if relative_level+1 == len(B_levels):
                b_level_up = np.inf
            else:
                b_level_up = B_levels[relative_level+1]
            sliced_distribution = normal_distribution[(normal_distribution >= b_level_low) & (normal_distribution < b_level_up)] 
            median = np.median(sliced_distribution)
            paa_recon.append(median)
        return paa_recon
    
    def get_paa(self,ts,paa_value):
        data = np.array(ts)
        data_znorm = znorm(data)
        data_paa = paa(data_znorm, paa_value)
        return data_paa
    
    
    def distance2(self,paa1,paa2):
        sum1 = 0
        sum2 = 0
        for i in range(0,len(paa1)):
            sum1 += paa1[i]
            sum2 +=  paa2[i]
        if sum1 == sum2 and sum1 == 0:
            return 0
        if sum1 != sum2 and (sum1==0 or sum2 ==0):
            return 1
        return distance.cosine(paa1,paa2)
            
    def value_loss(self):
        table_VL = 0
        for key in self.final_table:
            row_vl = 0
            for j in range(0,len(self.final_table[key])-2):
                range_string = self.final_table[key][j]
                values = range_string[1:-1].split(' - ')
                row_vl += (float(values[1])-float(values[0]))**2
            table_VL += sqrt(row_vl/(len(self.final_table[key])-2))
        return table_VL

    def pattern_loss(self):
        table_PL = 0
        for leaf in self.leaf_list:
            if leaf:
                for member in leaf[0].members:
                    paa_reconstructed = self.reconstruct(leaf[0].pattern_representation,leaf[0].level,leaf[0].paa_value)
                    original_paa = self.get_paa(self.initial_table[member],leaf[0].paa_value)
                    table_PL += self.distance2(paa_reconstructed,original_paa)
        return table_PL
