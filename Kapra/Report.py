import os
import numpy as np
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm
from saxpy.paa import paa
from math import sqrt
from scipy.spatial import distance
import time


class Report:
    def __init__(self, time_list: list(), leaf_list: list(), initial_table, final_table, k_value, p_value, paa_value, max_level ):
        self.time = time_list
        self.leaf_list = leaf_list
        self.avg_level = 0
        self.k = k_value
        self.p = p_value
        self.paa = paa_value
        self.max = max_level
        for leaf in leaf_list:
            self.avg_level += leaf.level
        self.avg_level= self.avg_level/(len(leaf_list))
        self.initial_table = initial_table
        self.final_table = final_table
        self.table_PL = 0
        self.table_VL = 0

    def printReport(self, path_name):
        total_time = 0
        for time in self.time:
            total_time += time
        print("Report of {}".format(os.path.basename(os.path.normpath(path_name))))
        print("Total time occured {}s \n".format(round(total_time,2)))
        print("Phase 1: Create-tree phase lasted {}, {}%".format(round(self.time[0],2),round(100*self.time[0]/total_time),1))
        if len(self.time) > 2:
            print("Phase 2: Recycle bad-leaves phase  lasted {}, {}%".format(round(self.time[1],2),round(100*self.time[1]/total_time),1))
        print("Phase 3.1: Top-down approach phase {}, {}%".format(round(self.time[-2],2),round(100*self.time[-2]/total_time),1))
        print("Phase 3.2: Group formation phase {}, {}%".format(round(self.time[-1],2),round(100*self.time[-1]/total_time),1))
        print("The average pattern level is: {}".format(round(self.avg_level,1)))
        self.table_PL = self.pattern_loss()
        print("The table pattern loss is {}".format(round(self.table_PL,2)))
        self.table_VL = self.value_loss()
        print("The table value loss is {}".format(round(self.table_VL,2)))

    def writeReport(self, path_name):
        total_time = 0
        for time in self.time:
            total_time += time
        f = open(path_name, "a")
        f.write("*****************************************************************************************************\n")
        f.write("Report of {} with K={}, P={}, PAA={}, MAX_LEVEL={}\n".format(os.path.basename(os.path.normpath(path_name)),self.k,self.p,self.paa,self.max))
        f.write("Total time occured {}s \n".format(round(total_time,2)))
        f.write("Phase 1: Create-tree phase lasted {}, {}% \n".format(round(self.time[0],2),round(100*self.time[0]/total_time),1))
        if len(self.time) > 2:
            f.write("Phase 2: Recycle bad-leaves phase  lasted {}, {}% \n".format(round(self.time[1],2),round(100*self.time[1]/total_time),1))
        f.write("Phase 3.1: Top-down approach phase {}, {}% \n".format(round(self.time[-2],2),round(100*self.time[-2]/total_time),1))
        f.write("Phase 3.2: Group formation phase {}, {}% \n".format(round(self.time[-1],2),round(100*self.time[-1]/total_time),1))
        f.write("The average pattern level is: {} \n".format(round(self.avg_level,1)))
        f.write("The table pattern loss is {} \n".format(round(self.table_PL,2)))
        f.write("The table value loss is {} \n".format(round(self.table_VL,2)))
        f.close()


    def get_level(self,letter):
        if "a" <= letter < "t":
            return ord(letter) - 97

    def reconstruct(self, pr, level, paa_value):
        """
        In order to reconstruct the feature vector we only need the pattern and the level

        :param pr: the pattern of the time-series
        :param level: the level of the time-series
        
        """

        paa_recon = list()
        # if the level is 1 then the pattern will be "a"*(dimension of feature space)
        # and the relative feature vector will be of zero's
        if level == 1:
            for letter in pr:
                paa_recon.append(0)
            return paa_recon
        # otherwise we cut the gaussian bell into n = level equal slices
        B_levels = cuts_for_asize(level)
        normal_distribution = np.random.normal(size=1000)
        # every letter is related to one of this levels
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
    
    
    def distance(self,paa1,paa2):
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
    
    def get_paa(self,ts,paa_value):
        data = np.array(ts)
        data_znorm = znorm(data)
        data_paa = paa(data_znorm, paa_value)
        return data_paa

    def pattern_loss(self):
        """
        Citing Shou et al. 2013
            "The pattern loss metric should be defined based on the feature vector p(.) as we proposed in Section 3.
             For any time-series Q, we can obtain its original feature vector p(Q), which represents the pattern information 
             embodied in the original time-series. Meanwhile, we can obtain from PR[Q] the reconstructed feature vector p*(Q) which represents
             the pattern information preserved in PR[Q]. Therefore, the pattern loss can be measured by the distance between p(Q) and p*(Q).
             Where distance(.) is a distance measure defined in the feature vector space of patterns, for simplicity we use the well-known cosine distance.
        """
        table_PL = 0
        for leaf in self.leaf_list:
            for member in leaf.members:
                paa_reconstructed = self.reconstruct(leaf.pattern_representation,leaf.level,leaf.paa_value) # p*(Q)
                original_paa = self.get_paa(self.initial_table[member],leaf.paa_value)  # p(Q)
                table_PL += self.distance(paa_reconstructed,original_paa) # PL(Q) = cosine_distance(p(Q), p*(Q))
        return table_PL


        