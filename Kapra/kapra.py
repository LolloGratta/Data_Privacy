import os
import numpy as np
import pandas as pd
import sys
from loguru import logger
import random
from .Node import Node
from treelib import Node as Nodegraph, Tree
import math
from loguru import logger
from datetime import timedelta
from .dataset_anonymized import DatasetAnonymized
import time
from .Report import Report
import copy







def top_down(good_leaf_list, p_value):
     """


     [Preprocessing] The preprocessing step partitions all large P-subgroups into smaller ones.
                     We find al large P-subgroups s1 contained in PGL |s1| >= 2*P.
                     Each s1 will be partitioned into new subgroups no smaller thna P using a top-down
                     partitioning method similar to the top-down- greedy search algorithm proposed in [25].
                     This partitioning process is targeted at minimizing the total istant value loss in the partitions. 
                     The resultant partitions, each regarded as a new P-subgroup, will be added into PGL to replace s1.
     
                     
     :param good_leaf_list: List of Nodes, all labelled as "good-leaf"
     :param p_value: P value of (k,p)-anonymity


     """

     groups = list()
     for leaf in good_leaf_list.copy():
          if leaf.size >= 2*p_value:
               max,min = get_max_and_min(leaf.group)
               good_leaf_list.remove(leaf)
               top_down_on_sub_group(leaf.group,max,min,leaf,good_leaf_list, p_value)

def top_down_on_sub_group(dict_of_ts,max,min,leaf,good_leaf_list, p_value):
     if len(list(dict_of_ts.values())) <  2*p_value:
          logger.debug("Less than {} elements in this group, stopping its recursion".format(2*p_value))
          #good_leaf_list.append(leaf)
          return
     else:
          keys = list(dict_of_ts.keys())
          rounds = 3
     # pick random tuple
     random_tuple = keys[random.randint(0,len(keys)-1)]
     group_u = dict()
     group_v = dict()
     group_u[random_tuple] = dict_of_ts[random_tuple]
     del dict_of_ts[random_tuple]
     last_row = random_tuple
     for round in range(0,rounds*2-1):
          if len(dict_of_ts) > 0:
               if round % 2 == 0:
                    v = find_tuple_with_maximum_ncp(group_u[last_row], dict_of_ts, last_row, max, min)
                    group_v[v] = dict_of_ts[v]
                    last_row = v
                    del dict_of_ts[v]
               else:
                    u = find_tuple_with_maximum_ncp(group_v[last_row], dict_of_ts, last_row, max, min)
                    group_u[u] = dict_of_ts[u]
                    last_row = u
                    del dict_of_ts[u]  
     # Now assigned to group with lower uncertain penalty
     index_keys_time_series = [x for x in range(0, len(list(dict_of_ts.keys())))]
     random.shuffle(index_keys_time_series)
     keys = [list(dict_of_ts.keys())[x] for x in index_keys_time_series]
     for key in keys:
            row_temp = dict_of_ts[key]
            group_u_values = list(group_u.values())
            group_v_values = list(group_v.values())
            group_u_values.append(row_temp)
            group_v_values.append(row_temp)

            ncp_u = compute_normalized_certainty_penalty_on_ai(group_u_values, max, min)
            ncp_v = compute_normalized_certainty_penalty_on_ai(group_v_values, max, min)
            if ncp_v < ncp_u:
               group_v[key] = row_temp
            else:
               group_u[key] = row_temp
            del dict_of_ts[key]
     if len(group_u) >= 2*p_value:
            # recursive partition group_u
            # maximum_value, minimum_value = get_list_min_and_max_from_table(list(group_u.values()))
          max, min = get_max_and_min(group_u)
          new_leaf = Node(leaf.level,leaf.pattern_representation,leaf.label,group_u,leaf.parent,leaf.paa_value)
          logger.debug("Creating Leaf with members: {} from starting leaf: {}".format(group_u.keys(),leaf.members))
          top_down_on_sub_group(dict_of_ts=group_u,
                                          max=max, min=min,leaf=new_leaf,good_leaf_list=good_leaf_list,p_value = p_value)
     else:
          new_leaf = Node(leaf.level,leaf.pattern_representation,leaf.label,group_u,leaf.parent,leaf.paa_value)
          logger.debug("Adding Leaf with members: {} from starting leaf: {}".format(group_u.keys(),leaf.members))
          good_leaf_list.append(new_leaf)

     if len(group_v) >= 2*p_value:
          # recursive partition group_v
          max, min = get_max_and_min(group_v)
          new_leaf = Node(leaf.level,leaf.pattern_representation,leaf.label,group_v,leaf.parent,leaf.paa_value)
          logger.debug("Creating Leaf with members: {} from starting leaf: {}".format(group_v.keys(),leaf.members))
          top_down_on_sub_group(dict_of_ts=group_v,
                                          max=max, min=min,leaf=new_leaf,good_leaf_list=good_leaf_list,p_value= p_value)
     else:
          new_leaf = Node(leaf.level,leaf.pattern_representation,leaf.label,group_v,leaf.parent,leaf.paa_value)
          logger.debug("Adding Leaf with members: {} from starting leaf: {}".format(group_v.keys(),leaf.members))
          good_leaf_list.append(new_leaf)
     return

def compute_normalized_certainty_penalty_on_ai(table=None, maximum_value=None, minimum_value=None):
    """
    Compute NCP(T)
    :param table:
    :return:
    """
    z_1 = list()
    y_1 = list()
    a = list()
    for index_attribute in range(0, len(table[0])):
        temp_z1 = 0
        temp_y1 = float('inf')
        for row in table:
            if int(row[index_attribute]) > temp_z1:
                temp_z1 = row[index_attribute]
            if int(row[index_attribute]) < temp_y1:
                temp_y1 = row[index_attribute]
        z_1.append(temp_z1)
        y_1.append(temp_y1)
        a.append(abs(maximum_value[index_attribute] - minimum_value[index_attribute]))
    ncp_t = 0
    for index in range(0, len(z_1)):
        try:
            ncp_t += (z_1[index] - y_1[index]) / a[index]
        except ZeroDivisionError:
            ncp_t += 0
    ncp_T = len(table)*ncp_t
    return ncp_T


def find_tuple_with_maximum_ncp(fixed_tuple, time_series, key_fixed_tuple, maximum_value, minimum_value):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes NCP(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_ncp = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            ncp = compute_normalized_certainty_penalty_on_ai([fixed_tuple, time_series[key]], maximum_value, minimum_value)
            if ncp >= max_value:
                tuple_with_max_ncp = key
                max_value = ncp
    return tuple_with_max_ncp

def get_max_and_min(list_of_time_series):
     max = dict()
     min = dict()
     for i in range(0,len(list(list_of_time_series.values())[0])):
          max[i] = list(list_of_time_series.values())[0][i]
          min[i] = list(list_of_time_series.values())[0][i]
     for ts in list(list_of_time_series.values()):
          for column_index in range(0,len(ts)):
               if ts[column_index] > max[column_index] :
                    max[column_index] = ts[column_index]
               if min[column_index] > ts[column_index]:
                    min[column_index] = ts[column_index]
     return max,min

def calculate_value_loss(group):
     max,min = get_max_and_min(group)
     vl = 0
     for i in range(0,len(list(max.values()))):
          vl = vl +  ((max[i] - min[i])**2)
     return math.sqrt(vl/len(list(max.values())))



                    
def group_formation_pt2(PGL,k_value,p_value,GL):
     """

     Core function of the group formation phase.
     "[STEP 1] All P-subgroups in PGL containing no fewer than k time series are taken as k-groups and simply moved into GL.
      [STEP 2] In the remaining P-subgroups in PGL, find the P-subgroup s1 with the minimum istant value loss, and then create a new group G = s1.
      [STEP 3] Find another P-subgroup s in {PGL}-s1, which, if merged with G, produces the minimal value loss.
      [STEP 4] Repeat [STEP 3] until the cardinality of G is greater or equal than K. 
      G is then added into GL and its respective subgroups in PGL are removed.
      [STEP 5] Repeat [STEP 2-4] until the total remaining time series in PGL are fewer than K.
      Each remaining P-subgroups s' in PGL whill choose to join a k-group G' in GL which again minimizes the total instant value loss.
      The computation complexity of this phase is O(|PGL|**2)." Supporting Pattern-Preserving Anonymiziation for Time-Series Data" Shou et al. 2013.

     :param PGL:  List of groups, where every group contains at least P time_series (and less than 2*P)
     :param k_value:  The size each group must be (or greater) after this function
     :param p_value: useless parameter :D
     :param GL: Final list of groups, where every group contains at least K time_series with at least P-subgroups of them having the same pattern.

     
     """
     size_of_PGL  = 0
     for group in PGL.copy():
          # [STEP 1] 
          if len(group) >= k_value:
               logger.debug("Adding {} to GL".format(group.keys()))
               GL.append(group)
               index_to_delete = PGL.index(group)
               del PGL[index_to_delete]
     # counting size of PGL
     for group in PGL:
          for member in group:
               size_of_PGL += 1
     # [STEP 2]
     PGL =  sorted(PGL, key=lambda x:calculate_value_loss(x))
     # [STEP 4]
     while size_of_PGL >= k_value:
          G = PGL[0]
          while len(G) < k_value:
               min_vl = float('inf')
               best_G = None
               # [STEP 3]
               for G2 in PGL.copy():
                    if G != G2 and G2:
                         G3 = G.copy()
                         G3.update(G2)
                         tmp = calculate_value_loss(G3)
                         if min_vl > tmp:
                              min_vl = tmp
                              best_G = G2
               index = PGL.index(G)
               if best_G:
                    logger.debug("Merging G: {} with G2: {}".format(G.keys(),best_G.keys()))
               G.update(best_G)
               PGL.remove(best_G)
               PGL[index] = G
               G2 = None
          GL.append(G)
          PGL.remove(G)
          size_of_PGL  = 0
          for group in PGL:
               for member in group:
                    size_of_PGL += 1
     # [STEP 5] where remaining elements are less than K
     for remaining_group in PGL.copy():
          PGL.remove(remaining_group)
          for G in GL:
               min_vl = float('inf')
               best_G = None
               G3 = remaining_group.copy()
               G3.update(G)
               tmp = calculate_value_loss(G3)
               if min_vl > tmp:
                    min_vl = tmp
                    best_G = G
          index = GL.index(best_G)
          GL[index].update(G3)
     return 


               

                        
def log_size(leaf_list):
     """"
     Logs how many members are contained in all the leaves, used to debug and to check for missing elements.

     :param leaf_list: list of leaves(class Node)
          """
     tot = 0
     for leaf in leaf_list:
          for member in leaf.members:
               tot +=1
     logger.debug("Number of time_series: {}".format(tot))


                    
                       
                       
                  


               

          

def kapra_main(k_value=None,p_value=None,paa_value=None,max_level=5,dataset_path=None,nrows=None,verbosity_level = "TRACE"):
     """


      Starts the Kapra algorithm described in the paper " Supporting Pattern-Preserving Anonymiziation for Time-Series Data" by Shou et al. 2013.
      Kapra is divided in three phases:
      1) Create-tree phase
      2) Recycle bad-leaves phase
      3) Group formation


     :param k_value:   Minimum Size for groups with same QI
     :param p_value:   Minimum time series with same pattern in one group
     :param paa_value:  Length of pattern
     :param max_level:  Alhabet dimension for pattern 1={a},2={a,b},3={a,b,c},ecc...
     :param dataset_path: Path of dataset
     :param nrows:  Number of rows to use from dataset, if nrows==None all will be used
     :param verbosity_level:  {TRACE,DEBUG,INFO}, where TRACE is VERY verbose and INFO is VERY concise


     """


     #logger settings
     logger.remove(0)
     logger.add(sys.stderr, colorize=True, format="<g>{elapsed}</g> | <level> {level}: </level>  <w>{message}</w>  ", level=verbosity_level )
     logger.level("INFO", color="<red>")
     logger.level("TRACE",color="<ly>")
     logger.level("DEBUG",color="<m>")

     if os.path.isfile(dataset_path):
          # read time_series_from_file
          time_series = pd.read_csv(dataset_path, nrows=nrows)

          # get columns name
          columns = list(time_series.columns)
          columns.pop(0)  # remove product code

          time_series_dict = dict()

          # save dict file instead pandas
          for index, row in time_series.iterrows():
               time_series_dict[row.iloc[0]] = list(row.iloc[1:])

          time_series_for_report = copy.deepcopy(time_series_dict) # We will use this to calculate pattern loss
          time_series_dict_copy = time_series_dict.copy()
          recycle_bad_leaves = list() # This will be the list of bad-leaves
          leaf_list = list() # This will be the list of all the good-nodes
          root = Node(1,pattern_representation="",label="intermediate",group=time_series_dict_copy,parent = None, paa_value=paa_value)
          logger.info("[START] Create-tree phase.")
          # start splitting, Start timer
          start_time = time.perf_counter()
          root.splitting(p_value,leaf_list, recycle_bad_leaves,max_level)
          end_time = time.perf_counter()
          logger.info("[END] Create-tree phase.")
          time_list = list()
          time_list.append(end_time-start_time)

          if verbosity_level == "DEBUG" or verbosity_level == "TRACE":
               log_size(leaf_list+recycle_bad_leaves)

          # Simple representation of the Tree (nice with little Datasets)
          logger.trace("******************Drawing Tree******************")
          if verbosity_level == "TRACE":
               root_tree_draw = root
               tree = Tree()
               nome = ' '.join(root_tree_draw.members)
               tree.create_node(root_tree_draw.members,str(nome))
               root_tree_draw.drawTree(tree,nome)
               tree.show()
          #root.remove_Bad_Leaves()
          logger.trace("******************END TREE******************")

          # For debug, it lists all members of good and bad leaves
          if verbosity_level == "DEBUG" or verbosity_level == "TRACE":
               for i in range(0,len(leaf_list)):
                    logger.debug("Leaf {} members: {}, pattern: {} level: {}.".format(i,leaf_list[i].members,leaf_list[i].pattern_representation,leaf_list[i].level))
               for i in range (0,len(recycle_bad_leaves)):
                    logger.debug("Bad-leaf {} members: {}, pattern: {}, level: {}. ".format(i,recycle_bad_leaves[i].members,recycle_bad_leaves[i].pattern_representation,recycle_bad_leaves[i].level))
          recycle_bad_leaves = sorted(recycle_bad_leaves, key=lambda x: x.level, reverse=True)
          
          # Recycle phase
          if len(recycle_bad_leaves) > 0:       
               logger.info("[START] Recycle Bad Leaves phase")
               start_time = time.perf_counter()
               root.Recycle_Bad_Leaves(recycle_bad_leaves,leaf_list,p_value,paa_value)
               end_time = time.perf_counter()
               time_list.append(end_time-start_time)
               logger.info("[END] Recycle Bad Leaves phase")
          else:
               logger.info("Recycle Bad Leaves phase will be skipped")

          # For debug, it lists all members of good leaves
          if verbosity_level == "DEBUG" or verbosity_level == "TRACE":
               logger.debug("*****Leaves and members after splitting and recycling.*****")
               for i in range(0,len(leaf_list)):
                    logger.debug("Leaf {} members:{} pr:{}".format(i,leaf_list[i].members,leaf_list[i].pattern_representation)) 
               log_size(leaf_list) 
               logger.debug("*****Leaves and members after splitting and recycling.*****") 

          # Top-down approach
          logger.info("[START] top_down approach generating PGL")
          start_time = time.perf_counter()
          top_down(leaf_list, p_value)
          end_time = time.perf_counter()
          time_list.append(end_time-start_time)
          logger.info("[END] top_down, generated PGL")
          if verbosity_level == "DEBUG" or verbosity_level == "TRACE":
               logger.debug("PGL:")
               for i in range(0,len(leaf_list)):
                    logger.debug("Leaf numero {}: members: {} and pr = {}".format(i,leaf_list[i].members,leaf_list[i].pattern_representation))
               log_size(leaf_list)
          # GL = final list of groups containing at least (k,p)-time_series
          # PGL = list of groups containing least P-time_series (and less than 2*P)
          GL = list()
          PGL = list()
          tot = 0
          for leaf in leaf_list:
               PGL.append(leaf.group)
          logger.info("[START] Group formation generating GL")
          start_time = time.perf_counter()
          group_formation_pt2(PGL,k_value,p_value,GL)
          end_time = time.perf_counter()
          time_list.append(end_time-start_time)
          logger.info("[END] Group formation, generated GL")
          if verbosity_level == "TRACE":
               logger.trace("Final Groups with (k,p) requirements.")
               for i in range(len(GL)):
                    logger.trace("Group {}: {}".format(i,GL[i].keys()))
                    for member in GL[i]:
                         tot +=1
               logger.trace("TOT: {}".format(tot)) 
          # Anonymization process for each istant value of the final time_series
          dataset_anonymized = DatasetAnonymized(GL,leaf_list)
          dataset_anonymized.compute_anonymized_data()
          path = os.path.normpath(dataset_path)
          path_list = path.split(os.sep)
          new_directory = os.sep.join(path_list[0:-1]) + os.sep + "Anonymized" + os.sep + path_list[-1].replace(".csv","")
          isdir = os.path.isdir(new_directory)
          if isdir == False:
               os.mkdir(new_directory)
          new_file_path =  os.sep.join(path_list[0:-1]) + os.sep + "Anonymized" + os.sep + path_list[-1].replace(".csv","") + os.sep + "({},{})Kapra_anonymized.csv".format(k_value,p_value)
          dataset_anonymized.save_on_file(new_file_path)
          # Report of the whole process with time, value_loss and pattern_loss
          final_report = Report(time_list,leaf_list,time_series_for_report,dataset_anonymized.anonymized_data_final,k_value,p_value,paa_value,max_level)
          final_report.printReport(dataset_path)
          report_path = os.sep.join(path_list[0:-1]) + os.sep + "Anonymized" + os.sep + path_list[-1].replace(".csv","")+ os.sep + "({},{})KAPRA_REPORT.txt".format(k_value,p_value)
          final_report.writeReport(report_path)


