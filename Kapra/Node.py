import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from loguru import logger
from saxpy.paa import paa


class Node:

    def __init__(self, level: int = 1, pattern_representation: str = "", label: str = "intermediate",
                 group: dict = None, parent=None, paa_value: int=None):
        self.level = level  # number of different char for representation
        self.paa_value = paa_value #size of pattern
        if pattern_representation == "":
            pr = "a"*self.paa_value  # using SAX
            self.pattern_representation = pr
        else:
            self.pattern_representation = pattern_representation
        self.members = list(group.keys())  # members   time series contained in N
        self.size = len(group)  # numbers of time series contained
        self.label = label  # each node has tree possible labels: bad-leaf, good-leaf or intermediate
        self.group = group  # group obtained from k-anonymity top-down
        self.child_node = list()  # all child nodes
        self.parent = parent  # parent


    def splitting(self, p_value: int, leaf_list: list(), recycle_bad_leaves: list(), max_level: int):
        logger.debug("Start of splitting routine for node with size:{}".format(self.size) )
        # Case in which leaf has less than p values
        # it will be labelled as "bad-leaf" and added to recycle_bad-leaves
        if self.size < p_value: 
            self.label= "bad-leaf"
            recycle_bad_leaves.append(self)
            logger.debug("self.size{} < p_value{}, leaf [{}] will be added to recycle_bad_leaves".format(self.size,p_value,self.members))
            return
        # Case in which the leaf has reached max-level
        # it will be labelled as "good-leaf" and added to the leaf_list 
        if self.level == max_level:
            self.label = "good-leaf"
            leaf_list.append(self)
            logger.debug(" leaf level [{}] == max_level [{}], leaf [{}] will be added to good-leaves ".format(self.level,max_level,self.members))
            return
        # Case in which the leaf size is higher than p but still not worth to be splitted since less than 2*P
        # it will be labelled as "good-leaf" and added to the leaf_list
        if p_value <= self.size < 2*p_value:
            self.maximize(max_level)
            self.label = "good-leaf"
            leaf_list.append(self)
            logger.debug("p_value[{}] <= self.size[{}] < 2*p_value[{}], leaf [{}] will be added to good-leaves".format(p_value,self.size,2*p_value,self.members))
            return
        # Sequence of instructions for all the leafs with more than 2*P elements and not at max-level
        level = self.level + 1
        pr_dict = dict()
        # We process all the Patterns with level = previous_level + 1, and store all the patterns
        for key,value in self.group.items():
            data = np.array(value)
            data_znorm = znorm(data)
            data_paa = paa(data_znorm, self.paa_value)
            pr = ts_to_string(data_paa, cuts_for_asize(level))
            if pr in pr_dict:
                pr_dict[pr].append(key)
            else:
                pr_dict[pr] = [key]
        # We check if all the different patterns would generate leaves with size only less than P
        size_of_nodes_after_split =  [len(x) for x in list(pr_dict.values())]
        all_less_than_P_check = np.all(np.array(size_of_nodes_after_split) < p_value)
        # In case this happens we stop here and declare the leaf as "good-leaf" and add it to the leaf_list
        if all_less_than_P_check:
            logger.debug("Impossible to split any further, leaf will be added to good-leaves")
            self.label = "good-leaf"
            leaf_list.append(self)
            return
        # Else, we can split
        total_number_of_TB_records = 0
        TB_nodes_series = dict()
        TB_pr = dict()
        for pr in pr_dict:
            TG_nodes_series = dict()
            # pr_dict is a dictionary which contains the time_series but discriminated for pattern
            # if the particular pr_dict["aaba..."] has less than p value it will be added to TB_nodes_series
            # TB stands for Tentative Bad
            if len(pr_dict[pr]) < p_value:
                total_number_of_TB_records += 1
                for key in pr_dict[pr]:
                    TB_nodes_series[key] = self.group[key] #dictionary [ID] = time_series of Tentaive Bad Nodes
                    TB_pr[key] = pr
            # else, the pr_dict["aaba...."] contains more than p elements, so it is declared as an intermediate node
            # we call the splitting on this new node
            else:
                for key in pr_dict[pr]:
                    TG_nodes_series[key] = self.group[key] #dictionary [ID] = time_series of Tentaive Good Nodes
                good_node = Node(self.level+1,pr,"intermediate", TG_nodes_series, self, self.paa_value)
                self.child_node.append(good_node)
                logger.trace("Node with members:{} can be split further".format(good_node.members))
                good_node.splitting(p_value=p_value, recycle_bad_leaves= recycle_bad_leaves, leaf_list=leaf_list, max_level=max_level)
        # Now we consider the possible "bad-leaves": if the sum of all the elements is greater or equal than P
        # we merge them into a "good-leaf", with level = previous_level, pattern = previous_pattern 
        if total_number_of_TB_records >=  p_value:
            merge_node = Node(self.level,self.pattern_representation,"good-leaf",TB_nodes_series,self,self.paa_value)
            logger.trace("Adding merged_node to good-leaves: {} ".format(merge_node.members))
            leaf_list.append(merge_node)
            self.child_node.append(merge_node)
            TB_nodes_series = dict()
        # This is the case in which the sum is less than P, in such a case we make n nodes, for n = different patterns
        else:
            if total_number_of_TB_records != 0:
                for key,value in TB_nodes_series.items():
                    tmp = dict()
                    tmp[key] = value
                    bad_leaf_node = Node(self.level+1,TB_pr[key],"bad-leaf",tmp,self,self.paa_value)
                    logger.trace("Adding leaf to bad-leaves: {}".format(bad_leaf_node.members))
                    recycle_bad_leaves.append(bad_leaf_node)
                    self.child_node.append(bad_leaf_node)
        





    def maximize(self, max_level: int):
        """
        :param max_level: Its the maximum level that the alphabet cardinality can reach.

        """
        logger.debug("Maximizing leaf, with initial pr={}".format(self.pattern_representation))
        # In order to maximize the pattern level without splitting we need
        # all the time-series records of the node to have the same pr (with the new level= previous_level +1)
        ts0 = self.group[self.members[0]] #first time series of the node
        equal = True
        temp_level = self.level
        while equal and temp_level < max_level:
            temp_level = self.level + 1
            data0 = np.array(ts0)
            data_znorm0 = znorm(data0)
            data_paa0 = paa(data_znorm0, self.paa_value)
            pr0 = ts_to_string(data_paa0, cuts_for_asize(temp_level))
            for i in range(1,len(self.members)):
                ts = self.group[self.members[i]]
                data = np.array(ts)
                data_znorm = znorm(data)
                data_paa = paa(data_znorm, self.paa_value)
                pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
                # If two higher-level patterns are different, we can't maximize without splitting, so we return without increasing the level any further.
                if pr0 != pr:
                    logger.debug("We can't maximize any further, pr={}".format(self.pattern_representation))
                    equal = False
                    return
            self.level = self.level + 1
            logger.trace("Pattern changes from {} to {}:".format(self.pattern_representation,pr))
            self.pattern_representation = pr
            

    @staticmethod
    def calculate_pattern(paa_value,ts,current_level):
         data = np.array(ts)
         data_znorm = znorm(data)
         data_paa = paa(data_znorm, paa_value)
         pr = ts_to_string(data_paa, cuts_for_asize(current_level))
         return pr
            


    def Recycle_Bad_Leaves(self,list_of_bad_leaves:list,list_of_good_leaves, p_value,paa_value):
        """
        :param list_of_bad_leaves: list of Nodes labelled as "bad-leaf"
        :param list_of_good_leaves: list of Nodes labelled as "good-leaf"
        :param p_value: p_value
        :param paa_value: feature vector space dimension
        """
        max_bad_level = list_of_bad_leaves[0].level
        current_level = max_bad_level
        sum = 0
        for leaf in list_of_bad_leaves:
            sum+=leaf.size
        logger.debug("Recycle process with initial leaf number = {}".format(sum))
        lista_copia = list_of_bad_leaves.copy()
        # We check every time how many members are contained in the remaning leaves.
        # In case they are less than P, we will never be able to recycle them with other bad leaves, and so they will be permanently removed
        while sum >=  p_value:
            counter = 0
            # We iterate through all the bad leaves
            for i in range(0,len(list_of_bad_leaves)):
                leaf1 = list_of_bad_leaves[i]
                pr1 = leaf1.pattern_representation
                if leaf1.pattern_representation == "":
                    continue
                for j in range(0,len(list_of_bad_leaves)):
                    if i==j:
                        continue
                    if leaf1.pattern_representation == "":
                        continue
                    leaf2 = list_of_bad_leaves[j]
                    pr2 = leaf2.pattern_representation
                    if leaf2.pattern_representation == "":
                        continue
                    # if two different leaves have same level and pattern we merge them
                    if pr1 == pr2 and leaf1.level == leaf2.level:
                        merge_leaf_list = leaf1.members + leaf2.members
                        merge_dict = leaf1.group | leaf2.group
                        total_size = leaf1.size + leaf2.size
                        # if the merge produces a leaf with size equal or greater than P, we declare it a good leaf
                        # the leaf is not going to be removed yet, first we wait to end the iteration to see if other's bad leaf have same pr and level as this one
                        if total_size >=  p_value:
                            logger.trace("Merged leaf {} with leaf {} pattern: {}".format(leaf1.members,leaf2.members,pr1))
                            logger.debug("Creating Good-leaf")
                            good_leaf = Node(leaf1.level,pr1,"good-leaf",merge_dict,leaf1.parent,paa_value)
                            list_of_bad_leaves[j] = good_leaf
                            list_of_bad_leaves[i].pattern_representation = ""
                        # if the merge produces a leaf with size less than P, we simply add it as a new leaf in the recycle list
                        else:
                            logger.trace("Merged leaf {} with leaf {} pattern: {}".format(leaf1.members,leaf2.members,pr1))
                            logger.debug("Creating Bad-Leaf")
                            new_node = Node(leaf1.level,pr1,"bad-leaf",merge_dict,leaf1.parent,paa_value)
                            list_of_bad_leaves[j] = new_node
                            list_of_bad_leaves[i].pattern_representation = ""
                            
                         
            current_level -=1     
            # simple iteration to remove all the leaves who should have been removed in the iteration cycle due to their merge
            for leaf in list_of_bad_leaves.copy():
                if leaf.pattern_representation == "":
                    list_of_bad_leaves.remove(leaf)
                    continue
                # Those who's size is greater or equal than P will be finally removed and added to the good leaves list
                if leaf.size >=  p_value:
                    list_of_good_leaves.append(leaf)
                    list_of_bad_leaves.remove(leaf)
                    logger.debug("Adding leaf with members{} to good-leaf.".format(leaf.members))
            # at the end of the iteration we force the downgrade of the level of all the current leaves with the highest level
            sum = 0
            if len(list_of_bad_leaves) > 0:
                for leaf in list_of_bad_leaves:
                    sum+=leaf.size
                    if leaf.level > current_level and leaf.level != 1:
                        leaf.level = current_level
                        leaf.pattern_representation = leaf.parent.pattern_representation    
                        leaf.parent = leaf.parent.parent
                        logger.trace("Leaf members {} pattern becomes {} with level {}".format(leaf.members,leaf.pattern_representation,leaf.level))
        # If the sum of the members in the remaining leaves are less than P but not zero, they will be removed.
        if sum != 0:
            for leaf in list_of_bad_leaves:
                logger.info("Leaf {} will be supressed".format(leaf.members))

                    


                        
            




    # Simple function used by drawTree
    def hasChild(self):
        if self.child_node:
            return True
        else:
            return False
    # Simple function to draw the Tree
    def drawTree(self, tree, parent=None, i = 0):
        if i!= 0 and ' '.join(self.members)!= parent:
            tree.create_node(self.members,' '.join(self.members),parent)
        for child in self.child_node:
            if i!= 0:
                parent = ' '.join(self.members)
            i = 1
            child.drawTree(tree,parent,i)
            
    



   
        
    
                    






    