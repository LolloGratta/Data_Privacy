import numpy as np
from loguru import logger
import copy


# General class that anonymize the istant values of each K group and adds two columns, one with Group Id 
# and one with the pattern of the time_series w.r.t. (k,p) group.


class DatasetAnonymized:
    def __init__(self, anonymized_data: list = list(), good_leaf_list: list = list()):
        self.anonymized_data = copy.deepcopy(anonymized_data)
        self.anonymized_data_final = dict()
        group_index = 0
        for ts_dict in anonymized_data:
            for key,value in ts_dict.items():
                value.append("Group {}".format(group_index))
                self.anonymized_data_final[key] = value
            group_index += 1
        for leaf in good_leaf_list:
            for member in leaf.members:
                self.anonymized_data_final[member].append(leaf.pattern_representation)
        print("end")



    def compute_anonymized_data(self):
        """
        create dataset ready to be anonymized
        :return:
        """
        logger.info("Start creating anonymized dataset ")
        for index in range(0, len(self.anonymized_data)):
            logger.trace("Start creating Group {}".format(index))

            group = self.anonymized_data[index]
            max_value = np.amax(np.array(list(group.values())), 0)
            min_value = np.amin(np.array(list(group.values())), 0)
            for key in group.keys():
                # key = row product
                group2 = self.anonymized_data_final[key]
                for column_index in range(0, len(max_value)):
                    group2[column_index] = ("[{} - {}]".format(round(min_value[column_index],2), round(max_value[column_index],2)))
            logger.trace("Finish creation Group {}".format(index))

    def save_on_file(self, name_file):
        with open(name_file, "w") as file_to_write:
            value_to_print_on_file = ""
            for key, value in self.anonymized_data_final.items():
                value_to_print_on_file = key
                value_to_print_on_file = "{},{}".format(value_to_print_on_file, ",".join(value))
                file_to_write.write(value_to_print_on_file+"\n")