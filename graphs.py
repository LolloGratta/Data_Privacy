import matplotlib.pyplot as plt
import copy
import numpy as np

#value_loss increasing K (5,5) (7,5) (10,5) (20,5)
value_loss_Kapra_increasing_K = [11062.22,11757.62,12875.95,14260.73 ]
experiments = list()
experiments_increasing_K = ["increasing_K",(5,5),(7,5),(10,5),(20,5),(40,5)]
experiments.append(experiments_increasing_K)
experiments_increasing_P = ["increasing_P",(15,3),(15,5),(15,6),(15,7),(15,8)]
experiments.append(experiments_increasing_P)
results = dict()
result_folder = dict()
increasing_P_dict = dict()
loss = dict()
loss["PL"] = list()
loss["VL"] = list()
loss["time"] = list()
increasing_P_dict["KAPRA"] = copy.deepcopy(loss)
increasing_P_dict["NAIVE"] = copy.deepcopy(loss)
increasing_K_dict = dict()
increasing_K_dict["KAPRA"] = copy.deepcopy(loss)
increasing_K_dict["NAIVE"] = copy.deepcopy(loss)
result_folder["increasing_P"] = copy.deepcopy(increasing_P_dict)
result_folder["increasing_K"] = copy.deepcopy(increasing_K_dict)


folders = ["./Dataset/Anonymized/Sales_Transaction_Dataset_Weekly_Final/", "./Dataset/Anonymized/exoStars/" ]

total_times = dict()

for folder in folders:
    print(folder)
    for i in range(0,len(experiments)):
        kapra_pattern_loss = list()
        kapra_value_loss = list()
        naive_pattern_loss = list()
        naive_value_loss = list()
        kapra_total_time = list()
        naive_total_time = list()
        times = list()
        for j in range(1,len(experiments[i])):
            file_name = folder + str(experiments[i][j]).replace(' ','') + "KAPRA_REPORT.txt"
            f = open(file_name, "r")
            content = f.readlines()
            kapra_pattern_loss.append(content[8].split(' ')[-2])
            kapra_value_loss.append(content[9].split(' ')[-2])
            kapra_total_time.append(content[2].replace('s','').split(' ')[-2])
            f.close()
            file_name = folder + str(experiments[i][j]).replace(' ','') + "NAIVE_REPORT.txt"
            f = open(file_name,'r')
            content = f.readlines()
            naive_pattern_loss.append(content[3].split(' ')[-2])
            naive_value_loss.append(content[4].split(' ')[-2])
            naive_total_time.append(content[2].replace('s','').split(' ')[-2])
            f.close()
        result_folder[experiments[i][0]]["KAPRA"]["PL"] = kapra_pattern_loss
        result_folder[experiments[i][0]]["KAPRA"]["VL"]= kapra_value_loss
        result_folder[experiments[i][0]]["NAIVE"]["PL"] = naive_pattern_loss
        result_folder[experiments[i][0]]["NAIVE"]["VL"] = naive_value_loss
        result_folder[experiments[i][0]]["KAPRA"]["time"] = kapra_total_time
        result_folder[experiments[i][0]]["NAIVE"]["time"] = naive_total_time
        times.append(kapra_total_time)
        times.append(naive_total_time)
    results[folder.split('/')[-2]] = copy.deepcopy(result_folder)
    total_times[folder.split('/')[-2]] = copy.deepcopy(times)


    #draw graphs

# plot time histograms





for experiment in experiments:
    kapra_pattern_loss = list()
    kapra_value_loss = list()
    naive_pattern_loss = list()
    naive_value_loss = list()
    plt.clf()
    for ds in results:
        kapra_pattern_loss = np.array(results[ds][experiment[0]]["KAPRA"]["PL"])
        kapra_value_loss = np.array(results[ds][experiment[0]]["KAPRA"]["VL"])
        naive_pattern_loss = np.array(results[ds][experiment[0]]["NAIVE"]["PL"])
        naive_value_loss = np.array(results[ds][experiment[0]]["NAIVE"]["VL"])
        kapra_time_forExperiment = np.array(results[ds][experiment[0]]["KAPRA"]["time"])
        naive_time_forExperiment = np.array(results[ds][experiment[0]]["NAIVE"]["time"])
        plt.subplot(121)
        x = list()
        for i in range(1,len(experiment)):
            if experiment[0] == "increasing_P":
                x.append(int(experiment[i][1]))
                plt.xlabel("P_values")
                plt.title("increasing_P with K={}".format(experiment[1][0]))
            else:
                x.append(int(experiment[i][0]))
                plt.xlabel("K_values")
                plt.title("increasing_K with P={}".format(experiment[1][1]))
        plt.plot(x,[float(i) for i in kapra_pattern_loss],label = ds + ",KAPRA")
        plt.plot(x,[float(i) for i in naive_pattern_loss],label = ds +  ",Naive ")
        plt.legend()
        plt.ylabel("Total Pattern Loss")
        plt.subplot(122)  
        plt.plot(x,[float(i) for i in kapra_value_loss],label = ds + ",KAPRA")
        plt.plot(x,[float(i) for i in naive_value_loss],label = ds +  ",Naive ")
        plt.legend()
        if experiment[0] == "increasing_P":
            plt.xlabel("P_values")
            plt.title("increasing_P with K={}".format(experiment[1][0]))
        else:
            plt.xlabel("K_values")
            plt.title("increasing_K with P={}".format(experiment[1][1]))
        plt.ylabel("Total Value Loss")
        plt.show()
        #times
        x = np.array([x for x in range(1,len(experiment))])
        my_xticks = [str(x) for x in experiment[1::]]
        plt.xticks(x,my_xticks)
        plt.bar(x-0.2,[float(i) for i in kapra_time_forExperiment],width = 0.5,label = ds + ",KAPRA") # A bar chart
        plt.bar(x+0.2,[float(i) for i in naive_time_forExperiment],width = 0.5,label = ds + ",NAIVE") # A bar chart
        plt.legend()
        plt.xlabel('Experiments')
        plt.ylabel('Time in seconds')
        plt.title("Time difference between Kapra and Naive "+ experiment[0])
        plt.show()



