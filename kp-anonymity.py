import sys
from Kapra.kapra import kapra_main
from Naive.naive import naive_main



if __name__ == "__main__":

    if len(sys.argv) >=7:
        algorithm = sys.argv[1]
        k_value = int(sys.argv[2])
        p_value = int(sys.argv[3])
        paa_value = int(sys.argv[4])
        max_level = int(sys.argv[5])
        dataset_path = sys.argv[6]
        verbosity_level = sys.argv[7]
        if len(sys.argv) ==9:
            nrows = int(sys.argv[8])
        else:
            nrows = None

        if k_value >= p_value:
            if algorithm == "both":
                kapra_main(k_value=k_value,p_value=p_value,paa_value=paa_value,max_level=max_level,dataset_path=dataset_path,nrows=nrows,verbosity_level = verbosity_level)
                naive_main(k_value=k_value,p_value=p_value,paa_value=paa_value,max_level=max_level,dataset_path=dataset_path,nrows=nrows,verbosity_level = verbosity_level)
            elif algorithm == "kapra.py":
                kapra_main(k_value=k_value,p_value=p_value,paa_value=paa_value,max_level=max_level,dataset_path=dataset_path,nrows=nrows,verbosity_level = verbosity_level)
            elif algorithm == "naive.py":
                naive_main(k_value=k_value,p_value=p_value,paa_value=paa_value,max_level=max_level,dataset_path=dataset_path,nrows=nrows,verbosity_level = verbosity_level)
            else:
                print("[*] Usage: python kp-anonymity.py algorithm.py k_value p_value paa_value max_level dataset.csv verbosity_level number_rows")
                print("[*] algorithm.py is either kapra.py, naive.py or both")
                print("[*] k_value must be greater or equal than p_value")
                print("[*] verbosity_level defines how verbose the logger will be [TRACE,DEBUG,INFO] (most verbose to lowest)")
                print("[*] nrows defines how many rows of the dataset.csv will be read, if None all dataset will be used")
                
    else:
        print("[*] Usage: python kp-anonymity.py (kapra.py/naive.py/both) k_value p_value paa_value max_level dataset.csv verbosity_level number_rows")