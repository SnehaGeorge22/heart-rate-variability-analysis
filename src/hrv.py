#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This python file processes the annotated ECG data and calculates the HRV metrics. 
   To fetch the data files, create a folder named "investigating_hrv_dataset" inside this current directory.
   Place the csv files inside the folder investigating_hrv_dataset for the python script to 
   automatically fetch and process the data files.
"""

# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setting the current directory and dataset directory to fetch the csv files  
current_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_dir,"investigating_hrv_dataset")


def calculate_HRV_metrics(file_in:str) -> dict:
    """
    This method calculates the Heart Rate Variability metrics based on the provided annotated ECG records of
    people of differnt age groups

    Arguments:
    ----------
    file_in : str
        A string containing the name of the csv file that contains the ECG data of a person.

    Returns:
    --------
    results : dictionary
        returns a dictionary containing the calculated hrv metrics for the provided input ECG data 
        of a person. Below are the details of the dictionary values:

        - "filename" : 
            the variable stores the name of the file for which HRV metrics is calculated.

        - "n" : 
            Total Number of N-N beat intervals for the given person.

        - "mean_nn" : 
            the mean duration of all N-N intervals in the recording.

        - "mean_bpm" : 
            the mean number of beats per minute calculated using the duration of all N-N intervals 
            in the recording.

        - "sdnn" : 
            the standard deviation of the set of all N-N intervals.

        - "rmssd" : 
            the square root of the mean squared value of the beat-to-beat difference for all sequential 
            N-type beats in the recording.

        - "pnn20" :
            the proportion of beat-to-beat interval differences, where the absolute difference is greater than 20ms 
            for all sequential N-type beats in the recording.

        - "pnn50" :
            the proportion of beat-to-beat interval differences, where the absolute difference is greater than 50ms 
            for all sequential N-type beats in the recording.
    """
    
    try:
        print("-------------------------------------------------")
        # Initializing the file dictionary variable
        results = {}

        # Concatenating the Dataset directory with the file name to access the csv file
        file_name = os.path.join(dataset_dir,file_in)

        # Checking whether the given file is a valid file inside the directory
        if os.path.isfile(file_name):
            print(f"Calculation of HRV metrics for {file_in} started..")
            # Reading the data from the given csv file as a dataframe "hrv_data"
            hrv_data = pd.read_csv(file_name, delimiter=',')

            # Making a copy of dataframe locally
            hrv_df = hrv_data.copy()

            # Storing the time of next beat in the dataframe as column 'time_next'
            hrv_df['time_next'] = hrv_df['time'].shift(-1)

            # Storing the type of next beat in the dataframe as column 'type_next'
            hrv_df['type_next'] = hrv_df['type'].shift(-1)

            # Storing the interval between the current and next beat using time and time_next columns
            hrv_df['rr'] = hrv_df['time_next']- hrv_df['time']

            # combining the type and type_next columns as a new column 'rr_type'
            hrv_df['rr_type'] = hrv_df['type']+hrv_df['type_next']

            # Storing the period of the next rr interval as a new column 'rr_next'
            hrv_df['rr_next'] = hrv_df['rr'].shift(-1)

            # Storing the difference between the current interval and the next interval
            hrv_df['diff'] = hrv_df['rr']-hrv_df['rr_next']

            # Storing the squared value of the diff column
            hrv_df['diff_squared'] = hrv_df['diff']**2

            # Storing the type of the next interval in the new column "rr_next_type"
            hrv_df['rr_next_type'] =hrv_df['rr_type'].shift(-1)

            # Filtering only the NN type intervals from the dataframe "hrv_data"
            hrv_df = hrv_df[hrv_df['rr_type']=='NN']

            # Getting the total number of intervals in variable 'n'
            n = int(hrv_df.shape[0])

            # Calculating the HRV metrics
            mean_nn = int(np.round(hrv_df['rr'].mean(),decimals=0))
            mean_bpm = np.round((1000/mean_nn)*60,decimals=1)
            sdnn = np.round(np.std(hrv_df['rr'], ddof=1),decimals=1)

            # filtering out the sequential NN intervals for calculating the metrics
            hrv_rr_df = hrv_df[hrv_df['rr_next_type']=='NN']
            rmssd = np.round(np.sqrt(np.mean(hrv_rr_df['diff_squared'].dropna())),decimals=1)
            pnn50 = np.round(100 * np.sum(np.abs(hrv_rr_df['diff']) > 50) / n,decimals=1)
            pnn20 = np.round(100 * np.sum(np.abs(hrv_rr_df['diff']) > 20) / n,decimals=1)
        
            # Checking whether total number of NN type intervals are more than 500
            if n >= 500:
                # Storing the HRV metrics in the results dictionary
                results['filename'] = file_in
                results['n'] = n
                results['mean_nn'] = mean_nn
                results['mean_bpm'] = mean_bpm
                results['sdnn'] = sdnn
                results['rmssd'] = rmssd
                results['pnn20'] = pnn20
                results['pnn50'] = pnn50
            else:

                # if the NN intervals are less than 500, the results dictionary is populated with None values
                results['filename'] = file_in
                results['n'] = n
                results['mean_nn'] = None
                results['mean_bpm'] = None
                results['sdnn'] = None
                results['rmssd'] = None
                results['pnn20'] = None
                results['pnn50'] = None

            # Returning the calculated HRV metrics in the results dictionary object.
            print("-------------------------------------------------")
            for k,v in results.items():
                print(f"{k} : {v} ")
            print("-------------------------------------------------")
            print(f"Calculation of HRV metrics for {file_in} completed!")

            return results
        else:
            print(f"File '{file_in}' cannot be accessed. Please verify! ")
    except Exception as e:
        print(f"Exception : {e}")


def process_HRV_files(file_list_in:list[str], file_out:str) -> None:
    """
    This method processes the given list of files to fetch the annotated ECG data of people across 
    different age groups.

    Arguments:
    ----------
    file_list_in : List[str]
        A list of strings containing the file names of the annotated ECG data of different people.

    file_out : str
        A string containing the name of the output csv file which needs to be created with the calculated
        hrv metrics.

    Returns : 
    ---------
    None 

    Additional Information:
    -----------------------
    This method fetches the csv data files from the dataset directory and calculate the HRV metrics for the given data.
    Once the data has been calculated , the output results are added to the csv file and saved the same location.
    
    """
    try:
        print("-------------------------------------------------")
        print("Processing of the input data csv files started..")
        # Initializing the output list variable 'hrv_results'
        hrv_results = []

        # Iterating over the file in the provided list of files in the directory
        for file in file_list_in:

            # Checking whether the file ends with '.csv' format
            if file.endswith(".csv"):

                # Calling the 'calculate_HRV_metrics' method with the file name as input
                result_dict=calculate_HRV_metrics(file_in=file)
                #print(f"Results for file {file} : {result_dict}")

                # Appending the results dictionary to the final list
                hrv_results.append(result_dict)
            else:
                print("-------------------------------------------------")
                print(f"File format '{file}' is not csv. Please verify!!")
                print("-------------------------------------------------")
        #print(f"Total HRV results : {hrv_results}")
        
        # Converting the output list to the dataframe
        df = pd.DataFrame(hrv_results)
        print("Processing of the input data csv files completed!!")
        print("-------------------------------------------------")
        print(df)
        print("-------------------------------------------------")
        # Adding the output file name with the current directory
        file_out_name = os.path.join(current_dir,file_out)

        # Converting the dataframe into output csv file
        df.to_csv(file_out_name, index=False)

        if os.path.isfile(file_out_name):
            print(f"Output csv file {file_out} is generated successfully")
        else:
            print(f"Output csv file {file_out} is not generated. Please verify")
        print("-------------------------------------------------")

    except Exception as e:
        print(f"Exception : {e}")


if __name__== "__main__":
    """
    This is the entry point of HRV.PY python script. It is the main method.
    """
    print("WELCOME TO THE HRV.PY FILE")
    # Print the directory details
    print("-------------------------------------------------")
    print(f"Current Directory : {current_dir}")
    print(f"Dataset Directory : {dataset_dir}")
    
    # Checking whether the given directory is exists and valid. if not , raise Exception
    try:
        if os.path.exists(dataset_dir) and os.listdir(dataset_dir):

            # Creating a list of file names 
            file_list_in = os.listdir(dataset_dir)
            
            #print(file_list_in)

            # Invoking the process_HRV_files function with file list and output file name
            process_HRV_files(file_list_in=file_list_in, file_out='fantasia.csv')
        else:
            raise Exception("Directory/Folder doesnt exist. Please check the path!!")
    except Exception as e:
        print(f"Exception : {e}")