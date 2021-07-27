#!/usr/bin/python3

'''
Author: Ambareesh Ravi
Date: 26 July, 2021
File: data.py
Description:
    Handles the loading of dataset
'''

# imports
import pandas as pd

class Dataset:
    # Class to load data as csv file
    def __init__(self, file_name):
        '''
        Initializes the class object

        Args:
            file_name: file path as <str>
        Returns:
            -
        Exception:
            -
        '''
        self.df = pd.read_csv(file_name)
    
    def __call__(self,):
        '''
        Object call returns the dataframe

        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        return self.df