### Import external modules;

import pandas as pd;
import numpy as np;
from simple_salesforce import Salesforce, SFType
from simple_salesforce.exceptions import SalesforceError
import re
import copy
from tqdm import tqdm 
import json
import traceback
from ydata_profiling import ProfileReport
import sys
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches 
import os
import nltk
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import dask
import sys 
import time
import requests
import csv
import yaml

# Install and initialize the sklearn patch;
from sklearnex import patch_sklearn
patch_sklearn()



class Objeto_sf:
    def __init__(self, name, apiName, booleanGraph=0, environment='prod', columnsNameType='Label'):
        
        ### Initialize variables;
        self._name = name; # Object name as known at the operational level;
        self._apiName = apiName; # Object name in ApiName, at the development level;
        self._columns = {}; # Initialize a dictionary to store information related to the object columns;
        
        ''' 
        Architecture of self._columns:
        
        _columns: {
            _columnsData: (pd.DataFrame --> containing the data of each column),
            _columnsNames: [] --> List of all column names in the object,
            _columnsNameWorking: [] --> List of column names that did not return an error when testing a request to SF,
            _columnsNameWorking_compound; A list of column names that are listed as 'compound' in the metadata because they are not returned when using the Salesforce bulk API.
            _columnsNameWorking_NOT_compound; Same as above but for non-compound fields, meaning they can be requested from the bulk API without issues.
            _columnsNameNotWorking: [] --> List of column names that returned an error when tested in an SF request and cannot be queried,
            _columnsNameWorkingToQuery : '' --> String containing all working column names, ready to be added to a SOQL query to retrieve all object information,
            _columnsApiName_Field_DataType: ({} A dictionary containing three keys: _working, _notWorking, _full. Each has three columns: API Name of the field, Label as it appears in the Salesforce frontend, and DataType of each column. The difference is that _working is filtered by columns that passed the request validation, _notWorking contains the ones that failed, and _full includes all columns.),
        }
        '''
        
        if booleanGraph not in [0,1]:
            raise ValueError("The value must be 0 to disable graphs or 1 to enable them;")
        self._executeGraph = booleanGraph;
                
        self._data = pd.DataFrame(); # Stores the dataset of the object with all its records;
        self._metadata = pd.DataFrame(); # Stores the metadata dataset of the object with all its columns and information;
        self._dataSample = pd.DataFrame(); # Stores a sample dataset of 10 records but with all object columns as a preview;
        self._data_to_graph = pd.DataFrame();
        self._queryAllObjectsInEntityDefinition = 'SELECT FIELDS(ALL) FROM EntityDefinition';
        self._queryFieldsAll_FieldDefinition = f"SELECT FIELDS(ALL) FROM FieldDefinition WHERE EntityDefinition.QualifiedApiName = '{self._apiName}'"; # This query retrieves all fields and metadata of the object from FieldDefinition;
        self._queryFullColumns_oneObject = f"SELECT 'attributes', 'Id', 'DurableId' FROM '{self._apiName}'"; # This query retrieves all object records; 
        self._environment = environment;
        self._columnsNameType = columnsNameType;
        
        
        ### Initializing functions;
        self.connection_to_salesforce();
        self.getColumns();
        self.filterWorkingColumns();
        self.getFullDataObject();
        self.initial_prep_data_for_graphs();
        
        if self._executeGraph == 1:
            self.bar_graph_sf_object_NaN();
        else:
            print('Graphs are disabled. To enable them, please rerun with the value 1 in the third parameter of the constructor;')


    def connection_to_salesforce(self):
        
        # The following sets up the connection to Salesforce depending on the specified environment.
        
        try:
            
            # Start time measurement
            start_time = time.perf_counter();
            
            environment = self._environment;
            
            if (environment=='prod'):
                try:
                    self._sf = Salesforce(
                        instance_url='https://.com', session_id='',
                        username='@.com.prod',
                        password='',
                        security_token='',
                        version='58.0'
                        ) # Initialize an instance of the Salesforce object from the simple_salesforce library to establish a connection;                    
                except Exception as e:
                    print(f'An error occurred while connecting to Salesforce environment: prod, error details: {e}\n')
            
            elif (environment=='full'):
                try:
                    self._sf = Salesforce(
                        domain='test', session_id='',
                        username='@.com.full',
                        password='',
                        security_token='',
                        version='58.0'
                        ) # Connection to full copy environment.
                except Exception as e:
                    print(f'An error occurred while connecting to Salesforce environment: full, error details: {e}\n')
            
            elif (environment=='uat'):
                try:
                    self._sf = Salesforce(
                        domain='test', session_id='',
                        username='@.com.uat',
                        password='',
                        security_token='',
                        version='58.0'
                        ) # Connection to UAT copy environment.                    
                except Exception as e:
                    print(f'An error occurred while connecting to Salesforce environment: uat, error details: {e}\n')
            
            elif (environment=='qa'):
                try:
                    self._sf = Salesforce(
                        domain='test', session_id='',
                        username='@.com.qa',
                        password='',
                        security_token='',
                        version='58.0'
                        ) # Connection to QA copy environment.
                except Exception as e:
                    print(f'An error occurred while connecting to Salesforce environment: qa, error details: {e}\n')
                
            # End time measurement
            end_time = time.perf_counter()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
                
            print('\033[92m',
                    f'\nSuccessfully connected to Salesforce.', 
                    f'\nEnvironment: {self._environment}.' 
                    f'\nColumnsNameType: {self._columnsNameType}.' 
                    f'\nConnection time: {elapsed_time:.6f} seconds.', 
                    '\033[0m')            
        
        except Exception as e:
            print('Failed connection to Salesforce DB;')


    def sf_custom(self, query:str, stage):
        
        '''
        This function connects you to Salesforce to allow you to send requests to Salesforce in order to obtain the dataset resulting from the provided query;
        
        Do not indicate LIMIT or OFFSET values, this function is designed to return all object records;
        
        Basically, this function returns the metadata of the required object, saving it in a variable called self._metadata;
        '''
        
        try:    
            
            print('\033[96m',f'\nsf_custom method initialized.', '\033[0m')
            
            # Start point
            start_time = time.perf_counter();

            if query:         
                
                # Initialize an offset counter
                offset = 0

                # Initialize a list to store all records
                all_records = []
                
            ###########################################
                
                ### Match pattern for FIELDS;            
                match_FIELDS = re.search(r'FIELDS\s*\(?', query, re.IGNORECASE)
                
                
                ### Match pattern for ObjectPermissions;
                match_ObjectPermissions = re.search(r'objectpermissions', query, re.IGNORECASE)
                
            ###########################################
                
                if match_FIELDS: 
                    
                    query = query + ' LIMIT 200'
                    
                    
                    while True:
                        
                        # Execute the query with the current offset
                        result = self._sf.query_all(f'{query} OFFSET {offset}')
                        
                        # Add records to the list
                        all_records.extend(result['records'])
                        
                        # If there are more records, increase the offset
                        if len(result['records']) == 200:
                            
                            print('offset_1', offset)
                            if (bool(match_ObjectPermissions) & (offset <= 1800) | (bool(match_ObjectPermissions) != True)):
                                offset += 200 
                                
                                print('bool(offset <= 1800)', bool(offset <= 1800))
                                
                                print('offset_2', offset)
                            
                            else: 
                                break
                        else:
                            break

                    # Now, 'all_records' contains all retrieved records
                    
                    df_result = pd.DataFrame(all_records)
                    
                    self._metadata = df_result
                    
                    # End point
                    end_time = time.perf_counter()

                    # Calculate elapsed time
                    elapsed_time_sf_custom = end_time - start_time
                
                    print('\033[92m',
                            f'\nsf_custom method ended successfully.',
                            f'\nMethod time: {elapsed_time_sf_custom:.6f} seconds.',
                            f'\nNumber of columns in {stage} {len(all_records)}', 
                            '\033[0m')
                    
                    return df_result, elapsed_time_sf_custom
                    
                else:
                        
                    while True:
                        # Execute the query with the current offset
                        result = self._sf.query_all(query.format(offset))
                        
                        # Add records to the list
                        all_records.extend(result['records'])
                        
                        # If there are more records, increase the offset
                        if not result['done']:
                            offset += 2000
                        else:
                            break

                    # Now, 'all_records' contains all retrieved records
                    print(len(all_records))  # Show total number of records
                    
                    df_result = pd.DataFrame(all_records)
                    
                    return df_result
                
            # End point
            end_time = time.perf_counter()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
                
            print('\033[92m',
                    f'\nsf_custom method ended successfully.',
                    f'\nMethod time: {elapsed_time:.6f} seconds. \n', 
                    '\033[0m') 
            
        except SalesforceError as e:
            # If a Salesforce query error occurs, raise a custom exception
            raise Exception(f"Salesforce query error: {str(e)}")

        except Exception as e:
            # Handle any other exception that occurs during the query
            print(f"An exception occurred: {str(e)}")
            traceback.print_exc()


    def getColumns(self):
            
        '''
        This method retrieves a dataframe with all the columns of the queried object and stores the information in different internal variables: '_columnsData' with the entire dataframe, '_columnsApiNameAndField' with a dataframe containing two columns (one with the column name in apiName and another with the Label name), and '_columnsNames' which is a list of all column names.
        '''
        
        try:
            
            # Start point
            start_time = time.perf_counter();
            
            print('\033[96m',f'\ngetColumns method initialized.', '\033[0m')
            
            # Request the list of all fields the object has according to the apiName provided as an argument;
            
            df_result, elapsed_time_sf_custom =  self.sf_custom(self._queryFieldsAll_FieldDefinition, 'getColumns');            
            
            
            # Store the column data in an internal class variable;
            self.metadata = df_result;
            
            # Store in a list all column names returned from the initial request, to be able to filter them and make another query later;
            self._columns['_columnsNames'] = self.metadata["QualifiedApiName"].tolist();
            
            # End point
            end_time = time.perf_counter()

            # Calculate elapsed time
            elapsed_time = end_time - start_time - elapsed_time_sf_custom
                
            print('\033[92m',
                    f'\ngetColumns method ended successfully.',
                    f'\nMethod time: {elapsed_time:.6f} seconds.', 
                    f'\nNote: This method contains the sf_custom method, and the time is discriminated. \n', 
                    '\033[0m')

        except Exception as e:
            print(f'An error occurred in the getColumns function, the error is: {e}\n')
            traceback.print_exc()
            
            
    def filterWorkingColumns(self):
            
        '''
        Filters the object's columns based on whether they work in a test query request to Salesforce and creates two lists of column names to later query only the working ones.
        '''
        
        try:
            
            # Start point
            start_time = time.perf_counter();
            
            print('\033[96m',f'\nfilterWorkingColumns method initialized.', '\033[0m')
            
            # Create an independent copy of the column name list;
            columnsToQuery = copy.deepcopy(self._columns['_columnsNames'])
            
            # Initialize two variables to store the names of the columns that work in a Salesforce test query and those that do not.
            columnsWorking = [];
            
            columnsNotWorking = [];

            # Iterate over the column name list to run test queries against Salesforce and differentiate the working ones from the non-working ones.
            for column in tqdm(columnsToQuery, desc='Iter columnsToQuery'):
                
                try:
                    
                    self._sf.query_all(f"SELECT {column} FROM {self._apiName} LIMIT 1")['records']

                except:
                    columnsNotWorking.append(column)
                else:
                    columnsWorking.append(column)

            # Store the resulting lists in internal variables.
            self._columns['_columnsNameWorking'] = columnsWorking;
            self._columns['_columnsNameNotWorking'] = columnsNotWorking;
            
            
            ### For the bulk API, it is necessary to differentiate compound fields from non-compound fields, so they are filtered below.
            
            compound_metadata_list = self._metadata.loc[self._metadata['IsCompound']==True, 'QualifiedApiName'].tolist()
                        
            compound_columns = [];
            not_compound_columns = [];
            
            for column in columnsWorking:
                
                if column in compound_metadata_list:
                    compound_columns.append(column);
                else:
                    not_compound_columns.append(column);
            
            self._columns['_columnsNameWorking_compound'] = compound_columns;
            self._columns['_columnsNameWorking_NOT_compound'] = not_compound_columns;
            
            # Create a single string with the names of the working columns to prepare for the next step: querying the entire object.
            self._columns['_columnsNameWorkingToQuery'] = ', '.join(self._columns['_columnsNameWorking'])
            print("self._columns['_columnsNameWorkingToQuery']", self._columns['_columnsNameWorkingToQuery'])
            
            ### Generate the string with the column names required for bulk API that are not compound.
            self._columns['_columnsNameWorkingToQuery_to_bulk'] = ', '.join(self._columns['_columnsNameWorking_NOT_compound']);
            
            ### Generate the string with the column names required for the REST API that are compound.
            self._columns['_columnsNameWorkingToQuery_to_sf_rest_api'] = ', '.join(self._columns['_columnsNameWorking_compound'])
            
            ##### Next, filter the object's metadata information based on working and non-working columns.
            
            '''
                Filters the object's metadata dataset by separating working columns from non-working ones, then stores the results in three keys within the _columnsApiName_Field_DataType dictionary:
                - _working (containing information on working columns),
                - _notWorking (containing information on non-working columns),
                - _full (containing the original dataset with all columns and their information).
            '''
            
            # Store a dataframe in an internal class variable containing API Name, Label, and DataType for column name changes.
            
            toWorking = self._metadata.loc[self._metadata['QualifiedApiName'].isin(self._columns['_columnsNameWorking']), ['QualifiedApiName','Label', 'DataType']]
            
            toNotWorking = self._metadata.loc[(~self._metadata['QualifiedApiName'].isin(self._columns['_columnsNameWorking'])), ['QualifiedApiName','Label', 'DataType']]
            
            toFull = self._metadata[['QualifiedApiName','Label', 'DataType']];
                        
            # Create a key inside _columns to store a dictionary where processed data will be stored.
            self._columns['_columnsApiName_Field_DataType'] = {
                '_working': toWorking,
                '_notWorking': toNotWorking,
                '_full': toFull
                }; 
            
            # End point
            end_time = time.perf_counter()

            # Calculate elapsed time
            elapsed_time = end_time - start_time;
                
            print('\033[92m',
                    f'\nfilterWorkingColumns method ended successfully.',
                    f'\nMethod time: {elapsed_time:.6f} seconds.', 
                    '\033[0m')
        
        except Exception as e:
            print(f'An error occurred in the filterWorkingColumns function, the error is: {e}\n')
            traceback.print_exc()
            
            
    def getFullDataObject(self):
            
        '''
        Executes a query with the names of the working columns and stores the resulting dataframe in an internal variable.
        '''

        try:
            
            # Start point
            start_time = time.perf_counter();
            
            print('\033[96m',f'\ngetFullDataObject method initialized.', '\033[0m')
            
            # Use the column string directly in the SOQL query
            columnas = self._columns['_columnsNameWorkingToQuery_to_bulk'];

            # Create the SOQL query
            query = f'SELECT {columnas} FROM {self._apiName}';

            # Headers for the request
            headers = {
                'Authorization': f'Bearer {self._sf.session_id}',
                'Content-Type': 'application/json'
            };

            # Base URL for Bulk API 2.0
            bulk_api_url = f"https://{self._sf.sf_instance}/services/data/v{self._sf.sf_version}/jobs/query";

            # Create the query job
            job_data = {
                "operation": "query",
                "query": query,
                "contentType": "CSV"  # CSV is used here to obtain results
            };

            response = requests.post(bulk_api_url, headers=headers, data=json.dumps(job_data));
            job_info = response.json();

            # Print the complete API response
            print(f"API Response: {job_info}")

            job_id = job_info['id'];
            print(f"Job ID: {job_id}")

            # Check the job status
            job_status_url = f"{bulk_api_url}/{job_id}";

            while True:
                job_status_response = requests.get(job_status_url, headers=headers);
                job_status = job_status_response.json();
                print(f"Job Status: {job_status['state']}")

                if job_status['state'] == 'JobComplete':
                    break;
                elif job_status['state'] == 'Failed':
                    raise Exception(f"Job failed: {job_status['errorMessage']}");

            # Retrieve job results
            results_url = f"{job_status_url}/results";
            results_response = requests.get(results_url, headers=headers);

            # Convert CSV to JSON and then to DataFrame
            csv_data = results_response.text;
            csv_reader = csv.DictReader(csv_data.splitlines());
            json_data = pd.DataFrame([row for row in csv_reader]);

            # Check if there are any "Compound" fields, if so, activate the condition below;
            is_there_any_compound = True if self._columns['_columnsNameWorkingToQuery_to_sf_rest_api'] != '' else False
            
            if (is_there_any_compound):
                
                # Request all records with compound columns plus Id to merge with bulk API results.
                query_compound = 'Id,' + ' ' + self._columns['_columnsNameWorkingToQuery_to_sf_rest_api'];
                
                print('query_compound', query_compound)
                
                # Request all data and store it in a dataframe;
                data_compound = pd.DataFrame(self._sf.query_all(f"SELECT {query_compound} FROM {self._apiName}")['records']);
            
            data = pd.merge(json_data, data_compound, on='Id', how='inner') if is_there_any_compound else json_data;
            
            # Get a dictionary mapping current column names to new ones;
            mapeo_nombres = dict(zip(self._columns['_columnsApiName_Field_DataType']['_full']['QualifiedApiName'], self._columns['_columnsApiName_Field_DataType']['_full']['Label']));
            
            print('mapeo_nombres', mapeo_nombres)
            
            if self._columnsNameType == 'Label':
                # Rename columns with Field Names to make searching easier;
                data = data.rename(columns=mapeo_nombres);
            
            # Check if the 'attributes' column exists and remove it if so
            if 'attributes' in data.columns:
                data.drop(columns=['attributes'], inplace=True);
            
            # Store the prepared data in a class variable;
            self._data = data;
            
            display('Resulting object size: ', self._data.shape)
            
            # Store a sample of 10 records for quick access from the complete object dataset;
            self._dataSample = self._data.head(10);
            
            # End point
            end_time = time.perf_counter();

            # Calculate elapsed time
            elapsed_time = end_time - start_time;
                
            print('\033[92m',
                    f'\ngetFullDataObject method ended successfully.',
                    f'\nMethod time: {elapsed_time:.6f} seconds.', 
                    '\033[0m')
            
        except Exception as e:
            print(f'An error occurred in the getFullDataObject function, the error is: {e}\n')
            traceback.print_exc()  # Displays detailed error information
                
                
    def initial_prep_data_for_graphs(self):
            
        '''
        Filters the object's dataframe to retain a specific number of columns. Columns are added to this dataframe until the necessary values are created to graph the overall completeness level of the database, categorized by the number of null values in each column.
        '''
        
        # Start point
        start_time = time.perf_counter();

        print('\033[96m',f'\ninitial_prep_data_for_graphs method initialized.', '\033[0m')

        # -----------------------------------------------------------
        
        object_data = self._data;
        
        # -----------------------------------------------------------
        
        ### Create an independent copy of the party dataframe that stores the metadata; This independent copy serves as a checkpoint to revert to, avoiding repeated requests to Party.
        object_deepcopy = self.metadata.copy(deep=True);
        
        ### Remove the 'attributes' column;
        object_deepcopy.drop(columns='attributes', inplace=True);
        
        ### Create an independent copy of object_deepcopy;
        object_metadata = object_deepcopy.copy(deep=True);
        
        # -----------------------------------------------------------
        
        ### Define party metadata with the relevant columns;
        object_metadata_filtered =  object_metadata[['EntityDefinitionId', 'QualifiedApiName', 'DeveloperName', 'Label', 'DataType']];
        
        # -----------------------------------------------------------
        
        ### Create a dataframe storing data types of each column for later merging;
        object_data_dtypes = pd.DataFrame(object_data.dtypes, columns=['Columns_dtypes']);
        
        # -----------------------------------------------------------
        
        ### Merge the metadata dataframe with the dtypes dataframe;
        object_metadata_filtered_merge_dtypes = pd.merge(object_metadata_filtered, object_data_dtypes, left_on='Label', right_on=object_data_dtypes.index);
        
        # -----------------------------------------------------------
        
        object_data_NAN = pd.DataFrame(object_data.isna().sum(), columns=['NaN']);
        
        # -----------------------------------------------------------
        
        ### Merge the NaN dataframe with the metadata dataframe; This results in a column showing the number of NaNs for each field;
        object_metadata_filtered_merge_dtypes_merge_NAN_df = pd.merge(object_metadata_filtered_merge_dtypes, object_data_NAN, left_on='Label', right_on=object_data_NAN.index);

        # -----------------------------------------------------------
        
        ### Create the column indicating the percentage of NaNs in each field;
        object_metadata_filtered_merge_dtypes_merge_NAN_df['NaN_%'] = np.round(object_metadata_filtered_merge_dtypes_merge_NAN_df['NaN'] / object_data.shape[0] *100, 2);
        
        # -----------------------------------------------------------
        
        ### Create the Not_NaN column;
        object_metadata_filtered_merge_dtypes_merge_NAN_df['Not_NaN'] = object_data.shape[0]- object_metadata_filtered_merge_dtypes_merge_NAN_df['NaN'];
        
        # -----------------------------------------------------------
        
        ### Create the Not_NaN_% column;
        object_metadata_filtered_merge_dtypes_merge_NAN_df['Not_NaN_%'] = np.round(object_metadata_filtered_merge_dtypes_merge_NAN_df['Not_NaN'] / object_data.shape[0] *100, 2);
        
        # -----------------------------------------------------------
        
        ### Sort the dataframe by fields with the highest proportion of NaNs;
        object_metadata_filtered_merge_dtypes_merge_NAN_df.sort_values(by='Not_NaN', ascending=False, inplace=True);
        
        # -----------------------------------------------------------
        
        ### Verify if there are duplicates in the 'QualifiedApiName' column;
        is_there_duplicated = object_metadata_filtered_merge_dtypes_merge_NAN_df.duplicated(subset=['QualifiedApiName'], keep=False)
        
        object_metadata_filtered_merge_dtypes_merge_NAN_df_without_duplicates = pd.DataFrame();
        
        if is_there_duplicated.any():
            ### Remove duplicates based on 'QualifiedApiName'
            object_metadata_filtered_merge_dtypes_merge_NAN_df_without_duplicates = object_metadata_filtered_merge_dtypes_merge_NAN_df.drop_duplicates(subset=['QualifiedApiName'], keep='first')
        
        # -----------------------------------------------------------
        
        ### This is done to verify that the data is as expected;
        final_data = object_metadata_filtered_merge_dtypes_merge_NAN_df.loc[object_metadata_filtered_merge_dtypes_merge_NAN_df['QualifiedApiName'].str.contains('Date_of_Birth_Text__c')|object_metadata_filtered_merge_dtypes_merge_NAN_df['QualifiedApiName'].str.contains('litify_pm__Date_of_birth__c')]
        
        self._data_to_graph = object_metadata_filtered_merge_dtypes_merge_NAN_df_without_duplicates
        
        # End point
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time;
            
        print('\033[92m',
                f'\ninitial_prep_data_for_graphs method ended successfully.',
                f'\nMethod time: {elapsed_time:.6f} seconds.', 
                '\033[0m')
        
        return self._data_to_graph;


    def bar_graph_sf_object_NaN(self):
        '''
        Generates a horizontal bar graph using the resulting dataframe from the previous method (initial_prep_data_for_graphs).
        The graph displays the apiName of each column, showing how empty each one is.
        '''
        
        # Start point
        start_time = time.perf_counter();
        
        print('\033[96m',f'\nbar_graph_sf_object_NaN method initialized.', '\033[0m')
        
        data = self._data_to_graph;
        
        # Calculate the graph height based on the number of columns, with a base height of 0.5 per column
        num_columns = len(data)
        fig_height = num_columns * 0.5  # Adjust this factor as needed for desired spacing
        
        # Create a horizontal bar graph using Seaborn
        fig, ax = plt.subplots(figsize=(30, fig_height))  # Adjust size based on the number of columns

        # Assign colors based on the percentage of NaN values
        unique_names = data['QualifiedApiName'].unique()
        colors = {name: ('yellow' if 100 > data.loc[data['QualifiedApiName'] == name, 'Not_NaN_%'].iloc[0] >= 50 else
                        ('green' if data.loc[data['QualifiedApiName'] == name, 'Not_NaN_%'].iloc[0] == 100 else 'orange'))
                for name in unique_names}
        
        barplot = sns.barplot(
            y='QualifiedApiName', 
            x='Not_NaN', 
            data=data, 
            orient='h', 
            hue='QualifiedApiName',  # Assign hue to properly use palette
            palette=colors,
            dodge=False,  # Ensures colors do not overlap
            legend=False  # Disables the legend
        )

        # Add title with larger font size and spacing (pad)
        plt.title(f'Completeness percentage of each field in object {self._name}', fontsize=30, pad=20)
        plt.ylabel('Columns')
        plt.xlabel('Number of non-empty values')

        # Add additional space above the title
        plt.figtext(0.5, 0.888, ' ', ha='center', fontsize=20)

        # Get current axis labels
        labels = ax.get_yticklabels()

        # Set red color for labels matching values in 'lista_roja'
        for label in labels:
            if label.get_text() in (list(data.loc[data['Not_NaN_%']==0, 'QualifiedApiName'])):
                label.set_color('red')

        # Create graph legends distinguished by colors
        green_patch = mpatches.Patch(color='green', label=f'{data.loc[(data["Not_NaN_%"] == 100)].shape[0]} Fields 100% Complete')
        yellow_patch = mpatches.Patch(color='yellow', label=f'{data.loc[(data["Not_NaN_%"] < 100) & (data["Not_NaN_%"] >= 50)].shape[0]} Fields less than 100% and greater or equal to 50% empty')
        orange_patch = mpatches.Patch(color='orange', label=f'{data.loc[(data["Not_NaN_%"] < 50) & (data["Not_NaN_%"] > 0)].shape[0]} Fields less than 50% and more than 0% empty')
        red_patch = mpatches.Patch(color='red', label=f'{data.loc[(data["Not_NaN_%"] == 0)].shape[0]} Fields 100% Empty')

        # Add the legend with colors 
        plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch], loc='lower right')

        # Iterate over dataframe rows and their index using enumerate
        for idx, (index, row) in enumerate(data.iterrows()):
            if idx < len(barplot.patches):
                bar = barplot.patches[idx]  # Use the enumeration index to access the corresponding bar
                bar_width = bar.get_width()
                label_y_pos = bar.get_y() + bar.get_height() / 2
                label_x_pos = bar_width + (0.00750 * plt.xlim()[1])  # Adds 5% of x-limit as offset
                plt.text(label_x_pos, label_y_pos, f"{row['Not_NaN_%']:.2f}%", ha='left', va='center')

        # Adjust x-axis limits to include labels
        x_lim = plt.xlim()
        plt.xlim(x_lim[0], x_lim[1] * 1.1)  # Increase right limit by 10% to create space for labels

        # End point
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time;
            
        print('\033[92m',
                f'\nbar_graph_sf_object_NaN method ended successfully.',
                f'\nMethod time: {elapsed_time:.6f} seconds.', 
                '\033[0m')

        return plt.show()

