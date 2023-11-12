print('importing packages...')
import gzip
import pandas as pd
import json
import numpy as np
import joblib  # For saving the trained model
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import os
import pickle
print('done importing necessary packages')


main_path = './'
# col_list_path = 'input/col_list.txt'
dataset_directory = 'input/'

def preprocessing_data(dataset_path, files):
    # Check if it has been processed before
    processed_path = 'processed/' + files + '_processed.csv.gz'

    if os.path.isfile(processed_path):
        print('processed csv exists, skipping preprocessing step')
        with gzip.open(processed_path, 'rb') as file_in:
            result_data_df = pd.read_csv(file_in)
    # If not processed before, process and save df
    else:
        # Load data into a DataFrame
        print('preprocessed csv file does not exist, starting preprocessing...')
        with gzip.open(dataset_path, 'r') as file_in:
            data = file_in.read().decode("utf-8") #change byte to string

        # Split data line by line
        data_lines = data.split('\n')[:-1] #remove last empty string, (train has 121838 lines)
        i = 0
        n = len(data_lines)
        frames = []
        for line in data_lines:
            i += 1
            print('processing line ' + str(i)+'/'+str(n))
            # Parse the string as a JSON object
            data_dict = json.loads(line)

            # Convert the nested dictionary into a list of lists
            data_list = []
            for gene, values in data_dict.items():
                for pos, subvalues in values.items():
                    for base, sublist in subvalues.items():
                        data_list.append([gene, pos, base, sublist])

            # Create a DataFrame from the list of lists
            df = pd.DataFrame(data_list,columns=['transcript_id', 'transcript_position','nucleotides', 'reads']) #each row
            frames.append(df)
        # Putting lines into a dataframe
        data_df = pd.concat(frames,ignore_index=True)

        #(1) length of the direct RNA-Seq signal of the 5-mer nucleotides (dwelling time),
        #(2) standard deviation of the direct RNA-Seq signal, and
        #(3) mean of the direct RNA-Seq signal.

        print ('Combining the reads by aggregation using the different functions') 
        data_df['combined_reads'] = data_df['reads'].apply(agg_func)
        #print(data_df) 

        print('Splitting the aggregations into their own column')
        new_data_df = data_df.apply(split_list, axis=1).rename(columns=lambda x: f"combined_reads_p{((x//4)//3)-1}_t{((x//4)%3)+1}_v{(x%4)+1}")
        #print(new_data_df)

        print('Merge the new DataFrame with the original DataFrame')
        result_data_df = pd.concat([data_df, new_data_df], axis=1).drop('combined_reads', axis=1)
        #print(result_data_df)
        
        # Save to processed file path
        result_data_df.to_csv(processed_path, index=False, compression='gzip') # Doesn't save index col
        print('processed file is saved in ' + processed_path)

    return result_data_df

# Define the aggregation function
def agg_func(input_reads):
    #print(input_df)
    #input_reads = json.loads(input_reads)
    agg_reads = []
    for var in range(9): #each var each position
        val_list = []
        for read in input_reads:
            val_list.append(read[var])
        agg_reads.append(np.mean(val_list))
        agg_reads.append(min(val_list))
        agg_reads.append(max(val_list))
        agg_reads.append(np.std(val_list))
    return agg_reads

# Define a function to split the list into columns
def split_list(row):
    return pd.Series(row['combined_reads'])

# Handling nucleotides as category
def nucleotides_as_cat(df):
    cat_df = df.copy(deep=True)
    #split the nucleotides into the relevant positions (+1, 0 and -1)
    cat_df['nucleotides_position-1']= cat_df['nucleotides'].str.slice(stop=5)
    cat_df['nucleotides_position']= cat_df['nucleotides'].str.slice(start=1, stop=6)
    cat_df['nucleotides_position+1']= cat_df['nucleotides'].str.slice(start=2, stop=7)

    #as category for xg boost
    cat_df['nucleotides_position+1'] = cat_df['nucleotides_position+1'].astype("category")
    cat_df['nucleotides_position'] = cat_df['nucleotides_position'].astype("category")
    cat_df['nucleotides_position-1'] = cat_df['nucleotides_position-1'].astype("category")
    return cat_df

# Function to produce intermediate submission csv

def create_output(model_name, paths):
    # Preprocessing files
    print('preprocessing data file...')
    df = preprocessing_data(paths['download_file_path'], paths['dataset'])
    processed_data_df_model = nucleotides_as_cat(df)    
    
    # List of features selected
    col_list = ['nucleotides_position', 'combined_reads_p0_t2_v1', 'combined_reads_p0_t3_v1', 'combined_reads_p0_t3_v4', 'combined_reads_p1_t3_v4',
                'combined_reads_p1_t3_v1', 'combined_reads_p0_t2_v2', 'nucleotides_position+1']
    
    # # Keeping list of transcript ids to match to predictions
    dataset_pred, transcript_ids, transcript_position = create_pred_set(processed_data_df_model, col_list)

    loaded_model = joblib.load(paths[model_name])
    dataset_output_probs = pd.DataFrame(loaded_model.predict_proba(dataset_pred)[:,1])

    dataset_output_probs.columns = ['score']
    pdList = [transcript_ids, transcript_position, dataset_output_probs]  # List of your dataframes
    new_df = pd.concat(pdList, axis=1)
    new_df.to_csv( paths['main_path'] + paths['output_file_name'], index=False)
   

# Function to format prediction data and keep transcript id
def create_pred_set(df, col_list):
    drop_cols = ['nucleotides','reads','transcript_id']

    # Keeping list of transcript ids to match to predictions
    transcript_ids = df['transcript_id']
    
    df = df.drop(columns=drop_cols)
    df['transcript_position'] = df['transcript_position'].astype(int) #change to int because currently it's an object dtype
    transcript_position = df['transcript_position']
    
    if col_list:
        df = df[col_list]
    
    return df,transcript_ids, transcript_position

print('done defining functions')

for root, dirs, files in os.walk(dataset_directory):
        samples = []
        for filename in files:
            if filename.endswith('.json.gz'):
                dataset = filename.split('.json.gz')[0]
                #download_file_path = main_path + 'input/' + files + '.json.gz'
                #output_file_name = main_path + 'output/' + files + '_output.csv'
                download_file_path =  'input/' + dataset + '.json.gz'
                output_file_name = 'output/' + dataset + '_output.csv'
                
                # Define your own model paths to load the models
                paths = {'main_path': main_path, 
                    'xgboost_feature_selection': "model/" + "xgboost.pkl",
                    'download_file_path': download_file_path,
                    'output_file_name': output_file_name,
                    'dataset': dataset
                }
                
                create_output('xgboost_feature_selection', paths)
                print('prediction complete! output is saved as ' + output_file_name)
        
    
    
