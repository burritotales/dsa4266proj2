# Prediction of m6A RNA modifications using XGBoost

## About the Model

The project’s goal is the identification of m6A (N6-methyladenosine) RNA modifications within transcripts, achieved through the development of a robust machine learning model. By using the XGBoost machine learning model, we aim to effectively distinguish between modified and unmodified sites within RNA sequences by integrating direct RNA-Seq data. The success of the project hinges on efficiently utilising the data to extract features indicative of m6A modifications.

## Prediction on Sample Dataset

### 0. Set up the Ubuntu instance
- EBSVolumeSize: 100 (or more)
- InstanceType: t3.medium (or larger)

<details>
<summary><U> How to connect using SSH </U></summary>


Run the code below on terminal with the [path to .pem file] and [ip address] replaced. 

Windows:
```
ssh -i [path to .pem file] ubuntu@ec2-[ip-address].ap-southeast-1.compute.amazonaws.com
```

Mac:
```
ssh -i [path to .pem file] ubuntu@[ip.address]
```

</details>

### 1. On the Ubuntu instance run the following to install pip3
```
sudo apt install python3-pip
```

### 2. Run the following to download the necessary files and packages
```
pip3 install pandas scikit-learn xgboost imblearn
```

### 3. Create folders in the ubuntu instance
```
mkdir processed output  
```

### 4. Download the input files
```
wget -P model https://github.com/burritotales/dsa4266proj2/raw/main/xgboost.pkl
wget -P py https://raw.githubusercontent.com/burritotales/dsa4266proj2/main/aws_prediction.py
```
This is the git link to the sample dataset, it can be replaced with other datasets.
```
wget -P input https://github.com/burritotales/dsa4266proj2/raw/main/dataset2.json.gz
```

### 4. Run the py script 
```
python3 py/aws_prediction.py
```
\* The prediction will run all on `.json.gz` files in the 'input' directory, if the corresponding processed `.csv.gz` file is not in the 'processed' directory.

### 5.	View the output file 
This is to view the output of the sample dataset, change the path if you used other datasets.
```
nano output/dataset2_output.csv 
```
Press `CTRL + X` to exit view


