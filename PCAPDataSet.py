import numpy as np 
import torch
from torch.utils.data import Dataset

## EXAMPLE: this example data type can be used for labeled PCAP data. Copy and modify this class to fit your own data
class PCAPDataSet(Dataset):
    """PCAP dataset."""
    # 
    def __init__(self, csv_file, transform=None):
        # EXAMPLE: set your own conversion matrix for your dataset
        # ways to cast each read in field from CSV to a float 
        converter = {
            ' Label':self.conv, 
            ' Destination Port':self.conv_str, 
            ' Flow Duration':self.conv_str, 
            ' Total Fwd Packets':self.conv_str, 
            ' Total Backward Packets':self.conv_str, 
            'Total Length of Fwd Packets':self.conv_str,
            ' Total Length of Bwd Packets':self.conv_str,
            ' Fwd Packet Length Max':self.conv_str,
            ' Fwd Packet Length Min':self.conv_str,
            ' Fwd Packet Length Mean':self.conv_str,
            ' Fwd Packet Length Std':self.conv_str,
            'Bwd Packet Length Max':self.conv_str,
            ' Bwd Packet Length Min':self.conv_str,
            ' Bwd Packet Length Mean':self.conv_str,
            ' Bwd Packet Length Std':self.conv_str,
            'Flow Bytes/s':self.conv_str,
            ' Flow Packets/s':self.conv_str,
            ' Flow IAT Mean':self.conv_str,
            ' Flow IAT Std':self.conv_str,
            ' Flow IAT Max':self.conv_str,
            ' Flow IAT Min':self.conv_str,
            'Fwd IAT Total':self.conv_str,
            ' Fwd IAT Mean':self.conv_str,
            ' Fwd IAT Std':self.conv_str,
            ' Fwd IAT Max':self.conv_str,
            ' Fwd IAT Min':self.conv_str,
            'Bwd IAT Total':self.conv_str,
            ' Bwd IAT Mean':self.conv_str,
            ' Bwd IAT Std':self.conv_str,
            ' Bwd IAT Max':self.conv_str,
            ' Bwd IAT Min':self.conv_str,
            'Fwd PSH Flags':self.conv_str,
            ' Bwd PSH Flags':self.conv_str,
            ' Fwd URG Flags':self.conv_str,
            ' Bwd URG Flags':self.conv_str,
            ' Fwd Header Length':self.conv_str,
            ' Bwd Header Length':self.conv_str,
            'Fwd Packets/s':self.conv_str,
            ' Bwd Packets/s':self.conv_str,
            ' Min Packet Length':self.conv_str,
            ' Max Packet Length':self.conv_str,
            ' Packet Length Mean':self.conv_str,
            ' Packet Length Std':self.conv_str,
            ' Packet Length Variance':self.conv_str,
            'FIN Flag Count':self.conv_str,
            ' SYN Flag Count':self.conv_str,
            ' RST Flag Count':self.conv_str,
            ' PSH Flag Count':self.conv_str,
            ' ACK Flag Count':self.conv_str,
            ' URG Flag Count':self.conv_str,
            ' CWE Flag Count':self.conv_str,
            ' ECE Flag Count':self.conv_str,
            ' Down/Up Ratio':self.conv_str,
            ' Average Packet Size':self.conv_str,
            ' Avg Fwd Segment Size':self.conv_str,
            ' Avg Bwd Segment Size':self.conv_str,
            'Fwd Avg Bytes/Bulk':self.conv_str,
            ' Fwd Avg Packets/Bulk':self.conv_str,
            ' Fwd Avg Bulk Rate':self.conv_str,
            ' Bwd Avg Bytes/Bulk':self.conv_str,
            ' Bwd Avg Packets/Bulk':self.conv_str,
            'Bwd Avg Bulk Rate':self.conv_str,
            'Subflow Fwd Packets':self.conv_str,
            ' Subflow Fwd Bytes':self.conv_str,
            ' Subflow Bwd Packets':self.conv_str,
            ' Subflow Bwd Bytes':self.conv_str,
            'Init_Win_bytes_forward':self.conv_str,
            ' Init_Win_bytes_backward':self.conv_str,
            ' act_data_pkt_fwd':self.conv_str,
            ' min_seg_size_forward':self.conv_str,
            'Active Mean':self.conv_str,
            ' Active Std':self.conv_str,
            ' Active Max':self.conv_str,
            ' Active Min':self.conv_str,
            'Idle Mean':self.conv_str,
            ' Idle Std':self.conv_str,
            ' Idle Max':self.conv_str,
            ' Idle Min':self.conv_str
        }
        ## EXAMPLE: modify to match your desired labels 
        # map the `label` attribute of CSV file to a float representation
        label_translate = {
            'BENIGN':0.0,
            'DoS GoldenEye':1.0,
            'DoS Slowhttptest' : 2.0,
            'DoS slowloris' : 3.0,
            'DoS Hulk' : 4.0,
            'Heartbleed' : 5.0,
            'FTP-Patator' : 6.0,
            'SSH-Patator' : 7.0,
            'Infiltration' : 8.0,
            'Web Attack � XSS' : 9.0,
            'Web Attack � Sql Injection' : 10.0,
            'Web Attack � Brute Force' : 11.0,
            'DDoS' : 12.0,
            'PortScan' : 13.0,
            'Bot' : 14.0
        }
        # set of the pcap data stored in the CSV file
        self.pcap_data = []
        # read in CSV to `self.pcap_data`, converting all values to floats
        csvfile = open(csv_file).readlines()
        for i in range(len(csvfile)):
            if i == 0:
                continue
            row = csvfile[i].split(',')
            for j in range(len(row)-2):
                row[j] = self.conv_str(row[j])
            row[-1] = self.conv(row[-1])
            self.pcap_data.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.pcap_data)

    # Retrieves the `idx`th row/slice of rows. Returns tensor and float; `data` : float values of row(s), `label` : float representation of label
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rows = self.pcap_data[idx]
        label = rows[-1]
        data = rows[0:-2]
        data = torch.Tensor(data)
        data.requires_grad_(requires_grad=True)
        count = 0
        for item in data:
            # Only needed if no converstion to float method provided on read
            if isinstance(item,str): 
                data.iloc[count] = np.float32(item)
            count+=1
        if isinstance(label,str):
            label = self.conv(label)

        return data, label

    # Converts values to 32 bit float, failing back to 0.0 for any errors
    def conv_str(self,val,default_val=0.0):
        try:
            return np.float32(val)
        except Exception as e:
            return np.float32(default_val)

    ## EXAMPLE, create your own conversion function for your own labels
    # Converts potential labels to 32 bit float, failing back to 0.0 for any errors
    def conv(self, val, default_val=0.0):
        label_translate = {
            'BENIGN':0.0,
            'DoS GoldenEye':1.0,
            'DoS Slowhttptest' : 2.0,
            'DoS slowloris' : 3.0,
            'DoS Hulk' : 4.0,
            'Heartbleed' : 5.0,
            'FTP-Patator' : 6.0,
            'SSH-Patator' : 7.0,
            'Infiltration' : 8.0,
            'Web Attack � XSS' : 9.0,
            'Web Attack � Sql Injection' : 10.0,
            'Web Attack � Brute Force' : 11.0,
            'DDoS' : 12.0,
            'PortScan' : 13.0,
            'Bot' : 14.0
        }
        if val == " Label":
            return val
        try:
            return np.float32(label_translate[str(val).strip()])
        except KeyError:
            return np.float32(default_val)
