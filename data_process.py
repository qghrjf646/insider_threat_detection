import pandas as pd
import numpy as np
from datetime import datetime
import pickle

def load_data():
    device = pd.read_csv('data/r1/device.csv')
    http = pd.read_csv('data/r1/http.csv', header=None, names=['id', 'date', 'user', 'pc', 'url'])
    logon = pd.read_csv('data/r1/logon.csv')
    
    # Add type
    device['type'] = device['activity'].map({'Connect': 0, 'Disconnect': 1})
    http['type'] = 2
    logon['type'] = logon['activity'].map({'Logon': 3, 'Logoff': 4})
    
    # Combine
    data = pd.concat([device[['date', 'user', 'pc', 'type']], 
                      http[['date', 'user', 'pc', 'type']], 
                      logon[['date', 'user', 'pc', 'type']]], ignore_index=True)
    
    # Parse date
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M:%S')
    
    return data

def load_labels():
    # Insiders based on scenarios 1, 2, 3, 4
    malicious_users = {
        "DTAA/ABB0272",
        "DTAA/ABS0726",
        "DTAA/ACH0803",
        "DTAA/AFG0122",
        "DTAA/AFH0331",
        "DTAA/AFN0918",
        "DTAA/AFR0646",
        "DTAA/AGW0182",
        "DTAA/AIB0797",
        "DTAA/AJA0220",
        "DTAA/AJB0016",
        "DTAA/AJB0370",
        "DTAA/AJC0266",
        "DTAA/AJN0804",
        "DTAA/AJP0553",
        "DTAA/AKS0283",
        "DTAA/ALK0520",
        "DTAA/AMA0191",
        "DTAA/AQG0678",
        "DTAA/ARC0677",
        "DTAA/ARK0026",
        "DTAA/ARM0416",
        "DTAA/ARR0977",
        "DTAA/ARS0993",
        "DTAA/ASO0023",
        "DTAA/ASV0049",
        "DTAA/ATJ0284",
        "DTAA/AZC0030",
        "DTAA/BAF0379",
        "DTAA/BBM0499",
        "DTAA/BCJ0800",
        "DTAA/BGF0236",
        "DTAA/BIR0534",
        "DTAA/BJC0569",
        "DTAA/BJM0992",
        "DTAA/BKM0116",
        "DTAA/BLB0079",
        "DTAA/BMR0412",
        "DTAA/BNW0850",
        "DTAA/BPC0533",
        "DTAA/BTW0973",
        "DTAA/BYB0347",
        "DTAA/CAC0854",
        "DTAA/CAW0976",
        "DTAA/CCB0496",
        "DTAA/CCG0876",
        "DTAA/CDZ0056",
        "DTAA/CGM0343",
        "DTAA/CGM0994",
        "DTAA/CHA0174",
        "DTAA/CHM0645",
        "DTAA/CHS0324",
        "DTAA/CJG0638",
        "DTAA/CJW0317",
        "DTAA/CLB0995",
        "DTAA/CLK0383",
        "DTAA/CLN0999",
        "DTAA/CMF0339",
        "DTAA/COS0121",
        "DTAA/CPP0761",
        "DTAA/CRC0996",
        "DTAA/CSC0297",
        "DTAA/CTH0856",
        "DTAA/CVW0133",
        "DTAA/CWG0305",
        "DTAA/CYK0330",
        "DTAA/DAO0613",
        "DTAA/DAV0579",
        "DTAA/DBB0235",
        "DTAA/DBF0903",
        "DTAA/DCC0600",
        "DTAA/DDK0417",
        "DTAA/DFW0798",
        "DTAA/DFZ0063",
        "DTAA/DGM0240",
        "DTAA/DIS0382",
        "DTAA/DLM0723",
        "DTAA/DMR0377",
        "DTAA/DNS0937",
        "DTAA/DRN0845",
        "DTAA/DRP0428",
        "DTAA/DSH0627",
        "DTAA/DSM0990",
        "DTAA/EBP0037",
        "DTAA/ECM0214",
        "DTAA/EDH0146",
        "DTAA/EGM0401",
        "DTAA/ELA0741",
        "DTAA/ELD1000",
        "DTAA/EMR0242",
        "DTAA/EMZ0196",
        "DTAA/EOS0972",
        "DTAA/ERP0254",
        "DTAA/ESB0080",
        "DTAA/EWR0504",
        "DTAA/FDP0500",
        "DTAA/FFP0721",
        "DTAA/FHM0931",
        "DTAA/FKR0550",
        "DTAA/GHC0041",
        "DTAA/GKJ0573",
        "DTAA/GLB0715",
        "DTAA/GNH0394",
        "DTAA/GRG0693",
        "DTAA/GRS0372",
        "DTAA/GSL0697",
        "DTAA/HAG0649",
        "DTAA/HBB0402",
        "DTAA/HBL0517",
        "DTAA/HCC0258",
        "DTAA/HCG0731",
        "DTAA/HCR0795",
        "DTAA/HDR0261",
        "DTAA/HDR0823",
        "DTAA/HED0802",
        "DTAA/HIB0604",
        "DTAA/HKM0260",
        "DTAA/HNS0858",
        "DTAA/HSD0519",
        "DTAA/HSR0472",
        "DTAA/HTC0607",
        "DTAA/HTS0265",
        "DTAA/IKH0904",
        "DTAA/IRC0991",
        "DTAA/IWM0482",
        "DTAA/JAD0531",
        "DTAA/JAH0544",
        "DTAA/JAM0392",
        "DTAA/JBL0450",
        "DTAA/JBS0483",
        "DTAA/JCC0140",
        "DTAA/JCC0998",
        "DTAA/JCS0873",
        "DTAA/JCT0410",
        "DTAA/JGB0062",
        "DTAA/JGC0775",
        "DTAA/JHA0404",
        "DTAA/JHD0512",
        "DTAA/JHM0408",
        "DTAA/JJR0278",
        "DTAA/JLC0744",
        "DTAA/JLL0301",
        "DTAA/JMM0759",
        "DTAA/JMS0793",
        "DTAA/JNF0543",
        "DTAA/JSB0493",
        "DTAA/JSE0020",
        "DTAA/JTT0989",
        "DTAA/JZO0727",
        "DTAA/KAD0059",
        "DTAA/KAG0880",
        "DTAA/KBB0983",
        "DTAA/KDL0806",
        "DTAA/KEE0997",
        "DTAA/KGS0398",
        "DTAA/KIN0073",
        "DTAA/KJG0378",
        "DTAA/KKT0039",
        "DTAA/KKW0676",
        "DTAA/KKW0945",
        "DTAA/KLS0717",
        "DTAA/KRC0021",
        "DTAA/KRC0455",
        "DTAA/KSB0673",
        "DTAA/KSW0597",
        "DTAA/LBB0095",
        "DTAA/LCG0546",
        "DTAA/LCK0666",
        "DTAA/LDB0961",
        "DTAA/LID0942",
        "DTAA/LJR0772",
        "DTAA/LPG0099",
        "DTAA/LRH0747",
        "DTAA/LTM0418",
        "DTAA/MAB0532",
        "DTAA/MCF0427",
        "DTAA/MCH0530",
        "DTAA/MCM0346",
        "DTAA/MDT0356",
        "DTAA/MIC0914",
        "DTAA/MIK0034",
        "DTAA/MJF0892",
        "DTAA/MJM0469",
        "DTAA/MMF0289",
        "DTAA/MQL0135",
        "DTAA/MVW0630",
        "DTAA/MXM0271",
        "DTAA/MXM0838",
        "DTAA/MZW0111",
        "DTAA/NAC0161",
        "DTAA/NDM0893",
        "DTAA/NDS0490",
        "DTAA/NFD0706",
        "DTAA/NGC0624",
        "DTAA/NHL0253",
        "DTAA/NHL0521",
        "DTAA/NNB0443",
        "DTAA/NSC0842",
        "DTAA/NSP0910",
        "DTAA/ODW0839",
        "DTAA/OJH0425",
        "DTAA/OYR0801",
        "DTAA/PCM0846",
        "DTAA/PIP0746",
        "DTAA/PJS0809",
        "DTAA/PSP0555",
        "DTAA/PUH0847",
        "DTAA/PXB0069",
        "DTAA/QGF0292",
        "DTAA/QKR0651",
        "DTAA/QLW0941",
        "DTAA/QNC0832",
        "DTAA/RAH0968",
        "DTAA/RBC0583",
        "DTAA/RBC0879",
        "DTAA/RCW0436",
        "DTAA/RES0962",
        "DTAA/RLC0442",
        "DTAA/RME0699",
        "DTAA/RQH0770",
        "DTAA/RVS0852",
        "DTAA/RXE0207",
        "DTAA/SAH0192",
        "DTAA/SAM0954",
        "DTAA/SCW0707",
        "DTAA/SDC0032",
        "DTAA/SEF0446",
        "DTAA/SHR0025",
        "DTAA/SJA0691",
        "DTAA/SJC0487",
        "DTAA/SJH0588",
        "DTAA/SKH0467",
        "DTAA/SKW0871",
        "DTAA/SLC0778",
        "DTAA/SMC0139",
        "DTAA/SOM0478",
        "DTAA/SPK0818",
        "DTAA/SRN0275",
        "DTAA/SSH0144",
        "DTAA/STF0145",
        "DTAA/TAO0639",
        "DTAA/TGC0670",
        "DTAA/TKH0211",
        "DTAA/TLB0017",
        "DTAA/TPC0102",
        "DTAA/TRB0414",
        "DTAA/TUA0298",
        "DTAA/VAH0262",
        "DTAA/VCG0663",
        "DTAA/VFP0669",
        "DTAA/VKB0605",
        "DTAA/VMM0309",
        "DTAA/VNM0127",
        "DTAA/WCN0570",
        "DTAA/WJG0153",
        "DTAA/WMJ0364",
        "DTAA/WOT0549",
        "DTAA/WTA0867",
        "DTAA/XKR0971",
        "DTAA/XQW0354",
        "DTAA/YCB0005",
        "DTAA/YCV0635",
        "DTAA/YJJ0129",
        "DTAA/YQW0689",
        "DTAA/YVF0045",
        "DTAA/ZAB0889",
        "DTAA/ZKH0388",
    }
    return malicious_users

def create_sequences(data, max_len=100):
    users = data['user'].unique()
    sequences = {}
    labels = {}
    
    malicious_users = load_labels()
    
    for user in users:
        user_data = data[data['user'] == user].sort_values('date')
        seq = []
        for _, row in user_data.iterrows():
            pc_num = int(row['pc'].split('-')[1]) / 1100.0  # normalize PC ID
            seq.append((row['type'], pc_num))
        if len(seq) > max_len:
            seq = seq[:max_len]
        elif len(seq) < max_len:
            seq += [(5, 0.0)] * (max_len - len(seq))  # pad with (5, 0)
        
        sequences[user] = seq
        labels[user] = 1 if user in malicious_users else 0
    
    return sequences, labels

def one_hot_sequences(sequences, labels, num_classes=6):
    X = []
    y = []
    for user, seq in sequences.items():
        seq_oh = np.zeros((len(seq), num_classes + 1))  # 6 for type + 1 for pc
        for i, (t, p) in enumerate(seq):
            seq_oh[i, t] = 1
            seq_oh[i, 6] = p
        X.append(seq_oh)
        y.append(labels[user])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    data = load_data()
    sequences, labels = create_sequences(data)
    
    X, y = one_hot_sequences(sequences, labels)
    
    # Save
    with open('data/processed.pkl', 'wb') as f:
        pickle.dump((X, y), f)
    
    print(f"Data processed: {X.shape}, labels: {y.shape}")