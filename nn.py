# Nueral Network
import torch
import csv
import numpy as np 
from pathlib import Path

import NetworkClassifier
import PCAPDataSet

# Set these if you wish to start with a trained model and if you want to train further 
TRAIN=True
MODEL_FILE="/Users/obiwan/csec620/project1/src/py/model.pt"
ITERS = 25

def run():
    try:
        data = []
        labels = []
        ## Filesys location of Dataset in CSV format
        CSV_DATA = ""
        # Array of Possible Outputs in str format
        outputs = []

        reader = csv.DictReader(open(CSV_DATA))
        for row in reader:
            data.append(row)
            labels.append(row['Label'])


        ## UNCOMMENT IF CSV FILES NEED TO BE CREATED
        ## Split data into smaller csv for training and testing:
        csvfile = open(CSV_DATA, 'r').readlines()
        open('data_for_testing.csv', 'w+').writelines(csvfile[0])
        open('data_for_training.csv', 'w+').writelines(csvfile[0])
        for i in range(len(csvfile)):
            if i % 10 == 0 and i != 0:
                open('data_for_testing.csv', 'a').writelines(csvfile[i])
            else:
            	open('data_for_training.csv', 'a').writelines(csvfile[i])


        training_csv = open('data_for_training.csv', 'r').readlines()
        i = 1
        for int_ in range(ITERS):
            open(f'data_for_training_{int_}.csv', 'a').writelines(training_csv[0])
        while i < len(training_csv) - ITERS - 2:
            for int_ in range(ITERS):
                open(f'data_for_training_{int_}.csv', 'a').writelines(training_csv[i+int_])
            i += ITERS

        training_data_set = []
        for j in range(ITERS):
            training_data_set.append(PCAPDataSet(f'data_for_training_{j}.csv'))
        
        
        testing_data_set = PCAPDataSet('data_for_testing.csv')

        my_nn = NetworkClassifier(data, 4, outputs, 128)

        
        testing_dataset_loader = torch.utils.data.DataLoader(testing_data_set,
                                                     batch_size=64, shuffle=True,
                                                     num_workers=4)


        
        learning_rate = 1e-3
        batch_size = 64
        epochs = 5

        # Initialize the loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        # check if model is pre-trained
        trained = False
        if Path(MODEL_FILE).is_file():
            my_nn.model.load_state_dict(torch.load(MODEL_FILE), strict=False)
        if not TRAIN:
            trained = True

        optimizer = torch.optim.SGD(my_nn.model.parameters(), lr=learning_rate)

        for training_data in training_data_set:
            training_dataset_loader = torch.utils.data.DataLoader(training_data,
                                                     batch_size=64, shuffle=True,
                                                     num_workers=4)
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                if not trained:
                    my_nn.train_loop(training_dataset_loader, loss_fn, optimizer)
                my_nn.test_loop(testing_dataset_loader, loss_fn)
            print("Done!")
            torch.save(my_nn.model.state_dict(), MODEL_FILE)
    finally:
        torch.save(my_nn.model.state_dict(), MODEL_FILE)

if __name__ == '__main__':
    run()

