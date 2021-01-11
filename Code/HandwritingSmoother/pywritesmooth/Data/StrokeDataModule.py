import sys, os, logging as log
import pytorch_lightning as pl
import pywritesmooth.Data.StrokeDataset as sds

class StrokeDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
        and processing work in one place.   

    Code courtesy of:
    https://www.kaggle.com/tartakovsky/pytorch-lightning-lstm-timeseries-clean-code
    '''
    
def __init__(self, seq_len = 1, batch_size = 128, num_workers=0, osFiles = None):
    super().__init__()
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.X_train = None
    self.y_train = None
    self.X_val = None
    self.y_val = None
    self.X_test = None
    self.X_test = None
    self.columns = None
    self.preprocessing = None
    self.osFiles = osFiles

def prepare_data(self):
    pass

def setup(self, stage=None):
    '''

    '''
    # If data is already loaded, don't load again (because all GPUs execute this method)
    if stage == 'fit' and self.X_train is not None:
        return 
    if stage == 'test' and self.X_test is not None:
        return
    if stage is None and self.X_train is not None and self.X_test is not None:  
        return
 
    writingSample = sds.StrokeDataset(self.osFiles)  # Load the data to our internal data structure
    
    path = '/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt'
        
    df = pd.read_csv(
        path, 
        sep=';', 
        parse_dates={'dt' : ['Date', 'Time']}, 
        infer_datetime_format=True, 
        low_memory=False, 
        na_values=['nan','?'], 
        index_col='dt'
    )

    df_resample = df.resample('h').mean()

    X = df_resample.dropna().copy()
    y = X['Global_active_power'].shift(-1).ffill()
    self.columns = X.columns


    X_cv, X_test, y_cv, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_cv, y_cv, test_size=0.25, shuffle=False
    )

    preprocessing = StandardScaler()
    preprocessing.fit(X_train)

    if stage == 'fit' or stage is None:
        self.X_train = preprocessing.transform(X_train)
        self.y_train = y_train.values.reshape((-1, 1))
        self.X_val = preprocessing.transform(X_val)
        self.y_val = y_val.values.reshape((-1, 1))

    if stage == 'test' or stage is None:
        self.X_test = preprocessing.transform(X_test)
        self.y_test = y_test.values.reshape((-1, 1))
        

def train_dataloader(self):
    train_dataset = TimeseriesDataset(self.X_train, 
                                        self.y_train, 
                                        seq_len=self.seq_len)
    train_loader = DataLoader(train_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)
        
    return train_loader

def val_dataloader(self):
    val_dataset = TimeseriesDataset(self.X_val, 
                                    self.y_val, 
                                    seq_len=self.seq_len)
    val_loader = DataLoader(val_dataset, 
                            batch_size = self.batch_size, 
                            shuffle = False, 
                            num_workers = self.num_workers)

    return val_loader

def test_dataloader(self):
    test_dataset = TimeseriesDataset(self.X_test, 
                                        self.y_test, 
                                        seq_len=self.seq_len)
    test_loader = DataLoader(test_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

    return test_loader