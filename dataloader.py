from SpatialTransform import SpatialTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

#Create PyTorch dataset
#We do this to make data compatible with DataLoader
class SWEStationDataset(Dataset):
    #Convert NumPy array to tensor for PyTorch
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) #ensures column vectors

    def __len__(self):
        return len(self.X) #number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] #feature/target pair for batching

#--------------------

class SWEDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self):
        swe = pd.read_csv(self.cfg.swe_path) 
        meta = pd.read_csv(self.cfg.meta_path) #station metadata
        #^This is parameterized using cfg rather than simply loading a CSV 

        #Data cleanup/DataFrame creation
        swe_long = swe.melt(id_vars='Date', var_name='Station', value_name='SWE')
        swe_long["Station"] = swe_long["Station"].str.replace("_", " ").str.lower()
        meta["Station_clean"] = meta["Station Name"].str.lower()

        merged = pd.merge(swe_long, meta, left_on="Station", right_on="Station_clean", how="inner")
        merged = merged.rename(columns={"Station_clean": "Station"})
        merged = merged.dropna()

        daily = merged[merged["Date"] == self.cfg.sample_date].copy() #filtering now using cfg

        train_df, val_df = train_test_split(daily, test_size=0.2, random_state=42)

        def build_features(df):
            return pd.DataFrame({
                "Elevation": df["Elevation_x"],
                "Slope": df["Slope_tif1_x"],
                "Aspect": df["Aspect_tif_x"],
                "Latitude": df["Latitude_x"],
                "Longitude": df["Longitude_x"],
                "DayOfYear": pd.to_datetime(df["Date"]).dt.dayofyear
            })

        #training/validation table creation
        X_train = build_features(train_df)
        X_train["SWE"] = train_df["SWE"]
        X_val = build_features(val_df)
        X_val["SWE"] = val_df["SWE"]
        
        #SpatialTransformer (from Heaton, et al (2024))
        transformer = SpatialTransformer()
        spatial_obj = transformer.transform_to_ind(
            target="SWE",
            trainData=X_train,
            trainLocs=train_df[["Latitude_x", "Longitude_x"]].to_numpy(),
            testData=X_val,
            testLocs=val_df[["Latitude_x", "Longitude_x"]].to_numpy(),
            smoothness=0.5, range_param=1.0, nugget=0.01, M=30, ncores=1 #set ncores to 1 but could be increased if we start doing more in-depth predictions/incorporate temporal data
        )

        #create PyTorch datasets from training/validation sets from SpatialTrasnfroemr
        train_dataset = SWEStationDataset(spatial_obj["trainData"].drop(columns=["SWE"]).values,
                                        spatial_obj["trainData"]["SWE"].values)
        val_dataset = SWEStationDataset(spatial_obj["testData"].values,
                                        val_df["SWE"].values)

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size)

        return train_loader, val_loader, transformer, val_df, spatial_obj
