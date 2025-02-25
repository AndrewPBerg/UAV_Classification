import pytorch_lightning as pl
from torch.utils.data import DataLoader
from helper.util import train_test_split_custom, MelSpectrogramFeatureExtractor
from config.model_config import TrainingConfig, FeatureExtractorConfig

class UAVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        training_config: TrainingConfig,
        feature_extractor_config: FeatureExtractorConfig
    ):
        super().__init__()
        self.data_path = data_path
        self.training_config = training_config
        self.feature_extractor_config = feature_extractor_config
        self.feature_extractor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: str = None):
        """Sets up datasets for training, validation, and testing"""
        if self.feature_extractor is None:
            self.feature_extractor = MelSpectrogramFeatureExtractor(
                **vars(self.feature_extractor_config)
            )
            
        if stage in (None, "fit"):
            # Split dataset for training
            self.train_dataset, self.val_dataset, self.test_dataset = train_test_split_custom(
                data_path=self.data_path,
                feature_extractor=self.feature_extractor,
                test_size=0.2,
                val_size=0.1,
                seed=42
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers
        )
        
    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset"""
        return len(self.train_dataset.classes) if self.train_dataset else None
