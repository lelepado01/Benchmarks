
class ModelType:
    SwinTransformerV2 = "SwinTransformerV2"
    HieraTiny = "HieraTiny"

    Linformer = "Linformer"
    Performer = "Performer"
    LongFormer = "LongFormer"
    SRFormer = "SRFormer"

    MaskedAutoencoder = "MaskedAutoencoder"

class ModelVariant: 
    PARAMS_100M = "100M"
    PARAMS_600M = "600M"
    PARAMS_1B = "1B"
    PARAMS_3B = "3B"

class RunConfigs: 

    def __init__(self):
        self.NUM_NODES = 1

        self.model_type = ModelType.MaskedAutoencoder
        self.model_variant = ModelVariant.PARAMS_100M

        self.STRATEGY = "ddp"
        self.EPOCHS = 1
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.0001

        self.train_samples = 1000
        self.val_samples = int(self.train_samples * 0.1) if self.train_samples is not None else None
        self.test_samples = int(self.train_samples * 0.2) if self.train_samples is not None else None

    def default():
        return RunConfigs()

    def get_experiment_name(self):
        return f"exp_m{self.model_type}_p{self.model_variant}_e{self.EPOCHS}_n{self.NUM_NODES}"