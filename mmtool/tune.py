import os
import pyarrow as pa
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import math
from .module import CLIPModule
from transformers import CLIPProcessor
from ray import tune
from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning.callbacks import LearningRateMonitor

from .datautils import MultiModalDataModule, Dataset

def clip_fine_tune(config, num_epochs, num_gpus, model_init_path):
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]
    
    tmpdir = os.environ['TMPDIR'] + '/base'
    
    bird_tab = pa.parquet.read_table(f'{tmpdir}/data/bird_guide_single_parquet', 
                                     columns=['description', 'image_bytes'])
    bird_dataset = Dataset(bird_tab).filter(lambda tup : tup['description'] 
                                            is not None and tup['image_bytes'] is not None)
    bird_dataset = bird_dataset.rename_column('description', 'text')

    dataset = bird_dataset
    
    processor = CLIPProcessor.from_pretrained(model_init_path)
    
    data_mod = MultiModalDataModule(dataset=dataset, processor=processor,
                           test_size=config['test_size'], batch_size=config['batch_size'],
                                    val_batch_size=config['val_batch_size'],
                                    num_workers=config['num_workers'])
    
    model = CLIPModule(model_init_path, **config)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            TuneReportCallback(
                {
                    "loss": "val_loss",
#                  "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    
    trainer.validate(model, data_mod)
    trainer.fit(model, data_mod)