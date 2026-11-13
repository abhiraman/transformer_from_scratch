import Path from pathlib

def getConfig():
  return {
    "batchSize":8,
    "num_epochs":20,
    "lr": 1e-4,
    "max_src_seq_len":2178,
    "max_tgt_seq_len":2069,
    "src_lang": "en",
    "tgt_lang": "hi",
    "model_folder":"weights",
    "model_filename":"tmodel_",
    "preload":False,
    "experiment_name":"runs/tmodel",
    "tokenizePath":"transformer_from_scratch/"
  }

def get_weights_file_path(config,epoch):
  model_folder = config["model_folder"]
  model_filename = f"{config["model_filename"]}_{epoch}.pt"
  return str(Path('.')/model_folder/model_filename)

