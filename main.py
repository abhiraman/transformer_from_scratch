from config import getConfig
from tokenizer import get_or_build_lang
from datasets import load_dataset
from dataloader import translationDataset
from torch.utils.data import Dataset,DataLoader
from train import trainModel
import torch
import gc



config = getConfig()
ds = load_dataset("cfilt/iitb-english-hindi")
print(ds['train']['translation']["en"])

config["tokenizePath"] = "/content/"
src_tokenizer = get_or_build_lang(ds,'en',config)
print("Done tokenizing english!")
tgt_tokenizer = get_or_build_lang(ds,'hi',config)

trainDataset = translationDataset(ds, 'train', src_tokenizer, config["src_lang"],
                                  tgt_tokenizer=tgt_tokenizer, tgt_lang=config["tgt_lang"],
                                  )

validationDataset = translationDataset(ds, 'validation', src_tokenizer, config["src_lang"],
                                  tgt_tokenizer=tgt_tokenizer, tgt_lang=config["tgt_lang"],
                                  )

gc.collect()
torch.cuda.empty_cache()
trainModel(1, config["lr"],src_tokenizer,tgt_tokenizer,config,trainDataset,validationDataset,verbose=True)
