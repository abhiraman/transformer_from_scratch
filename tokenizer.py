
# !pip install tokenizers

import tokenizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from pathlib import Path
import os

def get_or_build_lang(ds,lang,config):
  def iterator(ds,lang):
    for ekey in ds.keys():
      for eSeq in ds[ekey]['translation'][lang]:
        yield eSeq
  # tokenizers.normalizers()
  parentPath = os.listdir(Path(config["tokenizePath"]))
  if f"translation_tokenizer_{lang}.json" not in parentPath:
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.normalizer = Lowercase()  # Fixed by instantiating the class
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]","[SOS]","[EOS]","[UNK]"],min_frequency=2)
    tokenizer.train_from_iterator(iterator(ds,lang),trainer=trainer,length=len(ds))
    tokenizer.save(os.path.join(config["tokenizePath"], f"translation_tokenizer_{lang}.json"))
    return tokenizer
  else:
    file_path = os.path.join(config["tokenizePath"], f"translation_tokenizer_{lang}.json")
    return Tokenizer.from_file(file_path)


