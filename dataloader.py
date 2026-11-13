from torch.utils.data import Dataset,DataLoader
import json
import torch

def load_tokenizer(filePath):
  fh = open(filePath,'r')
  return json.load(fh)

# en = load_tokenizer('/content/translation_tokenizer_en.json')

def causalMask(size):
  mask = torch.triu(torch.ones(size,size),diagonal=1)
  print(mask.size())
  return mask == 0

class translationDataset(Dataset):
#   def __init__(self, ds, key: str, src_tokenizer, src_lang, max_src_seq_len,
#                tgt_tokenizer, tgt_lang, max_tgt_seq_len):

  def __init__(self, ds, key: str, src_tokenizer, src_lang,
               tgt_tokenizer, tgt_lang):
    super().__init__()
    self.ds = ds[key]['translation'][:20]
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    # self.max_src_seq_len = max_src_seq_len
    # self.max_tgt_seq_len = max_tgt_seq_len


    self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)


  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):

    src_sentence = self.ds[idx][self.src_lang]
    tgt_sentence = self.ds[idx][self.tgt_lang]

    encode_src_ids = self.src_tokenizer.encode(sequence=src_sentence).ids
    encode_tgt_ids = self.tgt_tokenizer.encode(sequence=tgt_sentence).ids

    # src_pad_tokens = self.max_src_seq_len - len(encode_src_ids) - 2
    # tgt_pad_tokens = self.max_tgt_seq_len - len(encode_tgt_ids) - 1 # Only Start of token


    # if src_pad_tokens < 0 or tgt_pad_tokens < 0:
    #   raise ValueError("Sentence more than max sequence length")

    # Assemble basic unpadded sequences with special tokens
    encoder_input = torch.cat([
        self.sos_token,
        torch.tensor(encode_src_ids, dtype=torch.int64),
        self.eos_token,
        # torch.tensor([self.pad_token]*src_pad_tokens,dtype=torch.int64)
    ])

    decoder_input = torch.cat([
        self.sos_token,
        torch.tensor(encode_tgt_ids, dtype=torch.int64),
        # torch.tensor([self.pad_token]*tgt_pad_tokens,dtype=torch.int64)

    ])

    label = torch.cat([
        torch.tensor(encode_tgt_ids, dtype=torch.int64),
        self.eos_token,
        # torch.tensor([self.pad_token]*tgt_pad_tokens,dtype=torch.int64)
    ])




    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "label": label,
        "src_text": src_sentence,
        "tgt_text": tgt_sentence
    }

def addDynamicPaddings(batch:list):
  ''' create dynamic paddings for each batch'''
  max_src_seq_len = max([len(eDict["encoder_input"]) for eDict in batch])
  max_tgt_seq_len = max([len(eDict["decoder_input"]) for eDict in batch])
  pad_token = src_tokenizer.token_to_id("[PAD]")

  encoder_input,decoder_input,label = [],[],[]
  for eSample in batch:
    src_pad_tokens = max_src_seq_len - len(eSample["encoder_input"])
    tgt_pad_tokens = max_tgt_seq_len - len(eSample["decoder_input"])

    encoder_input.append(torch.cat([eSample["encoder_input"],
                                          torch.full((src_pad_tokens,),pad_token,dtype=torch.int64)]))
    decoder_input.append(torch.cat([eSample["decoder_input"],
                                         torch.full((tgt_pad_tokens,),pad_token,dtype=torch.int64)]))
    label.append(torch.cat([eSample["label"],
                                         torch.full((tgt_pad_tokens,),pad_token,dtype=torch.int64)]))

  encoder_input_tensor = torch.stack(encoder_input)
  decoder_input_tensor = torch.stack(decoder_input)
  print(decoder_input_tensor.size())



  encoder_mask = (encoder_input_tensor!=torch.tensor([pad_token])).unsqueeze(1).unsqueeze(1).int()
  decoder_mask = (decoder_input_tensor!=torch.tensor([pad_token])).unsqueeze(1).unsqueeze(1)\
  &causalMask(decoder_input_tensor.size()[-1])


  return {"encoder_input":torch.stack(encoder_input),
          "decoder_input":torch.stack(decoder_input),
          "label":torch.stack(label),
          "encoder_mask":encoder_mask,
          "decoder_mask": decoder_mask,
          "src_text":[eSample["src_text"] for eSample in batch],
          "tgt_text":[eSample["tgt_text"] for eSample in batch]}

