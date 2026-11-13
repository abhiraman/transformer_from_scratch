import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
from model import buildTransformer
from dataloader import addDynamicPaddings
from torch.utils.data import DataLoader
from config import get_weights_file_path

def trainModel(epochs:int,lr:float,src_tokenizer,tgt_tokenizer,config,trainDataset,validationDataset,verbose=False,):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"device to be used : {device}")

  Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    # Build DataLoader instances using our dynamic padding collate function
  trainLoader = DataLoader(trainDataset, batch_size=config["batchSize"], shuffle=True,collate_fn=addDynamicPaddings)
  valLoader = DataLoader(validationDataset, batch_size=1, shuffle=True,collate_fn = addDynamicPaddings)


  model =  buildTransformer(src_vocab_size=src_tokenizer.get_vocab_size(),
                           tgt_vocab_size=tgt_tokenizer.get_vocab_size(),src_seq_len=config["max_src_seq_len"],
                           tgt_seq_length=config['max_tgt_seq_len']).to(device)

  writer = SummaryWriter(config["experiment_name"])
  optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'])
  loss_fn = nn.NLLLoss(ignore_index=src_tokenizer.token_to_id("[PAD]")).to(device)

  global_step = 0
  for epoch in range(epochs):
    model.train()
    batch_iterator = tqdm(trainLoader,desc = f"Processing for Epochs : {epoch}")
    for eBatch in batch_iterator:
      encoder_input = eBatch["encoder_input"].to(device) #(B,Seq_len)
      decoder_input = eBatch["decoder_input"].to(device) #(B,Seq_len)
      encoder_mask = eBatch["encoder_mask"].to(device) #(B,1,1,Seq_len)
      decoder_mask = eBatch["decoder_mask"].to(device) #(B,1,Seq_len,Seq_len)

      # Pass inputs through embeddings and positional encodings first
      embedded_encoder_input = model.ipos(model.iemb(encoder_input))
      embedded_decoder_input = model.opos(model.oemb(decoder_input))

      # Forward Pass with embedded inputs
      encoder_output = model.encoder(embedded_encoder_input, encoder_mask) # (B,seq_len,d_model)
      decoder_output = model.decoder(embedded_decoder_input, encoder_output, decoder_mask, encoder_mask) # (B,seq_len,d_model)
      proj_output = model.finalProjection(decoder_output) # (B,seq_len,output_vocab_size)

      if verbose:
        print("encoder input size : ",encoder_input.size())
        print("decoder_input size : ",decoder_input.size())
        print("encoder_output size : ",encoder_output.size())
        print("decoder_output size : ",decoder_output.size())
        print("proj_output  size :", proj_output.size())

      label = eBatch["label"].to(device)

      loss = loss_fn(proj_output.view(-1,tgt_tokenizer.get_vocab_size()),label.view(-1))
      batch_iterator.set_postfix(loss=loss.item())

      # Corrected positional arguments: tag, scalar_value, global_step
      writer.add_scalar("Loss/train", loss.item(), global_step=global_step)
      writer.flush()

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      global_step+=1

    model_filename = get_weights_file_path(config,epoch)
    print(model_filename)

    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "global_step":global_step
    },model_filename)
