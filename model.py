import torch
import torch.nn as nn
import math
from pathlib import Path
import json
import config as cfg

config = cfg.getConfig()

config

class inputEmbeddings(nn.Module):
  def __init__(self,vocab_size:int,d_model:int):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size,d_model)
  def forward(self,x):
    return self.embedding(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model) ## Add across all batch
        return self.dropout(x)

class layerNormalization(nn.Module):
  def __init__(self,eps:float=10**-6):
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.beta = nn.Parameter(torch.zeros(1))

  def forward(self,x):
    mean = x.mean(-1,keepdim=True)
    std = x.std(-1,keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.beta

class feedForwardLayer(nn.Module):
  def __init__(self,d_model:int,dff:int,dropout:float):
    super().__init__()
    self.linear1 = nn.Linear(d_model,dff)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dff,d_model)

  def forward(self,x):
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.depth = num_heads, d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.shape[0]
        T_q = q.shape[1]
        T_k = k.shape[1]
        T_v = v.shape[1]

        # Project and split into heads individually using their respective sequence lengths
        # (B, T, d_model) -> (B, h, T, depth)
        q = self.wq(q).view(B, T_q, self.h, self.depth).transpose(1, 2)
        k = self.wk(k).view(B, T_k, self.h, self.depth).transpose(1, 2)
        v = self.wv(v).view(B, T_v, self.h, self.depth).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.depth)   # (B, h, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.wo(out)

class residualConnection(nn.Module):
  def __init__(self,dropout:float):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.layerNorm = layerNormalization()

  def forward(self,x,sublayer):
    return x + self.dropout(sublayer(self.layerNorm(x)))

class encoderLayer(nn.Module):
  def __init__(self,selfAttention:MultiHeadAttentionBlock,feedforward:feedForwardLayer,dropout:float):
    super().__init__()
    self.mha = selfAttention
    self.ffn = feedforward
    self.residualList = nn.ModuleList([residualConnection(dropout) for _ in range(2)])


  def forward(self,x,src_mask): # x is after input embedding and positional embedding
    x = self.residualList[0](x,lambda x : self.mha(x,x,x,mask=src_mask))
    x = self.residualList[1](x,lambda x : self.ffn(x))
    return x


class encoder(nn.Module):
  def __init__(self,layers:nn.ModuleList) -> None:
     super().__init__()
     self.layers = layers
     self.layernorm = layerNormalization()

  def forward(self,x,src_mask):
    for eLayer in self.layers:
      x = eLayer(x,src_mask)
    return self.layernorm(x)

class decoderLayer(nn.Module):
  def __init__(self,maskedSelfAttention:MultiHeadAttentionBlock,crossAttention:MultiHeadAttentionBlock,
               feedforward:feedForwardLayer,dropout):
    super().__init__()
    self.maskedMHA = maskedSelfAttention
    self.crossAttention = crossAttention
    self.ffn = feedforward
    self.residualList = nn.ModuleList([residualConnection(dropout) for _ in range(3)])

  def forward(self,x,encoder_output,decoder_mask,src_mask):
    _,T,D = x.size()
    x = self.residualList[0](x,lambda x : self.maskedMHA(x,x,x,decoder_mask))
    x = self.residualList[1](x,lambda x : self.crossAttention(x,encoder_output,encoder_output,
                                                             src_mask))
    x = self.residualList[2](x,lambda x : self.ffn(x))
    return x

class decoder(nn.Module):
  def __init__(self,layers:nn.ModuleList):
    super().__init__()
    self.layers = layers
    # self.linear = nn.Linear(d_model,output_vocab_size)
    self.norm = layerNormalization()

  def forward(self,x,encoder_output,decoder_mask, src_mask):
    for eLayer in self.layers:
      x = eLayer(x,encoder_output,decoder_mask, src_mask)
    return self.norm(x)

class outputProjection(nn.Module):
  def __init__(self,d_model,vocab_size):
    super().__init__()
    self.d_model = d_model
    self.dff = vocab_size
    self.linear = nn.Linear(self.d_model,self.dff)

  def forward(self,x):
    x = self.linear(x)
    return torch.log_softmax(x,dim=-1) #(B,T,dff)

class Transformer(nn.Module):
  def __init__(self,encoder:encoder,decoder:decoder,iemb:inputEmbeddings,oemb:inputEmbeddings,
               ipos:PositionalEncoding,opos:PositionalEncoding,finalProjection:outputProjection):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.iemb = iemb
    self.oemb = oemb
    self.ipos = ipos
    self.opos = opos
    self.finalProjection = finalProjection

def encode(self,src,mask_tensor):
  src = self.iemb(src)
  src = self.ipos(src)
  return self.encode(src,mask_tensor)

def decode(self,tgt,encoder_output,src_mask,tgt_mask):
  tgt = self.oemb(tgt)
  tgt = self.opos(tgt)
  return self.decoder(tgt,encoder_output,tgt_mask,src_mask)

def projection(self,x):
  return self.finalProjection(x)

def buildTransformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_length:int,
                     d_model:int=512,num_layers:int=6,num_heads:int=8,dff:int=2048,dropout:float=0.1)->Transformer:

  src_emb = inputEmbeddings(src_vocab_size,d_model)
  tgt_emb = inputEmbeddings(tgt_vocab_size,d_model)

  src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
  tgt_pos = PositionalEncoding(d_model,tgt_seq_length,dropout)

  encoderLayers = []
  for _ in range(num_layers):
    selfAttn = MultiHeadAttentionBlock(d_model = d_model,num_heads=num_heads,dropout=dropout)
    feedforward = feedForwardLayer(d_model=d_model,dff=dff,dropout=dropout)
    enc_layer = encoderLayer(selfAttn,feedforward,dropout)
    encoderLayers.append(enc_layer)

  decoderLayers = []
  for _ in range(num_layers):
    maskedSelfAttn = MultiHeadAttentionBlock(d_model=d_model,num_heads=num_heads,dropout=dropout)
    crossAttn = MultiHeadAttentionBlock(d_model=d_model,num_heads=num_heads,dropout=dropout)
    feedforward = feedForwardLayer(d_model,dff,dropout)
    dec_layer = decoderLayer(maskedSelfAttn,crossAttn,feedforward,dropout=dropout)
    decoderLayers.append(dec_layer)

  transformer_encoder = encoder(nn.ModuleList(encoderLayers))
  transformer_decoder = decoder(nn.ModuleList(decoderLayers))

  finalProjection = outputProjection(d_model,tgt_vocab_size)

  transformer = Transformer(transformer_encoder,transformer_decoder,src_emb,tgt_emb,src_pos,tgt_pos,finalProjection)

  for p in transformer.parameters():
    if p.dim()>1:nn.init.xavier_uniform_(p)

  return transformer
