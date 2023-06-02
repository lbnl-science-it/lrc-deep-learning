from __future__ import print_function
from __future__ import division
import sys
import torch
import os
import tqdm
import pdb
import numpy as np
import platform
import hashlib
import pytorch_transformer
import re
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformProtein import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='PyTorch code for generating from CTRL')

#parser.add_argument('--model_dir', type =str, default='model_v0.pth', help='location of training model checkpoint')
#parser.add_argument('--model_path', type=str, default='/home/amadani/ctrl/ckpt/seqlen256_36layers_v0.ckpt/model.ckpt-684000', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')

parser.add_argument('--model_dir', type =str, default='./checkpoints_cur/finetune_progen_full.pth', help='location of training model checkpoint')
parser.add_argument('--model_path', type=str, default='../checkpoints/pretrain_progen_full.pth', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')

parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for PyTorch, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=511,
                                        help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=36, help='number of transfomer layers. used for loading checkpoint')
parser.add_argument('--batch_size', type=int, default = 4, help='batch size for dataloader')
parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=1000, help='save model checkpoint every X iterations')


args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
# length of the vocabulary
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

# define the numericalization map
# idx2word maps the numericalized ID to the word
# word2idx maps the word to the numericalized ID
#word2idx = {u:i for i, u in enumerate(vocab)}
#idx2word = np.array(vocab)

# sequence length to use for transfomer
seq_length = args.sequence_len
embedding_dim = 1280

class TiedEmbeddingSoftmax(torch.nn.Module):

  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  def __init__(self):
    super(CTRLmodel,self).__init__()
    self.tied_embedding_softmax = TiedEmbeddingSoftmax()
    self.encoder = pytorch_transformer.Encoder()

  def forward(self, inputs):
    x = self.tied_embedding_softmax(inputs, embed = True)
    x = self.encoder(x)
    x = self.tied_embedding_softmax(x, embed = False)
    return x

  def loadCheckpoint(self, model_path, num_layers):
    #pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()
    pytorch_model_hash = model_path

    if os.path.exists(pytorch_model_hash):
      print('Found PyTorch checkpoint @', pytorch_model_hash)
      print('Loading instead of converting from TensorFlow')
      checkpoint = torch.load(pytorch_model_hash)
      
      #self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
      #self.encoder.load_state_dict(checkpoint['encoder'])
      ## load state dict has KeyError, because checkpoint is ready the state_dict
      ## can load checkpoint direclty 
      ## https://discuss.pytorch.org/t/keyerror-state-dict/18220/5
      self.load_state_dict(checkpoint)

      self.tied_embedding_softmax.to('cuda')
      self.encoder.to('cuda')

    else:
      print('Error: Could not find PyTorch checkpoint')
      sys.exit(1)

# initialize ctrl object
# load checkpoint with args.model_path
model = CTRLmodel()
print('model initialized')
model.loadCheckpoint(model_path=args.model_path, num_layers = args.num_layers)
print('previous checkpoint loaded')
model = model.cuda()

# freeze all weights except embedding
for p in model.parameters():
    p.requires_grad=False
model.tied_embedding_softmax.w.requires_grad=True
model.tied_embedding_softmax.b.requires_grad=True

class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26 # Assuming that the pad token is the last token and AAs are at the end
        
        self.optimizer = torch.optim.Adam(self.model.parameters()) #lr, betas
        lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        
        self.transformFull = transformProtein(maxSampleLength = seq_length+1, 
                                              selectSwiss = 1.0, selectTrembl = 0, 
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
        self.transformPartial = transformProtein(maxSampleLength = seq_length+1,   
                                                 selectSwiss = 1.0, selectTrembl = 0,
                                                 maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
        self.transformNone = transformProtein(maxSampleLength = seq_length+1,   
                                              selectSwiss = 1.0, selectTrembl = 0,
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
        
        self.writer = SummaryWriter()

    def train(self, num_epochs):
        self.model.train()

        iter_num = 0
        for epoch in range(num_epochs):
            loss_e = 0.0
            num_e = 0

            for chunknum in range(1):
                pklpath = '../miBIG/mibig_train_new.p'
                chunk_dataset = ProteinDataset(pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull, 
                                               transformPartial = self.transformPartial, transformNone = self.transformNone)
                dataloader = DataLoader(chunk_dataset, shuffle = True, batch_size = self.batch_size,
                                        num_workers = self.num_workers, pin_memory = False) #TODO pinmem?
                
                for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()
                    output = self.model(sample)
                    #pdb.set_trace()
                    loss = self.criterion(output.permute(0,2,1), labels)
                    loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    self.scheduler.step()
                    loss_e += loss.item()
                    num_e += sample.shape[0]
                    iter_num += 1
                    self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)

                    if (iter_num+1)%self.save_iter==0:
                        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':iter_num,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss,
                                   }, self.model_dir)
                loss_e/=num_e
            print("Epoch: {0} ; loss_e: {1}".format(epoch, loss_e))
            self.writer.add_scalar('Loss_epoch',loss_e, epoch)


##########################################################
# Check the format of training data before the model train
##########################################################
pklpath = '../miBIG/mibig_train_new.p'
obj = transformProtein(mapfold = "./mapping_files", selectSwiss = 1.0, selectTrembl = 0, maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
with open(pklpath, 'rb') as handle:
    train_chunk = pickle.load(handle)
for uid in train_chunk.keys():
  try:
    sample_arr, existence, thePadIndex = obj.transformSample(train_chunk[uid])
    print("loaded UID:", uid)
  except:
    print("Error UID:", uid)


##########################################################
# Train the model
##########################################################
training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                   model_dir = args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)
