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
import json

from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


vocab_size = 129407
embedding_dim = 1280
DEVICE = 'cuda'

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        scheduler, criterion,
        args
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.args = args
        self.loss = torch.tensor(float('inf')) # set initial loss as inf

        #self.model.tied_embedding_softmax = model.tied_embedding_softmax.to(gpu_id)
        #self.model.encoder = model.encoder.to(gpu_id)

    def _run_batch(self, source, targets, existence, padIndex):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output.permute(0,2,1), targets)
        loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        self.loss = loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Loss: {round(self.loss.item(),5)}")
        self.train_data.sampler.set_epoch(epoch)
        for i, (source, targets, existence, padIndex, begAAindex) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            existence = existence.to(self.gpu_id)
            padIndex = padIndex.to(self.gpu_id)
            self._run_batch(source, targets, existence, padIndex)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        #PATH = "./checkpoint.pt"
        PATH = self.args.model_dir
        torch.save(ckp, PATH)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss
                   }, PATH)

        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch)


class TrainerCTRL(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter, pklpath):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.pklpath = pklpath
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

    def get_optimizer(self):
        return(self.optimizer, self.scheduler, self.criterion)

    def get_data(self):
        chunk_dataset = ProteinDataset(self.pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull, 
                                       transformPartial = self.transformPartial, transformNone = self.transformNone)
        return(chunk_dataset)
                
        
    def train(self, num_epochs):
        self.model.train()

        iter_num = 0
        for epoch in range(num_epochs):
            loss_e = 0.0
            num_e = 0

            for chunknum in range(1):
                #pklpath = '../miBIG/mibig_train_new2.p'
                #pklpath = '../miBIG/mibig_train_new.p'
                chunk_dataset = ProteinDataset(self.pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull, 
                                               transformPartial = self.transformPartial, transformNone = self.transformNone)
                dataloader = DataLoader(chunk_dataset, shuffle = False, batch_size = self.batch_size,
                                        num_workers = self.num_workers, pin_memory = True) #TODO pinmem?
                
                for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                    #sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()
                    sample, labels, existence, padIndex = sample.to(DEVICE), labels.to(DEVICE), existence.to(DEVICE), padIndex.to(DEVICE)
                    self.optimizer.zero_grad()
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

                    if (iter_num+1)%self.save_iter==0 or (epoch+1==num_epochs):
                        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':iter_num,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss,
                                   }, self.model_dir)
                loss_e/=num_e
            print("Epoch: {0} ; loss_e: {1}".format(epoch, loss_e))
            self.writer.add_scalar('Loss_epoch',loss_e, epoch)

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
    self.encoder = pytorch_transformer.Encoder(device=DEVICE)

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
      ## can load checkpoint directly 
      ## https://discuss.pytorch.org/t/keyerror-state-dict/18220/5
      self.load_state_dict(checkpoint)

      #self.tied_embedding_softmax.to('cuda')
      #self.encoder.to('cuda')
      self.tied_embedding_softmax.to(DEVICE)
      self.encoder.to(DEVICE)
      #self.tied_embedding_softmax.cuda()
      #self.encoder.cuda()

    else:
      print('Error: Could not find PyTorch checkpoint')
      sys.exit(1)


def load_train_objs(args):
    # initialize ctrl object
    model = CTRLmodel()
    print('model initialized')
    # load checkpoint with args.model_path
    model.loadCheckpoint(model_path=args.model_path, num_layers = args.num_layers)
    print('previous checkpoint loaded')
    # freeze all weights except embedding
    for p in model.parameters():
        p.requires_grad=False
    model.tied_embedding_softmax.w.requires_grad=True
    model.tied_embedding_softmax.b.requires_grad=True
    # user origianl Trainer to get Dataset and Optimizer
    trainerCTRL = TrainerCTRL(model=model, warmup_iteration=args.warmup_iteration, seq_length=args.sequence_len,
                       batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                       model_dir = args.model_dir, save_iter=args.save_iter,
                       pklpath=args.pklpath)
    train_set = trainerCTRL.get_data()
    print(f'Size of progen training data: {len(train_set)}')
    optimizer, scheduler, criterion = trainerCTRL.get_optimizer()
    #train_set = MyTrainDataset(2048)  # load your dataset
    #model = torch.nn.Linear(20, 1)  # load your model
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer, scheduler, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers = 0,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, args):
    print(f'rank {rank}, world_size {world_size}')
    ddp_setup(rank, world_size)
    dataset, model, optimizer , scheduler, criterion = load_train_objs(args)
    print(f'dataset size: {len(dataset)}')
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, scheduler, criterion, args)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch code for generating from CTRL')
    parser.add_argument('--total_epochs', default=200, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=50, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--model_dir', type =str, default='./checkpoints_cur/finetune_progen_multiGPU_demo.pth', help='location of training model checkpoint')
    parser.add_argument('--model_path', type=str, default='../checkpoints/pretrain_progen_full.pth', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')
    #parser.add_argument('--pklpath', type=str, default='../miBIG/mibig_train_new2.p', help='location of training data')
    parser.add_argument('--pklpath', type=str, default='../miBIG/S3_pickle/data-1.pickle', help='location of training data')

    parser.add_argument('--seed', type=int, default=313, help='random seed for PyTorch, numpy and PythonHash')
    parser.add_argument('--sequence_len', type=int, default=511*1,
                                            help='sequence len of model being fine-tuned')
    parser.add_argument('--num_epochs', type=int, default=16, help='number of epochs to train for')
    parser.add_argument('--num_layers', type=int, default=36, help='number of transfomer layers. used for loading checkpoint')
    #parser.add_argument('--batch_size', type=int, default = 4*2, help='batch size for dataloader')
    parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
    parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
    parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
    parser.add_argument('--save_iter', type=int, default=1000, help='save model checkpoint every X iterations')

    args = parser.parse_args()
    print(args)

    world_size = torch.cuda.device_count()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{DEVICE} with {torch.cuda.device_count()} gpu')    

    # load the vocabulary from file
    use_py3 = platform.python_version()[0] == '3'
    vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    # length of the vocabulary
    vocab_size = len(vocab)
    print('-----vocab size',vocab_size,'------')

    # sequence length to use for transfomer
    seq_length = args.sequence_len
    embedding_dim = 1280

    # Train the model on multiple GPUs
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args), nprocs=world_size)
