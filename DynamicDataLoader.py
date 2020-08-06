from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import Sampler
import torch


# ======================================================
# Sampler that samples datset of sample_size
# from every whole glue dataset.  
# If dynamic is False, we just concatenate the whole
# dataset and do random sampling, which is equaivalent
# to sampling data from each dataset with probability
# distribution relative to its relative size.
# if dynmaic is True, we do dynmaic sampling.
# That is, 
# ======================================================
class InstanceSampler():
    def __init__(self, sample_size, cola=None, sst2=None, mrpc=None, qqp=None, mnli=None, rte=None, qnli=None, wnli=None, dynamic=False):
        self.sample_size = sample_size
        self.cola = cola
        self.sst2 = sst2
        self.mrpc = mrpc
        self.qqp = qqp
        self.mnli = mnli
        self.rte = rte
        self.qnli = qnli
        self.wnli = wnli
        self.dynamic = dynamic
        self.total_size = len(self.cola) + len(self.sst2) + len(self.mrpc) + len(self.qqp) + len(self.mnli) + len(self.rte) + len(self.qnli) + len(self.wnli)
        self.concat_data = self.concat()
        self.weights = self.get_init_weights()

    def __len__(self):
        return self.sample_size
    
    def get_init_weights(self):
        ws = [len(self.cola)/self.total_size] * len(self.cola)
        ws += [len(self.sst2)/self.total_size] * len(self.sst2)
        ws += [len(self.mrpc)/self.total_size] * len(self.mrpc)
        ws += [len(self.qqp)/self.total_size] * len(self.qqp)
        ws += [len(self.mnli)/self.total_size] * len(self.mnli)
        ws += [len(self.rte)/self.total_size] * len(self.rte)
        ws += [len(self.qnli)/self.total_size] * len(self.qnli)
        ws += [len(self.wnli)/self.total_size] * len(self.wnli)
        assert len(ws) == self.total_size
        return ws

    def shuffle_all(self):
        self.cola = self.cola.shuffle()
        self.sst2 = self.sst2.shuffle()
        self.mrpc = self.mrpc.shuffle()
        self.qqp = self.qqp.shuffle()
        self.mnli = self.mnli.shuffle()
        self.rte = self.rte.shuffle()
        self.qnli = self.qnli.shuffle()
        self.wnli = self.wnli.shuffle()

    def update_weights(self, cola_score, sst2_score, mrpc_score, qqp_score, mnli_score, rte_score, qnli_score, wnil_score):
        tot_score = sum([cola_score, sst2_score, mrpc_score, qqp_score, mnli_score, rte_score, qnli_score, wnil_score])
        ws = [cola_score/tot_score] * len(self.cola)
        ws += [sst2_score/tot_score] * len(self.sst2)
        ws += [mrpc_score/tot_score] * len(self.mrpc)
        ws += [qqp_score/tot_score] * len(self.qqp)
        ws += [mnli_score/tot_score] * len(self.mnli)
        ws += [rte_score/tot_score] * len(self.rte)
        ws += [qnli_score/tot_score] * len(self.qnli)
        ws += [wnli_score/tot_score] * len(self.wnli)
        self.weights = ws

    def concat(self):
        next_ds = ConcatDataset([self.cola, self.sst2, self.mrpc, self.qqp, self.mnli, self.rte, self.qnli, self.wnli])
        return next_ds
        
    def get_sample(self):
        """
        If self.dynamic is set to true, we sample from each dataset
        proportionately to the validation loss of the dataset. 
        If not, we just do random sampling from the whole bunch.
        """
        if self.dynamic:
            assert self.concat_data != None
            assert self.weights != None and len(self.weights) == self.total_size
            sampled_idx = torch.multinomial(torch.tensor(self.weights), self.sample_size)
            next_instance = Subset(self.concat_data, sampled_idx.tolist())
            return next_instance
        else:
            return self.concat_data
                                             # torch.utils.data.random_split(dataset, lengths)  <- try this instead
        

        