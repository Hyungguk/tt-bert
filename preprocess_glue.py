from nlp import load_dataset
import pandas as pd
import os


# Target Tokens for each glue subtask.
TOKENS_TO_ADD = {'cola' : '[ACCEPTABLE] [UNACCEPTABLE] ',
                 'sst2' : '[POSITIVE] [NEGATIVE] ',
                 'mrpc' : '[PARAPHRASE] [NOT_PARAPHRASE] ',
                 'qqp' : '[SIMILAR] [NOT_SIMILAR] ',
                 'mnli' : '[ENTAILMENT] [NEUTRAL] [CONTRADICTION] ',
                 'rte'  : '[ENTAILMENT] [NOT_ENTAILMENT] ',
                 'qnli' : '[ANSWERABLE] [NOT_ANSWERABLE] ',
                 'wnli' : '[ENTAILMENT] [NOT_ENTAILMENT] '    
                }

RED_TOKENS_TO_ADD = {'cola' : '[ACCEPTABLE] [UNACCEPTABLE] ',
                 'sst2' : '[POSITIVE] [NEGATIVE] ',
                 'mrpc' : '[SIMILAR] [NOT_SIMILAR] ',
                 'qqp' : '[SIMILAR] [NOT_SIMILAR] ',
                 'mnli' : '[ENTAILMENT] [NEUTRAL] [CONTRADICTION] ',
                 'rte'  : '[ENTAILMENT] [NOT_ENTAILMENT] ',
                 'qnli' : '[ENTAILMENT] [NOT_ENTAILMENT]] ',
                 'wnli' : '[ENTAILMENT] [NOT_ENTAILMENT] '    
                }

TASK_SENT_CNT = {'cola' : 1,     
                 'sst2' : 1,
                 'mrpc' : 2,
                 'qqp' : 2,
                 'mnli' : 2,
                 'rte'  : 2,
                 'qnli' : 2,
                 'wnli' : 2    
                }

TASK_SENT_NAME = {'cola' : ['sentence'],
                 'sst2' : ['sentence'],
                 'mrpc' : ['sentence1', 'sentence2'],
                 'qqp' : ['question1', 'question2'],
                 'mnli' : ['premise', 'hypothesis'],
                 'rte'  : ['sentence1', 'sentence2'],
                 'qnli' : ['question', 'sentence'],
                 'wnli' : ['sentence1', 'sentence2']    
                }

POINTER_MASKS = {'two-targets' : [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0],
                'three-targets' : [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0]      
                }

TASK_TARGET_CNT = {'cola' : 'two-target',
                 'sst2' : 'two-target',
                 'mrpc' : 'two-target',
                 'qqp' : 'two-target',
                 'mnli' : 'three-target',
                 'rte'  : 'two-target',
                 'qnli' : 'two-target',
                 'wnli' : 'two-target'   
                }


# ====================================================
# For every input sentence (or every first sentence if
# the task is a pair-sentence task) we add tast target
# tokens. We add label by 1 because the 0th index
# of the final output corresponds to [CLS] token.
# ====================================================
def stokens_and_labels_adder(dtype, redundant):
    def add_stokens_and_labels(ds):
        if redundant:
          spec_tok = RED_TOKENS_TO_ADD[dtype]
        else:
          spec_tok = TOKENS_TO_ADD[dtype]
        ds[TASK_SENT_NAME[dtype][0]] = spec_tok + ds[TASK_SENT_NAME[dtype][0]]
        ds['label'] = ds['label'] + 1
        return ds
    return add_stokens_and_labels

# ===========================================================
# Add pointer masks to the dataset.
# Tasks with two labels have mask [0, 1, 1, 0, 0,..., 0].
# Tasks with three labels have mask [0, 1, 1, 1, 0, ..., 0].
# Among glue tasks, mnli is the only task with three targets.
# ===========================================================

def add_pointer_mask_two(ds):
    ds['pointer_mask'] = POINTER_MASKS['two-targets']
    return ds
    

def add_pointer_mask_three(ds):
    ds['pointer_mask'] = POINTER_MASKS['three-targets']
    return ds


# ====================================================
# Tokenize the input sentences of the dataset
# with the given tokenizer.
# ====================================================
def make_encoder(dtype, tokenizer):
    def encode(ds):
        if TASK_SENT_CNT[dtype] == 1:
            return_dic = tokenizer(ds[TASK_SENT_NAME[dtype][0]],
                        truncation=True, max_length=128, padding='max_length')
        else:         
            return_dic = tokenizer(ds[TASK_SENT_NAME[dtype][0]], ds[TASK_SENT_NAME[dtype][1]],
                        truncation=True, max_length=128, padding='max_length')
        return return_dic
    return encode
    
# ====================================================
# Get train and validation datasets of the
# given GLUE dataset dtype. 
# dtype should be lower cased string name of one of
# the glue tasks. 
# ====================================================
def get_train_val_datasets(dtype, redundant=False):
    train_fn = './glue_dataset/' + dtype + '_train.csv'
    val_fn = './glue_dataset/' + dtype + '_val.csv'
    if dtype == 'mnli':
        val_fn = './glue_dataset/' + dtype + '_matched_val.csv'
    if os.path.isfile(train_fn):
        train_dataset = load_dataset('csv', data_files=train_fn)['train']
    else:
        train_dataset = load_dataset('glue', dtype, split='train')
        train_dataset = train_dataset.map(stokens_and_labels_adder(dtype, redundant))
        #train_dataset.set_format(type='pandas')
        #train_dataset[:].to_csv(train_fn, index=False)
        #train_dataset.reset_format()
    if os.path.isfile(val_fn):
        val_dataset = load_dataset('csv', data_files=val_fn)['train']
    else:
        if dtype == 'mnli':
            val_dataset = load_dataset('glue', dtype, split='validation_matched')
        else:
            val_dataset = load_dataset('glue', dtype, split='validation')
        val_dataset = val_dataset.map(stokens_and_labels_adder(dtype, redundant))
        #val_dataset.set_format(type='pandas')
        #val_dataset[:].to_csv(val_fn, index=False)
        #val_dataset.reset_format()
    
    # Since Pointer Mask is a vector, it is a bit tricky
    # to save it in a csv file. Therefore, we just add Pointer Masks
    # everytime we load a dataset.
    if dtype == 'mnli':
        train_dataset = train_dataset.map(add_pointer_mask_three)
        val_dataset = val_dataset.map(add_pointer_mask_three)
    else:
        train_dataset = train_dataset.map(add_pointer_mask_two)
        val_dataset = val_dataset.map(add_pointer_mask_two)
    print("Loaded {:} train data of length {:} and validation data of length {:}".format(dtype, len(train_dataset), len(val_dataset)))
    if redundant:
        print("++ Used redundant target tokens.")
    return train_dataset, val_dataset



def get_small_train_val_datasets(dtype):
    train_fn = './glue_dataset/' + dtype + '_train.csv'
    val_fn = './glue_dataset/' + dtype + '_val.csv'
    
    train_dataset = load_dataset('glue', dtype, split='train[:5%]')
    train_dataset = train_dataset.map(stokens_and_labels_adder(dtype))

    if dtype == 'mnli':
        val_dataset = load_dataset('glue', dtype, split='validation_matched[:5%]')
    else:
        val_dataset = load_dataset('glue', dtype, split='validation[:5%]')
    val_dataset = val_dataset.map(stokens_and_labels_adder(dtype))
    
    # Since Pointer Mask is a vector, it is a bit tricky
    # to save it in a csv file. Therefore, we just add Pointer Masks
    # everytime we load a dataset.
    if dtype == 'mnli':
        train_dataset = train_dataset.map(add_pointer_mask_three)
        val_dataset = val_dataset.map(add_pointer_mask_three)
    else:
        train_dataset = train_dataset.map(add_pointer_mask_two)
        val_dataset = val_dataset.map(add_pointer_mask_two)
    print("Loaded train data of length {:} and validation data of length {:}".format(len(train_dataset), len(val_dataset)))
    return train_dataset, val_dataset



