import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import time
import datetime
import numpy as np
from PointerModel import PointerBert
from preprocess_glue import *
import os
from nlp import load_dataset
import tensorflow as tf
from DynamicDataLoader import InstanceSampler

flags =  tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_float('epsilon', 1e-8, 'epsilon.')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs when training.')
flags.DEFINE_integer('batch_size', 32, 'Size of each batch. Default is 32.')
flags.DEFINE_boolean('dynamic_sampling', False, 'Use dynamic sampling?')

NEW_SPEC_TOKENS = ['[ACCEPTABLE]', '[UNACCEPTABLE]', 
                   '[POSITIVE]', '[NEGATIVE]', 
                   '[PARAPHRASE]', '[NOT_PARAPHRASE]', 
                   '[SIMILAR]', '[NOT_SIMILAR]',
                   '[ENTAILMENT]', '[NEUTRAL]', '[CONTRADICTION]',
                   '[ANSWERABLE]', '[NOT_ANSWERABLE]',
                   '[REFERENT]', '[NOT_REFERENT]']

single_accuracy = {'cola' : 0.831,
                 'sst2' : 0.924,
                 'mrpc' : 0.869,
                 'qqp' : 0.911,
                 'mnli' : 0.839,
                 'rte'  : 0.799,
                 'qnli' : 0.911,
                 'wnli' : 0.565
                }

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# TODO : dddd
# =======================================================
# Get positive weights from accuracies(or other metrics) 
# by getting the differentials of those with those of
# single-task models. Add some positive number 
# if any of the scores are negative to make them positive.
# ========================================================
def get_scores_from_acc(cola=None, sst2=None, mrpc=None, qqp=None, mnli=None, rte=None, qnli=None, wnli=None):
    cola_score = single_accuracy['cola'] - cola + 0.2
    sst2_score = single_accuracy['sst2'] - sst2 + 0.2
    mrpc_score = single_accuracy['mrpc'] - mrpc + 0.2
    qqp_score = single_accuracy['qqp'] - qqp + 0.2
    mnli_score = single_accuracy['mnli'] - mnli + 0.2
    rte_score = single_accuracy['rte'] - rte + 0.2
    qnli_score = single_accuracy['qnli'] - qnli + 0.2
    wnli_score = single_accuracy['wnli'] - wnli + 0.2 
    return cola_score, sst2_score, mrpc_score, qqp_score, mnli_score, rte_score, qnli_score, wnli_score


if __name__ == "__main__":
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # ========================================================
    # Call appropriate tokenizer of the Bert model.
    # If there is a saved version, load it.
    # If not, load a new pretrained BertTokenizer with 
    # special tokens addded. Then save it.
    # ========================================================
    if os.path.isdir('./tokenizer/custom-tok'):       # Check if we already have a customzied tokenizer.
        tokenizer = BertTokenizer.from_pretrained('custom-tok') # If we do, load it.    
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, additional_special_tokens=NEW_SPEC_TOKENS)
        tokenizer.save_pretrained('./tokenizer/custom_tok')

    # =========================================================
    # Call Pointer Bert model that uses Bert as its embedding
    # and has an extra dot-producting layer that 
    # finds the right answer among the added task tokens.
    # =========================================================

    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.resize_token_embeddings(len(tokenizer))
    model = PointerBert(bert)

    if torch.cuda.is_available():
            model.cuda()

    batch_size = FLAGS.batch_size

    cola_train_dataset, cola_val_dataset = get_small_train_val_datasets('cola')
    sst2_train_dataset, sst2_val_dataset = get_small_train_val_datasets('sst2')
    mrpc_train_dataset, mrpc_val_dataset = get_small_train_val_datasets('mrpc')
    qqp_train_dataset, qqp_val_dataset = get_small_train_val_datasets('qqp')
    mnli_train_dataset, mnli_val_dataset = get_small_train_val_datasets('mnli')
    rte_train_dataset, rte_val_dataset = get_small_train_val_datasets('rte')
    qnli_train_dataset, qnli_val_dataset = get_small_train_val_datasets('qnli')
    wnli_train_dataset, wnli_val_dataset = get_small_train_val_datasets('wnli')

    cola_train_dataset = cola_train_dataset.map(make_encoder('cola', tokenizer))
    cola_val_dataset = cola_val_dataset.map(make_encoder('cola', tokenizer))
    sst2_train_dataset = sst2_train_dataset.map(make_encoder('sst2', tokenizer))
    sst2_val_dataset = sst2_val_dataset.map(make_encoder('sst2', tokenizer))
    mrpc_train_dataset = mrpc_train_dataset.map(make_encoder('mrpc', tokenizer))
    mrpc_val_dataset = mrpc_val_dataset.map(make_encoder('mrpc', tokenizer))
    qqp_train_dataset = qqp_train_dataset.map(make_encoder('qqp', tokenizer))
    qqp_val_dataset = qqp_val_dataset.map(make_encoder('qqp', tokenizer))
    mnli_train_dataset = mnli_train_dataset.map(make_encoder('mnli', tokenizer))
    mnli_val_dataset = mnli_val_dataset.map(make_encoder('mnli', tokenizer))
    rte_train_dataset = rte_train_dataset.map(make_encoder('rte', tokenizer))
    rte_val_dataset = rte_val_dataset.map(make_encoder('rte', tokenizer))
    qnli_train_dataset = qnli_train_dataset.map(make_encoder('qnli', tokenizer))
    qnli_val_dataset = qnli_val_dataset.map(make_encoder('qnli', tokenizer))
    wnli_train_dataset = wnli_train_dataset.map(make_encoder('wnli', tokenizer))
    wnli_val_dataset = wnli_val_dataset.map(make_encoder('wnli', tokenizer))
    
    cola_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    sst2_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mrpc_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qqp_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mnli_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    rte_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qnli_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    wnli_train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    
    cola_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    sst2_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mrpc_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qqp_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    rte_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    wnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])


    total_size = len(cola_train_dataset) + len(sst2_train_dataset) + len(mrpc_train_dataset) + len(qqp_train_dataset) + len(mnli_train_dataset) + len(rte_train_dataset) + len(qnli_train_dataset) + len(wnli_train_dataset)
    sample_size = total_size // 10

    inst_sampler = InstanceSampler(sample_size, cola_train_dataset, sst2_train_dataset, mrpc_train_dataset, qqp_train_dataset, mnli_train_dataset, rte_train_dataset, qnli_train_dataset, wnli_train_dataset, dynamic=True)
    
    train_dataset = inst_sampler.get_sample()
    print("total size of training data is", total_size)
      
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size 
            )

    # =========================================================
    # We make one validation dataloader for each dataset.
    # Validation accuracy of each dataset is used to set 
    # the weights of dynamic sampling.
    # =========================================================
    cola_val_dataloader = DataLoader(
                cola_val_dataset,
                sampler = SequentialSampler(cola_val_dataset), 
                batch_size = batch_size 
            )
    sst2_val_dataloader = DataLoader(
                sst2_val_dataset,
                sampler = SequentialSampler(sst2_val_dataset), 
                batch_size = batch_size 
            )
    mrpc_val_dataloader = DataLoader(
                mrpc_val_dataset,
                sampler = SequentialSampler(mrpc_val_dataset), 
                batch_size = batch_size 
            )
    qqp_val_dataloader = DataLoader(
                qqp_val_dataset,
                sampler = SequentialSampler(qqp_val_dataset), 
                batch_size = batch_size 
            )
    mnli_val_dataloader = DataLoader(
                mnli_val_dataset,
                sampler = SequentialSampler(mnli_val_dataset), 
                batch_size = batch_size 
            )
    rte_val_dataloader = DataLoader(
                rte_val_dataset,
                sampler = SequentialSampler(rte_val_dataset), 
                batch_size = batch_size 
            )
    qnli_val_dataloader = DataLoader(
                qnli_val_dataset,
                sampler = SequentialSampler(qnli_val_dataset), 
                batch_size = batch_size 
            )
    wnli_val_dataloader = DataLoader(
                wnli_val_dataset,
                sampler = SequentialSampler(wnli_val_dataset), 
                batch_size = batch_size 
            )

    # ====================================================
    # Set up optimizer and scheduler for training.
    # ====================================================
    optimizer = AdamW(model.parameters(),
                  lr =  FLAGS.learning_rate, 
                  eps = FLAGS.epsilon
                )

    epochs = FLAGS.num_epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print("size of instance sample is", len(train_dataset))
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        total_train_accuracy = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 400 batches.
            if step % 400 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask, 
                                labels=b_labels,
                                pointer_mask=b_pointer_mask)
            
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_train_accuracy += flat_accuracy(logits, label_ids)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)  

        avg_train_accuracy = total_train_accuracy / len(train_dataloader)        
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training accuracy: {0:.2f}".format(avg_train_accuracy))
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validations...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # ========================================
        #             Cola Validation
        # ========================================

        total_cola_eval_accuracy = 0
        total_cola_eval_loss = 0

        for batch in cola_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_cola_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_cola_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_cola_val_accuracy = total_cola_eval_accuracy / len(cola_val_dataloader)
        print("  Cola Accuracy: {0:.2f}".format(avg_cola_val_accuracy))

        avg_cola_val_loss = total_cola_eval_loss / len(cola_val_dataloader)
        print("  Cola Validation Loss: {0:.2f}".format(avg_cola_val_loss))

        # ========================================
        #             SSt-2 Validation
        # ========================================

        total_sst2_eval_accuracy = 0
        total_sst2_eval_loss = 0

        for batch in sst2_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_sst2_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_sst2_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_sst2_val_accuracy = total_sst2_eval_accuracy / len(sst2_val_dataloader)
        print("  SST-2 Accuracy: {0:.2f}".format(avg_sst2_val_accuracy))

        avg_sst2_val_loss = total_sst2_eval_loss / len(sst2_val_dataloader)
        print("  SST-2 Validation Loss: {0:.2f}".format(avg_sst2_val_loss))

        # ========================================
        #             MRPC Validation
        # ========================================

        total_mrpc_eval_accuracy = 0
        total_mrpc_eval_loss = 0

        for batch in mrpc_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_mrpc_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_mrpc_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_mrpc_val_accuracy = total_mrpc_eval_accuracy / len(mrpc_val_dataloader)
        print("  MRPC Accuracy: {0:.2f}".format(avg_mrpc_val_accuracy))

        avg_mrpc_val_loss = total_mrpc_eval_loss / len(mrpc_val_dataloader)
        print("  MRPC Validation Loss: {0:.2f}".format(avg_mrpc_val_loss))

        # ========================================
        #             QQP Validation
        # ========================================

        total_qqp_eval_accuracy = 0
        total_qqp_eval_loss = 0

        for batch in qqp_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_qqp_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_qqp_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_qqp_val_accuracy = total_qqp_eval_accuracy / len(qqp_val_dataloader)
        print("  QQP Accuracy: {0:.2f}".format(avg_qqp_val_accuracy))

        avg_qqp_val_loss = total_qqp_eval_loss / len(qqp_val_dataloader)
        print("  QQP Validation Loss: {0:.2f}".format(avg_qqp_val_loss))

        # ========================================
        #             MNLI Validation
        # ========================================

        total_mnli_eval_accuracy = 0
        total_mnli_eval_loss = 0

        for batch in mnli_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_mnli_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_mnli_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_mnli_val_accuracy = total_mnli_eval_accuracy / len(mnli_val_dataloader)
        print("  MNLI Accuracy: {0:.2f}".format(avg_mnli_val_accuracy))

        avg_mnli_val_loss = total_mnli_eval_loss / len(mnli_val_dataloader)
        print("  MNLI Validation Loss: {0:.2f}".format(avg_mnli_val_loss))

        # ========================================
        #             RTE Validation
        # ========================================

        total_rte_eval_accuracy = 0
        total_rte_eval_loss = 0

        for batch in rte_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_rte_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_rte_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_rte_val_accuracy = total_rte_eval_accuracy / len(rte_val_dataloader)
        print("  RTE Accuracy: {0:.2f}".format(avg_rte_val_accuracy))

        avg_rte_val_loss = total_rte_eval_loss / len(rte_val_dataloader)
        print("  RTE Validation Loss: {0:.2f}".format(avg_rte_val_loss))

        # ========================================
        #             QNLI Validation
        # ========================================

        total_qnli_eval_accuracy = 0
        total_qnli_eval_loss = 0

        for batch in qnli_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_qnli_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_qnli_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_qnli_val_accuracy = total_qnli_eval_accuracy / len(qnli_val_dataloader)
        print("  QNLI Accuracy: {0:.2f}".format(avg_qnli_val_accuracy))

        avg_qnli_val_loss = total_qnli_eval_loss / len(qnli_val_dataloader)
        print("  QNLI Validation Loss: {0:.2f}".format(avg_qnli_val_loss))

        # ========================================
        #             WNLI Validation
        # ========================================

        total_wnli_eval_accuracy = 0
        total_wnli_eval_loss = 0

        for batch in wnli_val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)
            b_pointer_mask = batch['pointer_mask'].to(device)
            
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    pointer_mask=b_pointer_mask)
                
            total_wnli_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_wnli_eval_accuracy += flat_accuracy(logits, label_ids)      

        avg_wnli_val_accuracy = total_wnli_eval_accuracy / len(wnli_val_dataloader)
        print("  WNLI Accuracy: {0:.2f}".format(avg_wnli_val_accuracy))

        avg_wnli_val_loss = total_wnli_eval_loss / len(wnli_val_dataloader)
        print("  WNLI Validation Loss: {0:.2f}".format(avg_wnli_val_loss))


        avg_tot_val_accuracy = (avg_cola_val_accuracy + avg_sst2_val_accuracy + avg_mrpc_val_accuracy + avg_qqp_val_accuracy
                                + avg_mnli_val_accuracy + avg_rte_val_accuracy + avg_qnli_val_accuracy + avg_wnli_val_accuracy) / 8

        avg_tot_val_loss = (avg_cola_val_loss + avg_sst2_val_loss + avg_mrpc_val_loss + avg_qqp_val_loss 
                            + avg_mnli_val_loss + avg_rte_val_loss + avg_qnli_val_loss + avg_wnli_val_loss) / 8

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Total Validation took: {:}".format(validation_time))


        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Accur.': avg_train_accuracy,
                'Training Loss': avg_train_loss,
                'COLA Valid. Loss': avg_cola_val_loss,
                'COLA Valid. Accur.': avg_cola_val_accuracy,
                'SST-2 Valid. Loss': avg_sst2_val_loss,
                'SST-2 Valid. Accur.': avg_sst2_val_accuracy,
                'MRPC Valid. Loss': avg_mrpc_val_loss,
                'MRPC Valid. Accur.': avg_mrpc_val_accuracy,
                'QQP Valid. Loss': avg_qqp_val_loss,
                'QQP Valid. Accur.': avg_qqp_val_accuracy,
                'MNLI Valid. Loss': avg_mnli_val_loss,
                'MNLI Valid. Accur.': avg_mnli_val_accuracy,
                'RTE Valid. Loss': avg_rte_val_loss,
                'RTE Valid. Accur.': avg_rte_val_accuracy,
                'QNLI Valid. Loss': avg_qnli_val_loss,
                'QNLI Valid. Accur.': avg_qnli_val_accuracy,
                'WNLI Valid. Loss': avg_wnli_val_loss,
                'WNLI Valid. Accur.': avg_wnli_val_accuracy,
                'Avg Total Valid. Accur.' : avg_tot_val_accuracy,
                'Avg Total Valid. Loss' : avg_tot_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if FLAGS.dynamic_sampling:
            print("Updating weights for dynamic sampling...")
            cola_score, sst2_score, mrpc_score, qqp_score, mnli_score, rte_score, qnli_score, wnli_score = get_scores_from_acc(avg_cola_val_accuracy, avg_sst2_val_accuracy, 
                                        avg_mrpc_val_accuracy, avg_qqp_val_accuracy, avg_mnli_val_accuracy, avg_rte_val_accuracy, avg_qnli_val_accuracy, avg_wnli_val_accuracy)
            inst_sampler.update_weights(cola_score, sst2_score, mrpc_score, qqp_score, mnli_score, rte_score, qnli_score, wnli_score)
            train_dataset = inst_sampler.get_sample()
            train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size 
            )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    pd.set_option('precision', 4)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    df_stats.head(5)
