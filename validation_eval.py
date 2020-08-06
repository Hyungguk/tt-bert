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
from sklearn.metrics import matthews_corrcoef

flags =  tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_float('epsilon', 1e-8, 'epsilon.')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs when training.')
flags.DEFINE_integer('batch_size', 32, 'Size of each batch. Default is 32.')
flags.DEFINE_string('saved_model_path', None, 'Path of the saved model to start training with.')
flags.DEFINE_boolean('dynamic', False, 'Use dynamic sampling?')
flags.DEFINE_boolean('redundant', False, 'Use redundant target tokens?')
flags.DEFINE_string('bert_type', 'bert-base-uncased', 'Type of bert model to use.')

NEW_SPEC_TOKENS = ['[ACCEPTABLE]', '[UNACCEPTABLE]', 
                   '[POSITIVE]', '[NEGATIVE]', 
                   '[PARAPHRASE]', '[NOT_PARAPHRASE]', 
                   '[SIMILAR]', '[NOT_SIMILAR]',
                   '[ENTAILMENT]', '[NEUTRAL]', '[CONTRADICTION]',
                   '[ANSWERABLE]', '[NOT_ANSWERABLE]',
                   '[REFERENT]', '[NOT_REFERENT]']

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


if __name__ == '__main__':
    print("!!!! We will start evalating our model on all 8 glue tasks (dev set)!!!!")
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    bert_type = FLAGS.bert_type
    if 'red' in FLAGS.saved_model_path:
        redundant = True
    else:
        redundant = False
    # ========================================================
    # Call appropriate tokenizer of the Bert model.
    # If there is a saved version, load it.
    # If not, load a new pretrained BertTokenizer with 
    # special tokens addded. Then save it.
    # ========================================================

    if os.path.isdir('./tokenizer/' +  bert_type + '_custom-tok'):       # Check if we already have a customzied tokenizer.
        tokenizer = BertTokenizer.from_pretrained('./tokenizer/' +  bert_type + '_custom-tok') # If we do, load it. 
        print('Loaded tokenizer from local...')   
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type, do_lower_case=True, additional_special_tokens=NEW_SPEC_TOKENS)
        print('Loaded tokenizer from hugginface...')   
        tokenizer.save_pretrained('./tokenizer/' +  bert_type + '_custom-tok')

    # =========================================================
    # Call Pointer Bert model that uses Bert as its embedding
    # and has an extra dot-producting layer that 
    # finds the right answer among the added task tokens.
    # =========================================================
    if FLAGS.saved_model_path == None:
        bert = BertModel.from_pretrained(bert_type)
        bert.resize_token_embeddings(len(tokenizer))
        model = PointerBert(bert)
        
    else:
        model = torch.load(FLAGS.saved_model_path)
        saved_model_name = FLAGS.saved_model_path.split('/')[-1]
        
    if torch.cuda.is_available():
            model.cuda()

    batch_size = FLAGS.batch_size

    # =================================================
    # Call every Glue dataset.
    # We tokenize every dataset with the tokenizer
    # we just loaded.
    # =================================================
        
    cola_train_dataset, cola_val_dataset = get_train_val_datasets('cola', redundant)
    sst2_train_dataset, sst2_val_dataset = get_train_val_datasets('sst2', redundant)
    mrpc_train_dataset, mrpc_val_dataset = get_train_val_datasets('mrpc', redundant)
    qqp_train_dataset, qqp_val_dataset = get_train_val_datasets('qqp', redundant)
    mnli_train_dataset, mnli_val_dataset = get_train_val_datasets('mnli', redundant)
    rte_train_dataset, rte_val_dataset = get_train_val_datasets('rte', redundant)
    qnli_train_dataset, qnli_val_dataset = get_train_val_datasets('qnli', redundant)
    wnli_train_dataset, wnli_val_dataset = get_train_val_datasets('wnli', redundant)

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
        
    cola_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    sst2_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mrpc_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qqp_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    mnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    rte_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    qnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])
    wnli_val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])



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

    
    training_stats = []
    total_t0 = time.time()
    

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


    cola_pred = []
    cola_true = []
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

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()  
        cola_pred = np.hstack([cola_pred, pred_flat])
        cola_true = np.hstack([cola_true, labels_flat]) 

    avg_cola_val_accuracy = total_cola_eval_accuracy / len(cola_val_dataloader)
    matthews = matthews_corrcoef(cola_true,cola_pred)

    print("  Cola Matthews: {0:.3f}".format(matthews))
    print("  Cola Accuracy: {0:.3f}".format(avg_cola_val_accuracy))

    avg_cola_val_loss = total_cola_eval_loss / len(cola_val_dataloader)
    print("  Cola Validation Loss: {0:.3f}".format(avg_cola_val_loss))

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
    print("  SST-2 Accuracy: {0:.3f}".format(avg_sst2_val_accuracy))

    avg_sst2_val_loss = total_sst2_eval_loss / len(sst2_val_dataloader)
    print("  SST-2 Validation Loss: {0:.3f}".format(avg_sst2_val_loss))

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
    print("  MRPC Accuracy: {0:.3f}".format(avg_mrpc_val_accuracy))

    avg_mrpc_val_loss = total_mrpc_eval_loss / len(mrpc_val_dataloader)
    print("  MRPC Validation Loss: {0:.3f}".format(avg_mrpc_val_loss))

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
    print("  QQP Accuracy: {0:.3f}".format(avg_qqp_val_accuracy))

    avg_qqp_val_loss = total_qqp_eval_loss / len(qqp_val_dataloader)
    print("  QQP Validation Loss: {0:.3f}".format(avg_qqp_val_loss))

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
    print("  MNLI Accuracy: {0:.3f}".format(avg_mnli_val_accuracy))

    avg_mnli_val_loss = total_mnli_eval_loss / len(mnli_val_dataloader)
    print("  MNLI Validation Loss: {0:.3f}".format(avg_mnli_val_loss))

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
    print("  RTE Accuracy: {0:.3f}".format(avg_rte_val_accuracy))

    avg_rte_val_loss = total_rte_eval_loss / len(rte_val_dataloader)
    print("  RTE Validation Loss: {0:.3f}".format(avg_rte_val_loss))

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
    print("  QNLI Accuracy: {0:.3f}".format(avg_qnli_val_accuracy))

    avg_qnli_val_loss = total_qnli_eval_loss / len(qnli_val_dataloader)
    print("  QNLI Validation Loss: {0:.3f}".format(avg_qnli_val_loss))

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
    print("  WNLI Accuracy: {0:.3f}".format(avg_wnli_val_accuracy))

    avg_wnli_val_loss = total_wnli_eval_loss / len(wnli_val_dataloader)
    print("  WNLI Validation Loss: {0:.3f}".format(avg_wnli_val_loss))


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

print("")
print("Dev set evaluation complete!")


