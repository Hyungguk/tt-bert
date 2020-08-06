import transformers
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertModel
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
import pandas as pd
from sklearn.metrics import matthews_corrcoef


flags =  tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_float('epsilon', 1e-8, 'epsilon.')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs when training.')
flags.DEFINE_integer('batch_size', 32, 'Size of each batch. Default is 32.')
flags.DEFINE_string('dtype', 'cola', 'Which GLUE dataset to fine-tune with.')
flags.DEFINE_string('saved_model_path', None, 'Path of the saved model to start training with.')
flags.DEFINE_boolean('redundant', False, 'Use redundant target tokens?')
flags.DEFINE_boolean('save', True, 'Save trained model?')
flags.DEFINE_string('bert_type', 'bert-base-uncased', 'Type of bert model to use.')


NEW_SPEC_TOKENS = ['[ACCEPTABLE]', '[UNACCEPTABLE]', 
                   '[POSITIVE]', '[NEGATIVE]', 
                   '[PARAPHRASE]', '[NOT_PARAPHRASE]', 
                   '[SIMILAR]', '[NOT_SIMILAR]',
                   '[ENTAILMENT]', '[NEUTRAL]', '[CONTRADICTION]',
                   '[ANSWERABLE]', '[NOT_ANSWERABLE]',
                   '[REFERENT]', '[NOT_REFERENT]']

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels, mc=False):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    if mc:
        return matthews_corrcoef(labels_flat, pred_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




if __name__ == '__main__':
    dtype = FLAGS.dtype
    if dtype == 'cola':
        mc = True
    else:
        mc = False
    save = FLAGS.save
    print("")
    print("!!!! We will start single task fine-tuning on " + dtype + " task!!!!")
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    bert_type = FLAGS.bert_type
    redundant = FLAGS.redundant
    # ========================================================
    # Call appropriate tokenizer of the Bert model.
    # If there is a saved version, load it.
    # If not, load the BertTokenizer with special tokens addded.
    # Then save it.
    # ========================================================
    if os.path.isdir('./tokenizer/' +  bert_type + '_custom-tok'):       # Check if we already have a customzied tokenizer.
        tokenizer = BertTokenizer.from_pretrained('./tokenizer/' +  bert_type + '_custom-tok') # If we do, load it. 
        print('Loaded tokenizer from local...')   
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type, do_lower_case=True, additional_special_tokens=NEW_SPEC_TOKENS)
        print('Loaded tokenizer from hugginface...')   
        tokenizer.save_pretrained('./tokenizer/' +  bert_type + '_custom-tok')
        print('Saved tokenizer to local...')   

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
    
    # =========================================================
    # Load dtype datasets and make dataloaders out of them.
    # =========================================================
    train_dataset, val_dataset = get_train_val_datasets(dtype)

    train_dataset = train_dataset.map(make_encoder(dtype, tokenizer), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])

    val_dataset = val_dataset.map(make_encoder(dtype, tokenizer), batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'pointer_mask'])

  
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size 
            )

    val_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )


    # ====================================================
    #   Set up optimizer and scheduler for training.
    # ====================================================
    optimizer = AdamW(model.parameters(),
                  lr =  FLAGS.learning_rate, 
                  eps = FLAGS.epsilon
                )

    epochs = FLAGS.num_epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    print("-------Start training on {:} train_dataset and {:} val_dataset------".format(len(train_dataset), len(val_dataset)))
    training_stats = []
    total_t0 = time.time()
    last_val_accuracy = 0
    last_model_path = None
    for epoch_i in range(0, epochs):
        
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

            # Progress update every 1000 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)                
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

            total_train_accuracy += flat_accuracy(logits, label_ids, mc)
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

        print("")
        print("Running Validations...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        cola_pred = []
        cola_true = []
        for batch in val_dataloader:
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
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids, mc)

            if dtype == 'cola':
                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()  
                cola_pred = np.hstack([cola_pred, pred_flat])
                cola_true = np.hstack([cola_true, labels_flat]) 

        if dtype == 'cola':
            matthews = matthews_corrcoef(cola_true,cola_pred)
            print("  Cola Matthews: {0:.3f}".format(matthews))    
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Save current model if its validation accuracy is higher 
        # than that of the last one.
        if save and avg_val_accuracy > last_val_accuracy:
            save_path = './saved_models/' + bert_type + '/single/' + dtype +'/'
            if FLAGS.saved_model_path != None:
                save_path += ('/' + FLAGS.saved_model_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            model_name = bert_type + '_' + dtype + '_lr={0:.2e}'.format(FLAGS.learning_rate) + 'bs={:}'.format(FLAGS.batch_size) + 'epoch={0:.2f}'.format(epoch_i+1) + '_val_acc={0:.2f}'.format(avg_val_accuracy)
            if redundant:
                model_name = 'red_' + model_name
            model_path = save_path + model_name + '.pt'
            torch.save(model, model_path)
            last_val_accuracy = avg_val_accuracy
            
            # Delete outdated saved model.
            if last_model_path != None:
              os.remove(last_model_path)
            
            last_model_path = model_path
            

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Accur.': avg_train_accuracy,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    #  Save training_stats in csv file format.
    pd.set_option('precision', 4)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    if not os.path.isdir(save_path + 'model_stats/'):
        os.mkdir(save_path + 'model_stats/')

    df_stats.to_csv(save_path + 'model_stats/' + dtype + '_single.csv')
    