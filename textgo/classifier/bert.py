# coding: utf-8

import os
import sys
import tqdm
import time, datetime
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

class Bert():
    def __init__(self):
        self.device = self.get_device()

    def get_device(self):
        # If there's a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")

            logger.info('There are %d GPU(s) available.' ,torch.cuda.device_count())

            logger.info('We will use the GPU:%s' ,torch.cuda.get_device_name(0))
        else:
            logger.info('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    def print_model_structure(self,model):
        '''Print model structure.
        Input:
            model: pytorch bert model object.
        '''
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


    def train(self, X1_train, y_train, args, X2_train=None, X1_val=None, X2_val=None, y_val=None):
        '''Train bert.
        Input:
            X1_train: list of text strings 
            X2_train: list of text strings. If X2_train is not None, then run text-pair 
        classification 
            y_train: list of strings. Labels for classification.
            args: dict, params 
        Output:
            best_model: pytorch bert model
            tokenizer: pytorch bert tokenizer
            training_stat: DataFrame, training status records 
        '''
        # Set params
        max_len = args['max_len'] # max length of input text string
        batch_size = args['batch_size'] # size of batch
        num_labels = args['num_labels'] # num of classes
        learning_rate = args['learning_rate'] # learning rate
        epochs = args['epochs'] # num of training epochs
        evaluation_steps= args['evaluation_steps'] # evaluate every num of steps
        val_metric = args['val_metric'] # metric to choose best model, default "val_loss"
        val_threshold = args['val_threshold'] # threshold to choose best model
        model_path = args['pretrained_model'] # pretrained model path
        output_dir = args['output_dir'] # output model path/dir
        seed_val = 42 # random seed for reproduction

        # Load the BERT tokenizer.
        logger.info('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(model_path)
        logger.info('Get train dataloader...')
        train_dataloader = self.get_dataloader(tokenizer,X1_train,X2_train,y_train,max_len,batch_size,True) 

        model = BertForSequenceClassification.from_pretrained(
            model_path, # Use the 12-layer BERT model
            num_labels = num_labels, # The number of output labels--2 for binary classification. # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        if torch.cuda.is_available(): 
            model.cuda()

        self.print_model_structure(model)
        
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        optimizer = AdamW(model.parameters(),
                  lr = learning_rate, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

        # Create the learning rate scheduler.
        total_steps = epochs*len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
        # Train
        # Set the seed value all over the place to make this reproducible.
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []
        best_score = -999
        min_val_loss = 999
        best_epoch = 0
        best_step = 0
        best_model = None

        # For each epoch...
        for epoch_i in range(0, epochs):
            # Perform one full pass over the training set.
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            logger.info('Training...')
            # Measure how long the training epoch takes.
            t0 = time.time()
            # Reset the total loss for this epoch.
            total_train_loss = 0
            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train() 
            # For each batch of training data...
            for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
                loss = self.one_pass(batch, model, optimizer, scheduler)
                total_train_loss += loss.item()
                # Evaluate every evaluation_steps
                if (step+1)%evaluation_steps==0 and evaluation_steps!=-1: 
                    print("Epoch %i, Steps %i:"%(epoch_i,step))
                    stat = {
                        "epochs":epoch_i,
                        "steps":step,
                        "train_loss":total_train_loss/(step+1)}
                    print('Train loss: %.3f'%stat['train_loss'])
                    if X1_val is not None:
                        logger.info("Evaluate on dev dataset...")
                        val_output_dict,val_accuracy,val_loss = self.evaluate(X1_val,y_val,X2_val,model,tokenizer,batch_size=batch_size,max_len=max_len,num_labels=num_labels)
                        stat["val_loss"] = val_loss
                        stat["val_accuracy"] = val_accuracy
                        stat["val_class0_precision"] = val_output_dict['0']['precision']
                        stat["val_class0_recall"] = val_output_dict['0']['recall']
                        stat["val_class0_f1"] = val_output_dict['0']['f1-score']
                
                        if val_metric=='val_loss':
                            if stat['val_loss']<min_val_loss: 
                                min_val_loss = stat['val_loss']
                                best_epoch = epoch_i
                                best_step = step
                                best_model = model
                            print("Min val loss: %.3f"%min_val_loss)
                        else:
                            if stat[val_metric]>best_score:
                                best_score = stat[val_metric]
                                best_epoch = epoch_i
                                best_step = step
                                if best_score >= val_threshold:
                                    best_model = model
                    training_stats.append(stat)
            # Calculate the average loss over all of the batches in one epoch
            avg_train_loss = total_train_loss / len(train_dataloader)            
            print('Avg train loss: %.3f'%avg_train_loss)

        logger.info('Finish training!')
        
        # Save best model and training stats
        training_stats = pd.DataFrame(training_stats)
        if best_model is None:
            best_model = model
        else:
            training_stats['is_best'] = training_stats.apply(lambda x:True if x['epochs']==best_epoch and x['steps']==best_step else False,axis=1)
        self.save_model(best_model,tokenizer,output_dir)
        path = output_dir+'/training_stats.csv'
        training_stats.to_csv(path,index=0)
        print('Training stats:')
        print(training_stats)
        logger.info('Save to: %s',path)
        return best_model, tokenizer, training_stats

    def one_pass(self, batch, model, optimizer, scheduler):
        '''One training step for each batch, including forward and backward.
        '''
        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        if len(batch)==3: 
            b_token_type_ids = None
            b_labels = batch[2].to(self.device)
        else: # text-pair
            b_token_type_ids = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        loss, logits = model(b_input_ids, 
                             token_type_ids=b_token_type_ids, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
            
        return loss

    def evaluate(self,X1,y,X2=None,model=None,tokenizer=None,model_path='',batch_size=16,max_len=256,num_labels=2):
        '''Evaluate bert model.
        Input:
            X1: list of text strings. 
            y: list of label strings.
            X2: list of text strings. If X2 is not None, then run text-pair classification.
            model: pytorch bert model.
            tokenizer: pytorch bert tokenizer.
            model_path: bert model path.
            batch_size: size of batch.
            max_len: max length of input string.
            num_labels: number of classes.
        Output:
            report_dict: dict, classification report.
            accuracy: float.
            avg_loss: float, average loss.
        '''
        # Load model
        if model is None and model_path!='':
            model = BertForSequenceClassification.from_pretrained(
                model_path, # Use the 12-layer BERT model
                num_labels = num_labels, # The number of output labels--2 for binary classification
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False # Whether the model returns all hidden-states.
                )
        if torch.cuda.is_available():
            model.cuda()
        # Load tokenizer
        if tokenizer is None and model_path!='':
            tokenizer = BertTokenizer.from_pretrained(model_path)
        total_loss = 0
        dataloader = self.get_dataloader(tokenizer,X1,X2,y,max_len,batch_size,False)
        labels = []
        preds = []
        for batch in tqdm.tqdm(dataloader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            if X2 is None:
                b_token_type_ids = None
                b_labels = batch[2].to(self.device)
            else:
                b_token_type_ids = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)
            with torch.no_grad():
                loss, logits = model(b_input_ids, 
                             token_type_ids=b_token_type_ids, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
            total_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to('cpu').numpy().flatten()
            preds.extend(pred)
            labels.extend(label_ids)
        avg_loss = total_loss/len(dataloader)
        print(classification_report(labels,preds))
        report_dict = classification_report(labels,preds,output_dict=True)
        accuracy = accuracy_score(labels,preds)
        print("Accuracy: %.3f"%accuracy) 
        print("Loss: %.3f"%avg_loss)
        return report_dict, accuracy, avg_loss

    def predict(self,X1,X2=None,model=None,tokenizer=None,model_path='',batch_size=16,max_len=256,num_labels=2):
        '''Predict with bert.
        Input:
            X1: list of text strings. 
            X2: list of text strings. If X2 is not None, then run text-pair classification.
            model: pytorch bert model. 
            tokenizer: pytorch bert tokenizer. 
            model_path: bert model path. 
            batch_size: size of batch. 
            max_len: max length of input string. 
            num_labels: number of classes.
        Output:
            predpro: list of float. Prediction probability for corresponding class.
            predclass: list of int. Predicted class.
        '''
        # Load model
        if model is None and model_path!='':
            model = BertForSequenceClassification.from_pretrained(
                model_path, # Use the 12-layer BERT model
                num_labels = num_labels, # The number of output labels--2 for binary classification
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False # Whether the model returns all hidden-states.
                )
        if torch.cuda.is_available():
            model.cuda()
        # Load tokenizer
        if tokenizer is None and model_path!='':
            tokenizer = BertTokenizer.from_pretrained(model_path)
        total_loss = 0
        dataloader = self.get_dataloader(tokenizer,X1,X2,None,max_len,batch_size,False)
        predpro = []
        predclass = []
        for batch in tqdm.tqdm(dataloader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            if X2 is None:
                b_token_type_ids = None
            else:
                b_token_type_ids = batch[2].to(self.device)
            with torch.no_grad():
                res = model(b_input_ids, 
                             token_type_ids=b_token_type_ids, 
                             attention_mask=b_input_mask) 
                logits = res[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            batch_predclass = np.argmax(logits, axis=1).flatten()
            predclass.extend(batch_predclass)
            batch_predpro = softmax(logits, axis=1)[:,1].flatten()
            predpro.extend(batch_predpro)
        return predpro,predclass

    def get_dataloader(self,tokenizer,X1,X2=None,y=None,max_len=256,batch_size=16,shuffle=False):
        '''Get dataloader for bert.
        Input:
            tokenizer: Pytorch Tokenizer Object
            X1: list of text strings
            X2: list of text strings. If X2 is not None, then run text-pair classification
            y: list of strings. Labels for classification. If y is None, then only predict
            max_len: int. Max length for text string
            batch_size: int. Size of each batch
            shuffle: boolean. Whether shuffle the dataset or not. True for train, otherwise False
        Output:
            dataloader: Pytorch dataloader object
        '''
        sentences1 = X1
        sentences2 = X2
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        token_type_ids = []
        # For every sentence...
        for i in tqdm.tqdm(range(len(sentences1))):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            if sentences2 is None:
                text_pair = None
                return_token_type_ids = False
            else:
                text_pair = sentences2[i]
                return_token_type_ids = True
            encoded_dict = tokenizer.encode_plus(
                        sentences1[i], # Sentence1 to encode.
                        text_pair=text_pair, # Sentence2 to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_token_type_ids = return_token_type_ids, # For text pair.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            # Add its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            if sentences2 is not None:
                # Add token type ids
                token_type_ids.append(encoded_dict['token_type_ids'])
         
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        if sentences2 is not None:
            token_type_ids = torch.cat(token_type_ids, dim=0)
            if y is not None:
                labels = torch.tensor(labels)
                # Combine the training inputs into a TensorDataset.
                dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
            else:
                dataset = TensorDataset(input_ids, attention_masks, token_type_ids)
        else:
            if y is not None:
                labels = torch.tensor(y)
                dataset = TensorDataset(input_ids, attention_masks, labels)
            else:
                dataset = TensorDataset(input_ids, attention_masks)
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset) 
        dataloader = DataLoader( 
            dataset,  
            sampler = sampler, 
            batch_size = batch_size
            )
        return dataloader

    def save_model(self,model,tokenizer,output_dir):
        '''Save model.
        Input:
            model: pytorch bert model object.
            tokenizer: bert tokenizer object.
            output_dir: string. Path to save model.
        '''
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
