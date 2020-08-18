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
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, XLNetConfig, get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from scipy.special import softmax

import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

# import local modules
from .utils import load_config, set_random_seed, get_device, get_time_dif
from .train_eval import categorical_crossentropy_with_prior, get_prior
 
class Model():
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        set_random_seed(self.args['random_seed'])

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


    def train(self, X1_train, y_train, X1_dev=None, y_dev=None, X2_train=None, X2_dev=None, evaluate_test=False):
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
        start_time = time.time()
        # Set params
        max_len = self.args['max_len'] # max length of input text string
        batch_size = self.args['batch_size'] # size of batch
        num_classes = self.args['num_classes'] # num of classes
        learning_rate = self.args['learning_rate'] # learning rate
        epochs = self.args['num_epochs'] # num of training epochs
        evaluation_steps= self.args['evaluation_steps'] # evaluate every num of steps
        val_metric = self.args['val_metric'] # metric to choose best model, default "dev_loss"
        model_path = self.args['pretrained_model'] # pretrained model path
        output_dir = self.args['save_path'] # output model path/dir
        if self.args['loss_with_prior']:
            prior = get_prior(y_train)
        else:
            prior = None
        # Split train/dev/test
        if X2_train is None: # Single sentence Classification
            if X1_dev is None or y1_dev is None: # train: dev = 8:2 
                X1_train, X1_dev, y_train, y_dev = train_test_split(X1_train, y_train, random_state=self.args['random_seed'], test_size=0.2)
            if evaluate_test: # train: dev: test = 8:1:1 
                X1_dev, X1_test, y_dev, y_test = train_test_split(X1_dev, y_dev, random_state=self.args['random_seed'], test_size=0.5)
            X2_dev = None
            X2_test = None
        else: # Sentence pair Classification
            if X1_dev is None or y1_dev is None: # train: dev = 8:2  
                X1_train, X1_dev, X2_train, X2_dev, y_train, y_dev = train_test_split(X1_train, X2_train, y_train, random_state=self.args['random_seed'], test_size=0.2) 
            if evaluate_test: # train: dev: test = 8:1:1  
                X1_dev, X1_test, X2_dev, X2_test, y_dev, y_test = train_test_split(X1_dev, X2_dev, y_dev, random_state=self.args['random_seed'], test_size=0.5)

        # Load the BERT pretrained tokenizer and model
        logger.info('Load BERT tokenizer...')
        tokenizer = self.load_tokenizer(model_path)
        logger.info('Load BERT model...')
        model = self.load_model(model_path)
        logger.info('Get train dataloader...') 
        train_dataloader = self.get_dataloader(tokenizer,X1_train,X2_train,y_train,True,show_process=True)
        
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
        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []
        best_score = -999
        min_dev_loss = float('inf')
        total_steps = 0 # record how many batches/steps already run
        last_improve = 0 # record last improved batch/step (less dev loss)
        flag = False # record whether improve for a long time or not
        best_epoch = 0 
        best_step = 0

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
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                loss = self.one_pass(batch, model, optimizer, scheduler, prior)
                train_loss = loss.item()
                total_train_loss += train_loss
                
                # Evaluate every evaluation_steps
                if (total_steps)%evaluation_steps==0 and evaluation_steps!=-1: 
                    print("Epoch %i, Steps %i:"%(epoch_i,step))
                    stat = {
                        "epochs":epoch_i,
                        "epoch_steps":step,
                        "total_steps":total_steps,
                        #"train_loss":total_train_loss/(step+1)
                        "train_loss":train_loss}
                    if X1_dev is not None:
                        logger.info("Evaluate on dev dataset...")
                        dev_output_dict,dev_acc,dev_loss = self.evaluate(X1_dev,y_dev,X2_dev,model,tokenizer,show_process=True)
                        stat["dev_loss"] = dev_loss
                        stat["dev_accuracy"] = dev_acc
                        #stat["dev_class0_precision"] = dev_output_dict['0']['precision']
                        #stat["dev_class0_recall"] = dev_output_dict['0']['recall']
                        #stat["dev_class0_f1"] = dev_output_dict['0']['f1-score']
                        isImproved = False
                        if val_metric=='dev_loss':
                            if stat['dev_loss']<min_dev_loss: 
                                min_dev_loss = stat['dev_loss']
                                best_epoch = epoch_i
                                best_step = step
                                last_improve = total_steps
                                isImproved = True
                                # save model
                                self.save_model(model,tokenizer,output_dir)
                            time_dif = get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.3}, Val Loss: {2:>6.3}, Val Acc: {3:>6.3%}, Time: {4}, Improved: {5}' 
                            print(msg.format(total_steps, train_loss, dev_loss, dev_acc, time_dif, isImproved))
                            print("Min val loss: %.3f"%min_dev_loss)
                        else:
                            if stat[val_metric]>best_score:
                                best_score = stat[val_metric]
                                best_epoch = epoch_i
                                best_step = step
                                last_improve = total_steps
                                isImproved = True 
                                # save model 
                                self.save_model(model,tokenizer,output_dir)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.3}, Val Loss: {2:>6.3}, {3}: {4:>6.3%}, Time: {5}, Improved: {6}'  
                            print(msg.format(total_steps, train_loss, dev_loss, val_metric, stat[val_metric], time_dif, isImproved))
                    training_stats.append(stat)
                total_steps += 1
                if total_steps - last_improve > self.args['require_improvement']: 
                    # 验证集loss超过1000batch没下降，结束训练 
                    print("No optimization for %i steps, auto-stopping..."%self.args['require_improvement']) 
                    flag = True 
                    break
            if flag:
                break
            # Calculate the average loss over all of the batches in one epoch
            avg_train_loss = total_train_loss / len(train_dataloader)            
            print('Avg train loss: %.3f'%avg_train_loss)

        logger.info('Finish training!')
        
        # Save training stats
        training_stats = pd.DataFrame(training_stats)
        if best_step>0:
            training_stats['is_best'] = training_stats.apply(lambda x:True if x['epochs']==best_epoch and x['epoch_steps']==best_step else False,axis=1)
        path = output_dir+'/training_stats.csv'
        training_stats.to_csv(path,index=0)
        print('Training stats:')
        print(training_stats)
        logger.info('Save to: %s',path)
        
        # Evaluate best model on dev dataset
        if best_step>0:
            print("Load best model...")
            model = self.load_model(output_dir)
        print("Evaluate on dev dataset...") 
        dev_report,dev_acc,dev_loss = self.evaluate(X1_dev,y_dev,X2_dev,model,tokenizer)
        # Evaluate best model on test dataset
        if evaluate_test:
            print("Evaluate on test dataset...")
            test_report,test_acc,test_loss = self.evaluate(X1_test,y_test,X2_test,model,tokenizer)
            return test_report,test_acc

    def one_pass(self, batch, model, optimizer, scheduler, prior=None):
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
        model.train()
        model.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        if prior is not None:
            outputs = model(b_input_ids,  
                             token_type_ids=b_token_type_ids,  
                             attention_mask=b_input_mask)
            logits = outputs[0]
            loss = categorical_crossentropy_with_prior(b_labels, logits, prior, self.device, tau=1.0)
        else:
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

    def evaluate(self,X1,y,X2=None,model=None,tokenizer=None,model_path='',show_process=False):
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
        batch_size = self.args['batch_size'] 
        max_len = self.args['max_len'] 
        num_classes = self.args['num_classes'] 
        # Load model
        if model is None and model_path!='':
            model = self.load_model(model_path)
        # Load tokenizer
        if tokenizer is None and model_path!='':
            tokenizer = self.load_tokenizer(model_path)
        model.eval()
        total_loss = 0
        dataloader = self.get_dataloader(tokenizer,X1,X2,y,False)
        labels = []
        preds = []
        iters = dataloader 
        if show_process: 
            iters = tqdm.tqdm(dataloader)
        for batch in iters:
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
        print(classification_report(labels,preds,digits=3))
        report_dict = classification_report(labels,preds,output_dict=True,digits=3)
        accuracy = accuracy_score(labels,preds)
        print("Accuracy: %.3f"%accuracy) 
        print("Loss: %.3f"%avg_loss)
        return report_dict, accuracy, avg_loss

    def predict(self,X1,X2=None,model=None,tokenizer=None,model_path='',show_process=False):
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
        batch_size = self.args['batch_size']
        max_len = self.args['max_len']
        num_classes = self.args['num_classes'] 
        # Load model
        if model is None: 
            if model_path=='':
                model_path = self.args['save_path']
            model = self.load_model(model_path)
        # Load tokenizer
        if tokenizer is None:
            if model_path=='':
                model_path = self.args['save_path']
            tokenizer = self.load_tokenizer(model_path)
        model.eval()
        total_loss = 0
        dataloader = self.get_dataloader(tokenizer,X1,X2,None,False)
        predpro = []
        predclass = []
        iters = dataloader 
        if show_process: 
            iters = tqdm.tqdm(dataloader)
        for batch in iters:
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
        return predclass

    def get_dataloader(self,tokenizer,X1,X2=None,y=None,shuffle=False,show_process=False):
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
        batch_size = self.args['batch_size']
        max_len = self.args['max_len']
        sentences1 = X1
        sentences2 = X2
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        token_type_ids = []
        iters = range(len(sentences1))
        if show_process:
            iters = tqdm.tqdm(iters)
        # For every sentence...
        for i in iters:
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
                        truncation=True
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
            token_type_ids =XLNetch.cat(token_type_ids, dim=0)
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
    
    def load_model(self,model_path):
        model = XLNetForSequenceClassification.from_pretrained(
                model_path, # Use the 12-layer BERT model
                num_labels = self.args['num_classes'], # The number of output labels--2 for binary classification
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False # Whether the model returns all hidden-states.
                )
        if torch.cuda.is_available():
            model.cuda()
        return model

    def load_tokenizer(self,model_path):
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        return tokenizer

if  __name__ == '__main__':
    # load data 
    import pandas as pd 
    data = pd.read_csv('data/2_categories_data.csv')   
    X_train = data['text'].tolist()  
    y_train = data['label'].tolist() 
    # load config 
    config_path = "./config.ini" 
    model_name = "Bert" 
    args = load_config(config_path, model_name) 
    args['model_name'] = model_name 
    args['save_path'] = "output/%s"%model_name  
    print(args)
    # train
    clf = Model(args)
    clf.train(X_train, y_train, evaluate_test=True)
    predclass = clf.predict(X_train)
