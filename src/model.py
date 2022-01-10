import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        #resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features
    

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha, attention_weights
        
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def generate_caption(self,features,max_len=20,vocab=None):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        start_label = vocab.start_label
        end_label = vocab.end_label
        
        #starting input
        word = torch.tensor(vocab.word_to_idx[start_label]).view(1,-1).to(device)
        embeds = self.embedding(word)

        alphas = []
        captions = []
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.idx_to_word[predicted_word_idx.item()] == end_label:
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.idx_to_word[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            embed_size=embed_size,
            vocab_size = len(dataset.vocab),
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    
class ModelSaver():
    """
    Stores and loads model, criterion and optimizer
    """
    def __init__(self):
        # Initialize settigns
        
        self.file_name=None
        #hyper params
        self.embed_size = None
        self.vocab_size = None
        self.attention_dim = None
        self.encoder_dim = None
        self.decoder_dim = None
        self.learning_rate = None
        # todo learning rate decay?
        self.pad_idx = None
        self.epoch = None
        
        self.train_loss = []
        self.val_loss = []
            
    def initialize(self,
                 file_name,
                 device,
                 embed_size,
                 vocab_size,
                 attention_dim,
                 encoder_dim,
                 decoder_dim,
                 learning_rate,
                 #epoch=0,
                 pad_idx=0):
        
        if len(file_name) > 4 and file_name[-4:] == '.pth':
            self.file_name = file_name
        else:
            self.file_name = file_name + '.pth'
        #hyper params
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.learning_rate = learning_rate
        # todo learning rate decay?
        
        self.pad_idx = pad_idx
        
        self.epoch = 0
        
        model = self.__get_model(device)
        criterion = self.__get_criterion()
        optimizer = self.__get_optimizer(model)
        
        return model, criterion, optimizer
        
    def load(self, file_name, device):
        
        model_settings = torch.load(file_name)
        
        self.file_name = file_name
        self.embed_size = model_settings['embed_size']
        self.vocab_size = model_settings['vocab_size']
        self.attention_dim = model_settings['attention_dim']
        self.encoder_dim = model_settings['encoder_dim']
        self.decoder_dim = model_settings['decoder_dim']
        self.learning_rate = model_settings['learning_rate']
        self.pad_idx = model_settings['pad_idx']
        self.epoch = model_settings['epoch']
        
        self.train_loss = model_settings['train_loss']
        self.val_loss = model_settings['val_loss']
        
        model = self.__get_model(device)
        model.load_state_dict(model_settings['state_dict'])
        
        criterion = self.__get_criterion()
        optimizer = self.__get_optimizer(model)
        
        return model, criterion, optimizer
        
    def save(self, state_dict, train_loss=None, val_loss=None):
        """
        updates state_dict
        increase epoch by 1
        """
        self.epoch += 1
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        
        model_state = {
            'epoch':self.epoch,
            'state_dict':state_dict,
            
            'embed_size':self.embed_size,
            'vocab_size':self.vocab_size,
            'attention_dim':self.attention_dim,
            'encoder_dim':self.encoder_dim,
            'decoder_dim':self.decoder_dim,
            'learning_rate':self.learning_rate,
            
            'pad_idx':self.pad_idx,
            'train_loss':self.train_loss,
            'val_loss':self.val_loss,
        }
        
        torch.save(model_state, self.file_name)
        
    def __get_model(self, device):
        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=self.vocab_size,
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim
        ).to(device)
        
        return model
    
    def __get_criterion(self):
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        return criterion
    
    def __get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return optimizer
    
        
    
    
    