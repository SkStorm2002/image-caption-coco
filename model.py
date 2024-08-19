import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # super(DecoderRNN, self).__init__()
        # self.embed_size = embed_size
        # self.hidden_size = hidden_size
        # self.vocab_size = vocab_size
        # self.num_layers = num_layers
        super().__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size,embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size ,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        self.linear = nn.Linear(hidden_size,vocab_size)
          
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embed = self.embedding_layer(captions)
#         features = features.view(self.batch_size,1,-1)
#         self.hidden = self.init_hid_state(features.size(0)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embed), dim =1)
        
        lstm_outputs,_ = self.lstm(inputs)
        
#         lstm_outputs_shape = lstm_outputs.shape
#         lstm_outputs_shape = list(lstm_outputs_shape)
        
#         vocab_outputs = self.linear(lstm_outputs)
#         vocab_outputs = vocab_outputs.reshape(lstm_outputs_shape[0], lstm_outputs_shape[1], -1)
        vocab_outputs = self.linear(lstm_outputs)
        
        return vocab_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pred_seq = []
        length = 0
        
        while(length != max_len +1):
            output , states = self.lstm(inputs,states)
            output = output.squeeze(dim=1)
            output = self.linear(output)
            
            _,pred_index = torch.max(output, 1)
            #referred this, since its. a cuda tensor it needs to be first converted to cpu then numpy
#             pred_index = pred_index.cpu()
#             pred_index = pred_index.numpy()[0]
            pred_seq.append(pred_index.cpu().numpy()[0].item())
            
            #ending statement , <end> index = 1
            if pred_index == 1:
                break
                
#           output of prev unit is the input for current unit
            inputs = self.embedding_layer(pred_index)
            inputs = inputs.unsqueeze(1)
            
            length+=1
        return pred_seq