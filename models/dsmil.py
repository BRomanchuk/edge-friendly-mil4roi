import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
    
    def train_step(self, x, y):
        self.train()
        classes, prediction_bag, A, B = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(prediction_bag.view(-1), y.view(-1))
        return {"BCE": loss}
    
    def val_step(self, x, y):
        self.eval()
        with torch.no_grad():
            classes, prediction_bag, A, B = self.forward(x)
            loss = F.binary_cross_entropy_with_logits(prediction_bag.view(-1), y.view(-1))
        return {"BCE": loss}
    

class BatchMILNet(nn.Module):
    def __init__(self, mil_net):
        super(BatchMILNet, self).__init__()
        self.mil_net = mil_net
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, batch_of_bags):
        classes, prediction_bags, As, Bs = [], [], [], []
        for x in batch_of_bags:
            classes_bag, prediction_bag, A, B = self.mil_net(x)
            classes.append(classes_bag)
            prediction_bags.append(prediction_bag)
            As.append(A)
            Bs.append(B)
        classes = torch.stack(classes)
        prediction_bags = torch.stack(prediction_bags)
        As = torch.stack(As)
        Bs = torch.stack(Bs)
        return classes, prediction_bags, As, Bs
    
    def train_step(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        classes, prediction_bag, A, B = self(x)
        loss = F.binary_cross_entropy_with_logits(prediction_bag.view(-1), y.view(-1))
        loss.backward()
        self.optimizer.step()
        return {"BCE": loss}
    
    def val_step(self, x, y):
        self.eval()
        with torch.no_grad():
            classes, prediction_bag, A, B = self(x)
            loss = F.binary_cross_entropy_with_logits(prediction_bag.view(-1), y.view(-1))
        return {"BCE": loss}
    
    def is_better(self, current_losses, best_losses):
        if len(best_losses) == 0:
            return True
        return current_losses["BCE"] < best_losses["BCE"]