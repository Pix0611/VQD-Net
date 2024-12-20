import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Loss_bc import HardTripletLoss
from models.networks.UsdlHead import *

class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
    
    def forward(self, pred, confidence, tgt):

        uncertainty = (1 - confidence)**2
        B = pred.shape[0]

        return torch.sum(((pred-tgt)**2)*1/(uncertainty**2+1e-6) + uncertainty**2) / B

class VqLayer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, 
                 compute_confidence=False, quantized_output=False):
        super(VqLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.compute_confidence = compute_confidence
        self.quantized_output = quantized_output


        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)  
                     - 2 * torch.matmul(flat_x, self.embedding.weight.T)  
                     + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True)) 


        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view_as(x)


        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        if not self.quantized_output:

            similarities = -distances
            weights = F.log_softmax(similarities, dim=1)

            weighted_output = torch.matmul(weights, self.embedding.weight)
            weighted_output = weighted_output.view_as(x)


        output_vector = quantized if self.quantized_output else (weighted_output+quantized) / 2


        confidence = None
        if self.compute_confidence:
            if not self.quantized_output:

                entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
                confidence = 1 - entropy / torch.log(torch.tensor(self.num_embeddings, device=x.device))
                confidence = confidence.view(x.size(0), x.size(1))
            else:

                nearest_distance = distances.gather(1, encoding_indices).squeeze(1)
                max_distance = distances.max(dim=1)[0]
                confidence = 1 - nearest_distance / max_distance
                confidence = confidence.view(x.size(0), x.size(1))


        output_vector = x + (output_vector - x).detach()

        return output_vector, confidence, vq_loss
    
    
    def weighted_inference(self, x, similarity_metric='cosine'):

        flat_x = x.view(-1, self.embedding_dim)

        if similarity_metric == 'cosine':

            norm_x = F.normalize(flat_x, dim=1)
            norm_embeddings = F.normalize(self.embedding.weight, dim=1)
            similarities = torch.matmul(norm_x, norm_embeddings.T)

        elif similarity_metric == 'negative_distance':

            similarities = -((torch.sum(flat_x ** 2, dim=1, keepdim=True)
                              - 2 * torch.matmul(flat_x, self.embedding.weight.T)
                              + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True)))
        else:
            raise ValueError("Unsupported similarity_metric. Choose 'cosine' or 'negative_distance'.")


        weights = F.log_softmax(similarities, dim=1)




        return weights
    
    
    
    def calc_loss(self, x, output_vector, vq_loss):

        recon_loss = F.mse_loss(output_vector, x)

        total_loss = recon_loss + vq_loss
        return total_loss


class MainModel(nn.Module):
    def __init__(self, input_feature_dim=1024, hidden_dim=256, nhead=4, num_layers=4, decopuling_dim=128, num_quantized_embedding=64):

        super(MainModel, self).__init__()
        

        self.decoupling_T = nn.Sequential(



            nn.Linear(input_feature_dim, decopuling_dim)
        )
        self.decoupling_P = nn.Sequential(



            nn.Linear(input_feature_dim, decopuling_dim)
        )
        


        self.vector_quantized_T = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=False)
        self.vector_quantized_P = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=True)
        


        

        self.weight_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.clip_score_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decopuling_dim*2,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                activation="relu"
            ),
            num_layers=num_layers
        )
        

        self.score_regressor = nn.Sequential(
            nn.Linear(decopuling_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.confidence_regressor = nn.Sequential(
            nn.Linear(decopuling_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.pred_loss = ModelLoss()
        self.tri_loss = HardTripletLoss(margin=0.5, hardest=True)

    def forward(self, f, tgt=None):

        B, L, E = f.shape
        loss = 0.0
        confidence = 0.0
        
        f = f.reshape(B*L, -1)
        T = self.decoupling_T(f).reshape(B, L, -1)
        P = self.decoupling_P(f).reshape(B, L, -1)
        
        T0, P0 = T.reshape(B*L, -1), P.reshape(B*L, -1)
        output_seperation = torch.concat([T0, P0], dim=-1)
        seperation_label = torch.concat([torch.zeros(B*L,1), torch.ones(B*L,1)], dim=-1)

        loss_tri = self.tri_loss(output_seperation, seperation_label)
        
        loss += loss_tri
        
        T, condidence_T, loss_T = self.vector_quantized_T(T)
        P, confidence_P, loss_P = self.vector_quantized_P(P)

        loss += loss_T
        loss += loss_P
        
        

        use_transformer = True
        if use_transformer:
            encoding = torch.cat([T, P], dim=-1)

            f_transformer = encoding.permute(1, 0, 2)
            

            f_transformer = self.transformer_encoder(f_transformer)
        

            f_transformer = f_transformer[-1, :, :]
            pred = self.score_regressor(f_transformer).squeeze()

            confidence = self.confidence_regressor(f_transformer).squeeze()
            confidence = torch.mean(condidence_T*confidence_P, dim=-1).squeeze()
            
            if tgt is not None:
                loss += self.pred_loss(pred, confidence, tgt)
                


        
        




        





        







        
        return pred, confidence, loss


    
if __name__ == '__main__':
    device = torch.device('cuda:0')
    













    

    model = MainModel().to(device)
    inputs = torch.randn(4, 192, 1024).to(device)
    tgt = torch.randn(4, 1).to(device)
    outputs, confidence, loss = model(inputs, tgt)
    loss.backward()
    print(outputs.shape, confidence.shape, loss)