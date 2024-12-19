import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
    
    def forward(self, pred, confidence, tgt):
        # pred, confidence, tgt: B*1
        B = pred.shape[0]
        return torch.sum((pred-tgt)**2) / B

class VqLayer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, 
                 compute_confidence=False, quantized_output=False):
        """
        Args:
            embedding_dim (int): Dimensionality of the input embedding (last dimension of input tensor).
            num_embeddings (int): Number of discrete embeddings in the codebook.
            commitment_cost (float): Weight for the commitment loss term.
            compute_confidence (bool): Whether to compute confidence.
            quantized_output (bool): IF `False`, use similarity-weighted outputs instead of nearest neighbor.
        """
        super(VqLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.compute_confidence = compute_confidence
        self.quantized_output = quantized_output

        # Codebook: Randomly initialized embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
        
        Returns:
            output_vector (Tensor): Quantized or weighted output tensor.
            confidence (Tensor or None): Confidence values if compute_confidence=True, else None.
            vq_loss (Tensor): Vector quantization loss.
        """
        # Flatten the input to (batch_size * seq_length, embedding_dim) for easier processing
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances or similarities between input vectors and embedding vectors
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)  # (N, 1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.T)  # (N, num_embeddings)
                     + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True))  # (1, num_embeddings)

        # Calculate nearest neighbor (for quantized output)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (N, 1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)  # (N, num_embeddings)
        quantized = torch.matmul(encodings, self.embedding.weight)  # (N, embedding_dim)
        quantized = quantized.view_as(x)  # Reshape to match original input shape

        # Loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), x)  # Loss for embedding commitment
        q_latent_loss = F.mse_loss(quantized, x.detach())  # Loss for embedding optimization
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        if not self.quantized_output:
            # Weighted output (optional)
            similarities = -distances  # Negative distances as similarities
            weights = F.log_softmax(similarities, dim=1)  # Softmax over similarities
            # weights = self.weighted_inference(x)
            weighted_output = torch.matmul(weights, self.embedding.weight)  # Weighted sum
            weighted_output = weighted_output.view_as(x)  # Reshape to match original input shape

        # Select output based on the flag
        output_vector = quantized if self.quantized_output else (weighted_output+quantized) / 2

        # Confidence calculation (optional)
        confidence = None
        if self.compute_confidence:
            if not self.quantized_output:
                # Compute entropy-based confidence for weighted output
                entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)  # Entropy of weights
                confidence = 1 - entropy / torch.log(torch.tensor(self.num_embeddings, device=x.device))
                confidence = confidence.view(x.size(0), x.size(1))  # Reshape to (batch_size, seq_length)
            else:
                # Compute confidence for quantized output
                nearest_distance = distances.gather(1, encoding_indices).squeeze(1)  # Distance to nearest embedding
                max_distance = distances.max(dim=1)[0]  # Maximum distance for normalization
                confidence = 1 - nearest_distance / max_distance
                confidence = confidence.view(x.size(0), x.size(1))  # Reshape to (batch_size, seq_length)

        # Pass gradients only to the embedding via the straight-through estimator
        output_vector = x + (output_vector - x).detach()

        return output_vector, confidence, vq_loss
    
    
    def weighted_inference(self, x, similarity_metric='cosine'):
        """
        Perform inference using a weighted sum of all embeddings based on similarity.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).
            similarity_metric (str): The similarity metric to use ('cosine' or 'negative_distance').

        Returns:
            weighted_quantized (Tensor): Quantized output based on similarity weights.
        """
        # Flatten the input to (batch_size * seq_length, embedding_dim) for easier processing
        flat_x = x.view(-1, self.embedding_dim)

        if similarity_metric == 'cosine':
            # Normalize the input and embeddings for cosine similarity
            norm_x = F.normalize(flat_x, dim=1)  # (N, embedding_dim)
            norm_embeddings = F.normalize(self.embedding.weight, dim=1)  # (num_embeddings, embedding_dim)
            similarities = torch.matmul(norm_x, norm_embeddings.T)  # Cosine similarity (N, num_embeddings)

        elif similarity_metric == 'negative_distance':
            # Use negative Euclidean distance as similarity
            similarities = -((torch.sum(flat_x ** 2, dim=1, keepdim=True)  # (N, 1)
                              - 2 * torch.matmul(flat_x, self.embedding.weight.T)  # (N, num_embeddings)
                              + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True)))  # (1, num_embeddings)
        else:
            raise ValueError("Unsupported similarity_metric. Choose 'cosine' or 'negative_distance'.")

        # Softmax over similarities to get weights
        weights = F.log_softmax(similarities, dim=1)
        # weights = F.softmax(similarities, dim=1)  # (N, num_embeddings)
        # similarities = F.relu(similarities)
        # weights = similarities / (similarities.sum(dim=1, keepdim=True) + 1e-8)

        return weights
    
    
    
    def calc_loss(self, x, output_vector, vq_loss):
        """
        Calculate the overall loss for the VQ-VAE layer.

        Args:
            x (Tensor): Original input tensor of shape (batch_size, seq_length, embedding_dim).
            output_vector (Tensor): Quantized or weighted tensor output from the forward pass.
            vq_loss (Tensor): The vector quantization loss.

        Returns:
            total_loss (Tensor): The total loss including reconstruction and vector quantization loss.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(output_vector, x)
        # Combine losses
        total_loss = recon_loss + vq_loss
        return total_loss

# 定义网络结构
class MainModel(nn.Module):
    def __init__(self, input_feature_dim=1024, hidden_dim=256, nhead=4, num_layers=4, decopuling_dim=128, num_quantized_embedding=64):
        """
        Args:
            embedding_dim: 输入序列的特征维度
            hidden_dim: 全连接层隐藏层维度
            nhead: Transformer Encoder中多头注意力头数
            num_layers: Transformer Encoder的层数,
            decopuling_dim: 量化的维度
            num_quantized_embedding: 量化的数量
        Desc:
            Model一共有三种模式：
            - 预训练：仅计算量化和重建损失；
            - 训练：仅计算预测结果损失；
            - 推理：仅计算预测结果和置信度结果。
        """
        super(MainModel, self).__init__()
        
        # 解耦部分
        self.decoupling_T = nn.Sequential(
            # nn.Linear(input_feature_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, decopuling_dim)
            nn.Linear(input_feature_dim, decopuling_dim)
        )
        self.decoupling_P = nn.Sequential(
            # nn.Linear(input_feature_dim, 256),
            # nn.ReLU(),
            # nn.Linear(256, decopuling_dim)
            nn.Linear(input_feature_dim, decopuling_dim)
        )
        
        # 量化部分
        # T直接量化
        self.vector_quantized_T = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=False) # P不直接量化->加权
        self.vector_quantized_P = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=True) # 不需要量化，加权；在此分支计算置信度
        
        # # 重建部分
        # self.reconstruction_layer = nn.Linear(decopuling_dim*2, input_feature_dim)
        
        # 分数预测——重要性预测部分
        self.weight_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # 分数预测——片段分数部分
        self.clip_score_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decopuling_dim*2,  # 输入特征维度,接受T和P的双分支输入
                nhead=nhead,            # 多头注意力头数
                dim_feedforward=hidden_dim,  # 前馈网络的维度
                activation="relu"
            ),
            num_layers=num_layers  # Transformer Encoder的层数
        )
        
        # 全连接部分
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

    def forward(self, f, tgt=None):
        """
        Args:
            f: 输入数据，形状为 [batch_size, seq_len, embedding_dim]
            tgt: 目标预测值，shape[B, 1]； 当tgt取None时，不计算结果预测值
        """
        B, L, E = f.shape
        loss = 0.0
        confidence = 0.0
        
        f = f.reshape(B*L, -1) # B*L, feature_dim
        T = self.decoupling_T(f).reshape(B, L, -1) # B, L, decouling_dim
        P = self.decoupling_P(f).reshape(B, L, -1) # B, L, decouling_dim
        
        
        T, condidence_T, loss_T = self.vector_quantized_T(T) # B, L, decouling_dim
        P, confidence_P, loss_P = self.vector_quantized_P(P) # B, L, decouling_dim; B, L; B,1
        # confidence = torch.mean(confidence, dim=-1, keepdim=False)
        loss += loss_T
        loss += loss_P
        
        
        # Transformer 部分
        use_transformer = True
        if use_transformer:
            encoding = torch.cat([T, P], dim=-1)
            # 变形：因为默认batch_first = False
            f_transformer = encoding.permute(1, 0, 2)
            
            # Transformer 编码，输出形状为 [seq_len, batch_size, embedding_dim]
            f_transformer = self.transformer_encoder(f_transformer)
        
            # 取序列的最后一个时间步的输出作为特征向量
            f_transformer = f_transformer[-1, :, :]  # 形状变为 [batch_size, embedding_dim]
            pred = self.score_regressor(f_transformer).squeeze()
            # pred = torch.clamp(pred, 0.0, 1.0)
            confidence = self.confidence_regressor(f_transformer).squeeze()
            confidence = torch.mean(condidence_T*confidence_P, dim=-1).squeeze()
            
            if tgt is not None:
                loss += self.pred_loss(pred, confidence, tgt)
                
            # return pred, confidence, loss

        
        
        # pred_weight = self.weight_regressor(T.reshape(B*L, -1)).squeeze().reshape(B, L) # B, L
        # pred_clip_score = self.clip_score_regressor(P.reshape(B*L, -1)).squeeze().reshape(B, L) # B, L
        # pred_weight = pred_weight / pred_weight.sum(dim=1, keepdim=True) # B, L
        # pred = torch.sum(pred_weight*pred_clip_score, dim=1).reshape(B, 1) # B, 1
        
        # P = P.reshape(B, L, -1) # B, L, decopuling_dim
        # P = P.permute(1, 0, 2) # L, B, decopuling_Dim
        # pred = self.transformer_encoder(P)
        # P = P[-1, :, :].squeeze()
        # pred = self.clip_score_regressor(P).squeeze()
        
        # if tgt is not None:
        #     loss += self.pred_loss(pred, confidence, tgt)

        # reconstruct_feature = torch.cat([T, P], dim=-1).reshape(B*L, -1) # B, L, decouling_dim*2
        # reconstruct_feature = self.reconstruction_layer(reconstruct_feature) # B*L, embedding_dim; f is already this shape
        # loss_reconstruct = F.mse_loss(f, reconstruct_feature)
        # loss += loss_reconstruct
        
        return pred, confidence, loss


    
if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    # try vqlayer
    # vq_layer = VqLayer(embedding_dim=128, num_embeddings=64, 
    #                      compute_confidence=True, quantized_output=False).to(device)
    # # 输入张量
    # inputs = torch.randn(4, 10, 128).to(device)  # batch_size=4, seq_length=10, embedding_dim=128

    # # 前向传播
    # output, confidence, loss = vq_layer(inputs)

    # print("Output Shape:", output.shape)  # (4, 10, 32)
    # if confidence is not None:
    #     print("Confidence Shape:", confidence.shape)  # (4, 10)
    # print("Loss:", loss.item())
    
    # try mainmodel
    model = MainModel().to(device)
    inputs = torch.randn(4, 192, 1024).to(device)
    tgt = torch.randn(4, 1).to(device)
    outputs, confidence, loss = model(inputs, tgt)
    loss.backward()
    print(outputs.shape, confidence.shape, loss)