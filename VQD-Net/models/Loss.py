import torch
import torch.nn as nn
import torch.nn.functional as F


class UsdlLoss(nn.Module):
    def __init__(self, score_range=(0.0, 1.0), score_step=0.1, gaussian_sigma=0.05):
        super().__init__()
        self.register_buffer("score_distribution", torch.arange(
            score_range[0], score_range[1], score_step))
        self.sigma = gaussian_sigma
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mmd_loss = MMD_loss()

    def forward(self, predicted_score_distribution, gt_score):
        predicted_score_distribution = predicted_score_distribution.reshape(
            (-1, self.score_distribution.shape[0]))
        gt_score = gt_score.reshape(-1, 1)

        gt = 1/(((2*torch.pi)**1/2)*self.sigma) * torch.exp((-1/2 *
                (self.score_distribution-gt_score)**2)/self.sigma**2)
        # gt = F.softmax(gt, dim=-1)
        gt = torch.div(gt, torch.sum(gt, dim=1).reshape((-1, 1)))
        return self.kl_loss(torch.log(predicted_score_distribution), gt)
        # return self.mmd_loss(predicted_score_distribution, gt)

# class SegmentationLoss(nn.Module):
#     def __init__(self, seq_length, action_types):
#         super().__init__()
#         self.seq_length = seq_length
#         self.action_types = action_types
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(self, segmentation_seq, gt_seq):
#         assert gt_seq.shape[1]== segmentation_seq.shape[1] # seq length
#         assert segmentation_seq.shape[-1] == self.action_types # num of action_types

#         segmentation_seq = segmentation_seq.reshape(-1, self.action_types)
#         gt_seq = gt_seq.reshape(-1)

#         loss = self.ce_loss(segmentation_seq, gt_seq)
#         return loss


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()

        self.register_buffer("device_anchor", torch.tensor([0]))

        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)
        # 计算两两之间的距离（首先对数据归一化后再计算）

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(
                labels, self.device_anchor.device).float()  # 标出和自己是否是同一类，自己与自己不作考虑
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(
                valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(
                labels).float()  # 标出和自己是否是不同类，自己与自己不作考虑
            max_anchor_negative_dist, _ = torch.max(
                pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(
                anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist -
                                  hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)

            # import pdb
            # pdb.set_trace()
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels, self.device_anchor.device).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / \
                                     (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    # cor_mat = torch.matmul(x, x.t())
    # norm_mat = cor_mat.diag()
    # distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    # distances = F.relu(distances)
    #
    # if not squared:
    #     mask = torch.eq(distances, 0.0).float()
    #     distances = distances + mask * eps
    #     distances = torch.sqrt(distances)
    #     distances = distances * (1.0 - mask)

    # cosine distance
    norm_x = F.normalize(x, dim=-1)
    distances = 1 - torch.mm(norm_x, norm_x.T)

    # print(distances)
    # import pdb
    # pdb.set_trace()

    # print(torch.all(distances <= 2), torch.all(distances >= 0))

    return distances


def _get_anchor_positive_triplet_mask(labels, device):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ bool(1)

    return mask


def _get_triplet_mask(labels, device):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0),
                           torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ bool(1))

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                                for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class SubscoreMseLoss(nn.Module):
    def __init__(self, num_segments, num_subactions):
        # For a batch with N samples, previous modules will split and regress subscores for #segments parts. However, not every segment matches with a subscore in GT. Here we select part of them(#subactions) for trainning and infering.
        super().__init__()
        self.num_segments = num_segments-2
        self.num_subactions = num_subactions
        self.func_mse = nn.MSELoss()


    def forward(self, x, ground_truth):
        # x.ndim is 3:(#batchsize, #segments, (score,confidence,weight(for sum))
        assert x.ndim==3 and ground_truth.ndim==2
        assert x.shape[0]==ground_truth.shape[0]
        N = x.shape[0] # #samples
        assert x.shape[1] == self.num_segments
        assert ground_truth.shape[1] == self.num_subactions

        select_mask = torch.arange(0, 2*self.num_subactions-1, 2)
        predict_score = x[:, select_mask, 0]

        loss = self.func_mse(predict_score, ground_truth)
        return loss

class UncertainRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, predict, uncertainty, target):
        # Input:
        # predict: 1d-array of predicted value'
        # uncertainty: 1d-array of predicted uncertainty
        # target: 1d-array of ground-truth
        N = predict.shape[0]
        assert uncertainty.shape[0] == N and target.shape[0] == N
        assert predict.ndim==1 and uncertainty.ndim==1 and target.ndim==1


        loss = torch.sum(1 / uncertainty**2 *(predict-target)**2+ uncertainty**2).squeeze() / N

        return loss


        return 0

def gaussian_kernel(x, y, sigma=1.0):
    """
    计算高斯核矩阵
    :param x: 张量，形状为 (n_samples_x, n_features)
    :param y: 张量，形状为 (n_samples_y, n_features)
    :param sigma: 高斯核的带宽参数
    :return: 核矩阵，形状为 (n_samples_x, n_samples_y)
    """
    x = x.unsqueeze(1)  # (n_samples_x, 1, n_features)
    y = y.unsqueeze(0)  # (1, n_samples_y, n_features)
    return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))

def mmd_loss(source, target, sigma=1.0):
    """
    计算MMD损失
    :param source: 源域数据，形状为 (n_samples_source, n_features)
    :param target: 目标域数据，形状为 (n_samples_target, n_features)
    :param sigma: 高斯核的带宽参数
    :return: MMD损失值
    """
    source_kernel = gaussian_kernel(source, source, sigma)
    target_kernel = gaussian_kernel(target, target, sigma)
    cross_kernel = gaussian_kernel(source, target, sigma)
    
    mmd = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()
    return mmd

def LUSD_reverse_mmd_loss(batch_seq_embedding, sigma=1.0, maximum_gap=1.0):
    """
    特地为模型定值的计算 反MMD损失（因为我们需要拉开分布的距离，而不是拉近），传入的形状是：batch_size*seq_length*num_tasks*embedding_dim
    两两一组计算 abs(maximum_gap-MMD_loss)后求和
    :return: MMD损失值
    """
    assert len(batch_seq_embedding.shape) == 4
    
    batch_size, seq_length, num_tasks, embedding_dim = batch_seq_embedding.shape
    
    sum_mmd_loss = 0
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            source = batch_seq_embedding[:, :, i, :]
            target = batch_seq_embedding[:, :, i, :]
            source = source.reshape(-1, embedding_dim)
            target = target.reshape(-1, embedding_dim)
            source_kernel = gaussian_kernel(source, source, sigma)
            target_kernel = gaussian_kernel(target, target, sigma)
            cross_kernel = gaussian_kernel(source, target, sigma)
            mmd = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()

            sum_mmd_loss += torch.abs(maximum_gap-mmd)
        # return mmd
    return sum_mmd_loss

class BatchSeqDiscriminatorLoss(nn.Module):
    def __init__(self, class_num=2, embedding_dim=1024, hidden_size=512, loss_discriminator_ratio=1.0, loss_generator_ratio=1.0):
        """
        用来区分一组特征，输入是embedding的结果，即batch_size * seq_length * num_tasks * embedding_dim

        Args:
            class_num (int): 有几个子任务的特征需要分离，例如，对于finefs就是2个
            embedding_dim(int): 输入的嵌入向量的维度. Defaults to 1024
            hidden_size (int, optional): FC的隐藏层维度. Defaults to 512.
        """
        super().__init__()
        self.class_num = class_num
        self.loss_discriminator_ratio = loss_discriminator_ratio
        self.loss_generator_ratio = loss_generator_ratio
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, class_num)
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, seq_length, class_num, embedding_dim = x.shape
        assert class_num == self.class_num
        device = x.device
        # 生成标签
        label = torch.arange(class_num, device=device).repeat(batch_size * seq_length)
        
        # 判别损失
        x1 = x.reshape(-1, embedding_dim).detach()  # Detach to prevent gradients flowing to the generator
        x1 = self.fc(x1)
        loss_discriminator = self.ce_loss(x1, label)
        
        # 生成损失
        x2 = x.reshape(-1, embedding_dim)  # No detach here, we want gradients to flow to the generator
        x2 = self.fc(x2)
        # 生成损失的标签是反转的
        fake_label = (label + 1) % class_num
        loss_generator = self.ce_loss(x2, fake_label)
        
        return self.loss_discriminator_ratio * loss_discriminator + self.loss_generator_ratio * loss_generator

if __name__ == '__main__':
    # gpu = 7
    # t_loss = HardTripletLoss(margin=1.0, hardest=True, squared=False).cuda(gpu)
    # batch_size = 8
    # embeddings = torch.tensor([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],dtype=torch.float32).cuda(gpu)
    # labels = torch.tensor([0,0,0,1,1,1]).cuda(gpu)
    # print(t_loss(embeddings, labels))
    # dummy_input = torch.rand(4,5,3,1024).cuda()
    # dummy_output = LUSD_reverse_mmd_loss(dummy_input)
    # print(dummy_output)
    dummy_input = torch.rand(5, 32, 3, 1024).cuda()
    loss = BatchSeqDiscriminatorLoss(class_num=3).cuda()
    dummy_output = loss(dummy_input)
    print(dummy_output)