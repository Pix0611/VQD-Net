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

        gt = torch.div(gt, torch.sum(gt, dim=1).reshape((-1, 1)))
        return self.kl_loss(torch.log(predicted_score_distribution), gt)




















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


        if self.hardest:

            mask_anchor_positive = _get_anchor_positive_triplet_mask(
                labels, self.device_anchor.device).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(
                valid_positive_dist, dim=1, keepdim=True)


            mask_anchor_negative = _get_anchor_negative_triplet_mask(
                labels).float()
            max_anchor_negative_dist, _ = torch.max(
                pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(
                anchor_negative_dist, dim=1, keepdim=True)


            triplet_loss = F.relu(hardest_positive_dist -
                                  hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)



        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)





            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels, self.device_anchor.device).float()
            triplet_loss = loss * mask


            triplet_loss = F.relu(triplet_loss)


            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / \
                                     (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):














    norm_x = F.normalize(x, dim=-1)
    distances = 1 - torch.mm(norm_x, norm_x.T)







    return distances


def _get_anchor_positive_triplet_mask(labels, device):


    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1


    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):



    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ bool(1)

    return mask


def _get_triplet_mask(labels, device):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """


    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k


    label_equal = torch.eq(torch.unsqueeze(labels, 0),
                           torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ bool(1))

    mask = distinct_indices * valid_labels

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

        super().__init__()
        self.num_segments = num_segments-2
        self.num_subactions = num_subactions
        self.func_mse = nn.MSELoss()


    def forward(self, x, ground_truth):

        assert x.ndim==3 and ground_truth.ndim==2
        assert x.shape[0]==ground_truth.shape[0]
        N = x.shape[0]
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




        N = predict.shape[0]
        assert uncertainty.shape[0] == N and target.shape[0] == N
        assert predict.ndim==1 and uncertainty.ndim==1 and target.ndim==1


        loss = torch.sum(1 / uncertainty**2 *(predict-target)**2+ uncertainty**2).squeeze() / N

        return loss


        return 0

def gaussian_kernel(x, y, sigma=1.0):

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))

def mmd_loss(source, target, sigma=1.0):

    source_kernel = gaussian_kernel(source, source, sigma)
    target_kernel = gaussian_kernel(target, target, sigma)
    cross_kernel = gaussian_kernel(source, target, sigma)
    
    mmd = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()
    return mmd

def LUSD_reverse_mmd_loss(batch_seq_embedding, sigma=1.0, maximum_gap=1.0):

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

    return sum_mmd_loss

class BatchSeqDiscriminatorLoss(nn.Module):
    def __init__(self, class_num=2, embedding_dim=1024, hidden_size=512, loss_discriminator_ratio=1.0, loss_generator_ratio=1.0):
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

        label = torch.arange(class_num, device=device).repeat(batch_size * seq_length)
        

        x1 = x.reshape(-1, embedding_dim).detach()
        x1 = self.fc(x1)
        loss_discriminator = self.ce_loss(x1, label)
        

        x2 = x.reshape(-1, embedding_dim)
        x2 = self.fc(x2)

        fake_label = (label + 1) % class_num
        loss_generator = self.ce_loss(x2, fake_label)
        
        return self.loss_discriminator_ratio * loss_discriminator + self.loss_generator_ratio * loss_generator

if __name__ == '__main__':









    dummy_input = torch.rand(5, 32, 3, 1024).cuda()
    loss = BatchSeqDiscriminatorLoss(class_num=3).cuda()
    dummy_output = loss(dummy_input)
    print(dummy_output)