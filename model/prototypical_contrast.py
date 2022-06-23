from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  

class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()

        self.temperature = 1
        self.m = 1000
        self.n = 2000

    def _negative_construct(self, pros, labels_): 
        unique_labels = torch.unique(labels_)
        pro_dict = dict()
        for i in range(len(unique_labels)):
            index = torch.where(labels_ == unique_labels[i])
            pro_dict[unique_labels[i].item()]=pros[index].contiguous().view(-1,256)
        return pro_dict  

    def _contrastive(self, base_pro, pos_pro, neg_dict, labels_):

        loss = torch.zeros(1).cuda()

        for base, pos, cls in zip(base_pro, pos_pro, labels_):
            positive_dot_contrast = torch.div(F.cosine_similarity(base, pos,0),
                                        self.temperature)
            negative_samples = neg_dict[cls.item()]
            if negative_samples.shape[0]>self.m:
                perm = torch.randperm(negative_samples.shape[0])
                negative_samples = negative_samples[perm[:self.m]]
            negative_dot_contrast = torch.div(F.cosine_similarity(base, torch.transpose(negative_samples, 0, 1),0),
                                        self.temperature)

            pos_logits  = torch.exp(positive_dot_contrast)
            neg_logits = torch.exp(negative_dot_contrast).sum()
            mean_log_prob_pos = - torch.log((pos_logits/(neg_logits))+1e-8)

            loss = loss + mean_log_prob_pos.mean()

        return loss/len(labels_)

    def forward(self, Q_feats, S_feats, Q_predit, Q_labels, S_labels, query_bg_out, supp_bg_out, classes, negative_dict):     # feats:4,256,260,260 ; labels:4,520,520; predict:4,260,260
        classes = torch.cat(classes,0).clone()
        Q_labels = Q_labels.unsqueeze(1).float().clone()
        S_labels = S_labels.float().clone()

        Q_labels = F.interpolate(Q_labels,(Q_feats.shape[2], Q_feats.shape[3]), mode='nearest')
        S_labels = F.interpolate(S_labels,(S_feats.shape[2], S_feats.shape[3]), mode='nearest')

        zeros = torch.zeros_like(Q_labels).cuda()
        Q_labels = torch.where(Q_labels==255,zeros,Q_labels)
        S_labels = torch.where(S_labels==255,zeros,S_labels)

        Q_disrupt_labels = F.relu(1-query_bg_out.max(1)[1].unsqueeze(1) - Q_labels)
        S_disrupt_labels = F.relu(1-supp_bg_out.max(1)[1].unsqueeze(1) - S_labels)

        Q_dsp_pro = Weighted_GAP(Q_feats, Q_disrupt_labels).squeeze(-1) 
        S_dsp_pro = Weighted_GAP(S_feats, S_disrupt_labels).squeeze(-1)
        Q_predit_pro = Weighted_GAP(Q_feats, Q_predit.max(1)[1].unsqueeze(1)).squeeze(-1)
        S_GT_pro = Weighted_GAP(S_feats, S_labels).squeeze(-1)
        

        Q_dsp_dict = self._negative_construct(Q_dsp_pro, classes)
        S_dsp_dict = self._negative_construct(S_dsp_pro, classes)

        for key in Q_dsp_dict.keys():
            if key not in negative_dict:
                negative_dict[key] = torch.cat((Q_dsp_dict[key],S_dsp_dict[key]),0).detach()
            else:
                orignal_value = negative_dict[key]
                negative_dict[key] = torch.cat((Q_dsp_dict[key],S_dsp_dict[key],orignal_value),0).detach()
                if negative_dict[key].shape[0]>self.n:
                    negative_dict[key] = negative_dict[key][:self.n,:]

        loss = self._contrastive(Q_predit_pro, S_GT_pro, negative_dict, classes)

        return loss,negative_dict