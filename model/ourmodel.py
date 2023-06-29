import torch
from torch import nn
import torch.nn.functional as F

import model.base.resnet as models
import model.base.vgg as vgg_models
from model.prototypical_contrast import PrototypeContrastLoss

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class NTRENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False，bgpro_num = 1):
        super(NTRENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        models.BatchNorm = BatchNorm
        
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 
        self.bg_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )  
        
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(

            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )  
        self.down_bg = nn.Sequential(
            nn.Conv2d(reduce_dim*(bgpro_num+1), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        mask_add_num = 1
        self.init_merge1 = []
        self.init_merge2 = []
        self.init_merge3 = []

        self.inital_beta_conv = []
        self.inital_inner_cls = []        
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge1.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 , reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))   
            self.init_merge2.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                     
            self.init_merge3.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 , reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.inital_beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inital_inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge1 = nn.ModuleList(self.init_merge1) 
        self.init_merge2 = nn.ModuleList(self.init_merge2) 
        self.init_merge3 = nn.ModuleList(self.init_merge3) 
        self.inital_beta_conv = nn.ModuleList(self.inital_beta_conv)
        self.inital_inner_cls = nn.ModuleList(self.inital_inner_cls)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
        self.bg_res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        ) 
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     

        self.bg_prototype = nn.Parameter(torch.zeros(1, reduce_dim * bgpro_num,1,1))

        self.bg_loss = nn.CrossEntropyLoss(reduction='none')
        self.contrast_loss = PrototypeContrastLoss()

    def _optimizer(self, args):
        optimizer = torch.optim.SGD(
        [
            ##### background prototype ####
            {'params': self.bg_prototype},
            {'params': self.down_bg.parameters()},
            {'params': self.bg_res1.parameters()},
            {'params': self.bg_cls.parameters()},
            ###############################

            {'params': self.down_query.parameters()},
            {'params': self.down_supp.parameters()},

            {'params': self.init_merge1.parameters()},
            {'params': self.init_merge2.parameters()},
            {'params': self.init_merge3.parameters()},
            {'params': self.inital_beta_conv.parameters()},
            {'params': self.inital_inner_cls.parameters()},
            
            {'params': self.alpha_conv.parameters()},
            {'params': self.beta_conv.parameters()},
            {'params': self.inner_cls.parameters()},

            {'params': self.res1.parameters()},
            {'params': self.res2.parameters()},
            {'params': self.cls.parameters()},
        ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        return optimizer

    def forward(self, s_x, s_y,x, y, classes, prototype_neg_dict=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature     
        supp_feat_list = []
        supp_nomask_feat_list = []
        final_supp_list = []
        mask_list = []
        bg_mask_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            bg_mask = (s_y[:,i,:,:] == 0)
            mask_list.append(mask)
            bg_mask_list.append(bg_mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_nomask = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat_nomask, mask)
            supp_feat_list.append(supp_feat)
            supp_nomask_feat_list.append(supp_feat_nomask)


        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

            supp_feat_nomask = supp_nomask_feat_list[0]
            for i in range(1, len(supp_nomask_feat_list)):
                supp_feat_nomask += supp_nomask_feat_list[i]
            supp_feat_nomask /= len(supp_nomask_feat_list)

        bg = self.bg_prototype.expand(query_feat.size(0),-1,query_feat.size(2),query_feat.size(3))

        qrybg_feat = torch.cat((query_feat,bg),dim=1)
        qrybg_feat1 = self.down_bg(qrybg_feat)
        qrybg_feat2 = self.bg_res1(qrybg_feat1) + qrybg_feat1         
        query_bg_out = self.bg_cls(qrybg_feat2)
        
        supp_bg_out_list = []
        if self.training:
            for supp_feat_nomask in supp_nomask_feat_list:
                suppbg_feat = torch.cat((supp_feat_nomask,bg),dim=1)
                suppbg_feat = self.down_bg(suppbg_feat)
                suppbg_feat = self.bg_res1(suppbg_feat) + suppbg_feat          
                supp_bg_out = self.bg_cls(suppbg_feat)
                supp_bg_out_list.append(supp_bg_out)

        inital_out_list = []
        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)

            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            bg_feat_bin = self.bg_prototype.expand(query_feat.size(0),-1,bin,bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_binbg = torch.cat([query_feat_bin, bg_feat_bin],1)
            merge_feat_binbg = self.init_merge1[idx](merge_feat_binbg)
            merge_feat_binfg = torch.cat([merge_feat_binbg,supp_feat_bin, corr_mask_bin],1)
            merge_feat_binfg = self.init_merge2[idx](merge_feat_binfg)
            merge_feat_binfg = self.inital_beta_conv[idx](merge_feat_binfg) + merge_feat_binfg   
            inital_inner_out_bin = self.inital_inner_cls[idx](merge_feat_binfg)
            inital_out_list.append(inital_inner_out_bin)

            query_bg_out_bin = F.interpolate(query_bg_out, size=(bin, bin), mode='bilinear', align_corners=True)
            confused_mask = F.relu(1- query_bg_out_bin.max(1)[1].unsqueeze(1) -  inital_inner_out_bin.max(1)[1].unsqueeze(1)) 
            confused_prototype = nn.AdaptiveAvgPool2d(1)(confused_mask*query_feat_bin)
            confused_prototype_bin = confused_prototype.expand(-1,-1,bin,bin)
            merge_feat_bin = torch.cat([merge_feat_binfg,confused_prototype_bin],1)
            merge_feat_bin = self.init_merge3[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)

            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat_out = torch.cat(pyramid_feat_list, 1)
        query_feat_out = self.res1(query_feat_out)
        query_feat_out = self.res2(query_feat_out) + query_feat_out           
        out = self.cls(query_feat_out)
        
        if self.training:
            prototype_contrast_loss, prototype_neg_dict = self.contrast_loss(query_feat, supp_feat_nomask, out, \
                y, s_y, query_bg_out, supp_bg_out, classes, prototype_neg_dict)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            query_bg_out = F.interpolate(query_bg_out, size=(h, w), mode='bilinear', align_corners=True)
            supp_bg_out = F.interpolate(supp_bg_out, size=(h, w), mode='bilinear', align_corners=True)
            erfa = 0.5
            main_loss = self.criterion(out, y.long())
            aux_loss1 = torch.zeros_like(main_loss).cuda()    
            aux_loss2 = torch.zeros_like(main_loss).cuda()    

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss1 = aux_loss1 + self.criterion(inner_out, y.long())   
            aux_loss1 = aux_loss1 / len(out_list)

            for idx in range(len(inital_out_list)):    
                inner_out = inital_out_list[idx]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss2 = aux_loss2 + self.criterion(inner_out, y.long())   
            aux_loss2 = aux_loss2 / len(inital_out_list)

            mygt1 = torch.ones(query_bg_out.size(0),h,w).cuda()
            mygt0 = torch.zeros(query_bg_out.size(0),h,w).cuda()
            query_bg_loss = self.weighted_BCE(query_bg_out, mygt0, y)+erfa*self.criterion(query_bg_out,mygt1.long())
            for j,supp_bg_out in enumerate(supp_bg_out_list):
                supp_bg_out = F.interpolate(supp_bg_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_bg_loss = self.weighted_BCE(supp_bg_out, mygt0, s_y[:,j,:,:])+erfa*self.criterion(supp_bg_out,mygt1.long())
                aux_bg_loss = aux_bg_loss + supp_bg_loss

            bg_loss = (query_bg_loss + aux_bg_loss)/ (len(supp_bg_out_list)+1)     

            return out.max(1)[1], main_loss,bg_loss+0.4*aux_loss1+0.6*aux_loss2+0.01*prototype_contrast_loss,prototype_neg_dict
        else:
            return out

    def weighted_BCE(self,input, target,mask):
        loss_list =[]
        cmask = torch.where(mask.long() == 1,mask.long(),target.long())
        
        for x,y,z in zip(input,target,cmask):
            loss = self.bg_loss(x.unsqueeze(0),y.unsqueeze(0).long())
            area = torch.sum(z)+1e-5
            Loss = torch.sum(z.unsqueeze(0)*loss) /area
            loss_list.append(Loss.unsqueeze(0))
        LOSS = torch.cat(loss_list,dim=0)                     
        return torch.mean(LOSS)
