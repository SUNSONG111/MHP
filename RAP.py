import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
try:
    import model.RAP.modules.conv_blocks as cb
    from model.RAP.modules.voxelmorph_us import U_Network, SpatialTransformer
    from model.RAP.modules.cre import Correlation
    from model.RAP.SGCNNet import GCNChannel, GCNSpatial
    #如果vscode能跳转，但是运行找不到，很可能是有两个地方有相同的模块名，甚至内部有__init__.py
except:
    import modules.conv_blocks as cb
    from modules.voxelmorph_us import U_Network, SpatialTransformer
    from modules.cre import Correlation
    from SGCNNet import GCNChannel, GCNSpatial
    #如果vscode能跳转，但是运行找不到，很可能是有两个地方有相同的模块名，甚至内部有__init__.py


class RAP(nn.Module):
    def __init__(self, params):
        super(RAP, self).__init__()
        self.n_channels = params.num_channels
        self.n_classes = params.num_class
        self.shot = params.shot
        self.n_clusters = 2
        params['num_channels'] = 1
        params['num_filters'] = 16
        self.encode1 = cb.SDnetEncoderBlock(params)#就是卷积
        params['num_channels'] = 16
        params['num_filters'] = 32
        self.encode2 = cb.SDnetEncoderBlock(params)#就是卷积
        params['num_channels'] = 32
        params['num_filters'] = 64
        self.encode3 = cb.SDnetEncoderBlock(params)#就是卷积
        params['num_channels'] = 64
        params['num_filters'] = 64
        self.encode4 = cb.SDnetEncoderBlock(params)#就是卷积
        params['num_channels'] = 64
        params['num_filters'] = 64
        self.bottleneck = cb.GenericBlock(params)#就是卷积


        params['num_channels'] = 176
        params['num_filters'] = 32
        self.decode1 = cb.SDnetDecoderBlock(params)#就是卷积
        params['num_channels'] = 320
        params['num_filters'] = 32
        self.decode2 = cb.SDnetDecoderBlock(params)#就是卷积
        params['num_channels'] = 704
        params['num_filters'] = 32
        self.decode3 = cb.SDnetDecoderBlock(params)#就是卷积
        params['num_channels'] = 640
        params['num_filters'] = 128
        self.decode4 = cb.SDnetDecoderBlock(params)#就是卷积
        params['num_channels'] = 96
        params['num_class'] = 1

        # cood_conv = cb.CoordConv(16, 1, kernel_size=3, padding=1)  #24
        self.soft_max = nn.Softmax2d()
        # self.sigmoid = nn.Sigmoid()
        self.mask_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sim_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        # nn.Sequential(cood_conv)

        self.unet = U_Network(2, [16, 32, 32, 32], [32, 32, 32, 32, 8, 8])#产生2维输出，就是位移场
        self.stn = SpatialTransformer((params.img_size, params.img_size))#这里要和图像尺寸一致

        self.conv00 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(inplace=True)
        )


        self.conv11 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        self.conv22 = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 3),
            nn.ReLU(inplace=True)
        )

        self.q4 = nn.Sequential(
            nn.Conv2d(64+49, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.q3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.q2 = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.fc2_1 = nn.Linear(32*9, 16*9)
        self.fc3_1 = nn.Linear(64*9, 16*9)
        self.fc4_1 = nn.Linear(64*9, 16*9)
        self.fc1_2 = nn.Linear(16*9, 32*9)
        self.fc3_2 = nn.Linear(64*9, 32*9)
        self.fc4_2 = nn.Linear(64*9, 32*9)
        self.fc1_3 = nn.Linear(16*9, 64*9)
        self.fc2_3 = nn.Linear(32*9, 64*9)
        self.fc4_3 = nn.Linear(64*9, 64*9)
        self.topo_loss = TopLoss_fewshot()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()#已有softmax
        self._init_weights()


    def seg_branch_encoder(self, input):
        e1, _, ind1 = self.encode1(input)#卷积 池化
        e2, _, ind2 = self.encode2(e1)#卷积 池化
        e3, _, ind3 = self.encode3(e2)#卷积 池化
        e4, out4, ind4 = self.encode4(e3)#卷积 池化
        bn = self.bottleneck(e4)#卷积

        return bn, ind4, ind3, ind2, ind1, e4, e3, e2, e1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def choose_grad_map(self, feat):
        '''
        用固定核找到疑似梯度0点 这些点应该在support监督中靠近边缘点  query直接找梯度为0点作为边缘 用于优化query预测的边缘
        feat: tensor (n, c, h, w)
        out: (n, 1, h, w)  仅在一个方向上梯度为0的点
        '''
        n, c, h, w = feat.shape
        vertical_1 = torch.tensor([
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        vertical_2 = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        horizontal_1 = torch.tensor([
            [0, 0, 0],
            [1, -1, 0],
            [0, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        horizontal_2 = torch.tensor([
            [0, 0, 0],
            [0, 1, -1],
            [0, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        left_cross_1 = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        left_cross_2 = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        right_cross_1 = torch.tensor([
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
        right_cross_2 = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [-1, 0, 0]
        ], requires_grad=False)[None, None,...].repeat(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)

        vertical_grad = (F.conv2d(feat, weight=vertical_1, padding=(1, 1)) * F.conv2d(feat, weight=vertical_2, padding=(1, 1)))
        vertical_grad = (vertical_grad <= 0).to(dtype=feat.dtype)
        horizontal_grad = (F.conv2d(feat, weight=horizontal_1, padding=(1, 1)) * F.conv2d(feat, weight=horizontal_2, padding=(1, 1)))
        horizontal_grad = (horizontal_grad <= 0).to(dtype=feat.dtype)
        left_cross_grad = (F.conv2d(feat, weight=left_cross_1, padding=(1, 1)) * F.conv2d(feat, weight=left_cross_2, padding=(1, 1)))
        left_cross_grad = (left_cross_grad <= 0).to(dtype=feat.dtype)
        right_cross_grad = (F.conv2d(feat, weight=right_cross_1, padding=(1, 1)) * F.conv2d(feat, weight=right_cross_2, padding=(1, 1)))
        right_cross_grad = (right_cross_grad <= 0).to(dtype=feat.dtype)

        ver_hor_grad = vertical_grad + horizontal_grad
        left_right_grad = left_cross_grad + right_cross_grad 
        ver_hor_choose = (ver_hor_grad == 1).to(dtype=feat.dtype)
        left_right_choose = (left_right_grad == 1).to(dtype=feat.dtype)
        total_grad = ver_hor_choose + left_right_choose

        ## 垂直方向只能有一个 并且最后也只能有一个方向
        choosed_grad = (total_grad == 1).to(dtype=feat.dtype)

        return choosed_grad

    def get_supp_sdm(self, img_gt):
        '''
        img_gt: (n, shot, 1, h, w)
        '''
        tmp_list = []
        for tmp_k in range(self.shot):
            curr_shot_img_gt = img_gt[:, tmp_k, ...]
            curr_shot_img_sdm = self.compute_sdf(curr_shot_img_gt)
            tmp_list.append(curr_shot_img_sdm)
        tmp_list = torch.stack(tmp_list, dim=1)
        return tmp_list

    def get_query_sdm(self, img_gt):
        '''
        img_gt: (n, 1, h, w)
        '''
        img_sdm = self.compute_sdf(img_gt)
        return img_sdm

    def compute_sdf(self, img_gt):
        """
        img_gt: tensor (n, 1, h, w)
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size,1, x, y)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """
        img_gt_np = img_gt.cpu().numpy().astype(np.uint8)
        sdm_list = []
        for tmp_k in range(len(img_gt_np)):
            posmask = img_gt_np[tmp_k,0]
            if posmask.any():
                negmask = 1-posmask
                posdis = distance(posmask)
                negdis = distance(negmask)

                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdm_pos = (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)+1e-7)
                sdm_neg = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)+1e-7)
                sdm = sdm_neg - sdm_pos
                sdm[boundary==1] = 0
                sdm_list.append(sdm)
        sdm_list = torch.from_numpy(np.array(sdm_list)).unsqueeze_(dim=1).to(device=img_gt.device)#(n,2,h,w)
        return sdm_list

    def get_coor_label_map_support(self, label):
        pts_list = []
        for tmp_k in range(self.shot):
            curr_pts = self.get_coor_label_map(label[:, tmp_k])
            pts_list.append(curr_pts)
        pts_list = torch.stack(pts_list, dim=1)
        return pts_list

    def get_coor_label_map_query(self, label):
        pts = self.get_coor_label_map(label)
        return pts

    def get_coor_label_map(self, label):
        '''
        label: tensor (n, 1, h, w)
        '''
        n, c, h, w = label.shape

        axis_x = torch.arange(0, w)#可以给坐标降低值大小 这样变相减小其重要性
        axis_y = torch.arange(0, h)
        grid_axes = torch.stack(torch.meshgrid(axis_y, axis_x), dim=2).to(device=label.device)#这个就是完整格网
        grid_axes_flatten = grid_axes.reshape((-1, 2))
        
        ## 横纵坐标必须代表相同的单位距离 否则不是欧式空间 与心肌代表的二维空间不同 这不被允许 height和width因为是四至框也会变 不能用  只能用固定值 暂时选true_map的第一维长度
        max_edge_length = max(h, w)
        axis_x_norm = axis_x / w
        axis_y_norm = axis_y / h
        grid_axes_norm = torch.stack(torch.meshgrid(axis_y_norm, axis_x_norm), dim=2).to(device=label.device)#(h,w,2)
        grid_axes_norm = grid_axes_norm[None, ...].repeat(n, 1, 1, 1).permute(0, 3, 1, 2) #(n, 2, h, w)

        # grid_axes_norm_flatten = grid_axes_norm.reshape((-1, 2))
        pts = torch.cat((grid_axes_norm, label), dim=1) #(n, 2+c, h, w)
        return pts
    
    def get_supp_feat_batch_tensor(self, batch, most_sim_idx):
        '''
        batch:  (n, shot c, h, w)
        out : (n, c, h, w)
        '''
        curr_s_y_sdms_pos = []
        for tmp_b in range(len(most_sim_idx)):
            curr_s_y_sdms_pos.append(batch[tmp_b,most_sim_idx[tmp_b],...])
        curr_s_y_sdms_pos = torch.stack(curr_s_y_sdms_pos, dim=0)
        return curr_s_y_sdms_pos

    def get_supp_feat_batch_list(self, batch, most_sim_idx):
        '''
        batch:  列表 shot个 (n, c, h, w)
        out : (n, c, h, w)
        '''
        curr_s_y_sdms_pos = []
        for tmp_b in range(len(most_sim_idx)):
            curr_s_y_sdms_pos.append(batch[most_sim_idx[tmp_b]][tmp_b])
        curr_s_y_sdms_pos = torch.stack(curr_s_y_sdms_pos, dim=0)
        return curr_s_y_sdms_pos



    def get_choosed_position_similarity(self, supp_feat, query_feat, supp_choosed_map, query_choosed_map, fold_window=3):
        '''
        query_choosed_map 中正类特征 要和 support正类部分差异足够小
        '''
        ## 中心原型########################  最大值周围一圈
        padded_supp_feat_k = F.pad(supp_feat*supp_choosed_map, [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_supp_feat_k = padded_supp_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)

        fold_supp_feat_k_reshape = fold_supp_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_supp_feat_k.shape[0], -1, fold_supp_feat_k.shape[2], fold_supp_feat_k.shape[3])#(n,c k k, h, w)

        curr_choose_supp = torch.stack(torch.where(supp_choosed_map > 0), dim=1)#(N,4)
        idx_list = [i for i in range(len(curr_choose_supp))]
        random.shuffle(idx_list)
        choose_num = min(len(curr_choose_supp)//len(supp_feat), 1000)

        
        supp_choose_value = fold_supp_feat_k_reshape[curr_choose_supp[idx_list[:choose_num],0],:,curr_choose_supp[idx_list[:choose_num],2], curr_choose_supp[idx_list[:choose_num],3]]#(N1,c)

        del fold_supp_feat_k_reshape,supp_choosed_map
        torch.cuda.empty_cache()
        ##  query计算特征距离
        padded_query_feat_k = F.pad(query_feat*query_choosed_map, [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_query_feat_k = padded_query_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)
        fold_query_feat_k_reshape = fold_query_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_query_feat_k.shape[0], -1, fold_query_feat_k.shape[2], fold_query_feat_k.shape[3])#(n,c k k, h, w)
        curr_choose_query = torch.stack(torch.where(query_choosed_map > 0), dim=1)

        idx_list = [i for i in range(len(curr_choose_query))]
        random.shuffle(idx_list)


        query_choose_value = fold_query_feat_k_reshape[curr_choose_query[idx_list[:choose_num],0],:,curr_choose_query[idx_list[:choose_num],2], curr_choose_query[idx_list[:choose_num],3]]#(N2,c)
        # query_choose_mask = query_choosed_map[curr_choose_query[idx_list[:choose_num],0],:,curr_choose_query[idx_list[:choose_num],2], curr_choose_query[idx_list[:choose_num],3]].detach()
        del fold_query_feat_k_reshape,query_choosed_map
        torch.cuda.empty_cache()
        query_cos_center = F.cosine_similarity((query_choose_value).unsqueeze(1), (supp_choose_value).unsqueeze(0), dim=-1)
        
        return query_cos_center

    def get_similarity(self, supp_feat, curr_s_y_sdms_pos, query_feat,fold_window=3):
        padded_supp_feat_k = F.pad(supp_feat*(curr_s_y_sdms_pos), [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_supp_feat_k = padded_supp_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)

        fold_supp_feat_k_reshape = fold_supp_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_supp_feat_k.shape[0], -1, fold_supp_feat_k.shape[2], fold_supp_feat_k.shape[3])
        curr_s_y_sdms_pos_resize = F.interpolate(curr_s_y_sdms_pos, supp_feat.shape[-2:])
        supp_sdm_weight_prototype = []
        for tmp_b in range(len(curr_s_y_sdms_pos)):#每个batch单算
            curr_choose = torch.stack(torch.where(curr_s_y_sdms_pos_resize[tmp_b] > 0), dim=1)#(N,3)

            choose_value = (fold_supp_feat_k_reshape)[tmp_b,:,curr_choose[:,1], curr_choose[:,2]]#(64,N)
            curr_pro = torch.mean(choose_value,dim=-1)#(64,)  TODO 这里直接mean不太合理   但是N不固定 
            supp_sdm_weight_prototype.append(curr_pro)
        supp_sdm_weight_prototype = torch.stack(supp_sdm_weight_prototype, dim=0).unsqueeze(dim=1)#(n, 1, c)

        ##  query计算特征距离
        padded_query_feat_k = F.pad(query_feat, [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_query_feat_k = padded_query_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)


        query_feat_k_reshape = fold_query_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_query_feat_k.shape[0], -1, fold_query_feat_k.shape[2], fold_query_feat_k.shape[3])
        query_feat_k_reshape = query_feat_k_reshape.reshape(query_feat_k_reshape.shape[0], query_feat_k_reshape.shape[1], -1).permute(0, 2, 1)#(n, hw, c)
        
        query_cos_center = F.cosine_similarity(query_feat_k_reshape, supp_sdm_weight_prototype, dim=2)

        query_cos_center = query_cos_center.reshape(query_feat.shape[0], 1, query_feat.shape[-2], query_feat.shape[-1])
        return query_cos_center

    def get_pro(self, supp_feat, curr_s_y_sdms_pos,fold_window=3):
        padded_supp_feat_k = F.pad(supp_feat*(curr_s_y_sdms_pos), [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_supp_feat_k = padded_supp_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)

        fold_supp_feat_k_reshape = fold_supp_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_supp_feat_k.shape[0], -1, fold_supp_feat_k.shape[2], fold_supp_feat_k.shape[3])
        curr_s_y_sdms_pos_resize = F.interpolate(curr_s_y_sdms_pos, supp_feat.shape[-2:])
        supp_sdm_weight_prototype = []
        for tmp_b in range(len(curr_s_y_sdms_pos)):#每个batch单算
            curr_choose = torch.stack(torch.where(curr_s_y_sdms_pos_resize[tmp_b] > 0), dim=1)#(N,3)

            choose_value = (fold_supp_feat_k_reshape)[tmp_b,:,curr_choose[:,1], curr_choose[:,2]]#(64,N)
            curr_pro = torch.mean(choose_value,dim=-1)#(64,)  TODO 这里直接mean不太合理   但是N不固定 
            supp_sdm_weight_prototype.append(curr_pro)
        supp_sdm_weight_prototype = torch.stack(supp_sdm_weight_prototype, dim=0)#(n, c)

        return supp_sdm_weight_prototype


    def get_neighbour(self, supp_feat, curr_s_y_sdms_pos=None,fold_window=3):
        if curr_s_y_sdms_pos is not None:
            padded_supp_feat_k = F.pad(supp_feat*(curr_s_y_sdms_pos), [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        else:
            padded_supp_feat_k = F.pad(supp_feat, [fold_window//2, fold_window//2, fold_window//2, fold_window//2])
        fold_supp_feat_k = padded_supp_feat_k.unfold(2, fold_window, 1).unfold(3, fold_window, 1)#(n,c,h,w,k,k)

        fold_supp_feat_k_reshape = fold_supp_feat_k.permute(0, 1, 4,5, 2, 3).reshape(fold_supp_feat_k.shape[0], -1, fold_supp_feat_k.shape[2], fold_supp_feat_k.shape[3])

        return fold_supp_feat_k_reshape

    def forward(self, q_x, s_x, s_y, q_y=None, curr_epoch=1):#inpt是query
        n, shot, c, h, w = s_x.shape

        s_y_sdms = self.get_supp_sdm(s_y).to(dtype=s_y.dtype)
        s_y_sdms = - s_y_sdms

        s_y_sdms_pos = (s_y_sdms*(s_y_sdms >= 0)).to(dtype=s_x.dtype)
        s_y_sdms_neg = ((1-s_y_sdms)*(s_y_sdms >= 0)).to(dtype=s_x.dtype)
        s_y_sdms_bg = (1+s_y_sdms*(s_y_sdms < 0)).to(dtype=s_x.dtype)
        if q_y is not None:
            q_y_sdm = self.get_query_sdm(q_y).to(dtype=q_y.dtype)
            q_y_sdm = - q_y_sdm#内部正[0,1] 外部负[-1,0] 
            q_y_sdms_pos = (q_y_sdm*(q_y_sdm >= 0)).to(dtype=q_y.dtype)
            q_y_sdms_neg = (q_y_sdm*(q_y_sdm < 0)).to(dtype=q_y.dtype)
            q_y_sdms_bg = ((1+q_y_sdm)*(q_y_sdm < 0)).to(dtype=q_y.dtype)
            

        ######空间分支###########################################################
        bn, _, _, _, _, e4, e3, e2, e1 = self.seg_branch_encoder(q_x)#bottle后,四个indice，四个池化结果 e1-e4 bn (/2, /4, /8, /16 /16)

        tmp_sp_poriors = []
        tmp_sp_img_poriors = []
        for i in range(s_x.shape[1]):#shot维度

            flow = self.unet(s_x[:,i, ...], q_x)#位移场

            tmp_sp_porior = self.stn(s_y[:,i, ...], flow)#mask转换 
            tmp_sp_img_porior = self.stn(s_x[:,i, ...], flow)#img转换

            tmp_sp_poriors.append(tmp_sp_porior)
            tmp_sp_img_poriors.append(tmp_sp_img_porior)


        sp_prior = torch.cat(tmp_sp_poriors, 1).mean(1, keepdim=True)#转换后平均mask  确实有用
        sim1_list = []
        sim2_list = []
        sim3_list = []
        sim4_list = []
        e1_sp_pro_list = []
        e2_sp_pro_list = []
        e3_sp_pro_list = []
        e4_sp_pro_list = []
        ## support相关运算
        for i in range(s_x.shape[1]):
            bn_sp, _, _, _, _, e4_sp, e3_sp, e2_sp, e1_sp = self.seg_branch_encoder(s_x[:,i,...])#单独img编码 64 64 64 32 16

            ## support应该自己能区分背景和前景 但是背景部分直接均值是不行的 因为里面混杂类别会导致难以区分
            ## 如果把背景类按照

            sp_mask = F.interpolate(s_y[:,i, ...], size=(e1_sp.shape[-2:]), mode='nearest')
            
            e1_sp_pro = self.get_pro(e1_sp, sp_mask)#(n, 1, c)
            e1_sp_pro_list.append(e1_sp_pro)

            sp_mask = F.interpolate(s_y[:,i, ...], size=(e2_sp.shape[-2:]), mode='nearest')
            e2_sp_pro = self.get_pro(e2_sp, sp_mask)#(n, 1, c)
            e2_sp_pro_list.append(e2_sp_pro)

            sp_mask = F.interpolate(s_y[:,i, ...], size=(e3_sp.shape[-2:]), mode='nearest')
            e3_sp_pro = self.get_pro(e3_sp, sp_mask)#(n, 1, c)
            e3_sp_pro_list.append(e3_sp_pro)

            sp_mask = F.interpolate(s_y[:,i, ...], size=(e4_sp.shape[-2:]), mode='nearest')
            e4_sp_pro = self.get_pro(e4_sp, sp_mask)#(n, 1, c)
            e4_sp_pro_list.append(e4_sp_pro)

            sp_prior_r = F.interpolate(sp_prior, size=(e1.shape[-2:]), mode='nearest')
            e1_nei = self.get_neighbour(e1, sp_prior_r)# 这里用sp_prior_r就会导致其他区域全0 需要改 空间先验究竟怎么用

            sim1 = F.cosine_similarity(e1_nei, e1_sp_pro.unsqueeze(-1).unsqueeze(-1), 1)
            sim1_2 = F.cosine_similarity(e1_nei, self.fc2_1(e2_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim1_3 = F.cosine_similarity(e1_nei, self.fc3_1(e3_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim1_4 = F.cosine_similarity(e1_nei, self.fc4_1(e4_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim1_cat = torch.stack([sim1, sim1_2, sim1_3, sim1_4], dim=1)
            sim1_cat = torch.mean(sim1_cat, 1, keepdim=True)
            sim1_list.append(sim1_cat)

            sp_prior_r = F.interpolate(sp_prior, size=(e2.shape[-2:]), mode='nearest')
            e2_nei = self.get_neighbour(e2, sp_prior_r)
            sim2_1 = F.cosine_similarity(e2_nei, self.fc1_2(e1_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim2 = F.cosine_similarity(e2_nei, e2_sp_pro.unsqueeze(-1).unsqueeze(-1), 1)
            sim2_3 = F.cosine_similarity(e2_nei, self.fc3_2(e3_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim2_4 = F.cosine_similarity(e2_nei, self.fc4_2(e4_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)

            sim2_cat = torch.stack([sim2_1, sim2, sim2_3, sim2_4], dim=1)

            sim2_cat = torch.mean(sim2_cat, 1, keepdim=True)
            sim2_list.append(sim2_cat)

            sp_prior_r = F.interpolate(sp_prior, size=(e3.shape[-2:]), mode='nearest')
            e3_nei = self.get_neighbour(e3, sp_prior_r)
            sim3_1 = F.cosine_similarity(e3_nei, self.fc1_3(e1_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim3_2 = F.cosine_similarity(e3_nei, self.fc2_3(e2_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)
            sim3 = F.cosine_similarity(e3_nei, e3_sp_pro.unsqueeze(-1).unsqueeze(-1), 1)
            sim3_4 = F.cosine_similarity(e3_nei, self.fc4_3(e4_sp_pro).unsqueeze(-1).unsqueeze(-1), 1)

            sim3_cat = torch.stack([sim3_1, sim3_2, sim3, sim3_4], dim=1)
            sim3_cat = torch.mean(sim3_cat, 1, keepdim=True)
            sim3_list.append(sim3_cat)

             
        sim1 = torch.mean(torch.stack(sim1_list, 1), 1)
        sim2 = torch.mean(torch.stack(sim2_list, 1), 1)
        sim3 = torch.mean(torch.stack(sim3_list, 1), 1)

        e1_sp_pro_mean = torch.mean(torch.stack(e1_sp_pro_list, dim=0), dim=0)
        e2_sp_pro_mean = torch.mean(torch.stack(e2_sp_pro_list, dim=0), dim=0)
        e3_sp_pro_mean = torch.mean(torch.stack(e3_sp_pro_list, dim=0), dim=0)
        e4_sp_pro_mean = torch.mean(torch.stack(e4_sp_pro_list, dim=0), dim=0)
        expand_e4_sp_pro_mean = e4_sp_pro_mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,e4.shape[2], e4.shape[3])
        d4 = self.decode4(torch.cat([bn, expand_e4_sp_pro_mean], 1), None, None)#加上e1_sp_pro

        expand_e3_sp_pro_mean = e3_sp_pro_mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,e3.shape[2], e3.shape[3])
        d3 = self.decode3(torch.cat([d4*(sim3), expand_e3_sp_pro_mean], 1), None, None)

        expand_e2_sp_pro_mean = e2_sp_pro_mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,e2.shape[2], e2.shape[3])
        d2 = self.decode2(torch.cat([d3*(sim2), expand_e2_sp_pro_mean], 1), None, None)

        expand_e1_sp_pro_mean = e1_sp_pro_mean.unsqueeze(-1).unsqueeze(-1).repeat(1,1,e1.shape[2], e1.shape[3])
        d1 = self.decode1(torch.cat([d2*(sim1), expand_e1_sp_pro_mean], 1), None, None)


        logit = self.mask_conv(torch.cat([d1], 1))#(b, c, h/4, w/4)
        # logit = self.sim_fuse(torch.cat([logit, sim1, sim2, sim3], dim=1))
        logit = F.interpolate(logit, size=(q_x.shape[-2:]), mode='nearest')
        query_predict_active = torch.sigmoid(logit)

        
        ## 
        if q_y is not None:

            supp_predict_active_list = []
            aux_loss_1 = 0
            ## 对称计算 就是用query算原型 然后给support 分类
            for i in range(s_x.shape[1]):
                bn_sp, _, _, _, _, e4_sp, e3_sp, e2_sp, e1_sp = self.seg_branch_encoder(s_x[:,i,...])#单独img编码 64 64 64 32 16

                ## support应该自己能区分背景和前景 但是背景部分直接均值是不行的 因为里面混杂类别会导致难以区分
                ## 如果把背景类按照
                q_mask = F.interpolate(q_y, size=(e1.shape[-2:]), mode='nearest')
                e1_pro = self.get_pro(e1, q_mask)#(n, 1, c)
                q_mask = F.interpolate(q_y, size=(e2.shape[-2:]), mode='nearest')
                e2_pro = self.get_pro(e2, q_mask)#(n, 1, c)
                q_mask = F.interpolate(q_y, size=(e3.shape[-2:]), mode='nearest')
                e3_pro = self.get_pro(e3, q_mask)#(n, 1, c)
                q_mask = F.interpolate(q_y, size=(e4.shape[-2:]), mode='nearest')
                e4_pro = self.get_pro(e4, q_mask)#(n, 1, c)
                
                q_mask = F.interpolate(s_y[:,i], size=(e1_sp.shape[-2:]), mode='nearest')
                e1_sp_nei = self.get_neighbour(e1_sp, q_mask)

                sim1_sp = F.cosine_similarity(e1_sp_nei, e1_pro.unsqueeze(-1).unsqueeze(-1), 1)
                sim1_2_sp = F.cosine_similarity(e1_sp_nei, self.fc2_1(e2_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim1_3_sp = F.cosine_similarity(e1_sp_nei, self.fc3_1(e3_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim1_4_sp = F.cosine_similarity(e1_sp_nei, self.fc4_1(e4_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim1_cat_sp = torch.stack([sim1_sp, sim1_2_sp, sim1_3_sp, sim1_4_sp], dim=1)

                sim1_cat_sp = torch.mean(sim1_cat_sp, 1, keepdim=True)

                q_mask = F.interpolate(s_y[:,i], size=(e2_sp.shape[-2:]), mode='nearest')
                e2_sp_nei = self.get_neighbour(e2_sp, q_mask)
                sim2_1_sp = F.cosine_similarity(e2_sp_nei, self.fc1_2(e1_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim2_sp = F.cosine_similarity(e2_sp_nei, e2_sp_pro.unsqueeze(-1).unsqueeze(-1), 1)
                sim2_3_sp = F.cosine_similarity(e2_sp_nei, self.fc3_2(e3_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim2_4_sp = F.cosine_similarity(e2_sp_nei, self.fc4_2(e4_pro).unsqueeze(-1).unsqueeze(-1), 1)

                sim2_cat_sp = torch.stack([sim2_1_sp, sim2_sp, sim2_3_sp, sim2_4_sp], dim=1)

                sim2_cat_sp = torch.mean(sim2_cat_sp, 1, keepdim=True)


                q_mask = F.interpolate(s_y[:,i], size=(e3.shape[-2:]), mode='nearest')
                e3_sp_nei = self.get_neighbour(e3_sp, q_mask)
                sim3_1_sp = F.cosine_similarity(e3_sp_nei, self.fc1_3(e1_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim3_2_sp = F.cosine_similarity(e3_sp_nei, self.fc2_3(e2_pro).unsqueeze(-1).unsqueeze(-1), 1)
                sim3_sp = F.cosine_similarity(e3_sp_nei, e3_pro.unsqueeze(-1).unsqueeze(-1), 1)
                sim3_4_sp = F.cosine_similarity(e3_sp_nei, self.fc4_3(e4_pro).unsqueeze(-1).unsqueeze(-1), 1)

                sim3_cat_sp = torch.stack([sim3_1_sp, sim3_2_sp, sim3_sp, sim3_4_sp], dim=1)
                sim3_cat_sp = torch.mean(sim3_cat_sp, 1, keepdim=True)


                d4_sp = self.decode4(torch.cat([bn_sp, expand_e4_sp_pro_mean], 1), None, None)

                d3_sp = self.decode3(torch.cat([d4_sp*sim3_cat_sp, expand_e3_sp_pro_mean], 1), None, None)

                d2_sp = self.decode2(torch.cat([d3_sp*sim2_cat_sp, expand_e2_sp_pro_mean], 1), None, None)

                d1_sp = self.decode1(torch.cat([d2_sp*sim1_cat_sp,expand_e1_sp_pro_mean], 1), None, None)


                logit_sp = self.mask_conv(torch.cat([d1_sp], 1))#(b, c, h/4, w/4)
                logit_sp = F.interpolate(logit_sp, size=(q_x.shape[-2:]), mode='nearest')

                supp_predict_active = torch.sigmoid(logit_sp)
                supp_predict_active_list.append(supp_predict_active)


                ## 无监督loss
                # if q_y is not None:
                query_position = F.interpolate(q_y, size=(d4.shape[-2:]), mode='nearest')
                # else:
                #     query_position = F.interpolate(query_predict_active, size=(d4.shape[-2:]), mode='nearest')
                low_feat_query = d4
                low_feat_supp = d4_sp

                s_y_r = F.interpolate(s_y[:,i,...], size=(d4.shape[-2:]), mode='nearest')
                # query所有正类 与support负类 相似性越小越好
                # try:

                cos_minus1 = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, s_y_r, query_position < 0.5, fold_window=3) 
                cos_minus2 = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, 1-s_y_r, query_position > 0.5, fold_window=3)
                cos_plus1 = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, s_y_r, query_position > 0.5, fold_window=3)
                cos_plus2 = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, 1-s_y_r, query_position < 0.5, fold_window=3)

                    # rand_idx = np.random.randint(0, 4)
                    # if rand_idx == 0:
                    #     cos_loss = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, 1-s_y_r, query_position < 0.5, fold_window=3)
                    # if rand_idx == 1:
                    #     cos_loss = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, s_y_r, query_position > 0.5, fold_window=3) 
                    # if rand_idx == 2:
                    #     cos_loss = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, s_y_r, query_position < 0.5, fold_window=3)
                    # if rand_idx == 3:
                    #     cos_loss = self.get_choosed_position_similarity(low_feat_supp, low_feat_query, 1-s_y_r, query_position > 0.5, fold_window=3)
            if curr_epoch >= 3:
                cos_plus1 = torch.mean(cos_plus1)
                cos_plus2 = torch.mean(cos_plus2)
                cos_minus1 = torch.mean(cos_minus1)
                cos_minus2 = torch.mean(cos_minus2)
                if cos_plus1.isnan():
                    print("cos_plus1")
                    cos_plus1 = 0
                if cos_plus2.isnan():
                    print("cos_plus2")
                    cos_plus2 = 0
                if cos_minus1.isnan():
                    print("cos_minus1")
                    cos_minus1 = 0
                if cos_minus2.isnan():
                    print("cos_minus2")
                    cos_minus2 = 0
                aux_loss_1 += (cos_plus1+0.1*cos_plus2-cos_minus1-0.1*cos_minus2)
            else:
                 aux_loss_1 += torch.tensor(0).to(device=s_y.device, dtype=low_feat_supp.dtype)
                   
        else:
            aux_loss_1 = torch.tensor(0).to(device=s_y.device, dtype=s_y.dtype) 

        if q_y is not None:
            # supp_bce_loss = torch.mean(torch.stack([self.bce(supp_predict_active_list[i], s_y[:,i]) for i in range(len(supp_predict_active_list))], dim=0))
            query_bce_loss = self.bce(query_predict_active, q_y) 
        else:
            query_bce_loss = 0

        final_dict = {
            'query_predict':logit,
            'query_bce_loss':query_bce_loss,
            'aux_loss_1':aux_loss_1,
        }

        return final_dict






if __name__ == '__main__':

    import argparse
    import sys
    sys.path.append(r"../../")
    from util import config

    def get_configs():
        parser = argparse.ArgumentParser(description='PyTorch Few Shot Semantic Segmentation')
        parser.add_argument('--config', type=str, default='../../config/LV/settings.yaml', help='config file')
        # parser.add_argument('--mode', '-m', default='train',
        #                     help='run mode, valid values are train and eval')
        # parser.add_argument('--device', '-d', default=0,
        #                     help='device to run on')
        args = parser.parse_args()#命令行给的
        assert args.config is not None
        cfg = config.load_cfg_from_cfg_file(args.config)#yaml中解析的
        return cfg, args

    cfgs, args = get_configs()#args没用到 其实也不需要有
    print(cfgs.DATA)
    net_params= cfgs.NETWORK


    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    few_shot_model = RAP(net_params)#其实只需要 params['num_channels'] = 1 params['num_filters'] 
    few_shot_model.to(device=device)

    import torchinfo
    nchannel = 1
    batch_size = 2
    shot = 1
    img_size = 224


    q_x = torch.rand((batch_size, nchannel, img_size, img_size)).to(device=device)
    
    s_x = torch.rand(( batch_size,shot,nchannel, img_size, img_size)).to(device=device)
    s_y = (torch.ones(( batch_size,shot,1, img_size, img_size))>0.5).to(dtype=torch.float32).to(device=device)
    q_y = (torch.ones((batch_size,1, img_size, img_size))>0.5).to(dtype=torch.float32).to(device=device)
    a = few_shot_model(q_x, s_x, s_y, q_y)
    # torchinfo.summary(few_shot_model, input_size=[(batch_size, 1, img_size, img_size), ( batch_size,shot, 1, img_size, img_size),  ( batch_size,shot, 1, img_size, img_size)])
