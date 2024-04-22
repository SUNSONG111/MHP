import torch
import torch.nn.functional as F

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float() # shape = 2,h,w
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords,r=3, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)#(n h w, 2r+1, 2r+1, 1) 值在-r 到 r  每个像素都采样中心邻域 (2r+1)*(2r+1)
    xgrid = 2*xgrid/(W-1) - 1#这里没懂 为什么  全小于0了 就是全在图像左上角
    ygrid = 2*ygrid/(H-1) - 1
    # xgrid = xgrid/W#除以W 代表取的是中心区域 如果W足够大 就是中间那一个点 这种采样的意义是什么？
    # ygrid = ygrid/H

    grid = torch.cat([xgrid, ygrid], dim=-1)## 不是-1到1 
    img = F.grid_sample(img, grid, align_corners=True)#(392,1,14,14) (392,7,7,2) -> (392,1,7,7)  按照grid的坐标 对img进行采样  最后形状是和grid的中间两个维度相同  F.grid_sample的相对坐标是从左上角-1开始算 右下角1  

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def Correlation(fmap1, fmap2, r=3):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)#这里已经计算了相关性
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())
    corr = corr.view(-1, 1, ht, wd)#(b h w, 1, h, w)  这个和坐标没关系

    coords = coords_grid(batch, ht, wd).to(fmap1.device)#(n,2,h,w) 是特征图的坐标格网 整数
    coords = coords.permute(0, 2, 3, 1)#(n,h,w,2)
    batch, h1, w1, _ = coords.shape
    dx = torch.linspace(-r, r, 2 * r + 1)
    dy = torch.linspace(-r, r, 2 * r + 1)# n h w个像素 每个像素的坐标都加了-r到r 然后得到coords_lvl  
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)#偏移的距离 -r到r (2r+1, 2r+1, 2) 偏移到底代表啥 

    centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
    delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    coords_lvl = centroid_lvl + delta_lvl#整数的取邻域坐标  (n h w, 1, 1, 2)和(1, 2r+1, 2r+1, 2)相加 得到(n h w, 2r+1, 2r+1, 2) 其中 nhw 代表所有的像素点， 每个像素点都有 2r+1个邻域

    corr = bilinear_sampler(corr, coords_lvl, r=r)#这里把相关性图根据coords_lvl坐标重采样 
    corr = corr.view(batch, h1, w1, -1)#(n,h, w, 2r+1*2r+1)
    out = corr.permute(0, 3, 1, 2).contiguous().float()#(n,2r+1*2r+1,h,w)

    return out

if __name__ == '__main__':
    fmap1 = torch.ones(1,1,3,3)
    fmap2 = torch.ones(1, 1, 3, 3)
    out = Correlation(fmap1, fmap2)
    print(out.shape)
