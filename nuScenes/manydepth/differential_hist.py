import torch
import torch.nn.functional as F

class SmoothDifferentiableHistogram(torch.nn.Module):
    def __init__(self, bins, min_val, max_val, sigma=0.1):
        """
        初始化可微分直方图构造器。
        :param bins: 直方图的bin数量
        :param min_val: 值的最小范围
        :param max_val: 值的最大范围
        :param sigma: 控制平滑程度的参数
        """
        super(SmoothDifferentiableHistogram, self).__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma

        # 计算 bin 的宽度
        self.bin_width = (max_val - min_val) / bins

        # 计算 bin 的中心点
        self.bin_centers = torch.linspace(min_val + 0.5 * self.bin_width, max_val - 0.5 * self.bin_width, bins)

    def logistic_function(self, z):
        """
        计算逻辑回归函数 σ(z)。
        :param z: 输入值
        :return: σ(z) 值
        """
        return 1 / (1 + torch.exp(-z))

    def kernel(self, z):
        """
        核函数 K(z) = σ(z) * σ(-z)，即为逻辑回归函数的导数。
        :param z: 输入值
        :return: K(z) 核值
        """
        sigma_z = self.logistic_function(z)
        return sigma_z * (1 - sigma_z)

    def forward(self, inputs):
        """
        输入为一张图片或一个 batch 的张量，计算可微分直方图。
        :param inputs: 输入的图像或深度值, (B, H, W)
        :return: 可微分直方图
        """
        # 将输入展平为一维
        inputs = inputs.view(-1, 1)  # (N, 1)，将输入转换为 (N, 1) 形式

        # 计算每个输入值到各 bin 中心点的距离
        distances = (inputs - self.bin_centers.to(inputs.device)) / self.bin_width

        # 计算核函数 K(z) = σ(z)σ(-z)
        kernel_values = self.kernel(distances / self.sigma)

        # 对每个 bin 求和并归一化
        histogram = kernel_values.sum(dim=0)

        # 归一化直方图
        histogram = histogram / histogram.sum()

        return histogram


def differentiable_histogram(img, num_bins, min_depth, max_depth):
    """
    Compute a differentiable histogram for an image tensor using two sigmoids.
    
    Args:
        img (torch.Tensor): The input image tensor of size (N, H, W).
        num_bins (int): The number of bins for the histogram.
        min_depth (float): The minimum depth value.
        max_depth (float): The maximum depth value.
        width (float): The width W that controls the smoothness of the sigmoid.
    
    Returns:
        hist (torch.Tensor): A tensor of shape (N, num_bins) representing the histogram.
    """
    N, H, W = img.shape  # Assume input is (N, H, W)
    L = (max_depth - min_depth) / num_bins  # Bin width
    width = L/2
    bin_centers = torch.linspace(min_depth + L / 2, max_depth - L / 2, num_bins).to(img.device)  # bin centers

    img_flat = img.view(N, -1)  # Flatten to (N, H*W)

    # Calculate the bin probabilities using two sigmoids
    hist = torch.zeros((N, num_bins), device=img.device)  # Initialize histogram

    for i, mu_k in enumerate(bin_centers):
        upper_sigmoid = torch.sigmoid((img_flat - (mu_k - L / 2)) / width)
        lower_sigmoid = torch.sigmoid((img_flat - (mu_k + L / 2)) / width)
        hist[:, i] = torch.sum(upper_sigmoid - lower_sigmoid, dim=1)  # Sum over pixels

    # Normalize the histogram by the total number of pixels
    hist = hist / (H * W)

    return hist

# 示例
if __name__ == "__main__":
    # 创建模拟数据 (例如深度图)
    depth_map = torch.rand((1, 64, 64)) * 10  # 假设深度值在 0 到 10 之间
    min_depth = 0
    max_depth = 10
    bins = 100

    # 初始化平滑可微分直方图生成器
    hist_gen = SmoothDifferentiableHistogram(bins=bins, min_val=min_depth, max_val=max_depth, sigma=0.05)

    # 生成直方图
    histogram = hist_gen(depth_map)
    print("Smooth Differentiable Histogram:", histogram)

    # 反向传播测试
    histogram.sum().backward()
    print("Gradients successfully propagated!")
