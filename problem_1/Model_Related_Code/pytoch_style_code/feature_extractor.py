import numpy as np
import cv2
from skimage.feature import hog
from skimage.morphology import skeletonize
from skimage.filters import gabor
# from mahotas.features import zernike_moments
from scipy.ndimage import convolve
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    MNIST 手写数字特征提取类

    输入:
        28x28 灰度图 (numpy array)
    输出:
        一维特征向量
    """

    def __init__(self, use_binary=True, zoning_grid=(4,4), zernike_radius=14):
        """
        参数
        ----------
        use_binary : bool
            是否进行二值化处理（轮廓/投影/骨架/Hu矩等特征通常需要）
        zoning_grid : tuple[int,int]
            Zoning 分区格子 (行, 列)
        zernike_radius : int
            Zernike矩半径
        """
        self.use_binary = use_binary
        self.zoning_grid = zoning_grid
        self.zernike_radius = zernike_radius

    def preprocess(self, img):
        """
        基本预处理

        - 转换为 uint8
        - 可选二值化

        返回
        -------
        img : 原始灰度图
        binary : 二值图（用于形状特征）
        """
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if self.use_binary:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        else:
            binary = img

        return img, binary

    def contour_features(self, binary):
        """
        轮廓特征

        返回：
            [轮廓面积, 轮廓周长]
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return [0, 0]
        cnt = max(contours, key=cv2.contourArea)
        return np.array([cv2.contourArea(cnt), cv2.arcLength(cnt, True)])

    def projection_features(self, binary):
        """
        水平和垂直投影特征

        返回：
            拼接的投影向量
        """
        horizontal = np.sum(binary, axis=1)
        vertical = np.sum(binary, axis=0)
        return np.concatenate([horizontal, vertical])

    def fourier_features(self, img):
        """
        傅里叶低频特征（中心系数）
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        center = magnitude[10:18, 10:18]
        return center.flatten()

    def structural_features(self, img):
        """
        HOG特征（方向梯度直方图）
        """
        return hog(img, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=False)

    def statistical_features(self, img):
        """
        基本统计特征：均值、标准差、方差、最大、最小
        """
        return np.array([np.mean(img), np.std(img), np.var(img), np.max(img), np.min(img)])

    def zoning_features(self, binary):
        """
        分区像素密度特征
        """
        h, w = binary.shape
        gh, gw = self.zoning_grid
        features = []
        for i in range(gh):
            for j in range(gw):
                cell = binary[i*h//gh:(i+1)*h//gh, j*w//gw:(j+1)*w//gw]
                features.append(np.sum(cell)/255)
        return np.array(features)

    def hu_moments_features(self, binary):
        """
        Hu矩特征（7维，不变形状特征）
        """
        moments = cv2.moments(binary)
        hu = cv2.HuMoments(moments).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu)+1e-10)

    def skeleton_features(self, binary):
        """
        骨架特征：端点数和交叉点数
        """
        skel = skeletonize(binary // 255)
        kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
        conv = convolve(skel.astype(np.uint8), kernel)
        endpoints = np.sum(conv==11)
        junctions = np.sum(conv>=13)
        return np.array([endpoints, junctions])

    def gabor_features(self, img):
        """
        Gabor滤波特征（多方向、多频率）
        """
        features = []
        frequencies = [0.1,0.2,0.3]
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for freq in frequencies:
            for theta in thetas:
                filt_real, _ = gabor(img, frequency=freq, theta=theta)
                features.append(filt_real.mean())
                features.append(filt_real.std())
        return np.array(features)

    # def zernike_features(self, binary):
    #     """
    #     Zernike矩特征（旋转不变全局形状）
    #     """
    #     img_square = binary.astype(np.uint8)
    #     try:
    #         features = zernike_moments(img_square, radius=self.zernike_radius)
    #     except:
    #         features = np.zeros(25)
    #     return features

    def flatten_image(self, img):
        """
        将图像展平为一维向量

        参数:
            img : np.ndarray, 灰度图 (28x28)

        返回:
            np.ndarray, shape (784,)
        """
        # 确保是 uint8 灰度图
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # 展平
        return img.flatten()

def extract_features_batch(images):
    """
    将图像批量提取特征

    参数:
        images : np.ndarray, 形状 (n_samples, 28, 28)

    返回:
        np.ndarray, 形状 (n_samples, n_features)
    """
    extractor = FeatureExtractor()
    all_feats = []
    for img in images:
        feat = np.concatenate([
            extractor.projection_features(img),
            extractor.fourier_features(img),
            extractor.structural_features(img),
            extractor.hu_moments_features(img),
            extractor.zoning_features(img)
        ])
        all_feats.append(feat)
    all_feats = StandardScaler().fit_transform(all_feats)
    return np.array(all_feats)

def fourier_batch_process(images):
    """
    将图像批量提取特征

    参数:
        images : np.ndarray, 形状 (n_samples, 28, 28)

    返回:
        np.ndarray, 形状 (n_samples, n_features)
    """
    extractor = FeatureExtractor()
    all_feats = []
    for img in images:
        feat = extractor.fourier_features(img)
        all_feats.append(feat)
    all_feats = StandardScaler().fit_transform(all_feats)
    return np.array(all_feats)