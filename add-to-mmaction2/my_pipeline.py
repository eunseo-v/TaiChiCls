import numpy as np
from ..builder import PIPELINES
from .augmentations import Flip
import copy as cp

@PIPELINES.register_module()
class GenerateTaiChi17PoseTarget:
    '''
        与COCO-kp17的关节点相匹配
        0鼻子	 13脖子 	
        1左眼	 14头
        2右眼	 14头
        3左耳	 15头ES
        4右耳	 15头ES

        5左肩    44左肩
        7左手肘	 45左臂
        9左手腕  47左手

        6右肩	 16右肩
        8右手肘	 17右臂
        10右手腕 19右手

        11左臀部 5左腿上部
        13左膝盖 6左腿
        15左脚踝 7左脚

        12右臀部 1右腿上部
        14右膝盖 2右腿
        16右脚踝 3右脚

        对应skeletons
        (0,1)	(13,14)
        (0,2)	(13,14)
        (1,3)	(14,15)
        (2,4)	(14,15)
        (0,5)	(13,44)
        (5,7)	(44,45)
        (7,9)	(45,47)
        (0,6)	(13,16)
        (6,8)	(16,17)
        (8,10)	(17,19)
        (5,11)	(44,5)
        (11,13)	(5,6)
        (13,15)	(6,7)
        (6,12)	(16,1)
        (12,14)	(1,2)
        (14,16)	(2,3)
        (11,12)	(5,1)
    '''
    pass
    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 keypoints = (13, 14, 14, 15, 15, 
                            44, 16, 45, 17, 47, 19, 
                            5, 1, 6, 2, 7, 3),
                 skeletons=((13, 14), (13, 14), (14, 15), (14, 15), (13, 44),
                            (44,45), (45, 47), (13, 16), (16, 17), (17, 19), (44, 5),
                            (5, 6), (6, 7), (16, 1), (1, 2), (2, 3), (5, 1)),
                 double=False,
                 left_kp=(5, 6, 7, 44, 45, 47),
                 right_kp=(1, 2, 3, 16, 17, 19)):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.keypoints = keypoints
        self.skeletons = skeletons
    
    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap
    
    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        '''Generate pseudo heap for specified keypoints and limbs
        Args:
            img_h, img_w : height and width of the heatmap
            kps: [N, V, C]
        Returns: np.ndarray
        '''
        heatmaps = []
        if self.with_kp:
            for kp in self.keypoints:
                heatmap = self.generate_a_heatmap(
                    img_h, img_w, kps[:, kp],
                    sigma, max_values[:, kp]
                )
                heatmaps.append(heatmap)
        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]
                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(
                    img_h, img_w, starts, ends,
                    sigma, start_values, end_values
                )
                heatmaps.append(heatmap)
        return np.stack(heatmaps, axis=-1)


    def gen_an_aug(self, results):
        '''Generate pseudo heatmaps for all frames.
        Args:
            results (dict): The dictionary that contains all info of a sample.
        Returns:
            list[np.ndarray]: The generated pseudo heatmaps
        '''
        all_kps = results['keypoint'] # [N, T, V, C]
        kp_shape = all_kps.shape
        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score'] # [N, T, V]
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)
        
        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]
        imgs=[]
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i] # [N, V, C]
            kpscores = all_kpscores[:, i] # [N, V]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores
            hmap = self.generate_heatmap(
                img_h, img_w, kps, sigma, max_values
            )
            imgs.append(hmap)
        return imgs            
    
    def __call__(self, results):
        '''
         results['imgs'] size[N, T, V=节点数, C]
         期望输出:
            [T, H, W, Vnew=17]
        '''
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)]) # [2T, H, W, C]
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'keypoints={self.keypoints}'
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str

@PIPELINES.register_module()
class GenerateNTUPoseTarget:
    '''
        原方法将17个关节点作为通道数，这里为了与太极拳数据集格式相同
        将人体分成躯干和四肢一共5个部分，每一部分作为一个通道
        对于keypoint: 
        躯干部分: 0, 1, 2, 3, 4  鼻子/左眼/右眼/左耳/右耳
        左臂部分: 5, 7, 9 左肩/左手肘/左手腕
        右臂部分: 6, 8, 10 右肩/右手肘/右手腕
        左下肢: 11, 13, 15 左臀部/左膝盖/左脚踝
        右下肢: 12, 14, 16 右臀部/右膝盖/右脚踝

        对于limmb:
        躯干部分: (0, 1), (0, 2), (1, 3), (2, 4), (11, 12) 
        左臂部分: (0, 5), (5, 7), (7, 9)
        右臂部分: (0, 6), (6, 8), (8, 10)
        左下肢: (5, 11), (11, 13), (13, 15)
        右下肢: (6, 12), (12, 14), (14, 16)
    '''
    def __init__(
        self, 
        sigma= 0.6,
        use_score = True,
        with_kp = True,
        with_limb = False,
        keypoints_group = [
            [0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10],
            [11, 13, 15], [12, 14, 16]
        ],
        limbs_group = [
            ((0, 1), (0, 2), (1, 3), (2, 4), (11, 12)),
            ((0, 5), (5, 7), (7, 9)),
            ((0, 6), (6, 8), (8, 10)),
            ((5, 11), (11, 13), (13, 15)),
            ((6, 12), (12, 14), (14, 16))
        ],
        double = False,
        left_kp = (1, 3, 5, 7, 9, 11, 13, 15),
        right_kp = (2, 4, 6, 8, 10, 12, 14, 16)
    ):
        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.left_kp = left_kp
        self.right_kp = right_kp
        # 判断有效输入
        self.eps = 1e-4
        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.'
        )
        self.keypoints_group = keypoints_group
        self.limbs_group = limbs_group

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_kpgroup_heatmap(self, img_h, img_w, centers, sigma, max_values):
        '''
           生成一帧内的一个通道的heatmap
           Args: 
                centers: [N, 一个通道内的节点数, C=2]
                max_values: [N, 一个通道内的节点数]
            Returns:
                np.ndarray: [H, W] 
        '''
        heatmap = np.zeros([img_h, img_w], dtype = np.float32)
        for center, max_value in zip(centers, max_values):
            for joint, score in zip(center, max_value):
                # joint: [C], score: float
                mu_x, mu_y = joint[0], joint[1]
                if score < self.eps:
                    continue
                st_x = max(int(mu_x - 3* sigma), 0) # 点坐标的最小值
                ed_x = min(int(mu_x + 3* sigma)+1, img_w) # 点坐标的最大值
                st_y = max(int(mu_y - 3* sigma), 0)
                ed_y = min(int(mu_y + 3* sigma)+1, img_h)
                x = np.arange(st_x, ed_x, 1, np.float32)
                y = np.arange(st_y, ed_y, 1, np.float32)
                # if the keypoint not in the heatmap coordinate system
                if not (len(x) and len(y)):
                    continue
                y = y[:, None]
                patch = np.exp(
                    -((x - mu_x)**2 + (y - mu_y)**2)/2/sigma**2
                )
                patch = patch* score
                # 因为定义图像是H, W，所以y在前，x在后
                heatmap[
                    st_y: ed_y,
                    st_x: ed_x
                ] = np.maximum(
                    heatmap[
                        st_y: ed_y,
                        st_x: ed_x
                    ], patch
                ) # 按点更新数值
        return heatmap
    
    def generate_lbgroup_heatmap(self, img_h, img_w, sigma, kps, max_values, limbs):
        '''
            生成一帧内的一个通道包含的limbs的heatmap
            Args:
                img_h, img_w: int 图片的尺寸
                sigma: float
                kps: [N, V, C] 各关节点的数据
                max_values: [N, V] 各关节点的置信度
                limbs: 该通道的输出包含哪些骨骼信息
            Return:
                heatmap: [H, W] np.ndarray 该通道生成的heatmap
        '''
        heatmap = np.zeros([img_h, img_w], dtype = np.float32)
        for limb in limbs:
            start_idx, end_idx = limb
            starts = kps[:, start_idx]  # [N, C]
            ends = kps[:, end_idx] # [N, C]

            start_values = max_values[:, start_idx] 
            end_values = max_values[:, end_idx]
            for start, end, start_value, end_value in zip(
                starts, ends, start_values, end_values
            ): # 每一个人的数据
                value_coeff = min(start_value, end_value)
                if value_coeff < self.eps:
                    continue

                min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
                min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

                min_x = max(int(min_x - 3 * sigma), 0)
                max_x = min(int(max_x + 3 * sigma)+1, img_w)
                min_y = max(int(min_y - 3 * sigma), 0)
                max_y = min(int(max_y + 3 * sigma)+1, img_h)

                x = np.arange(min_x, max_x, 1, np.float32)
                y = np.arange(min_y, max_y, 1, np.float32)

                if not (len(x) and len(y)):
                    continue
            
                y = y[:, None]
                x_0 = np.zeros_like(x)
                y_0 = np.zeros_like(y)

                # 其中点x,y到起始关节点的距离
                d2_start = ((x - start[0])**2 + (y - start[1])**2)
                # 到终止关节点的距离
                d2_end = ((x - end[0])**2 + (y - end[1])**2)
                # 骨骼长度
                d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

                # 如果长度小于一个像素值，直接画点
                if d2_ab <1:
                    full_heatmap = self.generate_a_heatmap(
                        img_h, img_w, [start], sigma, [start_value]
                    )
                    heatmap = np.maximum(heatmap, full_heatmap)
                    continue
                
                coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

                a_dominate = coeff <= 0
                b_dominate = coeff >= 1
                seg_dominate = 1 - a_dominate - b_dominate

                position = np.stack([x + y_0, y + x_0], axis=-1)
                projection = start + np.stack([coeff, coeff], axis=-1) * (
                    end - start)
                d2_line = position - projection
                d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
                d2_seg = (
                    a_dominate * d2_start + b_dominate * d2_end +
                    seg_dominate * d2_line)

                patch = np.exp(-d2_seg / 2. / sigma**2)
                patch = patch * value_coeff

                heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                    heatmap[min_y:max_y, min_x:max_x], patch)
        return heatmap        

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        '''
            生成一帧内的keypoints或者limbs的heatmap
            Returns:
                np.ndarray: The generated pseudo heatmap [H, W, Vnew=5]
        '''
        heatmaps = []
        if self.with_kp:
            num_channel = len(self.keypoints_group)
            for i in range(num_channel):
                heatmap = self.generate_kpgroup_heatmap(
                    img_h = img_h, img_w = img_w,
                    centers = kps[:, self.keypoints_group[i]],
                    sigma=sigma,
                    max_values= max_values[:, self.keypoints_group[i]]
                )
                heatmaps.append(heatmap) # list,元素内size是[H, W]
        
        if self.with_limb:
            for limbs in self.limbs_group: # 一个通道内的骨架连接坐标对
                heatmap = self.generate_lbgroup_heatmap(
                    img_h=img_h, img_w=img_w, sigma=sigma,
                    kps=kps, max_values=max_values, limbs = limbs
                )
                heatmaps.append(heatmap)
        return np.stack(heatmaps, axis = -1)
            

    def gen_an_aug(self, results):
        '''
            按帧生成heatmap
            Returns:
                list[np.ndarray]: The generated pseudo heatmap
        '''
        all_kps = results['keypoint'] # [N, T, V, C]
        kp_shape = all_kps.shape
        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score'] # [N, T, V]
        else:
            all_kpscores = np.ones(
                kp_shape[:-1], dtype = np.float32
            )
        img_h, img_w = results['img_shape']
        num_frames = kp_shape[1]
        imgs = []
        for i in range(num_frames):
            sigma = self.sigma
            kps = all_kps[:, i] # [N, V, C]
            kpscores = all_kpscores[:, i] # [N, V]
            max_values = np.ones(kpscores.shape, dtype = np.float32)
            if self.use_score:
                max_values = kpscores
            
            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)
        return imgs



    def __call__(self, results):
        '''
         results['imgs'] size[N, T, V=节点数, C]
         期望输出:
            [T, H, W, Vnew=5]
        '''
        if not self.double:
            results['imgs'] = np.stack(
                self.gen_an_aug(results)
            )
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio = 1, left_kp=self.left_kp, right_kp = self.right_kp
            )
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [
                    self.gen_an_aug(results),
                    self.gen_an_aug(results_)
                ]
            ) # [2T, H, W, C]
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'sigma = {self.sigma}, '
            f'use_score = {self.use_score}, '
            f'with_kp = {self.with_kp}'
            f'with_limb = {self.with_limb}'
            f'keypoints_group = {self.keypoints_group}'
            f'limbs_group = {self.limbs_group})'
        )
        return repr_str

@PIPELINES.register_module()
class GenerateTaiChiPoseTarget:
    '''
        与GenerateNTUPoseTarget编码格式相同
        对于keypoint:
        躯干部分: 0, 9, 10, 11, 12, 13, 14, 15 臀/脊柱0.1.2.3/脖子/头/头endsite
        左臂部分: 44, 45, 46, 47, 48-71 左肩/左臂/左前臂/左手/左手指节点相关
        右臂部分: 16, 17, 18, 19, 20-43 右...
        左下肢: 5, 6, 7, 8 左腿上部/左腿/左脚/左脚endsite
        右下肢: 1, 2, 3, 4 右...

        对于limb:
        躯干部分: (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        左臂部分: (12, 44), (44, 45), (45, 46), (46, 47),
                (47, 48), (47, 52), (47, 57), (47, 62), (47, 67),
                (48, 49), (49, 50), (50, 51),
                (52, 53), (53, 54), (54, 55), (55, 56),
                (57, 58), (58, 59), (59, 60), (60, 61),
                (62, 63), (63, 64), (64, 65), (65, 66),
                (67, 68), (68, 69), (69, 70), (70, 71)
        右臂部分: (12, 16), (16, 17), (17, 18), (18, 19),
                (19, 20), (19, 24), (19, 29), (19, 34), (19, 39),
                (20, 21), (21, 22), (22, 23),
                (24, 25), (25, 26), (26, 27), (27, 28),
                (29, 30), (30, 31), (31, 32), (32, 33),
                (34, 35), (35, 36), (36, 37), (37, 38),
                (39, 40), (40, 41), (41, 42), (42, 43),
        左下肢: (5, 6), (6, 7), (7, 8),
        右下肢: (1, 2), (2, 3), (3, 4),
    '''
    def __init__(
        self, 
        sigma= 0.6,
        use_score = True,
        with_kp = True,
        with_limb = False,
        keypoints_group = [
            [0, 9, 10, 11, 12, 13, 14, 15], 
            [
                44, 45, 46, 47, 48, 49, 
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
                70, 71
            ], 
            [
                16, 17, 18, 19, 
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43
            ],
            [5, 6, 7, 8], 
            [1, 2, 3, 4]
        ],
        limbs_group = [
            (
                (0, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)
            ),
            (
                (12, 44), (44, 45), (45, 46), (46, 47),
                (47, 48), (47, 52), (47, 57), (47, 62), (47, 67),
                (48, 49), (49, 50), (50, 51),
                (52, 53), (53, 54), (54, 55), (55, 56),
                (57, 58), (58, 59), (59, 60), (60, 61),
                (62, 63), (63, 64), (64, 65), (65, 66),
                (67, 68), (68, 69), (69, 70), (70, 71)
            ),
            (
                (12, 16), (16, 17), (17, 18), (18, 19),
                (19, 20), (19, 24), (19, 29), (19, 34), (19, 39),
                (20, 21), (21, 22), (22, 23),
                (24, 25), (25, 26), (26, 27), (27, 28),
                (29, 30), (30, 31), (31, 32), (32, 33),
                (34, 35), (35, 36), (36, 37), (37, 38),
                (39, 40), (40, 41), (41, 42), (42, 43)
            ),
            (
                (5, 6), (6, 7), (7, 8)
            ),
            (
                (1, 2), (2, 3), (3, 4)
            )
        ],
        double = False,
        left_kp = (
            5, 6, 7, 8, 44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
            70, 71
        ),
        right_kp = (
            1, 2, 3, 4, 16, 17, 18, 19, 
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43
        )
    ):
        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.left_kp = left_kp
        self.right_kp = right_kp
        # 判断有效输入
        self.eps = 1e-4
        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.'
        )
        self.keypoints_group = keypoints_group
        self.limbs_group = limbs_group

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_kpgroup_heatmap(self, img_h, img_w, centers, sigma, max_values):
        '''
           生成一帧内的一个通道的heatmap
           Args: 
                centers: [N, 一个通道内的节点数, C=2]
                max_values: [N, 一个通道内的节点数]
            Returns:
                np.ndarray: [H, W] 
        '''
        heatmap = np.zeros([img_h, img_w], dtype = np.float32)
        for center, max_value in zip(centers, max_values):
            for joint, score in zip(center, max_value):
                # joint: [C], score: float
                mu_x, mu_y = joint[0], joint[1]
                if score < self.eps:
                    continue
                st_x = max(int(mu_x - 3* sigma), 0) # 点坐标的最小值
                ed_x = min(int(mu_x + 3* sigma)+1, img_w) # 点坐标的最大值
                st_y = max(int(mu_y - 3* sigma), 0)
                ed_y = min(int(mu_y + 3* sigma)+1, img_h)
                x = np.arange(st_x, ed_x, 1, np.float32)
                y = np.arange(st_y, ed_y, 1, np.float32)
                # if the keypoint not in the heatmap coordinate system
                if not (len(x) and len(y)):
                    continue
                y = y[:, None]
                patch = np.exp(
                    -((x - mu_x)**2 + (y - mu_y)**2)/2/sigma**2
                )
                patch = patch* score
                # 因为定义图像是H, W，所以y在前，x在后
                heatmap[
                    st_y: ed_y,
                    st_x: ed_x
                ] = np.maximum(
                    heatmap[
                        st_y: ed_y,
                        st_x: ed_x
                    ], patch
                ) # 按点更新数值
        return heatmap
    
    def generate_lbgroup_heatmap(self, img_h, img_w, sigma, kps, max_values, limbs):
        '''
            生成一帧内的一个通道包含的limbs的heatmap
            Args:
                img_h, img_w: int 图片的尺寸
                sigma: float
                kps: [N, V, C] 各关节点的数据
                max_values: [N, V] 各关节点的置信度
                limbs: 该通道的输出包含哪些骨骼信息
            Return:
                heatmap: [H, W] np.ndarray 该通道生成的heatmap
        '''
        heatmap = np.zeros([img_h, img_w], dtype = np.float32)
        for limb in limbs:
            start_idx, end_idx = limb
            starts = kps[:, start_idx]  # [N, C]
            ends = kps[:, end_idx] # [N, C]

            start_values = max_values[:, start_idx] 
            end_values = max_values[:, end_idx]
            for start, end, start_value, end_value in zip(
                starts, ends, start_values, end_values
            ): # 每一个人的数据
                value_coeff = min(start_value, end_value)
                if value_coeff < self.eps:
                    continue

                min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
                min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

                min_x = max(int(min_x - 3 * sigma), 0)
                max_x = min(int(max_x + 3 * sigma)+1, img_w)
                min_y = max(int(min_y - 3 * sigma), 0)
                max_y = min(int(max_y + 3 * sigma)+1, img_h)

                x = np.arange(min_x, max_x, 1, np.float32)
                y = np.arange(min_y, max_y, 1, np.float32)

                if not (len(x) and len(y)):
                    continue
            
                y = y[:, None]
                x_0 = np.zeros_like(x)
                y_0 = np.zeros_like(y)

                # 其中点x,y到起始关节点的距离
                d2_start = ((x - start[0])**2 + (y - start[1])**2)
                # 到终止关节点的距离
                d2_end = ((x - end[0])**2 + (y - end[1])**2)
                # 骨骼长度
                d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

                # 如果长度小于一个像素值，直接画点
                if d2_ab <1:
                    full_heatmap = self.generate_a_heatmap(
                        img_h, img_w, [start], sigma, [start_value]
                    )
                    heatmap = np.maximum(heatmap, full_heatmap)
                    continue
                
                coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

                a_dominate = coeff <= 0
                b_dominate = coeff >= 1
                seg_dominate = 1 - a_dominate - b_dominate

                position = np.stack([x + y_0, y + x_0], axis=-1)
                projection = start + np.stack([coeff, coeff], axis=-1) * (
                    end - start)
                d2_line = position - projection
                d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
                d2_seg = (
                    a_dominate * d2_start + b_dominate * d2_end +
                    seg_dominate * d2_line)

                patch = np.exp(-d2_seg / 2. / sigma**2)
                patch = patch * value_coeff

                heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                    heatmap[min_y:max_y, min_x:max_x], patch)
        return heatmap        

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        '''
            生成一帧内的keypoints或者limbs的heatmap
            Returns:
                np.ndarray: The generated pseudo heatmap [H, W, Vnew=5]
        '''
        heatmaps = []
        if self.with_kp:
            num_channel = len(self.keypoints_group)
            for i in range(num_channel):
                heatmap = self.generate_kpgroup_heatmap(
                    img_h = img_h, img_w = img_w,
                    centers = kps[:, self.keypoints_group[i]],
                    sigma=sigma,
                    max_values= max_values[:, self.keypoints_group[i]]
                )
                heatmaps.append(heatmap) # list,元素内size是[H, W]
        
        if self.with_limb:
            for limbs in self.limbs_group: # 一个通道内的骨架连接坐标对
                heatmap = self.generate_lbgroup_heatmap(
                    img_h=img_h, img_w=img_w, sigma=sigma,
                    kps=kps, max_values=max_values, limbs = limbs
                )
                heatmaps.append(heatmap)
        return np.stack(heatmaps, axis = -1)
            

    def gen_an_aug(self, results):
        '''
            按帧生成heatmap
            Returns:
                list[np.ndarray]: The generated pseudo heatmap
        '''
        all_kps = results['keypoint'] # [N, T, V, C]
        kp_shape = all_kps.shape
        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score'] # [N, T, V]
        else:
            all_kpscores = np.ones(
                kp_shape[:-1], dtype = np.float32
            )
        img_h, img_w = results['img_shape']
        num_frames = kp_shape[1]
        imgs = []
        for i in range(num_frames):
            sigma = self.sigma
            kps = all_kps[:, i] # [N, V, C]
            kpscores = all_kpscores[:, i] # [N, V]
            max_values = np.ones(kpscores.shape, dtype = np.float32)
            if self.use_score:
                max_values = kpscores
            
            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)
        return imgs



    def __call__(self, results):
        '''
         results['imgs'] size[N, T, V=节点数, C]
         期望输出:
            [T, H, W, Vnew=5]
        '''
        if not self.double:
            results['imgs'] = np.stack(
                self.gen_an_aug(results)
            )
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio = 1, left_kp=self.left_kp, right_kp = self.right_kp
            )
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [
                    self.gen_an_aug(results),
                    self.gen_an_aug(results_)
                ]
            ) # [2T, H, W, C]
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'sigma = {self.sigma}, '
            f'use_score = {self.use_score}, '
            f'with_kp = {self.with_kp}'
            f'with_limb = {self.with_limb}'
            f'keypoints_group = {self.keypoints_group}'
            f'limbs_group = {self.limbs_group})'
        )
        return repr_str