### 模型与数据处理
#### 数据处理
* 针对数据集的pipeline处理中，class GeneratePoseTarget，当设置double=False时，输出的数据size [T, H, W, C]；设置double=True时，输出的数据size [2T, H, W, C]. 但是后续都连接相同的FormatShape+Collect+ToTensor类后，double= True时，一个batch的输入size为[B, 2*num_clip, C, T, H, W]; double= False时，一个batch的输入size为[B, num_clip, C, T, H, W]。若将double=True应用在train pipeline，训练时会报错；在val pipeline或者test pipeline，配合average_clips(在test_cfg参数中设置)可以运行。
* 出现上一条状况的原因解释。在class Recognizer3D的forward_train中，输入的6维数据[B, N_crops*N_clips, C, T, H, W]被reshape成5维数据 [B\* N_crops\* N_clips, C, T, H, W]，但label数目未变，训练时会出现维度不匹配的现象。在forward_test中，会首先记录N_crops\*N_clips的值，进行average_clips操作，得到该样本不同clip的平均分类结果作为该样本的分类结果。


### 训练配置
#### 如何使用训练过的识别器作为主干网络的预训练模型？
1.如果想对整个网络使用预训练模型，可以在配置文件中，将 load_from 设置为预训练模型的链接。

2.如果只想对主干网络使用预训练模型，可以在配置文件中，将主干网络 backbone 中的 pretrained 设置为预训练模型的地址或链接。 在训练时，预训练模型中无法与主干网络对应的参数会被忽略。

#### 微调模型参数时，如何冻结主干网络中的部分参数？
* 可以参照 def _freeze_stages() 和 frozen_stages。在分布式训练和测试时，还须设置 find_unused_parameters = True。

* 实际上，除了少数模型，如 C3D 等，用户都能通过设置 frozen_stages 来冻结模型参数，因为大多数主干网络继承自 ResNet 和 ResNet3D，而这两个模型都支持 _freeze_stages() 方法。

```
    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1_s.eval()
            self.conv1_t.eval()
            for param in self.conv1_s.parameters():
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
```
* frozen_stage的设置：
    + -1：not freezing any parameters
    + ≥0：就可以冻结stem layer中的3DCNN的权重
    + 大于0的整数i: 冻结res stage 1到res stage i的权重

### Test.py的编写
* 记录测试样本的标签类别和实际类别
* 保存样本输出的特征，t-SNE可视化，所以我需要模型输出FC层前的输出，或者就用最后的输出，先实现