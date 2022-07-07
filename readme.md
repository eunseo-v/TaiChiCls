1. install mmaction2
2. add files to mmaction2
    1) mmaction2/mmaction/datasets/pipelines/__init__.py 
        from .my_pipeline import GenerateNTUPoseTarget, GenerateTaiChiPoseTarget, GenerateTaiChi17PoseTarget
    2) in folder ``mmaction2/mmaction/datasets/pipelines`` add the file
        ``add-to-mmaction2/my_pipeline.py`` 

### experiment configurations are at folder `configs`

### experiment records and results are at folder `model_pth`