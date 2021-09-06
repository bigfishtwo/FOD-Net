# FOD-Net

This initial repo is for https://www.biorxiv.org/content/10.1101/2021.01.17.427042v2

Please kindly cite below if you feel FOD-Net is helpful and inspiring. Thanks.

```
@article{zeng2021fod,
title={FOD-Net: A Deep Learning Method for Fiber Orientation Distribution Angular Super Resolution},
author={Zeng, Rui and Lv, Jinglei and Wang, He and Zhou, Luping and Barnett, Michael and Calamante, Fernando and Wang, Chenyu},
journal={bioRxiv},
year={2021},
publisher={Cold Spring Harbor Laboratory}
}
```

------

Basically I added the train phase code. Since the patch-wise input is hard to load with torch.utils.data.DataLoader, I wrote my own dataoloder, which achives loading a FOD subject to GPU and feeding the model batch-wise patches.
