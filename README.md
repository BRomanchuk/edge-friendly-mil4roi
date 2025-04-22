# edge-friendly-mil4roi
Edge-Friendly Multiple Instance Learning for Region-of-Interest Proposal


## Repo Structure
```
├── models              # module with MIL models 
│   ├── feature_extractor.py  # MobileNetV3/V4, MIT-EfficientViT
│   └── mil  
│       ├── naive.py    # max / avg pooling 
│       └── dsmil.py    # dual-stream MIL pooling
├── custom_dataset.py   # dataset wrapper
├── metrics.py          # eval metrics 
├── pipeline.py         # training pipeline
└── prediction_pipeline.py   # test pipeline
```
