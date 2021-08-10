# Interactive Segmentation
## Preview
<img src='./misc/interactive_seg_demo.gif>

## Setting up an environment
- dockerfile build
-  `cd ritm_interactive_segmentation`
-  ` docker build --no-cache=false -t {img_name}:{tag} .`
- run container
-  `docker run -it -d -p {gpu_server_port}:{container_jupyterlab_port} --gpus all --ipc=host --shm-size=8g -v /home:/home {img_name}:{tag}`

## Custom model training
- setting dataset directory
```
data
└── {project_name}
├── train
│ ├── images
│ └── labels
└── valid
├── images
└── labels
```
- config.yaml
- EXPS_PATH: Where pipeline outputs will saved. (tensorboard log, ckpts, vis, ...)
- DATASET_PATH: Dataset root path.
- CLASS_LIST: All of class name.
- IGNORE_CLASS: This class is excluded from training because it affects bad influence during training. The classes have ambiguous shape.(contain many object or shape)
- IMAGENET_PRETRAINED_MODELS: Prae-trained model path
```
EXPS_PATH: "./experiments"
DATASET_PATH: "./data/HD"
CLASS_LIST: ['Background',
			'Freespace',
			...
			'Pillar']
IGNORE_CLASS: ['Background', 'Freespace']
IMAGENET_PRETRAINED_MODELS: "./pretrained_models/hrnet_w18_small_model_v2.pth"
```
-  [download pre-trained model](https://onedrive.live.com/?cid=f7fd0b7f26543ceb&id=F7FD0B7F26543CEB%21153&authkey=!AJ909Hv1YFLrVCc)
- run train script
`python train.py models/hrnet18s_cocolvis_itermask_3p.py `

## Preset model inference API
-  `inferenceAPI_tutorial.ipynb`

## Contributors
- milla(@DataLab)