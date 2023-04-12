

## NuScenes
Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

**Download CAN bus expansion**

```shell
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md

*We generate custom annotation files which are different from mmdet3d's*

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes_train_pkl --extra-tag nuscenes --version v1.0-trainval --canbus ./data/nuscenes

python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes_test_pkl --extra-tag nuscenes --version v1.0-test --canbus ./data/nuscenes/test
```

Using the above code will generate `nuscenes_infos_{train,val}.pkl`.

**Folder structure**

```
MSBEVFusion
├── projects
│   ├── configs
│   ├── nuscenes
├── ckpts
├── tools
├── data
│   ├── nuscenes
│   │   ├── can_bus
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-trainval
│   │   ├── test
│   │   │   ├── can_bus
│   │   │   ├── maps
│   │   │   ├── samples
│   │   │   ├── sweeps
│   │   │   ├── v1.0-test
│   ├── nuscenes_train_pkl
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   ├── nuscenes_test_pkl
│   │   ├── nuscenes_infos_test.pkl
```
