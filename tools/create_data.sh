# trainval
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
     --root-path /home/zhangbin/datasets/nuscenes/complete \
     --out-dir /home/zhangbin/datasets/nuscenes/complete/zhangbin/with_canbus \
     --updated-out-dir /home/zhangbin/datasets/nuscenes/complete/zhangbin/with_canbus/updated \
     --extra-tag nuscenes \
     --version v1.0-trainval \
     --canbus /home/zhangbin/datasets/nuscenes/complete \
     --db-save-path /home/zhangbin/datasets/nuscenes/complete/zhangbin/with_canbus/

## test
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python tools/create_data.py nuscenes \
#     --root-path /home/zhangbin/datasets/nuscenes/complete \
#     --out-dir /home/zhangbin/datasets/nuscenes/complete/zhangbin/with_temporal \
#     --extra-tag nuscenes \
#     --version v1.0-test \
#     --canbus ./data/nuscenes/test
#
## mini
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python tools/create_data.py nuscenes \
#    --root-path /data/common/nuScenes/mini \
#    --out-dir /data/common/nuScenes/pkl/zhangbin/mini \
#    --updated-out-dir /data/common/nuScenes/pkl/zhangbin/mini \
#    --extra-tag nuscenes \
#    --version v1.0-mini \
#    --canbus /data/common/nuScenes/mini \
#    --db-save-path /data/common/nuScenes/pkl/zhangbin/mini