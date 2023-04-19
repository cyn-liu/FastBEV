# trainval
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
     --root-path /home/xxxx/datasets/nuscenes/complete \
     --out-dir /home/xxxx/datasets/nuscenes/complete/xxxx/with_canbus \
     --updated-out-dir /home/xxxx/datasets/nuscenes/complete/xxxx/with_canbus/updated \
     --extra-tag nuscenes \
     --version v1.0-trainval \
     --canbus /home/xxxx/datasets/nuscenes/complete \
     --db-save-path /home/xxxx/datasets/nuscenes/complete/xxxx/with_canbus/

## test
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python tools/create_data.py nuscenes \
#     --root-path /home/xxxx/datasets/nuscenes/complete \
#     --out-dir /home/xxxx/datasets/nuscenes/complete/xxxx/with_temporal \
#     --extra-tag nuscenes \
#     --version v1.0-test \
#     --canbus ./data/nuscenes/test
#
## mini
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python tools/create_data.py nuscenes \
#    --root-path /data/common/nuScenes/mini \
#    --out-dir /data/common/nuScenes/pkl/xxxx/mini \
#    --updated-out-dir /data/common/nuScenes/pkl/xxxx/mini \
#    --extra-tag nuscenes \
#    --version v1.0-mini \
#    --canbus /data/common/nuScenes/mini \
#    --db-save-path /data/common/nuScenes/pkl/xxxx/mini