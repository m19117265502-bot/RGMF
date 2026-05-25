#CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name medfuse --devices 1 \
#    --pooling_type attention \
#    --use_prototype \
#    --use_multiscale \
#    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&   ##这里融合的是医学图像

CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name proto_ts --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&
CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name ipnet --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&

CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name grud --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&
CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name seft --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&
CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name mtand --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&
CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name dgm2 --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&






CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name utde --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 &&

CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task ihm --model_name lstm --devices 1 \
    --pooling_type attention \
    --use_prototype \
    --use_multiscale \
    --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 --lamb4 0.05 
