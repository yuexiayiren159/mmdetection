#python3 tools/train.py \
#  ./configs/yolo/yolov3_d53_8xb8-320-273e_coco_cat_my.py \
#  --work-dir /root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my





#python3 tools/train.py \
#  ./configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#  --work-dir /root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco128_my
#

#  yolov3_d53_8xb8-320-273e_coco_cat_my.py
#/root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth

#python3 tools/train.py \
#  ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1.py \
#  --resume /root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco.pth \
#  --auto-scale-lr

#python3 tools/train.py \
#  ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1.py


#python3 tools/train.py \
#  ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1.py \
#  --resume /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_320_300e_coco_modified.pth \
#  --auto-scale-lr

# python3 tools/train.py \
#   ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco_my_1.py \
#   --resume /root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco.pth \
#   --auto-scale-lr



# python3 tools/train.py \
#   ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_cat.py



# python3 tools/train.py \
#   ./configs/yolo/yolov3_xceptionb0_1xb24-416-736-coco256.py



# python3 tools/train.py \
#   ./configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my.py \
#   --work-dir /root/lanyun-tmp/my_work/mmdetection/yolov3_mobilenetv2_8xb24-ms-416-300e_coco_my


# python3 tools/train.py \
#   configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py


# mklink /D "E:/workspace/openmmlab/mmdetection/work_dirs" "F:/openmmlab/mmdetection/work_dirs"



# my_configs/my_deformable-detr_r50_16xb2-50e_coco.py

# 训练detr
# my_configs\detr_r18_8xb2-500e_coco.py
# python tools/train.py my_configs/detr_r18_8xb2-500e_coco.py




# SSD
# configs\ssd\ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py
# python tools/train.py configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py
# my_configs\ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py
# python tools/train.py my_configs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py


# 自定义SSD
python tools/train.py configs/ssd/ssd_custom.py