# python ./demo/image_demo.py \
#   ./demo/demo.jpg \
#   ./configs/yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py \
#   --weights /root/lanyun-tmp/my_work/mmdetection/yolov3_d53_8xb8-320-273e_coco_cat_my/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth \
#   --out-dir ./my_work/coco


# python ./demo/image_demo.py \
#   F:/dataset/labelme-data/coco_maskdataset_chatgpt/val2017/fff.jpg \
#   my_configs/my_1_deformable-detr_r50_16xb2-50e_coco.py \
#   --weights work_dirs/deformable-detr_r50_16xb2-50e_coco/epoch_50.pth \
#   --out-dir ./my_work/deformable-detr_r50


# 测试mobilenetv2-ssd
python ./demo/image_demo.py \
  data/coco4/val2017/000000370677.jpg \
  my_configs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py \
  --weights work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco/epoch_5.pth \
  --out-dir ./my_work/ssdlite_mobilenetv2-scratch_8xb24-600e_coco