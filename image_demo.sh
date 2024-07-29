# python demo/image_demo.py \
#     ./demo/demo.jpg \
#     ./configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py \
#     ./checkpoints/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth \
#     --out-dir ./my_work/mobilenetv2


python demo/image_demo.py \
    ./demo/demo.jpg \
    my_configs/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py \
    --weights checkpoints/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth \
    --out-dir outputs