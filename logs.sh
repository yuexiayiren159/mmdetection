python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/deformable-detr_r50_16xb2-50e_coco/20250409_205537/vis_data/20250409_205537.json \
    --keys loss_cls --legend loss_cls

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/deformable-detr_r50_16xb2-50e_coco/20250409_205537/vis_data/20250409_205537.json --keys loss_cls --legend loss_cls


python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/deformable-detr_r50_16xb2-50e_coco/20250409_205537/vis_data/20250409_205537.json --keys loss_cls loss_bbox --out losses.pdf