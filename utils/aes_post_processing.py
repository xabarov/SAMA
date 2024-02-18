from utils import ml_config


def find_missed_objects(detected_shapes):
    missed_objects_cls_nums = {}
    # Rule 1. If RO nums == MZ_nums
    ro_nums = 0
    for res in detected_shapes:
        cls_eng = ml_config.CLASSES_ENG[int(res['cls_num'])]
        if cls_eng == "ro_pf" or cls_eng == "ro_sf":
            ro_nums += 1

    mz_nums = 0
    for res in detected_shapes:
        cls_eng = ml_config.CLASSES_ENG[int(res['cls_num'])]
        if cls_eng == "mz_v" or cls_eng == "mz_ot":
            mz_nums += 1

    if mz_nums != ro_nums:
        if mz_nums < ro_nums:
            missed_objects_cls_nums["mz"] = ro_nums - mz_nums
        else:
            missed_objects_cls_nums["ro"] = mz_nums - ro_nums

    return missed_objects_cls_nums
