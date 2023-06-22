def encode(mask_results):
    result = []
    for mask_res in mask_results:
        # {'cls_num': cls_num, 'points': points, 'conf': conf}
        cls_num = str(mask_res['cls_num'])
        conf = str(mask_res['conf'])
        points = []
        for p in mask_res['points']:
            p_str = [str(p[0]), str(p[1])]
            points.append(p_str)
        result.append({'cls_num': cls_num, 'points': points, 'conf': conf})
    return result