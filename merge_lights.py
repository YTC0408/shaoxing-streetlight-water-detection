"""合并 damaged_lights + pole_dst2328 成一套统一数据集。
输出 datasets/lights_merged/   images/{train,val} + labels/{train,val}
类别映射:
  damaged_lights:  0=Not Working -> 0=light_off   1=Working -> 1=light_on
  pole_dst2328:    0=lightening  -> 1=light_on   1=damaged pole -> 0=light_off
val: damaged_lights 92 + pole_dst2328 272 = 364 张混合验证集
"""
import os, shutil, glob

SRC = {
    'damaged_lights': {
        'root': 'datasets/damaged_lights',
        'splits': {  # src_subdir -> dst_split
            'train': 'train',
            'valid': 'val',
        },
        'remap': {0: 0, 1: 1},  # 0=Not Working 0=light_off; 1=Working 1=light_on
        'prefix': 'dl_',  # 文件名前缀,避免和 pole 重名
    },
    'pole_dst2328': {
        'root': 'datasets/pole_dst2328',
        'splits': {
            'train': 'train',
            'valid': 'val',
        },
        'remap': {0: 1, 1: 0},  # 0=lightening -> 1=light_on; 1=damaged pole -> 0=light_off
        'prefix': 'pd_',
    },
}

OUT = 'datasets/lights_merged'
for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)

counts = {'train': {'imgs': 0, 'lbls': 0, 'cls0': 0, 'cls1': 0},
          'val':   {'imgs': 0, 'lbls': 0, 'cls0': 0, 'cls1': 0}}

def remap_label(src, dst, mapping):
    """读 src 标签,remap 类别 ID,写 dst"""
    n0 = n1 = 0
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            p = line.split()
            if not p:
                continue
            old = int(p[0])
            if old not in mapping:
                continue
            new = mapping[old]
            p[0] = str(new)
            fout.write(' '.join(p) + '\n')
            if new == 0: n0 += 1
            elif new == 1: n1 += 1
    return n0, n1

for ds, info in SRC.items():
    for src_split, dst_split in info['splits'].items():
        # 图片路径
        if ds == 'damaged_lights':
            img_dir = f"{info['root']}/{src_split}/images"
            lbl_dir = f"{info['root']}/{src_split}/labels"
        else:
            img_dir = f"{info['root']}/images/{src_split}"
            lbl_dir = f"{info['root']}/labels/{src_split}"

        for lbl_path in glob.glob(f'{lbl_dir}/*.txt'):
            base = os.path.basename(lbl_path)
            stem = base.rsplit('.txt', 1)[0]
            new_stem = f"{info['prefix']}{stem}"
            # 找对应图片(.jpg/.png)
            src_img = None
            for ext in ('.jpg', '.png', '.jpeg'):
                cand = f'{img_dir}/{stem}{ext}'
                if os.path.exists(cand):
                    src_img = cand
                    break
            if src_img is None:
                print(f'  [skip] no image for {base}')
                continue
            ext = os.path.splitext(src_img)[1]
            # 复制图片
            dst_img = f'{OUT}/images/{dst_split}/{new_stem}{ext}'
            shutil.copy(src_img, dst_img)
            # remap 标签
            dst_lbl = f'{OUT}/labels/{dst_split}/{new_stem}.txt'
            n0, n1 = remap_label(lbl_path, dst_lbl, info['remap'])
            counts[dst_split]['imgs'] += 1
            counts[dst_split]['lbls'] += 1
            counts[dst_split]['cls0'] += n0
            counts[dst_split]['cls1'] += n1

print('\n=== merged dataset ===')
for split, c in counts.items():
    print(f'{split:>5}: imgs={c["imgs"]}  lbls={c["lbls"]}  '
          f'light_off(cls0)={c["cls0"]}  light_on(cls1)={c["cls1"]}  '
          f'ratio_off={c["cls0"]/(c["cls0"]+c["cls1"])*100:.1f}%' if (c['cls0']+c['cls1']) else f'{split}: empty')
