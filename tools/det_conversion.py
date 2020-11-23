import glob
import json
import os
import tqdm


if __name__ == "__main__":

    root_dir = "/home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/nonOverlap_small/objectness/"
    # root_dir = "/storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/A-Folder-containes-all-the-sequences/json/"
    outdir = "/home/kloping/OpenSet_MOT/Tracking/proposals/forSORT_masks/val/"
    scoring = "objectness"

    data_srcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M"]
    for data_src in data_srcs:
        print('Processing', data_src)
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, data_src, '*')))]
        for idx, video in enumerate(tqdm.tqdm(videos)):
            fpath = os.path.join(root_dir, data_src, video)
            # List all files in the current folder
            files = sorted(glob.glob(fpath + '/*' + '.json'))

            # Store in the format of MOT benchmark
            txt_fpath = os.path.join(outdir, data_src, video + ".txt")
            if not os.path.exists(os.path.join(outdir, data_src)):
                os.makedirs(os.path.join(outdir, data_src))
            with open(txt_fpath, 'w') as f_txt:
                for frame_id, json_file in enumerate(files):
                    with open(json_file, 'r') as f:
                        proposals = json.load(f)

                    for prop in proposals:
                        x1, y1, x2, y2 = prop['bbox']
                        # convert [x1, y1, x2, y2] to [x_left, y_left, w, c]
                        box = [x1, y1, x2-x1, y2-y1]
                        mask = prop['instance_mask']
                        img_h, img_w = mask['size']
                        rle_str = mask['counts']
                        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>,
                        # <x>, <y>, <z>, <img_h>, <img_w>, <rle_str>
                        string = ",".join([str(frame_id + 1), "-1", str(box[0]), str(box[1]), str(box[2]), str(box[3]),
                                           str(prop[scoring]), "-1", "-1", "-1", str(img_h), str(img_w), rle_str])
                        f_txt.write(string + '\n')
            f_txt.close()
