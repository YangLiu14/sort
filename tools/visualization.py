import sys
import os
import colorsys
import glob

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as rletools

from PIL import Image
from multiprocessing import Pool
# from mots_common.io import load_sequences, load_seqmap
from functools import partial
from subprocess import call


class SegmentedObject:
    def __init__(self, bbox, class_id, track_id):
        self.bbox = bbox
        self.class_id = class_id
        self.track_id = track_id


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
    """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6,
            10]
    colors = [colors[idx] for idx in perm]
    return colors


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
  """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(",")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            # class_id = int(fields[2])
            class_id = 1
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            # mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            bbox = [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])]  # [x, y, w, h]
            # if frame not in combined_mask_per_frame:
            #   combined_mask_per_frame[frame] = mask
            # elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
            #   assert False, "Objects with overlapping masks in frame " + fields[0]
            # else:
            #   combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)

            objects_per_frame[frame].append(SegmentedObject(
                bbox,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def load_sequences(seq_paths):
    objects_per_frame_per_sequence = {}
    for seq_path_txt in seq_paths:
        seq = seq_path_txt.split("/")[-1][:-4]
        if os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
        else:
            assert False, "Can't find data in directory " + seq_path_txt

    return objects_per_frame_per_sequence


def process_sequence(seq_fpaths, tracks_folder, img_folder, output_folder, max_frames, annot_frames_dict,
                     draw_boxes=True, create_video=True):
    folder_name = tracks_folder.split("/")[-1]
    # print("Processing sequence", seq_name)
    os.makedirs(output_folder, exist_ok=True)
    tracks = load_sequences(seq_fpaths)
    for seq_fpath in seq_fpaths:
        seq_id = seq_fpath.split('/')[-1][:-4]
        max_frames_seq = max_frames[seq_id]
        annot_frames = annot_frames_dict[seq_id]
        visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, annot_frames, output_folder, draw_boxes, create_video)


def visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, annot_frames, output_folder, draw_boxes=True, create_video=True):
    colors = generate_colors()
    dpi = 100.0
    # frames_with_annotations = [frame for frame in tracks.keys() if len(tracks[frame]) > 0]
    # img_sizes = next(iter(tracks[frames_with_annotations[0]])).mask["size"]
    frames_with_annotations = [frame.split('/')[-1] for frame in annot_frames]
    # img_sizes = next(iter(tracks[seq_id])).bbox
    for t in range(max_frames_seq):
        print("Processing frame", frames_with_annotations[t])
        filename_t = img_folder + "/" + seq_id + "/" + frames_with_annotations[t]
        img = np.array(Image.open(filename_t), dtype="float32") / 255
        img_sizes = img.shape
        fig = plt.figure()
        fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.subplots()
        ax.set_axis_off()

        for obj in tracks[seq_id][t+1]:
            color = colors[obj.track_id % len(colors)]
            if obj.class_id == 1:
                category_name = "object"
            elif obj.class_id == 2:
                category_name = "Pedestrian"
            else:
                category_name = "Ignore"
                color = (0.7, 0.7, 0.7)
            if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
                x, y, w, h = obj.bbox
                if draw_boxes:
                    import matplotlib.patches as patches
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                             edgecolor=color, facecolor='none', alpha=1.0)
                    ax.add_patch(rect)
                category_name += ":" + str(obj.track_id)
                ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                            fontsize=7, ha='center', va='center', alpha=1.0)

        ax.imshow(img)
        if not os.path.exists(os.path.join(output_folder + "/" + seq_id)):
            os.makedirs(os.path.join(output_folder + "/" + seq_id))
        fig.savefig(output_folder + "/" + seq_id + "/" + frames_with_annotations[t])
        plt.close(fig)

        if create_video:
            os.chdir(output_folder + "/" + seq_id)
            call(["ffmpeg", "-framerate", "10", "-y", "-i", "%06d.jpg", "-c:v", "libx264", "-profile:v", "high", "-crf",
                  "20",
                  "-pix_fmt", "yuv420p", "-vf", "pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'", "output.mp4"])



def main():
    # if len(sys.argv) != 5:
    #     print("Usage: python visualize_mots.py tracks_folder(gt or tracker results) img_folder output_folder seqmap")
    #     sys.exit(1)

    # tracks_folder = sys.argv[1]
    # img_folder = sys.argv[2]
    # output_folder = sys.argv[3]
    # seqmap_filename = sys.argv[4]

    data_srcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M"]
    tracks_folder = "/home/kloping/git-repos/sort/output2/" + data_srcs[2]
    img_folder = "/home/kloping/OpenSet_MOT/data/TAO/TAO_VAL/val/" + data_srcs[2]
    output_folder = "/home/kloping/OpenSet_MOT/Tracking/track_results/SORT/" + data_srcs[2]
    seqmap_filenames = sorted(glob.glob(tracks_folder + '/*' + '.txt'))

    # seqmap, max_frames = load_seqmap(seqmap_filename)

    max_frames = dict()
    annot_frames = dict()  # annotated frames for each sequence
    for seq_fpath in seqmap_filenames:
        seq_name = seq_fpath.split('/')[-1][:-4]

        # Get the max number of frames of the current sequence
        num_frames = 0
        with open(seq_fpath, 'r') as f:
            for line in f:
                line = line.strip()
                fields = line.split(",")
                num_frames = int(fields[0])
        max_frames[seq_name] = num_frames

    # Get the annotated frames in the current sequence.
    txt_fname = "../data/tao/val_annotated_{}.txt".format(data_srcs[2])
    with open(txt_fname) as f:
        content = f.readlines()
    content = ['/'.join(c.split('/')[1:]) for c in content]
    annot_seq_paths = [os.path.join(img_folder, x.strip()) for x in content]

    for s in annot_seq_paths:
        seq_name = s.split('/')[-2]
        if seq_name not in annot_frames.keys():
            annot_frames[seq_name] = []
        annot_frames[seq_name].append(s)

    process_sequence_part = partial(process_sequence, max_frames=max_frames, annot_frames_dict=annot_frames,
                                    tracks_folder=tracks_folder, img_folder=img_folder, output_folder=output_folder)

    process_sequence_part(seqmap_filenames)

    # with Pool(10) as pool:
    #     pool.map(process_sequence_part, seqmap)
    # for seq in seqmap:
    #  process_sequence_part(seq)


if __name__ == "__main__":
    main()
