"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import shutil
import time
import tqdm
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def  convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
      Assigns detections to tracked object (both represented as bounding boxes)
      Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns ther a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        tracker_idx2box_idx = dict()   # Yang
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            tracker_idx2box_idx[m[1]] = m[0]  # Yang

        # create and initialise new trackers for unmatched detections
        tracker_last_idx = len(self.trackers)
        for i in unmatched_dets:
            # Yang
            tracker_idx2box_idx[tracker_last_idx] = i
            tracker_last_idx += 1
            # YANG
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        # Yang: Reverse the keys in tracker_idx2box_idx
        ret_idx2det = dict()
        r_idx = 0
        # YANG
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                # Yang
                ret_idx2det[r_idx] = dets[tracker_idx2box_idx[i-1]]
                r_idx += 1
                # YANG
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret), ret_idx2det
        return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--datasrc", nargs='+', type=str)
    parser.add_argument("--outdir", help="Output directory", type=str, default='output')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    # if not os.path.exists(args.outdir):
    #     os.makedirs(args.outdir)
    # else:
    #     shutil.rmtree(args.outdir, ignore_errors=True)
    #     os.makedirs(args.outdir)

    # pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    # pattern = os.path.join(args.seq_path, phase)
    data_srcs = glob.glob(os.path.join(args.seq_path, phase, "*"))
    if args.datasrc:
        data_srcs = [src for src in data_srcs if src.split('/')[-1] in args.datasrc]

    for data_src in data_srcs:

        if not os.path.exists(args.outdir + '/{}'.format(data_src.split('/')[-1])):
            os.makedirs(args.outdir + '/{}'.format(data_src.split('/')[-1]))

        print("Processing", data_src.split('/')[-1])
        # pattern = glob.glob(os.path.join(data_src, "*.txt"))
        pattern = data_src
        num_seq = len(glob.glob(os.path.join(pattern, '*')))
        for seq_cnt, seq_dets_fn in enumerate(glob.glob(os.path.join(pattern, '*'))):
            seq_cnt += 1
            mot_tracker = Sort(max_age=args.max_age,
                               min_hits=args.min_hits,
                               iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
            # seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            # seq_dets_wMask = np.loadtxt(seq_dets_fn, dtype='str', delimiter=',')  # proposals in all sequences with mask
            # seq_dets = seq_dets_wMask[:, :10].astype(np.float)
            # seq_masks = np.concatenate((seq_dets_wMask[:, 0:1], seq_dets_wMask[:, 10:]), axis=1)
            seq = seq_dets_fn.split("/")[-1][:-4]
            # seq = seq_dets_fn[pattern.find('*'):].split('/')[0]

            print("{}/{} Processing {}".format(seq_cnt, num_seq, seq))
            # Using numpy to load all dets at once will cause memory overflow
            with open(seq_dets_fn, 'r') as f:
                contents = f.readlines()
            max_frames = int(contents[-1].split(',')[0])

            with open(args.outdir + '/{}/{}.txt'.format(data_src.split("/")[-1], seq), 'w') as out_file:
                for frame in tqdm.tqdm(range(max_frames)):
                    frame += 1  # detection and frame numbers begin at 1

                    seq_dets_wMask = [line.split(',') for line in contents if int(line.split(',')[0]) == frame]
                    # check if all the bbox are valid: w > 0 and h > 0
                    seq_dets_wMask = [det for det in seq_dets_wMask if (float(det[4]) > 0 and float(det[5]) > 0)]
                    seq_dets_wMask = np.array(seq_dets_wMask)
                    dets = seq_dets_wMask[:, 2:7].astype(np.float)
                    dets_mask = np.concatenate((seq_dets_wMask[:, 0:1], seq_dets_wMask[:, 10:]), axis=1)

                    # dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                    # dets_mask = seq_masks[seq_masks[:, 0] == str(frame), :]
                    dets[:, 2:4] += dets[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                    total_frames += 1

                    if (display):
                        fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame)
                        im = io.imread(fn)
                        ax1.imshow(im)
                        plt.title(seq + ' Tracked Targets')

                    start_time = time.time()
                    # Append mask information to the end of dets
                    placeholder = np.empty((dets.shape[0], 0), dtype=np.object)
                    dets = np.append(placeholder, dets, axis=1)
                    dets = np.append(dets, dets_mask, axis=1)

                    trackers, tracker_idx2det = mot_tracker.update(dets)
                    assert len(dets) == len(trackers)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    for t_idx, d in enumerate(trackers):
                        # sanity check
                        curr_bbox = np.array(tracker_idx2det[t_idx][:4])
                        # assert np.abs((np.array(d[:4]) - curr_bbox)).any() < 5, "Idx doesn't match"
                        curr_mask = tracker_idx2det[t_idx][-3:]
                        curr_conf = tracker_idx2det[t_idx][4]

                        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                        # <img_h>, <img_w>, <rle>
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%f,-1,-1,-1,%d,%d,%s' % (
                        frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1], curr_conf,
                        int(curr_mask[0]), int(curr_mask[1]), curr_mask[2]), file=out_file)

                        # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                        #       file=out_file)
                        if (display):
                            d = d.astype(np.int32)
                            ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                            ec=colours[d[4] % 32, :]))

                    if (display):
                        fig.canvas.flush_events()
                        plt.draw()
                        ax1.cla()

        print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

        if (display):
            print("Note: to get real runtime results run without the option: --display")
