import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, frames, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes, output_confidence, output_heatmaps = [], [], []

        # cap = cv.VideoCapture(videofilepath)
        # cap = cv.VideoCapture(videofilepath)
        # success, frame = cap.read()
        frame = frames[0]

        # display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        # # success, frame = cap.read()
        # cv.imshow(display_name, frame)

        # def _build_init_info(box):
        #     return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
        #             'sequence_object_ids': [1, ]}
        def _build_init_info(box):
            return {'init_bbox': box}

        assert optional_box is not None
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
        tracker.initialize(frame, _build_init_info(optional_box))

        for frame in frames:

            if frame is None:
                break

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out["target_bbox"]]
            conf = out["bbox_score"]
            heatmap = out["heatmap"]  # .squeeze().cpu().detach()
            # If the tracker box confidence is < threshold, kill the tracker
            # if conf < 0.1:
            #     return output_boxes, output_confidence, output_heatmaps
            # print({k: max(v) for k, v in out["max_score"].items()}, state)
            output_boxes.append(state)
            output_confidence.append(conf)
            output_heatmaps.append(heatmap)

            tracker.initialize(frame, _build_init_info(state))  # Reinit the template

        return output_boxes, output_confidence, output_heatmaps

    def extract_encodings(self, frames, states, debug=False, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        tracker = self.create_tracker(params)
        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

        def _build_init_info(box):
            return {'init_bbox': box}

        tracker.initialize(frames[0], _build_init_info(states[0]))
        output_heatmaps, output_boxes = [], []
        for frame, state in zip(frames, states):
            self.state = state  # Not tracking, so overwrite with existing tracks

            if frame is None:
                break

            # Draw box
            out = tracker.track(frame)
            encoding = out["encodings"]
            state = [int(s) for s in out["target_bbox"]]

            # Figure out how to summarize encodings
            encoding = encoding.squeeze(0).mean(0).detach().cpu().numpy().reshape(1, -1)
            output_heatmaps.append(encoding)
            output_boxes.append(state)

        return output_heatmaps, output_boxes

    def gradient_of_distance(self, framesa, framesb, statesa, statesb, smooth_iters=0, debug=False, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        tracker = self.create_tracker(params)
        if hasattr(tracker, 'initialize_features'):
            tracker.initialize_features()

        def _build_init_info(box):
            return {'init_bbox': box}


        # First track framesa
        tracker.initialize(framesa[0], _build_init_info(statesa[0]))
        encs_a = []
        for frame, state in zip(framesa, statesa):
            self.state = state  # Not tracking, so overwrite with existing tracks

            # Get encodings
            out = tracker.track(frame)
            encoding = out["encodings"]
            encoding = encoding.squeeze(0).mean(0).reshape(1, -1)
            encs_a.append(encoding)

        # Next track framesb, compute distance of ||encodingb|| - encodinga_t||, and store gradient of difference wrt framesb
        tracker.initialize(framesb[0], _build_init_info(statesb[0]))
        gradients, patches = [], []
        for frame, state, enc_a in zip(framesb, statesb, encs_a):
            self.state = state  # Not tracking, so overwrite with existing tracks

            # Get encodings
            if smooth_iters:
                smoothed = []
                for _ in range(smooth_iters):
                    noise_frame = frame + np.random.normal(scale=1e-3, size=frame.shape)
                    out = tracker.track(noise_frame, store_grad=True)
                    encoding = out["encodings"]
                    input_patch = out["input_patch"]  # Has a gradient
                    enc_b = encoding.squeeze(0).mean(0).reshape(1, -1)

                    # Get difference and gradient
                    dist = ((enc_a - enc_b) ** 2).mean()
                    tracker.network.zero_grad()
                    dist.backward()
                    gradient = input_patch.grad.data.cpu().numpy()
                    smoothed.append(gradient)
                gradient = np.stack(smoothed, 0).mean(0)
            else:
                out = tracker.track(frame, store_grad=True)
                encoding = out["encodings"]
                input_patch = out["input_patch"]  # Has a gradient
                enc_b = encoding.squeeze(0).mean(0).reshape(1, -1)

                # Get difference and gradient
                dist = ((enc_a - enc_b) ** 2).mean()
                tracker.network.zero_grad()
                dist.backward()
                gradient = input_patch.grad.data.cpu().numpy()
            gradients.append(gradient)
            patches.append(input_patch.detach().cpu().numpy())
        return gradients, patches

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



