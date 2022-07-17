from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" %
    #                                  (yaml_name, cfg.TEST.EPOCH))
    params.checkpoint = "/content/drive/MyDrive/tracking_cells/checkpoints/ost_cells_bak/OSTrack_ep0011.pth.tar"
    params.checkpoint = "/content/drive/MyDrive/tracking_cells/checkpoints/ost_cells_bak/OSTrack_ep0012.pth.tar"
    params.checkpoint = "/content/drive/MyDrive/tracking_cells/checkpoints/ost_cells_v1/OSTrack_ep0012.pth.tar"
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
