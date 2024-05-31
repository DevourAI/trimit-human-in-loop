from trimit.app import VOLUME_DIR
import os

global CONF
CONF = {"chunk_delay": 0.00}

LINEAR_WORKFLOW_OUTPUT_FOLDER = os.path.join(VOLUME_DIR, "linear_workflow_output")
WORKFLOWS_DICT_NAME = "linear_workflow_workflows"
RUNNING_WORKFLOWS_DICT_NAME = "linear_workflow_running_workflows"
VIDEO_PROCESSING_CALL_IDS_DICT_NAME = "linear_workflow_video_processing_call_ids"
