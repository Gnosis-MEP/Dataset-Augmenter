import os

from decouple import config

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SOURCE_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, 'inputs')
INPUT_DIR = config('INPUT_DIR', default=INPUT_DIR)

INPUT_BGS_DIR = os.path.join(INPUT_DIR, 'BGs')
INPUT_OIS_DIR = os.path.join(INPUT_DIR, 'OIs', 'origin')
INPUT_OI_SAMPLES_DIR = os.path.join(INPUT_DIR, 'OIs', 'Samples')
INPUT_OI_REGION_MAPS_DIR = os.path.join(INPUT_DIR, 'OIs', 'Regions')


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
OUTPUT_IMAGES_DIR_NAME = 'train'
OUTPUT_ANNOTATION_NAME = 'train.json'
BACKUP_PROGRESS_JSON = os.path.join(OUTPUT_DIR, 'bk_progress.json')

REDIS_ADDRESS = config('REDIS_ADDRESS', default='localhost')
REDIS_PORT = config('REDIS_PORT', default='6379')

TRACER_REPORTING_HOST = config('TRACER_REPORTING_HOST', default='localhost')
TRACER_REPORTING_PORT = config('TRACER_REPORTING_PORT', default='6831')

SERVICE_STREAM_KEY = config('SERVICE_STREAM_KEY')


LISTEN_EVENT_TYPE_OI_FOREGROUND_CREATED = config('LISTEN_EVENT_TYPE_OI_FOREGROUND_CREATED')
LISTEN_EVENT_TYPE_OI_BACKGROUND_CREATED = config('LISTEN_EVENT_TYPE_OI_BACKGROUND_CREATED')

SERVICE_CMD_KEY_LIST = [
    LISTEN_EVENT_TYPE_OI_FOREGROUND_CREATED,
    LISTEN_EVENT_TYPE_OI_BACKGROUND_CREATED,
]

PUB_EVENT_TYPE_DATASET_AUGMENTED = config('PUB_EVENT_TYPE_DATASET_AUGMENTED')

PUB_EVENT_LIST = [
    PUB_EVENT_TYPE_DATASET_AUGMENTED,
]

# Only for Content Extraction services
SERVICE_DETAILS = None

# Example of how to define SERVICE_DETAILS from env vars:
# SERVICE_DETAILS_SERVICE_TYPE = config('SERVICE_DETAILS_SERVICE_TYPE')
# SERVICE_DETAILS_STREAM_KEY = config('SERVICE_DETAILS_STREAM_KEY')
# SERVICE_DETAILS_QUEUE_LIMIT = config('SERVICE_DETAILS_QUEUE_LIMIT', cast=int)
# SERVICE_DETAILS_THROUGHPUT = config('SERVICE_DETAILS_THROUGHPUT', cast=float)
# SERVICE_DETAILS_ACCURACY = config('SERVICE_DETAILS_ACCURACY', cast=float)
# SERVICE_DETAILS_ENERGY_CONSUMPTION = config('SERVICE_DETAILS_ENERGY_CONSUMPTION', cast=float)
# SERVICE_DETAILS_CONTENT_TYPES = config('SERVICE_DETAILS_CONTENT_TYPES', cast=Csv())
# SERVICE_DETAILS = {
#     'service_type': SERVICE_DETAILS_SERVICE_TYPE,
#     'stream_key': SERVICE_DETAILS_STREAM_KEY,
#     'queue_limit': SERVICE_DETAILS_QUEUE_LIMIT,
#     'throughput': SERVICE_DETAILS_THROUGHPUT,
#     'accuracy': SERVICE_DETAILS_ACCURACY,
#     'energy_consumption': SERVICE_DETAILS_ENERGY_CONSUMPTION,
#     'content_types': SERVICE_DETAILS_CONTENT_TYPES
# }

LOGGING_LEVEL = config('LOGGING_LEVEL', default='DEBUG')