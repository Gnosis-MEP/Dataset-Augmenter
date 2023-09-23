import threading

from event_service_utils.logging.decorators import timer_logger
from event_service_utils.services.event_driven import BaseEventDrivenCMDService
from event_service_utils.tracing.jaeger import init_tracer

from dataset_augmenter.augmenter import Augmenter
from dataset_augmenter.conf import (
    LISTEN_EVENT_TYPE_OI_FOREGROUND_CREATED,
    LISTEN_EVENT_TYPE_OI_BACKGROUND_CREATED,
    PUB_EVENT_TYPE_DATASET_AUGMENTED,
)


class DatasetAugmenter(BaseEventDrivenCMDService):
    def __init__(self,
                 service_stream_key, service_cmd_key_list,
                 pub_event_list, service_details,
                 file_storage_cli,
                 stream_factory,
                 logging_level,
                 tracer_configs):
        tracer = init_tracer(self.__class__.__name__, **tracer_configs)
        super(DatasetAugmenter, self).__init__(
            name=self.__class__.__name__,
            service_stream_key=service_stream_key,
            service_cmd_key_list=service_cmd_key_list,
            pub_event_list=pub_event_list,
            service_details=service_details,
            stream_factory=stream_factory,
            logging_level=logging_level,
            tracer=tracer,
        )
        self.cmd_validation_fields = ['id']
        self.data_validation_fields = ['id']
        self.fs_client = file_storage_cli
        self.augmenter_cls = Augmenter


    def publish_dataset_augmented(self, event_data):
        self.publish_event_type_to_stream(event_type=PUB_EVENT_TYPE_DATASET_AUGMENTED, new_event_data=event_data)

    @timer_logger
    def process_data_event(self, event_data, json_msg):
        if not super(DatasetAugmenter, self).process_data_event(event_data, json_msg):
            return False
        # do something here
        pass

    def process_event_type(self, event_type, event_data, json_msg):
        if not super(DatasetAugmenter, self).process_event_type(event_type, event_data, json_msg):
            return False
        if event_type == LISTEN_EVENT_TYPE_OI_FOREGROUND_CREATED:
            # do some processing
            pass
        elif event_type == LISTEN_EVENT_TYPE_OI_BACKGROUND_CREATED:
            # do some other processing
            pass

    def log_state(self):
        super(DatasetAugmenter, self).log_state()
        self.logger.info(f'Service name: {self.name}')
        # function for simple logging of python dictionary
        # self._log_dict('Some Dictionary', self.some_dict)

    def run(self):
        super(DatasetAugmenter, self).run()
        self.agu
        self.log_state()
        self.run_forever(self.process_cmd)