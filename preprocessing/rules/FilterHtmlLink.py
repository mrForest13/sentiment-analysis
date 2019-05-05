from preprocessing.Processor import Processor

import re


class FilterHtmlLinkProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)
        self.regex=r'(https?://[^ ]+)|(www.[^ ]+)'

    def process(self, data):
        processed_data = self.remove_html_links(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def remove_html_links(self, data):
        return data['text'].apply(lambda x: re.sub(self.regex, '', x))
