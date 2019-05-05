from preprocessing.Processor import Processor

import re


class FilterHtmlLinkProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.regex = r'(https?://[^ ]+)|(www.[^ ]+)'

    def process(self, data):
        data['text'] = self.remove_html_links(data)

        return self.next_processor.process(data)

    def remove_html_links(self, data):
        return data['text'].apply(lambda x: re.sub(self.regex, '', x))
