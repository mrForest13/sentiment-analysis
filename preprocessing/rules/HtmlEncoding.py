from preprocessing.Processor import Processor

import html


class HtmlEncodingProcessor(Processor):

    def __init__(self, next_processor):
        super().__init__(next_processor)

    def process(self, data):
        processed_data = self.remove_html_encoding(data)
        super().process(processed_data)
        return self.next_processor.process(processed_data)

    @staticmethod
    def remove_html_encoding(data):
        return data['text'].apply(lambda x: html.unescape(x))
