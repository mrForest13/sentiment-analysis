from preprocessing.Processor import Processor

import swifter
import html


class HtmlEncodingProcessor(Processor):

    def process(self, data):
        data['text'] = self.remove_html_encoding(data)

        return self.next_processor.process(data)

    @staticmethod
    def remove_html_encoding(data):
        return data['text'].swifter.apply(lambda x: html.unescape(x))
