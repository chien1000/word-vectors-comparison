import re
import html

class Corpus(object):
    """docstring for Corpus"""
    def __init__(self):
        super(Corpus, self).__init__()

class Reuters(Corpus):
    """docstring for Reuters"""
    def __init__(self, data_dir):
        super(Reuters, self).__init__()
        self.data_dir = data_dir

    def read_all_data(self):
        data_dir = self.data_dir
        file_ids = list(range(0,22))
        data_paths = list(map(lambda x: '{}/reut2-{:03d}.sgm'.format(data_dir, x), file_ids))

        all_data = []
        for data_path in data_paths:
        #     print(data_path)
            articles = self.read_data(data_path)
            all_data.extend(articles)
            
        return all_data

    def iter_all_data(self):
        data_dir = self.data_dir
        file_ids = list(range(0,22))
        data_paths = list(map(lambda x: '{}/reut2-{:03d}.sgm'.format(data_dir, x), file_ids))

        for data_path in data_paths:
            articles = self.read_data(data_path)
            for a in articles:
                yield a

    def read_data(self, data_path):
        with open(data_path, 'r') as file:
            raw = file.read()
            
        pattern_reu = re.compile(r'<REUTERS .*>([\s\S]+?)<\/REUTERS>')
        pattern_title = re.compile(r'<TITLE>([\s\S]+?)<\/TITLE>')
        pattern_body = re.compile(r'<BODY>([\s\S]+?)<\/BODY>')
        pattern_unproc = re.compile(r'<TEXT TYPE="UNPROC">')
        pattern_brief = re.compile(r'<TEXT TYPE="BRIEF">')
        
        reuters = pattern_reu.findall(raw)
        
        articles = []
        for reu in reuters:
            if pattern_unproc.search(reu) is None and pattern_brief.search(reu) is None:
        #         m = pattern_title.search(reu) 
        #         title = m.group(1)
        #         print(title)
        #         print(html.unescape(title))

                m = pattern_body.search(reu) 
                body = m.group(1)
                body = html.unescape(body).strip().strip('Reuter')
            
                body = body.replace('\n',' ')
                
                articles.append(body)
            
        return articles
        
        #iterator