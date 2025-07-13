class Paper:
    def __init__(self, paper_id, title, abstract, label_opts,summary=None,citations=None, internal=False):
        self.id = paper_id
        self.title = title
        self.abstract = abstract
        self.summary = summary
        self.citations = citations
        self.labels = {l:[] for l in label_opts}
        self.internal = internal

    def add_label(self, label, dim):
        self.labels[dim].append(label)

    def __str__(self):
        return f"Paper(id: {self.id}, title: '{self.title}', abstract: '{self.abstract}', summary: '{self.summary}', labels: {self.labels}, internal: {self.internal})"

    def __repr__(self):
        return self.__str__()
    
class Paper_simple:
    def __init__(self, paper_id, title, abstract, internal=False):
        self.id = paper_id
        self.title = title
        self.abstract = abstract
        self.internal = internal
        
    def __str__(self):
        return f"Paper(id: {self.id}, title: '{self.title}', abstract: '{self.abstract}', internal: {self.internal})"
    