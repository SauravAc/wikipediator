import nltk
from nltk import word_tokenize, pos_tag
from nltk.tree import Tree
from nltk.corpus import brown
from collections import defaultdict
from heapq import nlargest
from math import log, sqrt
import wikipedia, re, webbrowser

phrase_limit = 15        # maximum number of NPs to use as Wikipedia entry candidates (default = 15)
include_summary = False  # turn off for evaluation, which ignores the summary portion of Wiki entries
open_browser = False     # turn on to open browser page instead of printing the final Wikipedia url


def main():
    processInput()
    # runEvaluation()

def processInput():
    include_summary = True
    wiki = Wikipediator()
    with open("input.txt") as f:
        for i, line in enumerate(f):
            print "\nProcessing line %d:\n" % (i+1)
            url = wiki.getWikiPage(line)
            if open_browser:
                webbrowser.open_new(url)
            else:
                print "    URL is: ", url

def runEvaluation():
    include_summary = False
    passed, failed = 0, 0
    wiki = Wikipediator()

    with open("evaluation.txt") as f:
        for i, line in enumerate(f):
            try:
                summaryText = wikipedia.summary(line, auto_suggest = True)
                actualUrl = wikipedia.page(line, auto_suggest = True).url
                url = wiki.getWikiPage(summaryText)
                if actualUrl == url:
                    print "Evaluation line %d  -  test PASSED." % (i+1)
                    passed += 1
                else:
                    print "Evaluation line %d  -  test FAILED." % (i+1)
                    print "    actual URL: " + actualUrl
                    print "       got URL: " + url
                    failed += 1
                if (i+1) % 10 == 0:
                    print "\nEvaluation (so far)  -  %d/%d passed (%.2f%%)\n" % \
                        (passed, passed + failed, float(passed) / (passed + failed) * 100)
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        print "\nEvaluation summary  -  %d/%d passed (%.2f%%)" % \
            (passed, passed + failed, float(passed) / (passed + failed) * 100)


class Wikipediator:
    def __init__(self):
        self.uniLM = UnigramLM()
        corpus = brown.sents()[:10000]
        self.uniLM.estimateUnigrams(corpus)

    def getWikiPage(self, query):
        queryToks = word_tokenize(self.cleanString(query))
        tagged_toks = pos_tag(queryToks)
        
        phraseCounts = self.extractNounPhrases(tagged_toks)
        wikiPages = self.collectWikiPages(phraseCounts)

        tfidf = TfidfModel(phraseCounts, wikiPages)
        return tfidf.getMostSimilarDoc()

    def cleanString(self, string):
        # use a whitelist to filter out unknown characters:
        return re.sub('''[^\s\w\d\?><;,\{\}\[\]\-_\+=!@\#\$%^&\*\|\']''', '', string).decode('utf-8')

    def extractNounPhrases(self, tagged_toks):
        commonNPcounts, properNPcounts = defaultdict(int), defaultdict(int)
        grammar = r"""
            NP_common:  {<NN|NNS|JJ.*>*<NN|NNS>}    # catches JJ-NN noun phrases
            NP_proper:  {<NNP.*>+}                  # catches proper noun phrases
        """
        chunker = nltk.RegexpParser(grammar)
        chunk_tree = chunker.parse(tagged_toks)

        def TreeToTuple(tree):
            return tuple(map(lambda (word, pos): word, tree.leaves()))

        for npTree in chunk_tree.subtrees():
            if npTree.label() == "NP_common":
                commonNPcounts[TreeToTuple(npTree)] += 1
            elif npTree.label() == "NP_proper":
                properNPcounts[TreeToTuple(npTree)] += 1

        phraseCounts = self.prunePhrases(commonNPcounts, properNPcounts)
        return phraseCounts

    def prunePhrases(self, commonNPcounts, properNPcounts):
        commonNPcounts, properNPcounts = list(commonNPcounts.iteritems()), list(properNPcounts.iteritems())
        
        def pruneProperNPs(phraseCounts, size):
            return nlargest(size, phraseCounts, key = lambda (phrase, count): count)

        def pruneCommonNPs(phraseCounts, size):
            return nlargest(size, phraseCounts, key = lambda (phrase, count): -self.uniLM.phraseLogProbability(phrase))

        phrases_left = phrase_limit
        properNPcounts = pruneProperNPs(properNPcounts, phrases_left)
        phrases_left -= len(properNPcounts)
        commonNPcounts = pruneCommonNPs(commonNPcounts, phrases_left)
        return properNPcounts + commonNPcounts

    def collectWikiPages(self, phraseCounts):
        wikiPages = dict()
        for phrase, count in phraseCounts:
            phraseStr = " ".join(phrase)
            try:
                page = wikipedia.page(phraseStr, auto_suggest = True)
                content = page.content
                if not include_summary:    # note: the summary is a prefix of the full page content
                    summary = wikipedia.summary(phraseStr, auto_suggest = True)
                    content = content[len(summary):]    # effectively cuts out the summary part of the page
                wikiPages[page.url] = word_tokenize(self.cleanString(content))
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        return list(wikiPages.iteritems())


class UnigramLM:
    def __init__(self):
        self.clearCounts();

    def clearCounts(self):
        self.unigram_counts = defaultdict(lambda: 1.0)   # starting count is 1 (Laplace smoothing)
        self.token_count = 0

    def estimateUnigrams(self, data_set):
        self.clearCounts()
        for sent in data_set:
            for word in sent:
                self.token_count += 1
                self.unigram_counts[word] += 1
        self.token_count += len(self.unigram_counts)    # normalize for Laplace smoothing

    def phraseLogProbability(self, phrase):
        log_prob = 0.0
        for word in phrase:
            log_prob += log(self.unigram_counts[word.lower()] / self.token_count)
        return log_prob


class TfidfModel:
    def __init__(self, queryPhraseCounts, collection):
        self.vocabulary     = set()
        self.queryTfs       = defaultdict(float)      # term: tf
        self.docsTfs        = []                      # (docUrl, dict(term: tf))
        self.idfs           = defaultdict(float)      # term: idf
        self.cosSims        = defaultdict(float)      # docUrl: cosSim
        self.mostSimilarDocUrl = ""

        self.getQueryTf(queryPhraseCounts)
        self.getDocsTfidf(collection)
        self.getCosineSimilarities()

    def getQueryTf(self, queryPhraseCounts):
        queryLen = 0.0
        for phrase, count in queryPhraseCounts:
            for word in phrase:
                queryLen += count
                self.vocabulary.add(word)
                self.queryTfs[word] += count

        for word, count in self.queryTfs.iteritems():
            self.queryTfs[word] = count / queryLen    # normalize query tfs

    def getDocsTfidf(self, collection):
        for url, doc in collection:
            docLen = len(doc)
            docTfs = defaultdict(float)
            for word in doc:
                if word in self.vocabulary:
                    docTfs[word] += 1
            for word, count in docTfs.iteritems():
                docTfs[word] = count / docLen    # normalize doc tfs
            self.docsTfs.append((url, docTfs))

        numDocs = float(len(collection))
        numDocsWithTerm = defaultdict(int)      # (term): numDocs
        for url, docTfs in self.docsTfs:
            for word in docTfs:
                numDocsWithTerm[word] += 1
        for word in self.vocabulary:
            self.idfs[word] = log(numDocs / (numDocsWithTerm[word] + 1))    # +1 prevents division by 0

    def getCosineSimilarities(self):
        qMag = 0.0
        for word in self.vocabulary:
            qMag += (self.queryTfs[word] * self.idfs[word]) ** 2
        qMag = sqrt(qMag)

        maxSim = 0.0
        for url, docTfs in self.docsTfs:
            dotProd, dMag = 0.0, 0.0
            for word in self.vocabulary:
                dMag += (docTfs[word] * self.idfs[word]) ** 2
                dotProd += self.queryTfs[word] * docTfs[word] * (self.idfs[word] ** 2)
            dMag = sqrt(dMag)
            sim = dotProd / (qMag * dMag) if dotProd != 0 else 0
            self.cosSims[url] = sim
            
            if sim > maxSim:
                maxSim = sim
                self.mostSimilarDocUrl = url

    def getMostSimilarDoc(self):
        return self.mostSimilarDocUrl


if __name__ == "__main__": 
    main()  