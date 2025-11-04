from analysis.open_info_extraction import OpenInformationExtraction
from analysis.evidence_alignment import calculate_entailment


class ArticleScoring:
    def __init__(
        self,
        claim: str,
        claim_triples: list[tuple],
        oie: OpenInformationExtraction,
    ):
        self.claim = claim
        self.claim_triples = claim_triples
        self.oie = oie

    def score_article(self, article: dict[str, any]):
        """
        Score an article based on claim
        """
        sentences = article["sentences"]

        # Triple comparison with claim
        common_count = 0
        for sentence in sentences:
            article_triples = self.oie.generate_triples(sentence)
            if article_triples:
                common = set(article_triples).intersection(set(self.claim_triples))
                common_count += len(common)

        if common_count != 0:
            print("Common triples found:", common_count)

        # Scores sentences based on entailment to claim
        # Returns list of {label, score, sentence}
        alignments = calculate_entailment(claim=self.claim, sentences=sentences)
        article["alignments"] = alignments

        # Debug output
        print(f"  Alignments for this article: {len(alignments)}")

        evidence_count = {
            "neutral": 0,
            "entailment": 0,
            "contradiction": 0,
        }
        evidence_values = {
            "neutral": 0,
            "entailment": 0,
            "contradiction": 0,
        }

        for alignment in alignments:
            label = alignment["label"]
            evidence_count[label] += 1
            evidence_values[label] += alignment["score"]

        entailment = evidence_values["entailment"] + common_count
        contradiction = evidence_values["contradiction"]

        print(f"  Entailment: {entailment}, Contradiction: {contradiction}")

        # Score calculation
        score = (entailment - contradiction) / (entailment + contradiction + 1)
        article["score"] = score
        print(f"  Final score: {score}")
