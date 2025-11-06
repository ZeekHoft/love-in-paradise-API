from analysis.open_info_extraction import OpenInformationExtraction
from analysis.evidence_alignment import calculate_entailment

import numpy as np


class ArticleScoring:
    def __init__(
        self,
        claim: str,
        claim_triples: list[tuple],
        oie: OpenInformationExtraction,
        min_similarity: float = None,
        max_similarity: float = None,
    ):
        self.claim = claim
        self.claim_triples = claim_triples
        self.oie = oie
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        if min_similarity or max_similarity:
            print(
                f"Article scoring initialized with min:{min_similarity:.1f} and max:{max_similarity:.1f}"
            )

    def score_article(self, article: dict[str, any]):
        """
        Score an article based on claim
        """
        sentences = article["sentences"]
        similarity_scores = article["similarity_scores"]

        if self.min_similarity is None:
            self.min_similarity = min(similarity_scores)
        if self.max_similarity is None:
            self.max_similarity = max(similarity_scores)

        print(article["headline"])

        # Triple comparison with claim
        common_count = 0
        for sentence in sentences:
            article_triples = self.oie.generate_triples(sentence)
            if article_triples:
                common = set(article_triples).intersection(set(self.claim_triples))
                common_count += len(common)

        # if common_count != 0:
        #     print("Common triples found:", common_count)

        # Scores sentences based on entailment to claim
        # Returns list of {label, score, sentence}
        alignments = calculate_entailment(claim=self.claim, sentences=sentences)
        article["alignments"] = alignments

        # Debug output
        # print(f"  Alignments for this article: {len(alignments)}")

        evidence_values = {
            "neutral": 0,
            "entailment": 0,
            "contradiction": 0,
        }

        normalized_scores = []
        weight_factor = 1.5

        for i in range(len(sentences)):
            sentence = sentences[i]
            alignment = alignments[i]
            similarity_score = similarity_scores[i]
            label = alignment["label"]
            alignment_score = alignment["score"]

            # Normalize similarity score
            if self.max_similarity > self.min_similarity:
                normalized_similarity = (similarity_score - self.min_similarity) / (
                    self.max_similarity - self.min_similarity
                )
            else:
                normalized_similarity = 1.0

            normalized_similarity = normalized_similarity**weight_factor

            # Weighted scoring
            match label:
                case "entailment":
                    sentence_score = alignment_score * normalized_similarity
                case "contradiction":
                    sentence_score = -alignment_score * normalized_similarity
                case _:
                    sentence_score = 0

            if label != "neutral":
                print(
                    f"{similarity_score:.3f} | {label[:4]} | {alignment_score:.4f} | {normalized_similarity:.4f} | {sentence_score: .4f} | {sentence[:100]}..."
                )

            normalized_scores.append(sentence_score)

            evidence_values[label] += alignment_score

        entailment = evidence_values["entailment"] + common_count
        contradiction = evidence_values["contradiction"]

        # print(f"  Entailment: {entailment}, Contradiction: {contradiction}")

        if normalized_scores:
            final_score = np.average(normalized_scores)
            final_score = np.tanh(final_score)
        else:
            final_score = 0.0

        # Score calculation
        score = (entailment - contradiction) / (entailment + contradiction + 1)
        article["score"] = final_score
        print(f"  Final score OLD: {score}, NEW:{final_score}\n")
