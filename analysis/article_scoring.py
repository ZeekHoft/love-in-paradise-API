from analysis.open_info_extraction import OpenInformationExtraction
from analysis.evidence_alignment import calculate_entailment

import numpy as np


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
        similarity_scores = article["similarity_scores"]

        min_similarity = min(similarity_scores)
        max_similarity = max(similarity_scores)

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

        normalized_scores = []

        for i in range(len(alignments)):
            alignment = alignments[i]
            sentence = sentences[i]
            similarity_score = similarity_scores[i]
            label = alignment["label"]
            alignment_score = alignment["score"]

            # Normalize similarity score
            if max_similarity > min_similarity:
                normalized_similarity = (similarity_score - min_similarity) / (
                    max_similarity - min_similarity
                )
            else:
                normalized_similarity = 1.0

            # Weighted scoring
            match label:
                case "entailment":
                    sentence_score = alignment_score * (1 + normalized_similarity)
                case "contradiction":
                    sentence_score = -alignment_score * (1 + normalized_similarity)
                case _:
                    sentence_score = 0

            print(
                f"{similarity_score:.3f} | {label[:4]} | {alignment_score:.4f} | {sentence_score: .4f} | {sentence[:100]}..."
            )

            normalized_scores.append(sentence_score)

            evidence_count[label] += 1
            evidence_values[label] += alignment["score"]

        entailment = evidence_values["entailment"] + common_count
        contradiction = evidence_values["contradiction"]

        # print(f"  Entailment: {entailment}, Contradiction: {contradiction}")

        final_score = np.average(normalized_scores)
        final_score = np.tanh(final_score)

        # Score calculation
        score = (entailment - contradiction) / (entailment + contradiction + 1)
        article["score"] = score
        print(f"  Final score: {score} or {final_score}\n")
