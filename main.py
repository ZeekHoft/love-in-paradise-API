from analysis.open_info_extraction import OpenInformationExtraction
from analysis.sentence_similarity import SentenceSimilarity
from webcrawling.article_scraper import ArticleScraper
from tokenization.english import Eng_Tokenization_NLP
from llm.fact_checker_agent import FactCheckerAgent
from analysis.article_scoring import ArticleScoring

from justification.generate import generate_justification
from webcrawling.search_articles import search_news
from clasification.check import classify_input

from rank_bm25 import BM25Okapi
from logs import DocumentLogs
from typing import Generator
import numpy as np
import traceback
import spacy
import time


ACCEPT_LIST = ["news claim", "statement", "question"]
# news = "High pressure area over China influencing Tropical Depression Salome"
news = "CLTG Builders worked with the Discayas for Davao projects"
# news = "Some celebrities participated in assisting Cebu earthquake victims."
# news = "A Chinese Coast Guard ship fired its water cannon at a Philippine vessel near the West Philippine Sea."

# Instantiate classes that have models or take time to load
print("Loading models...")
nlp = spacy.load("en_core_web_sm")
log_collector = DocumentLogs()
sentence_similarity = SentenceSimilarity(nlp)
info_ext = OpenInformationExtraction()


def love_in_paradise(claim, use_llm=False) -> Generator[dict, None, None]:
    """
    Algorithm for verifying if a claim is true or not using online sources.

    This is a generator that yields a dictionary containing its current process
    until the whole algorithm is finished.
    """

    # Data to return to server
    results = {
        "verdict": None,
        "justification": None,
        "confidence": None,
        "sources": [],
        "currentProcess": None,
        "progress": 0.0,
    }

    if claim is None or claim.strip() == "":
        print("No claim was provided.")
        results["justification"] = "No claim was provided. Please try again."
        yield results
        return

    try:
        results["currentProcess"] = "Checking if claim is verifiable"
        results["progress"] = 1 / 8
        yield results

        # Take claim input
        claim_input = claim

        # Tokenize
        tokenizer = Eng_Tokenization_NLP(nlp=nlp)
        tokenizer.tokenizationProcess(word_list=claim_input.split())
        print("Finished tokenization.")

        # Classify input if it is verifiable or not
        try:
            input_classification = classify_input(claim_input)
            if input_classification in ACCEPT_LIST:

                results["currentProcess"] = "Searching the web"
                results["progress"] = 2 / 8
                yield results

                print(f"Input is a {input_classification}; proceeding to tokenization.")

                # Build search query from proper nouns and nouns, with fallback
                propn_tokens = tokenizer.pos_tokens.get("PROPN", [])
                noun_tokens = tokenizer.pos_tokens.get("NOUN", [])
                search_query_tokens = propn_tokens + noun_tokens

                # Fallback: if no proper nouns or nouns, use the original claim
                if not search_query_tokens:
                    print(
                        "Warning: No PROPN or NOUN tokens found, using full claim for search"
                    )
                    search_query = claim_input
                else:
                    search_query = " ".join(search_query_tokens)

                print(f"Search query: {search_query}")

                # Search articles - Request 20 results (2 API calls, allows 50 fact-checks/day)
                articles = search_news(
                    search_query,
                    results_amt=20,  # 2 API calls per fact-check = 50 fact-checks/day
                )

                print(f"Found {len(articles)} articles from search")

                if len(articles) == 0:
                    print("No news articles found related to news claim.")
                    results["justification"] = (
                        "No news articles found related to news claim."
                    )
                    yield results
                    return
                # print(articles)
            else:
                print("Input is not considered a news claim!")
                results["justification"] = "Input is not considered a news claim!"
                yield results
                return
        except KeyError as e:
            print(f"Missing POS tag in tokenization: {e}")
            traceback.print_exc()
            results["justification"] = (
                f"Unable to extract key terms from claim. Please try rephrasing."
            )
            yield results
            return
        except Exception as e:
            print((f"News claim has missing some missing key elements: {e}"))
            traceback.print_exc()
            results["justification"] = (
                f"News claim has missing some missing key elements: {e}"
            )
            yield results
            return

        results["currentProcess"] = "Retrieving data from articles"
        results["progress"] = 3 / 8
        yield results

        # Scrape each article
        articleScraper = ArticleScraper()
        articleScraper.timeout = 60
        news_data = articleScraper.scrape_multithreaded(articles)
        if len(news_data) == 0:
            print("Problem occurred in scraping data")
            results["justification"] = "Problem occurred in scraping data"
            yield results
            return

        """
        News Data Dictionary Structure
        news_data = {
            url: {
                "headline": HEADLINE,
                "content": CONTENT,
                "source_name": SOURCE,
                "sentences": will be compared to claim,
                "link": url
            },
        }
        """

        print(f"Successfully scraped {len(news_data)} articles")

        print("Scraped Articles ==================================")
        for url, data in news_data.items():
            print(url)
            print(data["source_name"], "|", data["headline"])

        results["currentProcess"] = "Searching possible evidence using claim keywords"
        results["progress"] = 4 / 8
        yield results

        # BM25 Sentence Level Keyword Search
        # Searches related sentences based on bag of words algorithm.
        # Inaccurate if words are not exact match
        print("Performing BM25 sentence retrieval using query.")

        article_sentences = {}
        for url, data in news_data.items():
            scraped_content = data["content"]
            separated_sentences = tokenizer.separate_sentences(scraped_content)
            tokenized_sentences = [
                tokenizer.tokenize_sentence(sent) for sent in separated_sentences
            ]

            bm25 = BM25Okapi(tokenized_sentences)
            bm25_scores = bm25.get_scores(tokenizer.tokenize_sentence(claim_input))
            print(data["headline"])
            if bm25_scores.any():
                article_sentences[url] = []
                for i in range(len(bm25_scores)):
                    sentence = separated_sentences[i]
                    score = bm25_scores[i]
                    if score != 0.0:
                        article_sentences[url].append(sentence)
                        print(f"{score:.2f} - {sentence}")
                print(
                    f"{len(article_sentences[url])} out of {len(tokenized_sentences)} scored.\n"
                )
            else:
                print("No related sentences found to query.\n")
                article_sentences[url] = []

        results["currentProcess"] = "Ranking evidence based on relevance"
        results["progress"] = 4 / 8
        yield results

        # SEMANTIC SENTENCE SEARCH
        # Filtering out the most relevant data
        print("Finding relevant data:")
        sentence_similarity.set_main_sentence(claim_input)
        urls_to_remove = []
        min_score = 1.0
        max_score = 0.0
        for url, data in news_data.items():
            similar_sentences = sentence_similarity.find_similar_sentences(
                article_sentences[url],
                cutoff_score=0.40,
            )
            print(data["headline"])
            if similar_sentences == []:
                print("No similar sentences found.")
                urls_to_remove.append(url)

            else:
                print("SCORE | SENTENCE")
                for sentence, score in similar_sentences:
                    if score < min_score:
                        min_score = score
                    if score > max_score:
                        max_score = score
                    print(f"{score:.3f} | {sentence[:100]}")
                    if "sentences" not in news_data[url].keys():
                        news_data[url]["sentences"] = [sentence]
                        news_data[url]["similarity_scores"] = [score]
                    else:
                        news_data[url]["sentences"].append(sentence)
                        news_data[url]["similarity_scores"].append(score)
            print()

        # Discard urls with no relevant sentences
        for key_url in urls_to_remove:
            print(f"Removed News: {news_data[key_url]['headline']}")
            news_data.pop(key_url)

        print(f"\nTotal articles with relevant content: {len(news_data)}")

        # Sort url articles by their max similarity score to identify most related ones
        sorted_by_score = sorted(
            news_data.keys(),
            key=lambda url: max(news_data[url]["similarity_scores"]),
            reverse=True,
        )

        print(f"\nSorted by similarity score:")
        index = 1
        for url in sorted_by_score:
            print(
                f"{index:02}. {max(news_data[url]["similarity_scores"]):.2f} - {url}:"
            )
            index += 1
        print()

        # LARGE LANGUAGE MODEL FACT-CHECKING
        # Use LLM agent for determining the verdict
        if use_llm:
            print("==============================")
            print("LLM Response:")
            fca = FactCheckerAgent(claim=claim_input, knowledge=str(news_data))
            agent_response = fca.verify()
            if agent_response:

                # LLM Specific
                results["verdict"] = agent_response[0]
                results["justification"] = agent_response[1]
                results["confidence"] = 0

                # Frontend display
                results["sources"] = list(news_data.keys())
                results["headlines"] = {
                    key: value["headline"].replace('"', "'")
                    for key, value in news_data.items()
                }
                results["currentProcess"] = "Complete"
                results["progress"] = 8 / 8

                log_collector.paradise_logs(
                    log_collector.user_input(claim_input),
                    log_collector.valid_claim(results),
                    log_collector.search_log(search_query),
                )

                yield results
                return
            else:
                results["justification"] = "Error: No response from LLM"
                results["currentProcess"] = "Error"
                results["progress"] = 8 / 8
                yield results
                return

        # End of LLM section ======================================
        # Code afterwards is the manual NLP fact-checking algorithm

        # Information Extraction
        # ===============================================================
        # Gets subject, predicate, object triples
        claim_triples = info_ext.generate_triples(claim_input)
        print(f"Claim triples: {claim_triples}")

        results["currentProcess"] = "Comparing evidence to claim"
        results["progress"] = 5 / 8
        yield results

        # CLAIM-EVIDENCE ALIGNMENT & ENTAILMENT SCORING
        # ===============================================================
        # Given a list of the most relevant sentences from articles, evaluate them against the claim
        # -> evidences = {"agree", "disagree", "neutral"}

        print("\nScoring each article")
        article_scoring = ArticleScoring(
            oie=info_ext,
            claim=claim_input,
            max_similarity=max_score,
            min_similarity=min_score,
            claim_triples=claim_triples,
        )
        print("SIMILARITY | LABEL | ALIGNMENT | WEIGHTED SCORE | SENTENCE")
        last_score = None
        previous_consistent = True
        current_consistent = True
        stopped_early = False
        for url in sorted_by_score:
            article = news_data[url]
            # Score each article based on semantic entailment and matching triples
            latest_score = article_scoring.score_article(article=article)
            if latest_score != 0:
                print(f"Latest score is {latest_score:.2f}. ", end="")

                # Continue checking when scores are past threshold
                if abs(latest_score) > 0.5:
                    last_score = None
                    previous_consistent = True
                    current_consistent = True

                if last_score is not None:
                    # Stop scoring when inconsistency is encountered
                    current_consistent = (latest_score > 0) == (last_score > 0)
                    if current_consistent:
                        print("Last 2 numbers are consistent.")
                    elif not previous_consistent and not current_consistent:
                        print("Inconsistent scores. Further scoring stopped.")
                        stopped_early = True
                        break
                    else:
                        print("Last 2 numbers are inconsistent.")
                else:
                    print("Reset inconsistency counter.")
                last_score = latest_score
                previous_consistent = current_consistent
        if not stopped_early:
            print("Scoring process ended with all articles scored.\n")
        else:
            print("Done scoring. Some articles were not scored.\n")

        results["currentProcess"] = "Aggregating Scores"
        results["progress"] = 6 / 8
        yield results

        # Print all scores of articles and remove nonrelevant
        print("SCORE | ARTICLE")
        agree = []
        disagree = []
        urls_to_remove = []
        for url in sorted_by_score:
            article = news_data[url]
            score = article.get("score", 0)
            if score > 0.01 or score < -0.01:
                print(f"{score: .5f} | {article['headline'][:50]}...")
                if score > 0:
                    agree.append((url, score))
                else:
                    disagree.append((url, score))
            else:
                urls_to_remove.append(url)
        print()

        # Discard urls with no score
        print(f"Removing {len(urls_to_remove)} article/s that scored low or 0:")
        for key_url in urls_to_remove:
            print(f"Removed: {news_data[key_url]['headline'][:50]}")
            news_data.pop(key_url)
        print()

        article_scores = [a["score"] for a in news_data.values() if a["score"] != 0]
        if len(article_scores) == 0:
            print("No significant evidence found.")
            results["justification"] = "No significant evidence found."
            results["currentProcess"] = "Complete"
            results["progress"] = 8 / 8
            yield results
            return

        # SCORE AGGREGATION
        # ===============================================================
        # More agree-ing evidences: higher confidence, -> True
        # Few agree-ing evidences: low confidence,-> Likely True
        # Divisive, 50-50 agree/disagree: Unsure, Need more information
        # Few disagree-ing confidence: low confidence, -> Likely False
        # More disagree-ing confidence: high confidence, -> False

        print("Statistics")
        agree_scores = [s[1] for s in agree]
        disagree_scores = [s[1] for s in disagree]
        if agree and disagree:
            win_count = len(agree_scores) >= len(disagree_scores)
            win_highest = max(agree_scores) > abs(min(disagree_scores))
            win_total = sum(agree_scores) > abs(sum(disagree_scores))
            conditions = [win_count, win_highest, win_total]
            winning_side = None

            # Score verdict only based on side with majority evidence or if first article is highest rated
            if conditions.count(True) == 3 or (
                news_data.get(sorted_by_score[0], {}).get("score", 0)
                == max(agree_scores)
                and win_highest
            ):
                winning_side = "Agree"
                article_scores = agree_scores
            elif conditions.count(False) == 3 or (
                news_data.get(sorted_by_score[0], {}).get("score", 0)
                == min(disagree_scores)
                and not win_highest
            ):
                winning_side = "Disagree"
                article_scores = disagree_scores

            print(
                f"Article Count: {len(agree_scores)} Agree vs {len(disagree_scores)} Disagree"
            )
            print(
                f"Highest Score: {max(agree_scores):.2f} Agree vs {abs(min(disagree_scores)):.2f} Disagree"
            )
            print(
                f"Total Overall: {sum(agree_scores):.2f} Agree vs {abs(sum(disagree_scores)):.2f} Disagree"
            )
            print("Winner: ", end="")
            print("Agree" if win_count else "Disagree", end=", ")
            print("Agree" if win_highest else "Disagree", end=", ")
            print("Agree" if win_total else "Disagree")
            if winning_side is not None:
                print(f"Verdict will be based on: {winning_side}")

        results["currentProcess"] = "Finalizing score"
        results["progress"] = 7 / 8
        yield results

        # Verdict Score: Average of all scores
        verdict_score = sum(article_scores) / len(article_scores)
        print(f"\nFinal Verdict Score: {verdict_score}")

        # Calculate confidence based on standard deviation
        standard_deviation = np.std(article_scores)
        confidence = (1 - standard_deviation / 2) * 100
        confidence = max(0, min(100, confidence))
        print(f"Confidence Level: {confidence:.1f}%\n")

        # Verdict Assigment
        THRESHOLD1 = 0.3
        THRESHOLD2 = 0.45
        if -THRESHOLD1 < verdict_score < THRESHOLD1:
            verdict = "UNSURE"
        elif verdict_score <= -THRESHOLD2:
            verdict = "FALSE"
        elif verdict_score <= -THRESHOLD1:
            verdict = "LIKELY FALSE"
        elif verdict_score >= THRESHOLD2:
            verdict = "TRUE"
        elif verdict_score >= THRESHOLD1:
            verdict = "LIKELY TRUE"

        # JUSTIFICATION GENERATION
        # Manual justification based on top articles

        # Articles justification will base on
        evidence_scores = agree if verdict_score > 0 else disagree
        alignment_label = "entailment" if verdict_score > 0 else "contradiction"

        evidence_articles = []
        for url, score in evidence_scores:
            article = news_data[url]
            alignments = [
                a for a in article["alignments"] if a["label"] == alignment_label
            ]
            top_sentence = max(
                alignments, key=lambda alignment: abs(alignment["score"])
            )["sentence"]
            evidence_articles.append(
                {
                    "headline": article["headline"],
                    "source_name": article["source_name"],
                    "url": url,
                    "sentence": top_sentence,
                }
            )

        justification = generate_justification(evidence_articles, 3)

        # FINAL RETURN DATA
        results["verdict"] = verdict
        results["justification"] = justification
        results["confidence"] = confidence
        results["sources"] = list(news_data.keys())
        results["headlines"] = {  # Dict of headlines where url: headline
            key: value["headline"].replace('"', "'") for key, value in news_data.items()
        }
        results["currentProcess"] = "Complete"
        results["progress"] = 8 / 8

        # Log data to Firebase
        log_collector.paradise_logs(
            log_collector.user_input(claim_input),
            log_collector.valid_claim(results),
            log_collector.search_log(search_query),
        )

        yield results
        return

    except Exception as e:
        # Catch any unhandled exceptions
        print(f"Unexpected error in love_in_paradise: {e}")
        traceback.print_exc()
        results["justification"] = f"Unexpected Error occurred"
        results["currentProcess"] = "Error"
        results["progress"] = 8 / 8
        yield results
        return


def validate(expected: bool, claim: str, is_llm: bool):
    """
    Validate if algorithm returns the correct expected verdict
    """
    lip = love_in_paradise(claim, use_llm=is_llm)
    final = None
    for result in lip:
        final = result

    verdict_result = None
    verdict: str = final["verdict"]
    if "true" in verdict.lower():
        verdict_result = True
    elif "false" in verdict.lower():
        verdict_result = False

    if verdict_result is None:
        return None
    elif expected == verdict_result:
        return True
    else:
        return False


if __name__ == "__main__":
    # New way to run code:
    lip = love_in_paradise(input("Enter claim: "), use_llm=False)
    final = None
    process_name = None
    times = {}
    start = time.time()
    for result in lip:
        if process_name:
            times[process_name] = time.time() - start
        print(result["currentProcess"])
        start = time.time()
        process_name = result["currentProcess"]
        final = result
    times[process_name] = time.time() - start

    print("\nRETURN DATA")
    for key, val in final.items():
        print(f"{key.title()}: {val}")

    print("\nMEASURED TIME")
    for key, val in times.items():
        print(f"{val:.2f} seconds - {key}")
    print(f"{sum(list(times.values())):.2f} seconds - Total")
