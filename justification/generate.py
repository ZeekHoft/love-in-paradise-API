def generate_justification(
    evidence_articles: list[dict],
    justification_count: int,
) -> str:
    """
    Template based justification generation. Requires an ordered list of the top scored
    articles that support the verdict. Each article should consist of the
    headline, url, source_name, and top evidence.
    """
    justification = ""
    if len(evidence_articles) >= justification_count:

        justification += (
            "The verdict was evaluated based on the following news articles:\n"
            + "(Listed based on relevance)\n"
        )

        item_number = 1
        for article in evidence_articles[:justification_count]:
            justification += (
                f"{item_number}. {article['source_name']} | {article['headline']}\n"
                + f"{article["url"]}\n"
                + f"Evidence: {article['sentence']}\n"
            )
            item_number += 1

    elif len(evidence_articles) == 1:
        # Only one supporting evidence is found
        article = evidence_articles[0]
        justification += "The verdict was made on a single online article:\n"
        justification += (
            f"{article['source_name']} | {article['headline']}\n"
            + f"{article["url"]}\n"
            + f"Evidence: {article['sentence']}\n"
        )
        justification += "\nNote: Limited sources available. Be mindful to review the article yourself or rephrase your claim and try again."
        pass
    else:
        # Handle cases with fewer than the desired number of articles
        justification += (
            "The verdict was evaluated based on a limited number of news articles:\n"
        )
        item_number = 1
        for article in evidence_articles:
            justification += (
                f"{item_number}. {article['source_name']} | {article['headline']}\n"
                + f"{article["url"]}\n"
                + f"Evidence: {article['sentence']}\n"
            )
            item_number += 1
        justification += "\nNote: Limited sources available. This claim needs more information for a conclusive verdict."

    return justification
