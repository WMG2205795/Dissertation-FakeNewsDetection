def format_evidence(
    retrieved_chunks: list[dict],
) -> str:

    evidence_blocks = []

    for item in retrieved_chunks:

        contents = item.get(
            "contents"
        )

        if (
            contents is None
            or not str(contents).strip()
        ):
            continue


        evidence_blocks.append(
            "\n".join(
                [
                    f"[Evidence {item['rank']}]",
                    f"Text: {contents}",
                    f"Source URL: {item.get('source_url') or 'Unavailable'}",
                    f"Source type: {item.get('source_type') or 'Unknown'}",
                ]
            )
        )


    if not evidence_blocks:

        return (
            "[No usable evidence was retrieved.]"
        )


    return "\n\n".join(
        evidence_blocks
    )
