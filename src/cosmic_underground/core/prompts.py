

def tokens_for_poi(poi):
    if getattr(poi, "name", "") == "Boss Skuggs":
        return ["cosmic", "talkbox", "boogie"]
    if getattr(poi, "kind", None) == "npc":
        return ["alien", "funk", "bass"]
    if getattr(poi, "kind", None) == "object":
        return ["weird", "motif"]
    return []