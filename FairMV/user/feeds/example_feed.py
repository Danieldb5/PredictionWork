"""
Template plugin feed. Copy this file (any name not starting with _), set ENABLED = True.

FairMV calls read(ctx) for EVERY market it values, so return None fast for
markets you have nothing to say about. Exceptions are caught and shown as
warnings; they never crash the scalper.

ctx fields: token_id, slug, question, condition_id, outcome, outcome_index,
            event_slug, now (UTC Timestamp), p_market (local book+trades fair value),
            book (OrderBook), extra

Return any of:
    None                                     nothing to say
    0.63                                     a probability (sd = feed_default_sd)
    {"p": 0.63, "sd": 0.03}                  probability + confidence (prob sd)
    {"shift": 0.02}                          bias: +2c after fusion
    {"lo": 0.1, "hi": 0.4}                   hard bounds
    {"p": .., "sd": .., "ts": <when your data is from>}   staleness: fades with age
    Reading(...) or a list of any of these
Values are about THIS token (ctx.outcome). Use ctx.outcome_index to mirror if
your scraper only knows YES: p_no = 1 - p_yes.
"""

ENABLED = False          # set True to activate
NAME = "example"

_CACHE = {}              # keep expensive scrapes between calls (FairMV calls often)


def read(ctx):
    if "rotten-tomatoes" not in ctx.slug:
        return None
    # --- your scraping / model here; cache it, e.g. refresh every 10 minutes ---
    # score, n_reviews = scrape_rt(ctx.question)
    # p_yes = my_model(score, n_reviews)
    p_yes, sd = 0.60, 0.06
    p = p_yes if ctx.outcome_index == 0 else 1 - p_yes
    return {"p": p, "sd": sd, "note": "rt model"}
