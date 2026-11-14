# Resale Listing Assistant (ADK 2.0 graph workflow)

Photos in, reviewed listing out. A picky-buyer critic reviews each draft and the writer revises.
Nothing is posted online; the listing is saved to a file after you confirm, and you post it yourself.

```
START -> photo_review -> save_photo_review -> ask_seller (pause)
      -> intake -> store_facts -> pricing_agent -> ask_extras (pause if missing) -> ask_price (pause if missing)
      -> start_drafting -> drafter -> fact_check
fact_check --passed--> buyer_critic -> decide
           --failed------------------> decide      (skips the LLM critic)
decide --approved--> confirm_save (pause: yes/no, then code saves)
       --revise----> drafter                       (the worker/critic cycle)
       --give_up---> fallback                      (after 3 drafts, nothing is saved)
```

## LLM nodes vs code nodes

| Node | Type | Job |
|---|---|---|
| photo_review | LLM + web_search | Says what it can see and what it cannot tell |
| intake | LLM (typed output) | Builds a fact sheet; anything the seller did not say is NOT PROVIDED |
| pricing_agent | LLM + web_search, find_comps, price_stats | Looks up asking prices |
| drafter | LLM (typed output) | Writes title, category, condition, description (no price) |
| buyer_critic | LLM (typed output) | Picky buyer; fills a true/false rubric plus specific issues |
| ask_seller, ask_extras, ask_price, confirm_save | code + pause | Ask the seller; never assume "None" or a price |
| fact_check | code | Rejects placeholders, prices in text, unsupported claim words and numbers |
| decide | code | Approve, revise, or give up after 3 drafts |

The price is set by code from the seller's answer, so the writer cannot change it.

## Setup and run

This agent uses the shared model helper from the course repo (`shared/model.py`), so it runs from
the course's `src/gemini-adk/multi-agents` folder. Copy `resale_listing_graph` into that folder.
Then, from `src/`, with the course `.venv` active and `.env` set up (Gemini key):

```powershell
cd gemini-adk\multi-agents
adk web
```

Pick `resale_listing_graph`, attach photos, and type "I want to sell this".
Saved listings go to `resale_listing_graph\outputs\listings\`.

To compare the picky buyer critic with a polite reviewer, set `CRITIC_STYLE` before starting:

```powershell
$env:CRITIC_STYLE="polite"; adk web
```

## Testing

Unit tests for the code check in `fact_check.py` need no model or API key:

```powershell
pip install pytest
cd resale_listing_graph\tests
python -m pytest
```

Manual end-to-end checks in `adk web`:

- [ ] Attach photos: the photo review lists what it can see and what it cannot tell.
- [ ] Leave out the price: the agent pauses and asks for one.
- [ ] A draft with an invented detail is rejected by `fact_check` and revised.
- [ ] Answer "no" at the final confirmation: nothing is saved.
- [ ] Answer "yes": the listing is saved to `outputs\listings\`.
