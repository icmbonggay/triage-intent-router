## Data

### Source
- **Dataset used:** `mteb/banking77` on Hugging Face, pinned to commit `18072d2685ea682290f7b8924d94c62acc19c0b2`.
- **Original dataset:** BANKING77, from Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić, "Efficient Intent Detection with Dual Sentence Encoders" (2nd Workshop on NLP for Conversational AI, ACL 2020). Original release: https://github.com/PolyAI-LDN/task-specific-datasets
- **Note on the copy:** `mteb/banking77` is a Parquet repackaging published by the MTEB benchmark project. It has 14 fewer rows than the original paper reports (see Size). I used it because the original `PolyAI/banking77` Hub repo relies on a loading script, which `datasets` 5.x no longer supports.

### License
- **License followed:** CC BY 4.0, as declared on the original `PolyAI/banking77` Hugging Face card.
- **Discrepancy:** The `mteb/banking77` copy includes an MIT license file, but it is an unfilled template (no year or copyright holder). Since the copy's publisher did not create the data, I follow the original authors' CC BY 4.0 terms.
- **Attribution:** The original authors are credited in this section and cited below, with a link to the CC BY 4.0 license: https://creativecommons.org/licenses/by/4.0/

### Size

| Split | Rows |
| --- | --- |
| train | 9,993 |
| test | 3,076 |
| **total** | 13,069 |

The original paper reports 13,083 examples, so this copy is 14 rows short. In the test split, 73 intents have 40 examples and 4 have 39, which accounts for its missing rows. The cause of the gap has not been confirmed.

### Labels
- **Number of intents:** 77
- **Columns:** `text` (the customer's message), `label` (integer intent ID), `label_text` (intent name)
- **Example intents:** `card_arrival`, `contactless_not_working`

### Balance

| Split | Smallest intent (count) | Largest intent (count) | Max/min ratio |
| --- | --- | --- | --- |
| train | `contactless_not_working` (35) | `card_payment_fee_charged` (187) | 5.34 |
| test | 4 intents tied, e.g. `atm_support` (39) | 73 intents tied (40) | 1.03 |

The train split is imbalanced: its largest intent has about 5.3 times as many examples as its smallest. The test split is nearly uniform. Because test is balanced and train is not, a model may perform worse on rare training intents than overall accuracy suggests, so per-intent metrics should be reported alongside accuracy.

### Observations from reading examples
- Intent is sometimes implied rather than stated. Some `lost_or_stolen_card` messages ask to freeze the card or check recent activity without saying "lost" or "stolen".
- Some messages are ambiguous even for a human. For example, "what is this charge on my card?" is labelled `direct_debit_payment_not_recognised`, but it could fit several fee-related intents.
- Similar wording appears across different intents. "The ATM didn't give me..." appears in both `wrong_amount_of_cash_received` and `card_swallowed`, where only the missing item (cash or card) decides the intent.
- Writing style varies: full sentences and short phrases, lowercase starts, slang ("money machine"), greetings, and mixed British and American conventions.

### Citation

@inproceedings{Casanueva2020,
  author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title     = {Efficient Intent Detection with Dual Sentence Encoders},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.04807}
}

The `mteb/banking77` copy also asks users to cite MTEB. Copy its BibTeX from the dataset card's citation section.