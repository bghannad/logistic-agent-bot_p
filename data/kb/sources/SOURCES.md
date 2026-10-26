# Knowledge Base Sources

The Project 2 knowledge base is built from real U.S. federal government publications, not documents written for this project. All of these are U.S. federal government works and are in the public domain under **17 U.S.C. § 105** ("Copyright protection... is not available for any work of the United States Government"), so they can be freely reproduced and redistributed, including in a public GitHub repository. No permission was required; they are cited below as good practice and so the source of every fact in the KB is checkable.

Each raw file in `data/kb/raw/` carries its own header (source, title, URL, retrieval date, and a note on whether the text is the full page or the trimmed article body) directly above the verbatim text.

## U.S. Customs and Border Protection (CBP.gov)

| File | Title | URL | Last modified (per site) |
|---|---|---|---|
| `cbp_internet_purchases.txt` | Internet Purchases — Your Responsibility and Liability | https://www.cbp.gov/trade/basic-import-export/internet-purchases | Mar 6, 2024 |
| `cbp_prohibited_restricted_items.txt` | Prohibited and Restricted Items | https://www.cbp.gov/travel/us-citizens/know-before-you-go/prohibited-and-restricted-items | Mar 14, 2025 |
| `cbp_protecting_americans_shopping_online.txt` | Protecting Americans Shopping Online | https://www.cbp.gov/trade/basic-import-export/e-commerce/protecting-americans-shopping-online | Jul 22, 2019 |

`cbp_internet_purchases.txt` is the strongest single document for the core scenario ("why is my package stuck in customs," duty/informal-vs-formal-entry thresholds, lost packages, prohibited/restricted merchandise) and is the one to lean on most heavily in retrieval evaluation.

## Federal Trade Commission (FTC)

| File | Title | URL | Published |
|---|---|---|---|
| `ftc_online_order_never_arrives.txt` | What to do if your online order never arrives | https://consumer.ftc.gov/consumer-alerts/2022/12/what-do-if-your-online-order-never-arrives | Dec 2022 |
| `ftc_prompt_delivery_rules.txt` | Selling on the Internet: Prompt Delivery Rules (Mail/Telephone/Internet Order Merchandise Rule) | https://www.ftc.gov/business-guidance/resources/selling-internet-prompt-delivery-rules | Oct 2001 |
| `ftc_scams_orders_dont_arrive.txt` | Scams in online sales: when orders don't arrive | https://consumer.ftc.gov/scams-online-sales-when-orders-dont-arrive | Jul 2020 |

`ftc_prompt_delivery_rules.txt` is the one with actual regulatory teeth (the 30-day shipping rule, delay-notification and refund obligations) — best for grounding "can I get a refund" answers.

## Not yet pulled

- `Federal Register: Indefinite Suspension of the De Minimis Exemption for Mail Shipments` (June 2026) — fetched successfully during research (see project chat) but not yet saved as a KB doc; ~21 pages, dense regulatory text on the brand-new rule ending duty-free de minimis mail imports. Worth adding — it's exactly the "messy, real, currently relevant" material that makes retrieval harder than Project 1.
- CBP's `help.cbp.gov` article on goods ordered from overseas not received (Article-1174) — page is a JS single-page app that hasn't rendered cleanly yet.

## Extraction method (for the eventual README "how the KB was built" note)

Pages were opened in a real browser (not scraped blind) and their rendered text extracted via the DOM (`document.querySelector('main').innerText`), with any accordion/collapsed sections expanded first (`Open all`) so no content was missed. For `consumer.ftc.gov` articles, the raw extraction also includes reader comments and site navigation/sidebar clutter below the actual article body — that was trimmed by hand before saving, and each file's header notes when trimming happened. Nothing in the substantive article text itself was edited, summarized, or reworded.
