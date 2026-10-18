# Shopper pair 02 — local model

Submitted exactly `a bottle for drinking water under $40` in a new Chrome tab with `?model=local` preserved. Empty cart was verified. Tabs share origin storage; storage was not cleared and isolation is not claimed.

Independently selected **Everyday steel bottle, Graphite**, quantity 1, **$36**. The visible bottle name, image and price supported the choice; capacity, heat retention, leakproofing and stock were not assumed. Clicking its heading did not open detail navigation.

The submitted query first showed “ordering products; last available result shown”, then learned order with the bottle first and “Writing fresh copy…”. Heading click occurred before complete copy. Immediately before adding, the DOM readiness attribute already contained the generation 8 complete entry; add result showed “Complete copy 6.48 s · 19/20 fresh fields · 1 withheld”. Generation 8 rank/first-three/complete after-paint opportunities: 421.9 / 5539.8 / 6515.4 ms. These are DOM commit opportunities, not physical paint or image readiness.

Cart visibly contained Everyday steel bottle Graphite × 1 at $36; demo subtotal $36. Fake checkout showed **Demo order confirmed**, “No payment was processed or real order placed”, and 1 demo item · $36 demo subtotal.

9 input actions; approximately 35 seconds from opening tab through confirmation. First timing checkpoint was taken just after tab creation (03:50:49.411570 UTC, monotonic 936072.379701291); final checkpoint was 03:51:18.688862 UTC (936101.656753708), 29.277 seconds later. Exact pre-open and per-action timestamps were not collected; action order and tool durations are in JSON. No timeout or browser errors.

The native CUA transcript contains the confirmation screenshot. No durable screenshot path was returned; there is no Playwright trace. Full raw HTML readiness-attribute value captured before adding is in readiness-attribute.txt. No source/catalog/gold files or other shopper result were read.

