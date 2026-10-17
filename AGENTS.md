# Project instructions

## UI guidance imported from BeaverHacks

The user requested all UI-related guidance from the existing BeaverHacks instructions. The excerpts below are copied verbatim from `/Users/lukas/Documents/Projects/bh-website-v2-test/AGENTS.md` on 2026-10-17. Source SHA-256: `31df3f56937a02233814143ab65947092b8ed73cc30cb8a163397e7af28e6a6b`.

Only UI/design, frontend rendering, UI performance and React component guidance is imported. BeaverHacks deployment, database, framework, package-management and unrelated project rules are not imported. References to `apps/site`, `packages/ui`, or a frontend skill are historical source-project references; do not create those paths or install a skill solely to satisfy the quotation. Apply design guidance to the existing DubHacks UI and coordinate with the teammates changing it. The current local ranking/data work must not change storefront styling or competing UI files.

### Verbatim source excerpts

no eyebrows, no fully uppercase text

The main website (`apps/site`) is the visual standard. Keep shared typography, spacing, page widths, buttons, navigation, and brand assets in `packages/ui`. Both public websites use the same page width and responsive gutters.

The website should be beautiful. Never rely on browser primitives (e.g. select) or icons (e.g. ^); always use Base UI and the icon library (Lucide icons). Use the frontend skill. Avoid layers of cards. Always reference the existing design, and ensure that new designs fit in with that existing design.

Performance is of the utmost priority. Always prefer optimistic updates for UI items rather than waiting on the server response, unless it's something where that doesn't make sense (e.g. final project submission). The UI should feel instant and fast, and the packages we choose and the API decisions we make should reflect that.

Do not have multiple React components in the same file. (Unless the other component is truly tiny and tangential, and less than 20 or so LOC)

- Do not make the app SSR
