# Project vision

## One-sentence pitch

Build an agent-driven experimentation tool that helps ecommerce teams find shopping friction, make factual storefront improvements, and test whether those changes help diverse shoppers complete suitable purchases.

The merchant-facing product is an evidence-backed loop: **observe → propose → edit → retest → compare**. Shopper success and satisfaction should improve alongside merchant revenue. A larger basket is useful only when the purchase is relevant, meets the shopper's requirements, and stays within their constraints.

This document describes the intended product and how to evaluate it. The repository currently contains a working local scaffold with deterministic mocks and a scripted browser path. It does not yet establish that our agents predict human behavior or that a change increases human conversion or revenue. The final product name and model provider remain undecided.

## Who it serves and what problem it solves

The primary user is a merchant or ecommerce product team deciding what to change on a storefront. They can see a drop in a funnel without knowing whether the cause is unclear delivery information, poor product discovery, unsuitable recommendations, confusing navigation, or an interface failure. They need specific changes with inspectable evidence and a way to compare alternatives.

The shopper is the other stakeholder. A successful experience means finding something appropriate, understanding its cost and terms, and completing the task without unnecessary friction. Sometimes the correct result is to decline a purchase because the catalog cannot satisfy the request.

Our dashboard should let a merchant follow the full experiment: which shoppers were tested, where they struggled, what change was proposed, what actually changed, and how the candidate compared with the baseline. A chat response recommending products is not enough to demonstrate this product. The demo needs a storefront interaction, a visible intervention, and a measured retest.

## A concrete end-to-end demo

The following is the target workflow, not a claim that every step is implemented today.

1. **Define the experiment.** Choose a local fixture catalog, record its product and fee facts, snapshot the baseline, and define task success before running shoppers. For example: find a useful gift within a $40 budget including fixture-defined fees, meet a stated product requirement, and understand delivery terms before ordering.
2. **Define diverse shoppers.** Give each persona a shopping goal, budget, hard requirements, preferences, patience, and stopping behavior. A decisive shopper with a strict budget should behave differently from a patient explorer or a returning shopper with supplied history. Define what each can observe; do not give the shopper hidden evaluator answers.
3. **Separate development and final evaluation.** Reserve disjoint held-out personas and seeds. Use development runs to discover problems and choose changes. Keep final evaluation evidence out of that selection loop.
4. **Run the baseline.** Shopper agents navigate the page, inspect products and terms, select items, and either reach the fake confirmation screen or stop. Save steps, screenshots, traces, technical errors, and independently checked outcomes.
5. **Propose an intervention.** Turn a specific observation into a constrained change. If delivery terms exist but are hard to find, propose making those existing facts visible. Link the proposal to the relevant events. Do not invent free shipping, availability, discounts, reviews, or product capabilities.
6. **Apply and retest candidates.** Represent edits as versioned configuration. Show the merchant the proposed patch and resulting page. Compare candidates on development personas, including regressions and unsuccessful proposals; do not hide attempts that failed to improve the result.
7. **Freeze the final candidate.** Choose it using development evidence, then lock the configuration, model/prompt versions, evaluator rules, catalog, and evaluation plan. Run baseline and candidate on the same paired held-out personas and seeds from equivalent starting states.
8. **Explain the comparison.** Show per-persona outcomes, aggregate differences, supporting traces, constraint violations, and operational cost. Clearly label every outcome synthetic. A neutral or negative result is valid; the system must be able to explain it rather than manufacture a winning story.

The current fixture shop provides a safe place to demonstrate this flow. Its order confirmation is fake: no payment, address, real order, or external transaction is required.

## What exists today, and what comes next

| Area                            | Current repository behavior                                                                                                                          | Intended next capability                                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Merchant dashboard              | Polls runs, shows persona splits, baseline/variant settings, synthetic comparisons, events, artifacts, and suggestions                               | A visible experiment timeline, proposal review, candidate creation, and iteration history                                   |
| Shopper behavior                | A deterministic mock rule stops shipping-sensitive personas when shipping is hidden; a separate Playwright script follows a fixed fake-checkout path | Model-driven browser decisions reflecting goals, budgets, requirements, patience, and optional preferences/history          |
| Variants and optimizer          | Seeded configurations change headline, shipping visibility, and recommendation order; a rule suggests showing shipping from development-run evidence | Generate supported factual patches, review/apply them as candidates, then retest; no arbitrary generated code execution     |
| Evaluation                      | A synthetic binary task score and failure reasons; mock comparisons are separated from browser smoke runs                                            | Independent requirement checks, richer outcome categories, paired analysis, and recorded latency/cost                       |
| Personas and experiment control | Four seeded personas, disjoint development/held-out lists, immutable run snapshots and a recorded seed                                               | Structured budget/patience/requirements, versioned selection and evaluator metadata, and a frozen final evaluation workflow |
| Recommendations                 | A Go ranking adapter can use explicitly supplied history; the fixture shop's history option is placeholder ordering                                  | Controlled model-based ranking experiments connected to the same shopper/evaluation loop                                    |
| Models and training             | The app uses a no-network stub; isolated feature and 9B shared-backbone fusion proofs have trained heads and synthetic reports                       | Select and evaluate an inference adapter; explore post-training only against controlled baselines                           |

The current batch button runs the predefined baseline and variant across all seeded personas. It does not yet implement candidate selection followed by a locked final held-out evaluation. Its shipping-rule outcome is constructed by the mock, not discovered evidence of human shopping behavior. Reaching the scripted confirmation screen proves browser plumbing, not independent shopper competence.

## Architecture and parallel team work

Keep the existing local foundation and let teammates build against explicit boundaries. These are workstreams, not assigned people.

| Workstream               | Current location                                     | Boundary and responsibility                                                                                              |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Dashboard                | `apps/web/`                                          | Vite+ React/TypeScript UI with Tailwind/shadcn primitives; consume contracts and present evidence and comparisons        |
| Experiment orchestration | `apps/api/app.go`, `queue.go`, `store.go`            | Go `net/http`, SQLite metadata, immutable run identities, bounded in-process queue, local artifacts, and polling         |
| Shopper runner           | `apps/api/internal/runner/`, `apps/runner/`          | Go mock/subprocess adapter and Node/TypeScript Playwright execution; actions, observations, traces, and bounded stopping |
| Evaluator                | `apps/api/internal/evaluator/`                       | Check outcomes against task/catalog truth independently of the shopper's own success claim                               |
| Optimizer                | `apps/api/internal/optimizer/`                       | Evidence-linked factual proposals from development runs, represented as constrained configuration patches                |
| Personalized ranking     | `apps/api/internal/recommendation/`                  | Rank catalog items using only the supplied task context and explicitly permitted history/preferences                     |
| Model adapter            | `apps/api/internal/model/`                           | Server-side inference interface, provider configuration, and eventual latency/token/cost accounting                      |
| Fixture environment      | `apps/storefront/`, `apps/api/fixtures/catalog.json` | Local demo page, catalog/config fixtures, known behavior, and fake checkout; keep product IDs and facts consistent       |
| Shared contracts         | `packages/contracts/`                                | Zod schemas mirrored by Go JSON structs; coordinate schema changes before parallel implementation                        |

The shared v1 entities are `Persona`, `Experiment`, `Variant`, `Run`, `StepEvent`, `Artifact`, `EvaluationResult`, and `ProposedChange`. The Go API sends a run snapshot to the browser child process through JSON stdin/stdout; no additional runner service is required. New persona fields and evaluation metrics need coordinated contract changes, not incompatible local assumptions.

Vite+ handles frontend development/builds and shared checks. The actual `tsc-ts` compiler is installed and checked separately from Vite+'s built-in type checking; standard TypeScript and Go checks remain available. Portless exposes the loopback-only HTTP services:

- Dashboard: `http://dubhacks.localhost:1356`
- API: `http://api.dubhacks.localhost:1356`
- Fixture shop: `http://shop.dubhacks.localhost:1356`

Use the [README](README.md) for setup and commands. The default demo needs no API key, cloud database, Redis, or Docker.

## Experimentation methodology

**Define the target before optimizing.** Freeze task predicates, catalog facts, allowable actions, fees, and the meaning of a successful purchase. Preserve the distinction between hard constraints and preferences. The evaluator should check the selected item, total cost, and final state against known fixture facts; an agent saying “I succeeded” is insufficient.

**Make comparisons paired.** Baseline and candidate receive the same persona, goal, seed, catalog snapshot, initial cart/browser state, and run budget. Record model and prompt versions, policy/configuration, evaluator version, timings, and artifact IDs. A seed helps reproducibility but does not guarantee deterministic behavior from every model provider. Repeat stochastic trials and retain all results rather than selecting a favorable run.

**Protect the final evaluation.** Development evidence may inform proposals and candidate selection. Freeze that selection before inspecting final held-out results. If the team changes the candidate after seeing those results, treat the exposed set as development evidence and reserve a fresh final evaluation set. Keep personalization training data separate too.

**Separate outcome types.** Report valid task completion, deliberate abandonment, requirement/constraint failure, technical failure, and timeout separately. A broken selector, crashed browser, unavailable model, or infrastructure deadline is not evidence that a shopper abandoned the store. A safety step cap is not automatically a persona's patience limit. Mark interrupted or ambiguous sessions unresolved/censored rather than silently recoding them as rejection.

**Use independent checks and honest denominators.** Test evaluator predicates against known good and bad fixture outcomes. Keep the evaluator distinct from the policy choosing actions; model judgments can supplement deterministic checks but should not replace them. Report all assigned runs, technical failure rates, and task success among evaluable runs together so exclusions cannot improve the headline unnoticed. A process status of `completed` means execution finished; it does not imply the shopping task succeeded.

**Show uncertainty and variation.** Include paired wins/losses/ties and results by persona requirements and budget. Use paired uncertainty estimates when there are enough independent cases. With a small hackathon sample, show exact counts and traces instead of claiming statistical significance or population-wide lift. Repeat an unchanged baseline to expose simulator noise.

## Metrics that make the demo meaningful

| Metric                   | What to measure and how to interpret it                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Task success             | Fraction of evaluable runs satisfying the predeclared shopping goal; show technical failures alongside it                                                                            |
| Requirement satisfaction | Which hard requirements and preferences were met; a checkout does not override an unsuitable selection                                                                               |
| Checkout completion      | Reaching the verified fake confirmation state; report separately from task success                                                                                                   |
| Time and steps           | Actions, repeated searches, backtracking, time to decision, and total runtime; distinguish model/tool latency from simulated shopper patience                                        |
| Basket value             | Fixture-defined total conditional on a successful, relevant, constraint-satisfying purchase, together with success count; expensive unsuitable purchases do not count as improvement |
| Constraint violations    | Budget overruns, incompatible products, unavailable items, unsupported claims, hidden fees, and prohibited actions; serious violations are rejection criteria                        |
| Latency and cost         | Model calls, tokens where available, inference/tool time, and actual API cost if a provider is used; mock cost must not be represented as measured model spend                       |
| Satisfaction proxies     | Fulfilled needs, clarity of terms, unnecessary effort, and regret signals with an explicit rubric; self-reported agent satisfaction remains synthetic                                |

Choose a primary metric and guardrails before each experiment. The intent is more suitable successful purchases and satisfied shoppers, with merchant value measured alongside those outcomes. Do not maximize spend through irrelevant items, pressure, concealed fees, or relaxed budgets. A modest basket that meets the request can be a better outcome than a larger one that does not.

All displayed results must identify their provenance: deterministic mock, synthetic model-driven browser run, or later real-user evidence. Neither synthetic checkout rates nor synthetic basket values establish human conversion, revenue lift, or causal business impact.

## Optional personalized ranking within the same system

Personalization is an optional treatment inside this experimentation product, not a second standalone product. A persona may bring supplied preferences or history, such as prior product interactions and explicit likes/dislikes. Record whether that history is a fixture or permissioned user input. Do not collect unrelated personal data or infer access to accounts we do not have.

Use a controlled comparison:

| Arm                       | Model and context                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| A: no history             | The base model receives the current task, budget/requirements, and catalog, without prior history                                             |
| B: supplied history       | The same base model and catalog receive the same current task plus the allowed history/preferences                                            |
| C: optional post-training | A model derived from that same base receives the same history, task context, and catalog as B; training is a separately measured intervention |

A versus B tests the value of history in context. B versus C tests whether post-training adds value beyond access to that same history. Hold candidate generation, catalog availability, evaluation rules, and decoding settings constant where applicable, and disclose deliberate differences. Report context size and serving cost rather than attributing a larger inference budget to training quality.

Train only on the training partition and use held-out people/tasks or time splits appropriate to the data. Do not leak final evaluation preferences, labels, or future interactions into history, prompts, or training. Check ranking relevance against independently defined preferences and then test its effect on the same end-to-end shopping goals. Training can be explored if it produces a meaningful result; training alone is not a novelty claim. The isolated [feature-model baseline](experiments/ranker/README.md) has real trained checkpoints and mixed synthetic results: its ranking score improves slightly after adaptation, while private-constraint failures exceed the heuristic. That baseline has no pretrained language ability. The separate [pretrained experiment](experiments/ranker/pretrained/README.md) tests a frozen quantized Qwen language backbone with locally trained decision heads; it is not integrated into the storefront, and its report must distinguish verified runs from pending work.

[Netflix's GenRec article](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) is inspiration for representing history/context, catalog-aware ranking, and weighing recommendation quality against serving cost. It is not evidence that our ecommerce approach works, and we are not claiming to reproduce Netflix's data, model, architecture, or results.

## Positioning, credibility, and scope

[Shopify SimGym](https://help.shopify.com/en/manual/online-store/simgym) already uses AI shoppers to simulate store journeys, compare themes, and provide behavior feedback and recommendations. Our intended emphasis is a visible, integrated propose–edit–retest loop and testable personalization within that loop. This is a product direction to demonstrate and validate, not a claim that nobody else simulates shoppers or supports iteration.

The central research risk is simulator credibility. Agents may be unusually patient, follow prompts too literally, share the optimizer's biases, or exploit fixture shortcuts. An optimizer may learn to please the simulator without helping people. Use varied and adversarial cases, held-out evaluation, repeated runs, independent checks, and manual trace review to investigate those weaknesses. Later real-user research and properly designed merchant A/B tests are needed to validate business impact.

The demo scope is a merchant dashboard, local storefront, controlled shopper runs, inspectable evidence, constrained candidate changes, and a measured comparison. Optional personalization belongs inside that flow. Catalog import, production integrations, deployment, authentication, distributed infrastructure, and broader commerce operations are later decisions. No real payments, purchases, checkout submissions, or Visa API integration are currently implemented or required for the demo. Arbitrary generated-code execution, invented product claims, and changes that hide fees are out of scope.

## Demo success criteria

- A teammate can start the local services and run the key demo without secrets; mock behavior is visibly labeled.
- A merchant can follow one baseline journey from its task through the recorded actions to a checked outcome, including a useful failure case.
- At least one evidence-linked factual proposal becomes an inspectable candidate configuration and is visibly retested. This full workflow remains planned beyond the scaffold's current suggestions and seeded variants.
- The final comparison uses a frozen candidate and paired held-out cases, reports regressions and operational failures, and links aggregate results back to traces.
- A successful fake checkout is counted as goal success only when independent requirement checks pass. The demo cannot “win” by exceeding budgets or selecting unsuitable products.
- If personalization is included, the no-history/history comparison is controlled and clearly separate from any optional post-training experiment.
- The team can state what the result supports, what is synthetic, and what remains unvalidated. A credible negative result is preferable to an unsupported lift claim.

## Decisions still open

1. Which merchant segment, catalog, and shopping tasks make the most convincing first demonstration?
2. Which browser policy and model provider fit the interaction quality, latency, reproducibility, and cost targets?
3. What persona schema, patience model, and independent success predicates are sufficient for believable variation?
4. Which single primary outcome and constraint guardrails will govern candidate selection?
5. Which factual edits should the optimizer support first, and how should merchant review/application work?
6. Does personalization add enough value to the main demo, and is there a suitable permissioned or explicitly synthetic history dataset?
7. Is there evidence that post-training would add more than better context/ranking design, and what held-out evaluation would establish that?
8. What real-user observations or future A/B test would distinguish simulator improvement from improvement for actual shoppers?

These decisions should be recorded with the experiment configuration as they are made. They are not settled by the current mocks, scaffold defaults, or this vision document.
